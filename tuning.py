# --------------------------------------------------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------------------------------------------------

from pyspark.sql import SparkSession
import torch, torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import gc, math
from hyperopt import fmin, tpe, hp, Trials, SparkTrials, space_eval # allows to distribute tuning trials over multiple workers

from fastai.data.core import DataLoaders
from fastai.tabular.all import Learner
from fastai.metrics import rmse

import data_processing as dpr
import training as tr

# --------------------------------------------------------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------------------------------------------------------

# spark session
spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

tuning_batch_size = math.floor(tr.batch_size / tr.batch_size_ratio) # smaller batch size for a smaller dataset

# minimum and maximum layer widths for tuning (rule of thumb)
min_layer_size = 8
max_layer_size = 64

# hyperparameter space to search over
hyperparam_space = {
    'lr': hp.loguniform('lr', math.log(1e-4), math.log(1e-1)), # log scale to focus on order-of-magnitude differences
    'wd': hp.loguniform('wd', math.log(1e-4), math.log(3e-1)), # and to avoid disfavouring small values
    'dp': hp.uniform('dp', 1e-3, 0.5),
    'n_layers': hp.choice('n_layers', [1, 2, 3]),
    'layer_1': hp.qloguniform('layer_1', math.log(min_layer_size), math.log(max_layer_size), 1), # rounded to nearest integer
    'layer_2': hp.qloguniform('layer_2', math.log(min_layer_size), math.log(max_layer_size), 1), # by using 
    'layer_3': hp.qloguniform('layer_3', math.log(min_layer_size), math.log(max_layer_size), 1), # qloguniform with q = 1
}

# --------------------------------------------------------------------------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------------------------------------------------------------------------

class TupledTensorDataset(Dataset):
    """Custom dataset to match the output of SparkBatchDataset for fast hyperparam tuning."""

    def __init__(self, cats, conts, y):
        self.cats, self.conts, self.y = cats, conts, y

    def __len__(self):
        return self.y.shape[0]
    
    def __getitem__(self, i):
        # tuple of categorical and continuous features + target
        return (self.cats[i], self.conts[i]), self.y[i]

# --------------------------------------------------------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------------------------------------------------------

def build_tuning_dls(pd_df, mean, stddev, batch_size):
    """Build dataloaders for hyperparameter tuning."""

    # train valid split
    idx = math.floor(len(pd_df) * 0.8) # 80% for training
    shuffled_df = pd_df.sample(frac = 1, random_state = 42).reset_index(drop = True)
    train_df, valid_df = shuffled_df.iloc[:idx], shuffled_df.iloc[idx:]

    def to_ds(df): # converts to PyTorch dataset with format matching main training dataset
        cats = torch.tensor(df[dpr.cat_cols].values, dtype = torch.int64)
        conts = (torch.tensor(df[dpr.cont_cols].values, dtype = torch.float32) - mean) / stddev
        y = torch.tensor(df[dpr.y_col].values, dtype = torch.float32).unsqueeze(1)
        return TupledTensorDataset(cats, conts, y)

    # define train and valid PyTorch dataloaders
    train_dl = DataLoader(to_ds(train_df), batch_size = batch_size, shuffle = True)
    valid_dl = DataLoader(to_ds(valid_df), batch_size = batch_size, shuffle = False)

    # wrap in FastAI dataloaders object
    return DataLoaders(train_dl, valid_dl, device = tr.device)
    
# --------------------------------------------------------------------------------------------------------------------------------------------

def objective(params):
    """Objective function to be minimized."""

    # unpack learning rate, weight decay and dropout
    lr, wd, dp = float(params['lr']), float(params['wd']), float(params['dp'])

    # define architecture
    n_layers = int(params['n_layers'])
    layers = [int(params['layer_1'])]
    if n_layers >= 2:
        layers.append(int(params['layer_2']))
    if n_layers == 3:
        layers.append(int(params['layer_3']))

    emb_sizes = dpr.get_embedding_sizes()
    # get sample for tuning from temp view 
    sample_for_tuning = spark.table('sample_v').toPandas()
    # get means and standard deviations of continuous features from temp view as PyTorch tensors
    mean, stddev = dpr.convert_mean_stddev_into_tensors()
    # build a dataloaders object
    tuning_dls = build_tuning_dls(sample_for_tuning, mean, stddev, tuning_batch_size)

    # create instance of TabularModel
    model = tr.TabularModel(emb_sizes, len(dpr.cont_cols), layers, float(dp))
    # create instance of FastAI's learner
    learn = Learner(tuning_dls, model, loss_func = torch.nn.SmoothL1Loss(), metrics = rmse, wd = float(wd))

    # train
    with learn.no_bar(), learn.no_logging():
        learn.fit_one_cycle(tr.number_of_epochs, lr_max = float(lr))

    # run in eval mode and get the metric value
    rmse_val = float(learn.validate()[1])

    # free resources
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # return positive RMSE as Hyperopt miminizes
    return rmse_val
    
# --------------------------------------------------------------------------------------------------------------------------------------------

def tune_hyperparameters(max_evals, parallelism):
    """Run bayesian optimization to determine best hyperparameters."""

    # if parallelism > 1, use SparkTrials to distribute trials across worker nodes
    # otherwise use non-distributed Trials
    trials = SparkTrials(parallelism = parallelism) if parallelism > 1 else Trials()

    # run the optimization
    best = fmin(
        fn = objective, # function to minimize
        space = hyperparam_space, # hyperparameter space
        algo = tpe.suggest, # Tree-structured Parzen Estimator (Bayesian-ish - first 20 trials are random)
        max_evals = max_evals, # number of trials to perform (random + TPE-guided)
        trials = trials,
    )

    # return values mapped onto input parameter names
    return space_eval(hyperparam_space, best)