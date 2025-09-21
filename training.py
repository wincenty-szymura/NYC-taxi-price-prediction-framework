# --------------------------------------------------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------------------------------------------------

from pyspark.sql import functions as F, SparkSession
import torch, torch.nn as nn
from torch.utils.data import IterableDataset, DataLoader
import gc, math

from fastai.data.core import DataLoaders
from fastai.tabular.all import Learner
from fastai.metrics import rmse

import data_processing as dpr

# --------------------------------------------------------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------------------------------------------------------

# spark session
spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

number_of_epochs = 3
batch_size = 32768
batch_size_ratio = 8 # batch_size / tuning_batch_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') # switch to GPU if available

# --------------------------------------------------------------------------------------------------------------------------------------------
# Classes
# --------------------------------------------------------------------------------------------------------------------------------------------

class SparkBatchDataset(IterableDataset):
    """Iterable dataset streaming batches from a temp view for main training."""

    def __init__(self, view, cat_cols, cont_cols, y_col, mean, stddev, batch_size, n_partitions = 1):
        self.view = view # dataset in the form of temp view
        self.cat_cols = cat_cols
        self.cont_cols = cont_cols
        self.y_col = y_col
        self.mean = mean # PyTorch tensor with means of continuous columns
        self.stddev = stddev # PyTorch tensor with standard deviations of continuous columns
        self.n_partitions = n_partitions # the dataset will be split into n_partitions for reading into memory
        self.batch_size = batch_size # mini-batch training
        self.total_rows = spark.table(view).count()

    def __len__(self):
        return self.total_rows

    def __iter__(self):            
        for i in range(self.n_partitions):
            if self.n_partitions == 1: # if dataset fits into memory no need to iterate
                pd_df = spark.table(self.view).toPandas() # load entire to pandas
            else:
                # if dataset too big to fit in memory, partition using modulo of integer ID (defined for train dataset only) and load iteratively
                pd_df = spark.table(self.view).where((F.col('monotonic_id') % self.n_partitions) == i).toPandas()
            cats = torch.tensor(pd_df[self.cat_cols].values, dtype = torch.int64) # convert categorical features to PyTorch tensor
            # convert continuous features to PyTorch tensor, standardize
            conts = (torch.tensor(pd_df[self.cont_cols].values, dtype = torch.float32) - self.mean) / self.stddev
            y = torch.tensor(pd_df[self.y_col].values, dtype = torch.float32).unsqueeze(1) # convert target to PyTorch tensor
            # yield batches within a partition
            for i in range(0, len(pd_df), self.batch_size):
                yield (cats[i:i+self.batch_size], conts[i:i+self.batch_size]), y[i:i+self.batch_size]

# --------------------------------------------------------------------------------------------------------------------------------------------

class TabularModel(nn.Module):
    """Embedding + MLP architecture for tabular data."""

    def __init__(
        self, 
        emb_sizes, # list of tuples with pairs of categorical feature size and the corresponding embedding size
        n_cont, # number of continuous features
        layers, # list with layer sizes
        dp, # dropout probability
        out_size = 1 # output size (1 for regression tasks)
    ): 
        super().__init__()
        self.embeds = nn.ModuleList([nn.Embedding(ni, nf) for ni, nf in emb_sizes]) # categorical features embedding layer
        self.emb_drop = nn.Dropout(dp) # embedding dropout layer
        self.bn_cont = nn.BatchNorm1d(n_cont) # continuous features batch normalization layer

        sizes = [sum(e for _, e in emb_sizes) + n_cont] + layers # add layer containing continuous features + embeddings as input layer
        seq = [] # layers sequence
        for i in range(len(layers)):
            seq += [
                nn.Linear(sizes[i], sizes[i+1]), # linear 
                nn.ReLU(inplace = True), # activation function
                nn.BatchNorm1d(sizes[i+1]), # batch normalization
                nn.Dropout(dp) # dropout
            ]
        seq.append(nn.Linear(sizes[-1], out_size)) # no output activation - handled by loss function
        self.layers = nn.Sequential(*seq)

    def forward(self, x_cat_cont):
        # unpack tuple passed from SparkBatchDataset 
        x_cat, x_cont = x_cat_cont # x_cat is a tensor of shape (batch_size, n_cat), x_cont is a tensor of shape (batch_size, n_cont)
        x = torch.cat([e(x_cat[:, i]) for i, e in enumerate(self.embeds)], 1) # concatenate embeddings of each categorical feature along dim 1
        x = self.emb_drop(x) # perform dropout on embeddings
        x = torch.cat([x, self.bn_cont(x_cont)], 1) # perform initial batch normalization of continuous features
        return self.layers(x) # perform the rest of the forward pass
    
# --------------------------------------------------------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------------------------------------------------------

def train_model(best, utilisation = 0.1, number_of_epochs = number_of_epochs):
    """Train the model."""

    emb_sizes = dpr.get_embedding_sizes()
    # get means and standard deviations of continuous features from temp view as PyTorch tensors
    mean, stddev = dpr.convert_mean_stddev_into_tensors()

    # get number of partitions - round up to get enough partitions to never exceed max rows within a partition
    n_partitions = math.ceil(spark.table('train_v').count() / dpr.get_max_rows(spark.table('train_v'), dpr.available_memory, utilisation))

    # define train and valid PyTorch dataloaders
    # automatic batching is disabled as batching is handled in the SparkBatchDataset class 
    # this means the data do not have the extra batch dimension (https://docs.pytorch.org/docs/stable/data.html)
    train_dl = DataLoader(SparkBatchDataset('train_v', dpr.cat_cols, dpr.cont_cols, dpr.y_col, mean, stddev, batch_size, n_partitions), batch_size = None)
    valid_dl = DataLoader(SparkBatchDataset('valid_v', dpr.cat_cols, dpr.cont_cols, dpr.y_col, mean, stddev, batch_size), batch_size = None)

    # wrap in FastAI dataloaders object
    dls = DataLoaders(train_dl, valid_dl, device = device)

    # get best architecture
    layers = [int(round(best['layer_1']))]
    if int(round(best['n_layers'])) >= 2:
        layers.append(int(round(best['layer_2'])))
    if int(round(best['n_layers'])) == 3: 
        layers.append(int(round(best['layer_3'])))

    # create instance of TabularModel
    model = TabularModel(emb_sizes, len(dpr.cont_cols), layers, float(best['dp'])) 

    # create instance of FastAI's learner
    learn = Learner(
        dls, 
        model,
        loss_func = nn.SmoothL1Loss(), # balances MSE & MAE, robust to outliers
        metrics = rmse, # classical metric for regression tasks
        wd = float(best['wd']) # weight decay
    )

    adjusted_lr = batch_size_ratio*float(best['lr']) # lr is roughly linearly scaled by batch size
    learn.fit_one_cycle(number_of_epochs, lr_max = adjusted_lr)

    # run in eval mode and get the metric value
    rmse_val = float(learn.validate()[1])

    # free resources
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return learn, rmse_val

# --------------------------------------------------------------------------------------------------------------------------------------------

def save_model(learn, name):
    """Save trained model."""

    learn.save(name)