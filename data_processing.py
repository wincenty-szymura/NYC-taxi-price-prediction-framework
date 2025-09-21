# --------------------------------------------------------------------------------------------------------------------------------------------
# Imports
# --------------------------------------------------------------------------------------------------------------------------------------------

from pyspark.sql import functions as F, SparkSession
import torch
import psutil

# --------------------------------------------------------------------------------------------------------------------------------------------
# Parameters
# --------------------------------------------------------------------------------------------------------------------------------------------

# spark session
spark = SparkSession.getActiveSession() or SparkSession.builder.getOrCreate()

# an estimate of driver memory
available_memory = psutil.virtual_memory().available

# columns in the dataset
cat_cols = ['hour', 'am_pm', 'weekday']
cont_cols = ['pickup_latitude', 'pickup_longitude', 'dropoff_latitude', 'dropoff_longitude', 'passenger_count', 'distance_km']
y_col = 'fare_amount'
cols = [y_col] + cat_cols + cont_cols

# mean Earth's radius in km
r_earth = 6371.0

# --------------------------------------------------------------------------------------------------------------------------------------------
# Functions
# --------------------------------------------------------------------------------------------------------------------------------------------

def compute_mean_stddev():
    """Compute means and standard deviations of continuous columns."""

    stats_df = (spark.table('train_v')
                    # compute agg df with mean and standard deviation values
                    .agg(
                        *[F.mean(c).alias(f'{c}_mean') for c in cont_cols],
                        *[F.stddev(c).alias(f'{c}_stddev') for c in cont_cols]
                    )
    )

    # define temp view which can be referenced between modules
    stats_df.createOrReplaceTempView('stats_v')

# --------------------------------------------------------------------------------------------------------------------------------------------

def convert_mean_stddev_into_tensors():
    """Convert mean and standard deviation dataframe into tensors."""

    # get first (and only) row and turn it into dictionary
    stats_dict = spark.table('stats_v').first().asDict() 

    # convert into PyTorch tensors
    mean_tensor = torch.tensor([stats_dict[f'{c}_mean']  for c in cont_cols], dtype = torch.float32)
    stddev_tensor = torch.tensor([stats_dict[f'{c}_stddev'] for c in cont_cols], dtype = torch.float32).clamp_min(1e-5) # avoid division by zero

    return mean_tensor, stddev_tensor

# --------------------------------------------------------------------------------------------------------------------------------------------

def clean_and_split(path, train_fraction, chronological_split):
    """Perform data cleaning and feature engineering, and split dataset into train and valid sets."""

    df = spark.read.csv(path, header = True, inferSchema = True)

    df = (df
            # haversine distance 
            .withColumn(
                'distance_km', 
                F.lit(2) * F.lit(r_earth) * F.asin(F.sqrt(
                    F.pow(F.sin((F.radians(F.col('dropoff_latitude') - F.col('pickup_latitude')))/2), 2) + 
                    F.cos(F.radians('pickup_latitude')) * F.cos(F.radians('dropoff_latitude')) *
                    F.pow(F.sin((F.radians(F.col('dropoff_longitude') - F.col('pickup_longitude')))/2), 2)
                ))
            )
    )

    # compute 99th quantile of distance
    upper_distance_limit = df.approxQuantile('distance_km', [0.99], 0.01)[0]

    df = (df
            .where(
                # realistic latitudes and longitudes
                (F.col('pickup_latitude').between(40.0, 42.0)) &
                (F.col('dropoff_latitude').between(40.0, 42.0)) &
                (F.col('pickup_longitude').between(-75.0, -72.0)) &
                (F.col('dropoff_longitude').between(-75.0, -72.0)) &
                # realistic passenger count and fare amount
                (F.col('passenger_count').between(1, 6)) &
                (F.col('fare_amount').between(2.5, 200.0)) & 
                # very short / long trips can be error entries
                F.col('distance_km').between(0.01, F.lit(upper_distance_limit))          
            )
            # Eastern Daylight Time (NYC in April)
            .withColumn('edt_pickup_datetime', F.to_timestamp('pickup_datetime') - F.expr('INTERVAL 4 HOURS'))
            .withColumn('hour', F.hour('edt_pickup_datetime')) # from 0 to 23
            .withColumn('am_pm', F.when(F.col('hour') < 12, 0).otherwise(1)) # from 0 to 1
            .withColumn('weekday', F.dayofweek('edt_pickup_datetime') - 1) # from 0 to 6
    )

    # perform train valid split (randomly)
    if not chronological_split:
        train_df, valid_df = df.randomSplit([train_fraction, 1.0 - train_fraction], seed = 42)
    # alternatively perform chronological split
    else:
        df = df.withColumn('seconds_count', F.col('edt_pickup_datetime').cast('long'))
        threshold = df.approxQuantile('seconds_count', [train_fraction], 0.01)[0]
        train_df = df.where(F.col('seconds_count') < threshold)
        valid_df = df.where(F.col('seconds_count') >= threshold)
        train_df = train_df.drop(F.col('seconds_count'))
        valid_df = valid_df.drop(F.col('seconds_count'))

    # add monotonic ID for partitioning train dataset
    train_df = train_df.withColumn('monotonic_id', F.monotonically_increasing_id())

    # define temp views which can be referenced between modules
    train_df.select(*(cols + ['monotonic_id'])).createOrReplaceTempView('train_v')
    valid_df.select(*cols).createOrReplaceTempView('valid_v')

# --------------------------------------------------------------------------------------------------------------------------------------------

def get_max_rows(df, available_memory, utilisation):
    """Compute maximum number of rows that can be loaded into memory."""

    probe = df.sample(fraction = min(1.0, 1000/df.count()), seed = 42).toPandas() # sample to probe memory usage
    memory_per_row = probe.memory_usage(deep = True).sum() / len(probe) # in bytes
    return utilisation * available_memory / memory_per_row

# --------------------------------------------------------------------------------------------------------------------------------------------

def prepare_sample_for_tuning():
    """Prepare a sample of the dataset for hyperparameter tuning."""

    train_df = spark.table('train_v') # get train df from temp view
    rows_for_tuning = min(get_max_rows(train_df, available_memory, 0.1), 100000)
    sample_df = train_df.sample(fraction = min(1.0, rows_for_tuning/train_df.count()), seed = 42) # prepare the sample
    sample_df.createOrReplaceTempView('sample_v') # define temp view which can be referenced between modules

# --------------------------------------------------------------------------------------------------------------------------------------------

def get_embedding_sizes():
    """Compute embedding sizes for all categorical features."""

    # get unique value count of each categorical feature (each takes integer values starting from 0)
    n_unique_per_cat = [spark.table('train_v').agg(F.max(c)).first()[0] + 1 for c in cat_cols] 
    # get embedding sizes
    return [(n, min(50, (n + 1) // 2)) for n in n_unique_per_cat] # rule of thumb says embedding size should be around half the unique value count but not more than 50