# NYC-taxi-price-prediction-framework

Modularised code for training a neural network to predict prices of taxi rides in NYC based on data from a Kaggle competition: https://www.kaggle.com/c/new-york-city-taxi-fare-prediction. It consists of a notebook orchiestrating the execution, NYC_taxi_fare_prediction.ipynb, and 3 Python modules: data_processing.py, training.py and tuning.py. Developed on Azure Databricks, the code leverages Spark's distributed computing capabilities, notably the Hyperopt library for parallel tuning trials, and PyTorch's IterableDataset for handling a large training dataset. Spark version 3.5.2 and Python version 3.12.3 were used. The following versions of libraries were also used:
- Hyperopt 0.2.8,
- PyTorch 2.2.2,
- FastAI 2.7.14,
- psutil 5.9.0.
