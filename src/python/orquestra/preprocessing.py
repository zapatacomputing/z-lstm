"""
This module preprocesses the data into windows.
"""
import json
import sys
import numpy as np
import pandas as pd

def create_dataset(df, window_size=1):
  col1 = df.columns[0]
  col2 = df.columns[1]

  x = df[col1]
  y = df[col2]

  xs, ys = [], []
  for i in range(len(x) - window_size):
      v = x.iloc[i:(i + window_size)].values
      xs.append(v)
      ys.append(y.iloc[i + window_size])

  return np.array(xs), np.array(ys)

def preprocess_data(data, train_perc = 0.8, window_size = 10):
  # Load data into dataframe
  df = pd.DataFrame.from_dict(data)
  print("DataFrame head:")
  print(df.head())
  dfsize = df.shape[0]

  # Splitting up dataset into Training and Testing datsets
  train_size = int(dfsize * train_perc)
  test_size = dfsize - train_size
  train, test = df.iloc[0:train_size], df.iloc[train_size:]

  print("Train and test set sizes")
  print(len(train), len(test))

  # Reshape to dimensions required by tensorflow: [samples, window_size, n_features]
  train_windows, train_next_vals = create_dataset(train, window_size)
  test_windows, test_next_vals = create_dataset(test, window_size)

  # Save all 4 data sets to dictionaries
  train_dict = {}
  train_dict["data"] = train.to_dict()

  test_dict = {}
  test_dict["data"] = test.to_dict()

  train_window_dict = {"data":{}}
  train_window_dict["data"]["windows"] = train_windows.tolist()
  train_window_dict["data"]["next_vals"] = train_next_vals.tolist()

  test_window_dict = {"data":{}}
  test_window_dict["data"]["windows"] = test_windows.tolist()
  test_window_dict["data"]["next_vals"] = test_next_vals.tolist()

  return train_dict, test_dict, train_window_dict, test_window_dict

