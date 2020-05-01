"""
This module generates a set of data.
"""

import sys
import numpy as np
import pandas as pd
import json
from typing import TextIO

def noisy_sine_generation(time_range, time_step, noise_std) -> dict:
  print('time_range = ', time_range)
  print('time_step = ', time_step)
  print('noise_std = ', noise_std)

  data_dict = {}

  try:
    time = np.arange(0, time_range, time_step)
 
    # Generating data: sine function
    values = np.sin(time) + np.random.normal(scale=noise_std, size=len(time))
    print('Values shape from numpy: ', values.shape)
    
    # Making pandas DataFrame
    data_df = pd.DataFrame(data=np.transpose([time, values]), columns=['time','values'])

    print('Data shape from pandas:')
    print(data_df.shape)
    print('DataFrame head:')
    print(data_df.head())
    
    # Save data in dict for serialization into JSON
    data_dict["data"] = data_df.to_dict()
  except:
    e = sys.exc_info()[0]
    print(e)
    print('Something went wrong!')

  return data_dict

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

  print("Train and test set sizes: ", len(train), len(test))

  # Reshape to dimensions required by tensorflow: [samples, window_size, n_features]
  col = df.columns[1]
  train_windows, train_next_vals = create_dataset(train[col], train[col], window_size)
  test_windows, test_next_vals = create_dataset(test[col], test[col], window_size)

  # Save all 4 data sets to JSON serializable formats (dicts/lists)
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

def create_dataset(x, y, window_size=1):
  xs, ys = [], []

  # Create pairs of a window of data and the next value after the window
  for i in range(len(x) - window_size):
      v = x.iloc[i:(i + window_size)].values
      xs.append(v)
      ys.append(y.iloc[i + window_size])

  return np.array(xs), np.array(ys)

def save_data(datas: list, filenames: list) -> None:
  for i in range(len(datas)):
    data = datas[i]
    filename = filenames[i]

    data["schema"] = "orquestra-v1-data"

    with open(filename,'w') as f:
      f.write(json.dumps(data, indent=2)) # Write data to file as this will serve as output artifact


def load_data(file: TextIO) -> dict:
  if isinstance(file, str):
    with open(file, 'r') as f:
      data = json.load(f)
  else:
    data = json.load(file)

  return data