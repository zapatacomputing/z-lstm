"""
This module manipulates data.
"""

import sys
import numpy as np
import pandas as pd
import json
from typing import TextIO

def noisy_sine_generation(time_range:float, time_step:float, noise_std:float) -> dict:
  """
  Generates noisy sine data.

  Args:
    time_range (float):
      The upper limit of the time range for the data to generate. The time
      range starts at 0. The time_range is not included as the last point.
    time_step (float):
      The step between each of the time values.
    noise_std (float):
      The standard deviation of the noise. Noise follows a normal distribution
      centered at zero.

  Returns:
    data_dict (dict):
      A dict containing a dict representation of a Pandas dataframe within its
      "data" field.
  """

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

def preprocess_data(data:dict, train_frac:float=0.8, window_size:int=10) -> (dict, dict, dict, dict):
  """
  Preprocesses data into a format suitable for training a model, splits it 
  into training and testing sets, and creates datasets of lookback windows and 
  next values.
  
  Args:
    data (dict):
      A dict with two keys, each containing indexes as keys and data as values 
      (this is the dict format of Pandas DataFrames). Here is an example:
      {
        "x": {
          "0": 0.0,
          "1": 0.1
        },
        "y": {
          "0": 1.0,
          "1": 2.0
        }
      }
    train_frac (float):
      The fraction of the data to use for training. The remaining data will be
      returned as testing data.
    window_size (int):
      The number of data values in the rolling lookback window.

  Returns:
    train_dict (dict):
      A dict with a Pandas DataFrame of the data for training in input format 
      inside its "data" field.
    test_dict (dict):
      A dict with a Pandas DataFrame of the data for testing in input format 
      inside its "data" field.
    train_window_dict (dict):
      A dict of the data for training in the "data" field, with a list of 
      lookback windows in the "windows" field and a list of the corresponding 
      next values in the "next_vals" field.
    test_window_dict (dict):
      A dict of the data for testing in the "data" field, with a list of 
      lookback windows in the "windows" field and a list of the corresponding 
      next values in the "next_vals" field.
  """

  # Load data into dataframe
  df = pd.DataFrame.from_dict(data)
  print("DataFrame head:")
  print(df.head())
  
  dfsize = df.shape[0]

  # Splitting up dataset into Training and Testing datsets
  train_size = int(dfsize * train_frac)
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

def create_dataset(x: pd.Series, y: pd.Series, window_size:int=1) -> (np.ndarray, np.ndarray):
  """
  A helper function of `preprocess_data` to split data into lookback windows 
  and next values.

  Args:
    x (pd.Series):
      The data to make the lookback windows from
    y (pd.Series):
      The data to get the next values from
    window_size (int):
      The size of the lookback window.

  Returns:
    np.array(xs) (numpy.ndarray):
      An array of lookback windows.
    np.array(ys) (numpy.ndarray):
      An array of corresponding next values.
  """

  xs, ys = [], []

  # Create pairs of a window of data and the next value after the window
  for i in range(len(x) - window_size):
      v = x.iloc[i:(i + window_size)].values
      xs.append(v)
      ys.append(y.iloc[i + window_size])

  return np.array(xs), np.array(ys)

def save_data(datas:list, filenames:list) -> None:
  """
  Saves data as JSON.

  Args:
    datas (list):
      A list of dicts of data to save.
    filenames (list):
      A list of filenames corresponding to the data dicts to save the data in. 
      These should have a '.json' extension.
  """

  for i in range(len(datas)):
    data = datas[i]
    filename = filenames[i]

    data["schema"] = "orquestra-v1-data"

    with open(filename,'w') as f:
      f.write(json.dumps(data, indent=2)) # Write data to file as this will serve as output artifact

def load_data(file:TextIO) -> dict:
  """
  Loads data from JSON.

  Args:
    file (TextIO):
      The file to load the data from.
    
  Returns:
    data (dict):
      The data that was loaded from the file.
  """

  if isinstance(file, str):
    with open(file, 'r') as f:
      data = json.load(f)
  else:
    data = json.load(file)

  return data