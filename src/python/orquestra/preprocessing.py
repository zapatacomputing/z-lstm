"""
This module preprocesses the data into windows.
"""
import json
import sys
import numpy as np
import pandas as pd

def preprocess_data(data, trainperc = 0.8, time_steps = 10):
  time = data["time"]
  values = data["values"]

  # Data preprocessing
  df = pd.DataFrame(dict(values=values), index=time, columns=['values'])
  dfsize = df.shape[0]

  # Splitting up dataset into Training and Testing datsets
  train_size = int(dfsize * trainperc)
  test_size = dfsize - train_size
  train, test = df.iloc[0:train_size], df.iloc[train_size:dfsize]

  print("Train and test set sizes")
  print(len(train), len(test))

  # Reshape to dimensions required by tensorflow: [samples, time_steps, n_features]
  xtrain, ytrain = create_dataset(train, train.sine, time_steps)
  xtest, ytest = create_dataset(test, test.sine, time_steps)

  print(xtrain.shape, ytrain.shape)

def create_dataset(x, y, time_steps=1):
  xs, ys = [], []
  for i in range(len(x) - time_steps):
      v = x.iloc[i:(i + time_steps)].values
      xs.append(v)
      ys.append(y.iloc[i + time_steps])
  return np.array(xs), np.array(ys)


