"""
This module manipulates an LSTM model.
"""

import json
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json, load_model
import pandas as pd
import numpy as np

def build_model(data, hnodes=32, dropout=0.2, learning_rate=0.001) -> dict:
  # Load data into dataframe
  df = pd.DataFrame.from_dict(data)
  print("DataFrame head:")
  print(df.head())

  # Add a '1' as the third dimension of the DataFrame shape if it's not there
  if len(df.shape) != 3:
    df = np.expand_dims(df, axis=2)

  print("DataFrame shape:")
  print(df.shape)

  # Gets size of window from length of array in dataframe row 0 col 0
  window_size = len(df[0][0][0])

  model = keras.Sequential()

  # Adding one LSTM layer
  model.add(keras.layers.LSTM(
    units=hnodes,
    input_shape=(window_size, df.shape[2])
  ))
  
  # Adding Dropuut
  model.add(keras.layers.Dropout(dropout))
  
  # Adding a Dense layer at the end
  model.add(keras.layers.Dense(units=1))

  # Compile model using MSE as loss function to minimize, and Adam as optimiser
  model.compile(
    loss='mean_squared_error',
    optimizer=keras.optimizers.Adam(learning_rate)
  )

  return model

def train_model(model: Sequential, data: dict, nepochs=30, batchsize=32, valsplit=0.1, learning_rate=0.001):
  windows = np.array(data["windows"])
  next_vals = np.array(data["next_vals"])

  if len(windows.shape) == 2:
    windows = windows.reshape(windows.shape + (1,))
  
  # model.compile(
  #   loss='mean_squared_error',
  #   optimizer=keras.optimizers.Adam(learning_rate)
  # )

  fithistory = model.fit(
    windows, next_vals,
    epochs=nepochs,
    batch_size=batchsize,
    validation_split=valsplit,
    verbose=1,
    shuffle=False
  )

  return fithistory.history, model

def save_model_h5(model: Sequential, filename: str) -> None:
  model.save(filename)

def load_model_h5(filename: str) -> Sequential:
  model = keras.models.load_model(filename, compile=False)
  return model

def save_loss_history(history, filename: str) -> None:
  history_dict = {}
  history_dict["history"] = history
  history_dict["schema"] = "orquestra-v1-loss-function-history"
  with open(filename, "w") as f:
    f.write(json.dumps(history_dict, indent=2))

def save_model_json(model: Sequential, filename: str) -> None:
  model_dict = {"model":{}}

  model_json = model.to_json()
  model_dict["model"]["specs"] = json.loads(model_json)

  weights = model.get_weights()
  weights = nested_arrays_to_lists(weights)
  model_dict["model"]["weights"] = weights

  model_dict["schema"] = "orquestra-v1-model"

  with open(filename, "w") as f:
    f.write(json.dumps(model_dict, indent=2))

def nested_arrays_to_lists(obj):
  if isinstance(obj, np.ndarray):
    obj = obj.tolist()

  try:
    for i in range(len(obj)):
      obj[i] = nested_arrays_to_lists(obj[i])
  except TypeError:
    return obj

  return obj

def nested_lists_to_arrays(obj):
  if isinstance(obj, list):
    obj = np.array(obj)

  try:
    for i in range(len(obj)):
      obj[i] = nested_lists_to_arrays(obj[i])
  except TypeError:
    return obj

  return obj

def load_model_json(filename: str) -> Sequential:
  # load json and create model
  with open(filename) as json_file:
    loaded_model_artifact = json.load(json_file)

  loaded_model = json.dumps(loaded_model_artifact["model"]["specs"])

  loaded_model = model_from_json(loaded_model)
  
  weights = loaded_model_artifact["model"]["weights"]
  # Everything below the top-level list needs to be converted to a numpy array
  for i in range(len(weights)):
    weights[i] = nested_lists_to_arrays(weights[i])
  loaded_model.set_weights(weights)

  return loaded_model
