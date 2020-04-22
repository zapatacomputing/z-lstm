"""
This module manipulates an LSTM model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
import pandas as pd
import numpy as np

def build_model(data, hnodes=32, dropout=0.2, learning_rate=0.001) -> dict:
  # Load data into dataframe
  df = pd.DataFrame.from_dict(data)
  print("DataFrame head:")
  print(df.head())

  if len(df.shape) != 3:
    df = np.expand_dims(df, axis=2)

  print("DataFrame shape:")
  print(df.shape)

  model = keras.Sequential()

  # Adding one LSTM layer
  model.add(keras.layers.LSTM(
    units=hnodes,
    input_shape=(df.shape[1], df.shape[2])
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

def save_model(model: Sequential, filename: str) -> None:
  model_json = model.to_json()
  with open(filename, "w") as f:
    f.write(model_json)

def load_model(filename: str) -> Sequential:
  # load json and create model
  json_file = open(filename, 'r')
  loaded_model_json = json_file.read()
  json_file.close()
  loaded_model = model_from_json(loaded_model_json)
  return loaded_model