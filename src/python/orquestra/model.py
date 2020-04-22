"""
This module manipulates an LSTM model.
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json
import pandas as pd
import numpy as np
import json

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

# def train_model(model, data) -> dict:
#   historyfitting = model.fit(
#     xtrain, ytrain,
#     epochs=nepochs,
#     batch_size=batchsize,
#     validation_split=valsplit,
#     verbose=1,
#     shuffle=False
#   )

#   return historyfitting

def save_model(model: Sequential, filename: str) -> None:
  model_json = model.to_json()
  model_dict = {}
  model_dict["model"] = json.loads(model_json) 
  model_dict["schema"] = "orquestra-v1-model"

  with open(filename, "w") as f:
    f.write(json.dumps(model_dict, indent=2))

def load_model(filename: str) -> Sequential:
  # load json and create model
  with open(filename) as json_file:
    loaded_model_json = json_file.read()

  loaded_model = model_from_json(loaded_model_json)
  return loaded_model