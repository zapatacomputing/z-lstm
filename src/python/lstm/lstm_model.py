"""
This builds a LSTM model.
"""

import json
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json, load_model, save_model
from tensorflow.keras import callbacks


def build_model(data:dict, hnodes:int=32, dropout:float=0.2) -> Sequential:
    """
      Builds LSTM model with an LSTM layer, dropout layer, and dense layer.

      Args:
        data (dict):
          A dict of data to use the input shape of to build the model.
          It must have two keys, each containing indexes as keys and data as 
          values (this is the dict format of Pandas DataFrames). Here is an 
          example:
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
        hnodes (int):
          The number of nodes in the LSTM layer.
        dropout (float):
          The fraction of the LSTM layer to be the dropout layer.

      Returns:
        model (keras.models.Sequential):
          The model that was built.
    """

    # Load data into dataframe
    df = pd.DataFrame.from_dict(data)
    print("DataFrame head:")
    print(df.head())

    # Add a '1' as the third dimension of the DataFrame shape if it's not there
    if len(df.shape) != 3:
        df = np.expand_dims(df, axis=2)
    print("Shape of input data: ", df.shape)

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

    # Adding Dropout
    model.add(keras.layers.Dropout(dropout))

    # Adding a Dense layer at the end
    model.add(keras.layers.Dense(units=1))

    return model


def train_model(model:Sequential, data:dict, nepochs:int=30, batchsize:int=32, valsplit:float=0.1, learning_rate:float=0.01) -> (callbacks, Sequential):
    """
    Trains input model using input data.

    Args:
      model (Sequential):
        The model to train.
      data (dict):
        The data to train the model on. This should be in a dict with keys 
        "windows and "next_vals", where "windows" contains a list of lookback 
        windows and "next_vals" contains a list of the corresponding next values.
      nepochs (int):
        The number of training epochs to perform.
      batchsize (int):
        The batch size for training.
      valsplit (float):
        The fraction of the data to use for validation during training.
      learning_rate (float):
        The learning rate for training.

    Returns:
      fithistory.history (keras.History.history):
        The keras history object from training.
      model (Sequential):
        The trained model.
    """
    try:
        windows = np.array(data["windows"])
        next_vals = np.array(data["next_vals"])
    except KeyError:
        print(f'Error: Could not load windows and next_vals from data.')

    # Add a '1' as the third dimension of the data shape if it's not there
    if len(windows.shape) == 2:
        windows = np.expand_dims(windows, axis=2)
    print("Shape of input windows: ", windows.shape)

    model.compile(
        loss='mean_squared_error',
      optimizer=keras.optimizers.Adam(learning_rate)
    )

    fithistory = model.fit(
        windows, next_vals,
      epochs=nepochs,
      batch_size=batchsize,
      validation_split=valsplit,
      verbose=1,
      shuffle=False
    )

    return fithistory.history, model


def predict(model:Sequential, data:dict) -> dict:
    """
    Makes predictions about input data using input model.

    Args:
      model (Sequential):
        The model to use for predictions.
      data (dict):
        The data to make predictions about. This should be in a dict with keys 
        "windows and "next_vals", where "windows" contains a list of lookback 
        windows and "next_vals" contains a list of the corresponding next values.

    Returns:
      pred_dict (dict):
        A dict with a list of predictions in the "data" field.
    """

    windows = np.array(data["windows"])

    # Add a '1' as the third dimension of the data shape if it's not there
    if len(windows.shape) == 2:
        windows = np.expand_dims(windows, axis=2)
    print("Shape of input windows: ", windows.shape)

    pred = model.predict(windows)

    # Save predictions to a JSON serializable format (a dict of a list)
    pred_dict = {}
    pred_dict["data"] = pred.tolist()

    return pred_dict


def save_model_json(model:Sequential, filename:str) -> None:
    """
    Saves a model's architecture and weights as a JSON file. The output JSON will 
    contain a "model" field with "specs" and "weights" fields inside it. The 
    "specs" field contains the model's architecture and the "weights" field
    contains the weights.

    Args:
      model (keras.models.Sequential):
        The model to save.
      filename (str):
        The name of the file to save the model in. This should have a '.json'
        extension.
    """

    model_dict = {"model":{}}

    model_json = model.to_json()
    model_dict["model"]["specs"] = json.loads(model_json)

    weights = model.get_weights()
    # Convert weight arrays to lists because those are JSON compatible
    weights = nested_arrays_to_lists(weights)
    model_dict["model"]["weights"] = weights

    model_dict["schema"] = "orquestra-v1-model"

    try:
        with open(filename, "w") as f:
            f.write(json.dumps(model_dict, indent=2))
    except IOError:
        print('Error: Could not load {filename}')


def load_model_json(filename:str) -> Sequential:
    """
    Loads a keras model from a JSON file.

    Args:
      filename (str):
        The JSON file to load the model from. This file must contain a "model" 
        field with "specs" and "weights" fields inside it. The "specs" field 
        should contain the model's architecture and the "weights" field should
        contain the weights. (The format saved by the `save_model_json` function.)

    Returns:
      model (Sequential):
        The model loaded from the file. 
    """

    # load json and create model
    with open(filename) as json_file:
        loaded_model_artifact = json.load(json_file)

    loaded_model = json.dumps(loaded_model_artifact["model"]["specs"])
    loaded_model = model_from_json(loaded_model)

    try:
        weights = loaded_model_artifact["model"]["weights"]
    except KeyError:
        print(f'Error: Could not load weights from {weights}')

    # Everything below the top-level list needs to be converted to a numpy array
    # because those are the types expected by `set_weights`
    for i in range(len(weights)):
        weights[i] = nested_lists_to_arrays(weights[i])
    loaded_model.set_weights(weights)

    return loaded_model


def nested_arrays_to_lists(obj):
    """
    Helper function for saving models in JSON format. Converts nested numpy 
    arrays to lists (lists are JSON compatible).
    """

    if isinstance(obj, np.ndarray):
        obj = obj.tolist()

    try:
        for i in range(len(obj)):
            obj[i] = nested_arrays_to_lists(obj[i])
    except TypeError:
        return obj

    return obj


def nested_lists_to_arrays(obj):
    """
    Helper function for loading models in JSON format. Converts nested lists to numpy arrays (numpy arrays are the expected type to set the weights 
    of a Keras model).
    """

    if isinstance(obj, list):
        obj = np.array(obj)

    try:
        for i in range(len(obj)):
            obj[i] = nested_lists_to_arrays(obj[i])
    except TypeError:
        return obj

    return obj


def save_model_h5(model:Sequential, filename:str) -> None:
    """
    Saves a complete model as an H5 file. H5 files can be used to pass models 
    between tasks but cannot be returned in a workflowresult.

    Args:
      model (keras.models.Sequential):
        The model to save.
      filename (str):
        The name of the file to save the model in. This should have a '.h5'
        extension.
    """
    keras.models.save_model(
        model, filename, include_optimizer=True
    )


def load_model_h5(filename:str) -> Sequential:
    """
    Loads a keras model from an H5 file. H5 files can be used to pass models 
    between tasks but cannot be returned in a workflowresult.

    Args:
      filename (str):
        The H5 file to load the model from. This should have the format created 
        by `keras.models.save_model`.

    Returns:
      model (Sequential):
        The model loaded from the file. 
    """

    model = keras.models.load_model(filename, compile=True)
    return model

def save_loss_history(history, filename:str) -> None:
    """
    Saves a keras.History.history object to a JSON file.

    Args:
      history (keras.History.history):
        The history object to save.
      filename (str):
        The name of the file to save the history in. This should have a '.json'
        extension.
    """

    history_dict = {}
    history_dict["history"] = history
    history_dict["schema"] = "orquestra-v1-loss-function-history"

    try:
        with open(filename, "w") as f:
            f.write(json.dumps(history_dict, indent=2))
    except IOError:
        print(f'Could not write to {filename}')
