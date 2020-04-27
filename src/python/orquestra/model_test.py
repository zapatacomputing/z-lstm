from model import *
import unittest
import json
import os
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json, load_model
import numpy as np

class TestModel(unittest.TestCase):
  def test_build_model(self):
    # Get test data to build the model to fit the dimensions of
    test_data_file_path = Path("test/test_window_data.json")
    with open(test_data_file_path) as test_data_file:
      test_data = json.load(test_data_file)

    # Build the model
    model = build_model(test_data)

    # Get expected model out of file
    expected_model_file_path = Path("test/test_untrained_model_specs.json")
    with open(expected_model_file_path) as expected_model_file:
      expected_model_json = expected_model_file.read()

    expected_model = model_from_json(expected_model_json)

    self.assertEqual(model.to_json(), expected_model.to_json())
    window_size = len(test_data["windows"][0])
    self.assertEqual(model.layers[0].input_shape, (None, window_size, 1))

  def test_train_model(self):
    test_data_file_path = Path("test/test_window_data.json")
    with open(test_data_file_path) as test_data_file:
      test_data = json.load(test_data_file)

    # Make simple model
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
      units=2,
      input_shape=(10, 1)
    ))
    model.add(keras.layers.Dropout(0.2))
    model.add(keras.layers.Dense(units=1))
    model.compile(
      loss='mean_squared_error',
      optimizer=keras.optimizers.Adam(0.001)
    )

    old_weights = model.get_weights()
    history, model = train_model(model, test_data, nepochs=1, batchsize=1, valsplit=0, learning_rate=0.001)
    new_weights = model.get_weights()
    
    expected_model_file_path = Path("test/test_trained_model_specs.json")
    with open(expected_model_file_path) as expected_model_file:
      expected_model_json = expected_model_file.read()

    expected_model = model_from_json(expected_model_json)
    self.assertEqual(model.to_json(), expected_model.to_json())

    self.assertEqual(len(old_weights), len(new_weights))
    for i in range(len(old_weights)):
      self.assertTrue((old_weights[i] != new_weights[i]).any())

  def test_predict(self):
    np.random.seed(1)
    test_data_file_path = Path("test/test_window_data.json")
    with open(test_data_file_path) as test_data_file:
      test_data = json.load(test_data_file)

    # Make simple model
    model = keras.Sequential()
    model.add(keras.layers.Dense(units=1,
                kernel_initializer='ones',
                bias_initializer='zeros'))
    model.compile(
      loss='mean_squared_error',
      optimizer=keras.optimizers.Adam(0.001)
    )

    predictions = predict(model, test_data)

    print(predictions)
    expected_predictions = {'data': [[3.3053510189056396], [3.0758495330810547], [1.956718921661377]]}

    self.assertTrue(isinstance(predictions, dict))
    self.assertTrue(isinstance(predictions["data"], list))
    self.assertTrue(isinstance(predictions["data"][0], list))
    self.assertDictEqual(predictions, expected_predictions)

  def test_save_model_h5(self):
    model = keras.Sequential()
    model.add(keras.layers.Dense(
      units=1,
      input_shape=(1,1)
    ))
    model.compile(
      loss='mean_squared_error',
      optimizer=keras.optimizers.Adam(0.001)
    )
    
    test_file_name = "tmp.h5"

    save_model_h5(model, test_file_name)

    saved_model = keras.models.load_model(test_file_name)

    for i in range(len(saved_model.get_weights())):
      self.assertTrue((saved_model.get_weights()[i] == model.get_weights()[i]).all())
  
    try:
      os.remove(test_file_name)
    except OSError:
      pass

  def test_load_model_h5(self):
    model_file = Path("test/test_model.h5")
    model = load_model_h5(model_file)
    self.assertTrue(isinstance(model, Sequential))
    loaded_weights = model.get_weights()

    expected_weights = [
      np.array(
        [np.array(
           [1.0]
        ).astype('float32')]
      ),
      np.array(
          [2.0]
        ).astype('float32')
    ]

    for i in range(len(loaded_weights)):
      self.assertTrue((loaded_weights[i] == expected_weights[i]).all())

  def test_save_loss_history(self):
    history = {'loss': [0.3, 0.2, 0.1]}
    test_file_name = "tmp.json"

    save_loss_history(history, test_file_name)

    with open(test_file_name) as test_file:
      saved_history = json.load(test_file)

    history_dict = {
      "history": history,
      "schema": "orquestra-v1-loss-function-history"
    }

    self.assertEqual(history_dict, saved_history)

    try:
      os.remove(test_file_name)
    except OSError:
      pass

  # How to test this without just redoing code in save_model?
  def test_save_model_json(self):
    model = keras.Sequential()
    model.add(keras.layers.LSTM(
      units=1,
      input_shape=(1,1)
    ))
    
    test_file_name = "tmp.json"

    save_model_json(model, test_file_name)

    with open(test_file_name) as test_file:
      saved_model_artifact = json.load(test_file)

    expected_model_artifact = {
      "model": {
        "specs": json.loads(model.to_json()),
        "weights": nested_arrays_to_lists(model.get_weights())
      },
      "schema": "orquestra-v1-model"
    }

    self.assertEqual(expected_model_artifact, saved_model_artifact)

    try:
      os.remove(test_file_name)
    except OSError:
      pass

  def test_nested_arrays_to_lists(self):
    arr1 = np.array([1, 2])
    arr2 = np.array([3, 4])
    list_of_arrays = [arr1, arr2]

    converted = nested_arrays_to_lists(list_of_arrays)

    self.assertTrue(isinstance(converted, list))
    self.assertTrue(isinstance(converted[0], list))
    self.assertTrue(isinstance(converted[1], list))
    self.assertTrue(converted == [[1,2],[3,4]])

  def test_nested_lists_to_arrays(self):
    list1 = [1, 2]
    list2 = [3, 4]
    list_of_lists = [list1, list2]

    converted = nested_lists_to_arrays(list_of_lists)

    self.assertTrue(isinstance(converted, np.ndarray))
    self.assertTrue(isinstance(converted[0], np.ndarray))
    self.assertTrue(isinstance(converted[1], np.ndarray))
    expected = np.array([np.array([1,2]),np.array([3,4])])
    self.assertTrue((converted == expected).all())

  def test_load_model_json(self):
    model_file = Path("test/test_trained_model.json")
    model = load_model_json(model_file)
    self.assertTrue(isinstance(model, Sequential))
    loaded_weights = model.get_weights()

    expected_weights = [
      np.array(
        [np.array(
           [0.1, 0.2, 0.3, 0.4]
        ).astype('float32')]
      ),
      np.array(
        [np.array(
          [0.5, 0.6, 0.7, 0.8]
        ).astype('float32')]
      ),
      np.array(
        [0.9, 0.01, 0.11, 0.21]
      ).astype('float32'),
    ]

    for i in range(len(loaded_weights)):
      self.assertTrue((loaded_weights[i] == expected_weights[i]).all())

if __name__ == '__main__':
  unittest.main()