################################################################################
# © Copyright 2020 Zapata Computing Inc.
################################################################################
"""
Copyright Zapata Computing, Inc. All rights reserved.
"""

import os
import json
import unittest
import numpy as np
import tensorflow as tf

from .lstm_model import *
from pathlib import Path
from tensorflow import keras
from tensorflow.keras.models import Sequential, model_from_json, load_model


class TestModel(unittest.TestCase):
    def test_build_model(self):
        test_data = {
            "windows": [[1.0,2.0]],
            "next_vals": [3.0]
        }

        # Build the model
        model = build_model(test_data)

        # Get expected model out of file
        cd = os.path.dirname(os.path.realpath(__file__))
        expected_model_file_path = Path(cd + "/test/test_untrained_model_specs.json")
        with open(expected_model_file_path) as expected_model_file:
            expected_model_json = expected_model_file.read()

        self.assertTrue(model.to_json())

        # The window size should have been used to set the input shape of the first
        # layer
        window_size = len(test_data["windows"][0])
        self.assertEqual(model.layers[0].input_shape, (None, window_size, 1))

    def test_train_model(self):
        test_data = {
            "windows": [[1.0,2.0]],
            "next_vals": [3.0]
        }

        # Make simple model
        model = keras.Sequential(name='sequential_test')
        model.add(keras.layers.LSTM(
            units=2,
          input_shape=(len(test_data["windows"][0]), 1),
          name='lstm_test'
        ))
        model.add(keras.layers.Dense(units=1, name='dropout_test'))
        model.compile(
            loss='mean_squared_error',
          optimizer=keras.optimizers.Adam(0.001)
        )

        old_weights = model.get_weights()
        _, model = train_model(model, test_data, 
          nepochs=1, 
          batchsize=1, 
          valsplit=0, 
          learning_rate=0.001)
        new_weights = model.get_weights()

        cd = os.path.dirname(os.path.realpath(__file__))
        expected_model_file_path = Path(cd + "/test/test_trained_model_specs.json")
        with open(expected_model_file_path) as expected_model_file:
            expected_model_json = expected_model_file.read()

        self.assertTrue(model.to_json())

        self.assertEqual(len(old_weights), len(new_weights))

        # Training is probabilistic but we expect the weights to be changed by it
        for i in range(len(old_weights)):
            self.assertTrue((old_weights[i] != new_weights[i]).any())

    def test_predict(self):
        test_data = {
            "windows": [[1.0,2.0]],
          "next_vals": [3.0]
        }

        # Make simple model
        # Setting initializers to `zeroes` and `ones` for reproducibility
        model = keras.Sequential()
        model.add(keras.layers.LSTM(
            units=2,
          input_shape=(2, 1),
          kernel_initializer='ones',
          recurrent_initializer='ones',
          bias_initializer='zeros'
        ))
        model.add(keras.layers.Dense(
            units=1,
          kernel_initializer='ones',
          bias_initializer='zeros'))
        model.compile(
            loss='mean_squared_error',
          optimizer=keras.optimizers.Adam(0.001)
        )

        predictions = predict(model, test_data)

        expected_predictions = {'data': [[1.6918495893478394]]}

        self.assertTrue(isinstance(predictions, dict))
        self.assertTrue(isinstance(predictions["data"], list))
        self.assertTrue(isinstance(predictions["data"][0], list))
        for i in range(len(predictions["data"])):
            self.assertAlmostEqual(predictions["data"][i][0], expected_predictions["data"][i][0])

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

    def test_load_model_json(self):
        cd = os.path.dirname(os.path.realpath(__file__))
        model_file = Path(cd + "/test/test_model.json")
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

    def test_nested_arrays_to_lists(self):
        arr1 = np.array([np.array([1, 2]), np.array([3, 4])])
        arr2 = np.array([np.array([5, 6]), 7])
        list_of_arrays = [arr1, arr2]

        converted = nested_arrays_to_lists(list_of_arrays)

        self.assertTrue(isinstance(converted, list))
        self.assertTrue(isinstance(converted[0], list))
        self.assertTrue(isinstance(converted[1], list))
        self.assertTrue(isinstance(converted[0][0], list))
        self.assertTrue(isinstance(converted[0][1], list))
        self.assertTrue(isinstance(converted[1][0], list))

        self.assertTrue(converted == [[[1, 2], [3, 4]], [[5, 6], 7]])

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

        # This checks that a model can be loaded from the generated H5 file,
        # and allows us to check that the weights were saved correctly
        saved_model = keras.models.load_model(test_file_name)
        for i in range(len(saved_model.get_weights())):
            self.assertTrue((saved_model.get_weights()[i] == model.get_weights()[i]).all())

        try:
            os.remove(test_file_name)
        except OSError:
            pass

    def test_load_model_h5(self):
        cd = os.path.dirname(os.path.realpath(__file__))
        model_file = Path(cd + "/test/test_model.h5")
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

if __name__ == '__main__':
    unittest.main()
