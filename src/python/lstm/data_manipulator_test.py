################################################################################
# © Copyright 2020 Zapata Computing Inc.
################################################################################
"""
Copyright Zapata Computing, Inc. All rights reserved.
"""

import json
import unittest
import os.path
import numpy as np
import pandas as pd

from .data_manipulator import *
from pathlib import Path


class TestData(unittest.TestCase):
    def test_noisy_sine_generation(self):
        time_range = 10
        time_step = 0.1
        noise_std = 0.5

        data_dict = noisy_sine_generation(time_range, time_step, noise_std)

        # The number of data points should be within 1 of the range divided by the step
        self.assertAlmostEqual(len(data_dict["data"]["time"]), time_range/time_step, delta=1)
        self.assertAlmostEqual(len(data_dict["data"]["values"]), time_range/time_step, delta=1)
        # The last time value should be within one step of the end of the range
        last_index = list(data_dict["data"]["time"].keys())[-1]
        self.assertAlmostEqual(data_dict["data"]["time"][last_index], time_range-time_step, delta=time_step)

    def test_preprocess_data(self):
        train_frac = 0.8
        window_size = 10

        cd = os.path.dirname(os.path.realpath(__file__))
        test_file_path = Path(cd + "/test/test_data.json")
        with open(test_file_path) as test_file:
            test_data = json.load(test_file)

        data_length = len(test_data["time"])

        data = preprocess_data(test_data, train_frac, window_size)
        train_dict = data[0]
        test_dict = data[1]
        train_window_dict = data[2]
        test_window_dict = data[3]

        # The length of the training data should be equal to fraction train_frac of
        # the total data set, rounded down
        self.assertEqual(len(train_dict["data"]["time"]),
          int(data_length * train_frac))
        # The length of the testing data should be equal to the remainder of the above
        self.assertEqual(len(test_dict["data"]["time"]),
          data_length - int(data_length * train_frac))
        # The length of the training data in window format should be the length of the
        # training data minus the window size
        self.assertEqual(len(train_window_dict["data"]["windows"]),
          int(data_length * train_frac) - window_size)
        # The length of the testing data in window format should be the length of the
        # testing data minus the window size
        self.assertEqual(len(test_window_dict["data"]["next_vals"]),
          data_length - int(data_length * train_frac) - window_size)

    def test_create_dataset(self):
        x = [1,2,3,4]
        y = [5,6,7,8]
        time_steps = 2

        df = pd.DataFrame(data=np.transpose([x, y]), columns=['x','y'])

        xx, yy = create_dataset(df.y, df.y, time_steps)

        self.assertTrue((xx == [[5,6],[6,7]]).all())
        self.assertTrue((yy == [7, 8]).all())

    def test_save_data(self):
        data_1 = {"data":
          {
              "time": {
                  "0": 0,
              "1": 1
            },
            "vals": {
                "0": 2.0,
              "1": 3.0
            }
          }
        }
        # The saved artifact should be the same but have a schema field
        expected_artifact_1 = data_1
        expected_artifact_1["schema"] = "orquestra-v1-data"

        data_2 = {"data":
          {
              "time": {
                  "0": 0,
              "1": 1
            },
            "vals": {
                "0": 4.0,
              "1": 5.0
            }
          }
        }
        # The saved artifact should be the same but have a schema field
        expected_artifact_2 = data_2
        expected_artifact_2["schema"] = "orquestra-v1-data"

        datas = [data_1, data_2]
        filenames = ['data_1.json','data_2.json']
        expected_artifacts = [expected_artifact_1, expected_artifact_2]
        save_data(datas, filenames)

        for i in range(len(filenames)):
            self.assertTrue(os.path.isfile(filenames[i]))

            with open(filenames[i], 'r') as f:
                artifact = json.load(f)

            self.assertDictEqual(artifact, expected_artifacts[i])

            try:
                os.remove(filenames[i])
            except OSError:
                pass

    def test_load_data(self):
        expected_data = {
            "data":
          {
              "time": {
                  "0": 0,
              "1": 1
            },
            "vals": {
                "0": 2.0,
              "1": 3.0
            }
          },
          "schema": "orquestra-v1-data"
        }

        cd = os.path.dirname(os.path.realpath(__file__))
        data = load_data(cd + "/test/test_data_artifact.json")

        self.assertDictEqual(data, expected_data)

if __name__ == '__main__':
    unittest.main()

