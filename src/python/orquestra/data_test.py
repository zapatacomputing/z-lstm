import unittest
from data import *
import json
import numpy as np
import pandas as pd
from pathlib import Path
import os.path

class TestDataGeneration(unittest.TestCase):
  def test_noisy_sine_generation(self):
    time_range = 2
    time_step = 0.1
    noise_std = 0.5
    data_dict = noisy_sine_generation(time_range, time_step, noise_std)
    
    self.assertEqual(len(data_dict["data"]["time"]), time_range/time_step)
    self.assertEqual(len(data_dict["data"]["values"]), time_range/time_step)
    last_index = list(data_dict["data"]["time"].keys())[-1]
    self.assertAlmostEqual(data_dict["data"]["time"][last_index], time_range-time_step)

  def test_preprocess_data(self):
    
    train_perc = 0.8
    window_size = 10

    test_file_path = Path("test/test_data.json")
    with open(test_file_path) as test_file:
      test_data = json.load(test_file)

    data = preprocess_data(test_data, train_perc, window_size)
    train_dict = data[0]
    test_dict = data[1]
    train_window_dict = data[2]
    test_window_dict = data[3]

    self.assertEqual(len(train_dict["data"]["time"]), 400)
    self.assertEqual(len(test_dict["data"]["time"]), 100)
    self.assertEqual(len(train_window_dict["data"]["windows"]), 390)
    self.assertEqual(len(test_window_dict["data"]["next_vals"]), 90)
    
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

    data = load_data("test/test_data_artifact.json")

    self.assertDictEqual(data, expected_data)

if __name__ == '__main__':
  unittest.main()

