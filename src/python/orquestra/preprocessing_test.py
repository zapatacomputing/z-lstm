import unittest
from preprocessing import preprocess_data, create_dataset
import json
import numpy as np
import pandas as pd
from pathlib import Path

class TestPreprocessData(unittest.TestCase):
  def test_create_dataset(self):
    x = [1,2,3,4]
    y = [5,6,7,8]
    time_steps = 2

    df = pd.DataFrame(data=np.transpose([x, y]), columns=['x','y'])

    xx, yy = create_dataset(df.y, df.y, time_steps)

    self.assertTrue((xx == [[5,6],[6,7]]).all())
    self.assertTrue((yy == [7, 8]).all())

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

if __name__ == '__main__':
  unittest.main()
