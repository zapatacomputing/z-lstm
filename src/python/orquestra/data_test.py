import unittest
import json
from data import *

class TestDataGeneration(unittest.TestCase):
  def test_noisy_sine_generation(self):
    time_range = 2
    time_step = 0.1
    noise_std = 0.5
    data_dict = noisy_sine_generation(time_range, time_step, noise_std)

    data_df_dict = json.loads(data_dict["data"])

    self.assertEqual(len(data_df_dict["time"]), time_range/time_step)
    self.assertEqual(len(data_df_dict["values"]), time_range/time_step)
    last_index = list(data_df_dict["time"].keys())[-1]
    self.assertEqual(data_df_dict["time"][last_index], time_range-time_step)

if __name__ == '__main__':
  unittest.main()

