import unittest
from data import *

class TestDataGeneration(unittest.TestCase):
  def test_noisy_sine_generation(self):
    time_range = 100
    time_step = 0.1
    noise_std = 0.5
    data_dict = noisy_sine_generation(time_range, time_step, noise_std)

    self.assertEqual(len(data_dict["data"]["time"]), time_range/time_step)
    self.assertEqual(len(data_dict["data"]["values"]), time_range/time_step)
    last_index = list(data_dict["data"]["time"].keys())[-1]
    self.assertEqual(data_dict["data"]["time"][last_index], time_range-time_step)

if __name__ == '__main__':
  unittest.main()

