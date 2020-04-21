import unittest
from data import noisy_sine_generation
import json
import os

class TestDataGeneration(unittest.TestCase):

  def test_noisy_sine_generation(self):
    time_range = 100
    time_step = 0.1
    noise_std = 0.5
    data_dict = noisy_sine_generation(time_range, time_step, noise_std)

    self.assertEqual(len(list(data_dict["data"]["values"].keys())), time_range/time_step)
    self.assertEqual(len(list(data_dict["data"]["values"].values())), time_range/time_step)
    self.assertEqual(list(data_dict["data"]["values"].keys())[-1], time_range-time_step)

if __name__ == '__main__':
    unittest.main()

