import unittest
from data import noisy_sine_generation
import json
import os

class TestDataGeneration(unittest.TestCase):

  def test_noisy_sine_generation(self):
    timerange = 100
    timestep = 0.1
    noisestd = 0.5
    noisy_sine_generation(timerange, timestep, noisestd)

    filename = 'data.json'

    with open(filename) as json_file:
      datafile = json.load(json_file)

    self.assertEqual(len(datafile["data"]["time"]), 1000)
    self.assertEqual(len(datafile["data"]["values"]), 1000)
    self.assertEqual(datafile["data"]["time"][-1], timerange-timestep)

    try:
      os.remove(filename)
    except OSError:
      pass

if __name__ == '__main__':
    unittest.main()

