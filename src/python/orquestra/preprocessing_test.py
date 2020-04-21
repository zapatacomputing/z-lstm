import unittest
from data import noisy_sine_generation
import json
import os

class TestCreateData(unittest.TestCase):
    x = [1,2,3,4]
    y = [5,6,7,8]
    time_steps = 2
    xx, yy = create_dataset(x, y, time_steps)
    
    self.assertEqual(xx, [[1,2],[2,3],[3,4]])
    self.assertEqual(xx, [[5,6],[6,7],[7,8]])

class TestPreprocessData(unittest.TestCase):
    def test_preprocess_data(self):
        
        trainperc = 0.8
        timesteps = 10

        # Come up with a dummy dataset
        data = {"time": , "values":}

        preprocess_data(data, trainperc, time_steps)

        filename = 'data.json'

        with open(filename) as json_file:
            datafile = json.load(json_file)

        self.assertEqual(len(datafile["xtrain"]), 8)
        self.assertEqual(len(datafile["xtest"]), 2)
        self.assertEqual(len(datafile["ytrain"]), 8)
        self.assertEqual(len(datafile["ytest"]), 2)

        try:
        os.remove(filename)
        except OSError:
        pass

if __name__ == '__main__':
    unittest.main()
