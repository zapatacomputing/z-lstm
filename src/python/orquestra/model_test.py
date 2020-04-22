import unittest
import json
from model import build_model, save_model
from pathlib import Path

from tensorflow.keras.models import model_from_json

class TestModel(unittest.TestCase):
  def test_build_model(self):
    # Get test data to build the model to fit the dimensions of
    test_data_file_path = Path("test/test_model_data.json")
    with open(test_data_file_path) as test_data_file:
      test_data = json.load(test_data_file)

    # Build the model
    model = build_model(test_data)

    # Get expected model out of file
    expected_model_file_path = Path("test/test_model.json")
    with open(expected_model_file_path) as expected_model_file:
      expected_model_json = expected_model_file.read()

    expected_model = model_from_json(expected_model_json)

    self.assertEqual(model.to_json(), expected_model.to_json())

if __name__ == '__main__':
  unittest.main()