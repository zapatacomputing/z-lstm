import unittest
import json
from model import build_model

class TestModel(unittest.TestCase):
  def test_build_model(self):
    test_data_file_name = 'test_model_data.json'
    with open(test_data_file_name) as test_data_json_file:
      test_data = json.load(test_data_json_file)

    model = build_model(test_data)

    expected_model_file_name = 'test_model.json'
    with open(expected_model_file_name) as expected_model_json_file:
      expected_model = json.load(expected_model_json_file)
    
    # Remove keras version from result because that may change
    model_json = model.to_json()
    model_dict = json.loads(model_json) 
    model_dict.pop("keras_version", None)

    self.assertEqual(model_dict, expected_model)

if __name__ == '__main__':
  unittest.main()