import unittest
from data_management import *
import os.path

class TestDataManagement(unittest.TestCase):
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