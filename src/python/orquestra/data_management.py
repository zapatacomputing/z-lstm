"""
This module preprocesses the data into windows.
"""
import json
import sys
import numpy as np
import pandas as pd
from typing import TextIO

def save_data(datas: list, filenames: list):
  for i in range(len(datas)):
    data = datas[i]
    filename = filenames[i]

    data["schema"] = "orquestra-v1-data"

    with open(filename,'w') as f:
      f.write(json.dumps(data, indent=2)) # Write data to file as this will serve as output artifact


def load_data(file: TextIO) -> dict:
  if isinstance(file, str):
    with open(file, 'r') as f:
      data = json.load(f)
  else:
    data = json.load(file)

  return data