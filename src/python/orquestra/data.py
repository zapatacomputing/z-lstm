"""
This module generates a set of data.
"""

import json
import sys
import numpy as np
import pandas as pd

def noisy_sine_generation(timerange, timestep, noisestd) -> dict:
  print('timerange = ', timerange)
  print('timestep = ', timestep)
  print('noisestd = ', noisestd)

  data_dict = {}

  try:
    time = np.arange(0, timerange, timestep)
 
    # Generating data: sine function
    values = np.sin(time) + np.random.normal(scale=noisestd, size=len(time))
    print('Values shape from numpy: ', values.shape)
    
    # Making pandas DataFrame
    data_df = pd.DataFrame(values, index=time, columns=['values'])

    print('Data shape from pandas:')
    print(data_df.shape)
    print('Data frame header:')
    print(data_df.head())

    data_dict = {}
    data_dict["data"] = data_df.to_dict()
  except:
    e = sys.exc_info()[0]
    print(e)
    print('Something went wrong!')

  return data_dict

