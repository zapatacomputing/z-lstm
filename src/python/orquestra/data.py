"""
This module generates a set of data.
"""

import sys
import numpy as np
import pandas as pd

def noisy_sine_generation(timerange=100, timestep=0.1, noisestd=0.2):
  print(timerange)
  print(type(timerange))
  print(timestep)
  print(type(timestep))
  print(noisestd)
  print(type(noisestd))
  try:
    time = np.arange(0, int(timerange), float(timestep))
 
    # Generating data: sine function
    data = np.sin(time) + np.random.normal(scale=float(noisestd), size=len(time))

    # Testing numpy Array
    print('Data shape from numpy: ', data.shape)
    
    # Testing pandas DataFrame
    data_df = pd.DataFrame(data)
    print('Data shape from pandas: ', data_df.shape)
    print(data_df.head())
  except:
    e = sys.exc_info()[0]
    print(e)
    print('Something went wrong!')

