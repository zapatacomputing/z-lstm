"""
This module generates a set of data.
"""

import sys
import numpy as np
import pandas as pd

def noisy_sine_generation(timerange=100, timestep=0.1, noisestd=0.2):
  # timerange = int(timerange)
  # timestep = float(timestep)
  # noisestd = float(noisestd)

  try:
    time = np.arange(0, timerange, timestep)
 
    # Generating data: sine function
    data = np.sin(time) + np.random.normal(scale=noisestd, size=len(time))

    # Testing numpy Array
    print('Data shape from numpy: ', data.shape)
    
    # Testing pandas DataFrame
    data_df = pd.DataFrame(data, columns=['Values'])
    print('Data shape from pandas: ', data_df.shape)
    print(data_df.head(3))
  except:
    e = sys.exc_info()[0]
    print(e)
    print('Something went wrong!')

