"""
This module generates a set of data.
"""

import numpy as np
import pandas as pd

def noisy_sine_generation(numsamples):
  time = np.arange(0, 100, 0.1)
  noisestd = 0.2

  for i in range(1, numsamples+1):
    print('Sample ', i)

    # Generating data: sine function
    data = np.sin(time) + np.random.normal(scale=noisestd, size=len(time))

    # Testing numpy Array
    print('Data shape from numpy: ', data.shape)
    
    # Testing pandas DataFrame
    data_df = pd.DataFrame(data)
    print('Data shape from pandas: ', data_df.shape)
    print(data_df.head())
