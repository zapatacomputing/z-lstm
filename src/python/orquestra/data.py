"""
This module generates a set of data.
"""

import json
import sys
import numpy as np
import pandas as pd

def noisy_sine_generation(timerange, timestep, noisestd):
  print('timerange = ', timerange)
  print('timestep = ', timestep)
  print('noisestd = ', noisestd)

  try:
    time = np.arange(0, timerange, timestep)
 
    # Generating data: sine function
    data = np.sin(time) + np.random.normal(scale=noisestd, size=len(time))

    # Testing numpy Array
    print('Data shape from numpy: ', data.shape)
    
    # Testing pandas DataFrame
    data_df = pd.DataFrame(data, columns=['Values'])
    print('Data shape from pandas: ', data_df.shape)
    print(data_df.head())
  except:
    e = sys.exc_info()[0]
    print(e)
    print('Something went wrong!')

  data_dict = {}
  data_dict["data"] = {'time' : time.tolist(), 'values': data.tolist()}
  data_dict["schema"] = "orquestra-v1-data"

  with open("data.json",'w') as f:
      f.write(json.dumps(data_dict, indent=2)) # Write data to file as this will serve as output artifact

