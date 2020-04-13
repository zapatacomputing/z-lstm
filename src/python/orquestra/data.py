"""
This module generates a set of data.
"""

import numpy as np
import pandas as pd

def noisy_sine_generation():
  # Generating data: sine function
  time = np.arange(0, 100, 0.1)
  sin = np.sin(time) + np.random.normal(scale=noisestd, size=len(time))
  print(sin.shape)
  sin_df = pd.DataFrame(sin)
  print(sin_df.shape)
