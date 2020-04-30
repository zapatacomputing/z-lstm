"""Plot the energy vs the number of layers in the QAOA ansatz."""

import json
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd

# Insert the path to your JSON file here
with open('example_lstm_results.json') as f:
    results = json.load(f)

training_loss_vals = []
validation_loss_vals = []
predicted_vals = []

for step in results:
  if results[step]['class'] == 'train-model':
    training_loss_vals = results[step]['history']['history']['loss']
    validation_loss_vals = results[step]['history']['history']['val_loss']
  if results[step]['class'] == 'preprocess-data':  
    training_df = pd.DataFrame(results[step]['training_data']['data'])
    testing_df = pd.DataFrame(results[step]['testing_data']['data'])
  if results[step]['class'] == 'predict-using-model':
    predicted_vals_obj = results[step]['predictions']['data']
    for entry in predicted_vals_obj:
      predicted_vals.append(entry['data'][0]['data'])

# Converting indices to ints and sorting by indices
training_df.index = training_df.index.astype(int)
training_df.sort_index(inplace=True)
testing_df.index = testing_df.index.astype(int)
testing_df.sort_index(inplace=True)

# Plotting values from training process
plt.figure()
plt.plot(training_loss_vals, label='Train')
plt.plot(validation_loss_vals, label='Validation')
plt.xlim(left=0.0)
plt.ylim(bottom=0.0)
plt.grid()
plt.legend()
plt.title("Loss function: MSE")
plt.xlabel('Epoch')
plt.ylabel('Loss function value')

# Plotting results: train, test and perdicted datasets
fig, ax = plt.subplots()

ax.plot(training_df['time'], training_df['values'], color='g')
ax.plot(testing_df['time'], testing_df['values'], color='b')
ax.scatter(testing_df['time'].values[10:], predicted_vals, marker='o', color='r', label="Predicted data")

plt.ylabel('Value')
plt.xlabel('Time Step')
plt.grid()
ax.legend(['Training', 'Testing', 'Predicted'])
plt.title("Training, Test and Predicted datasets")

plt.show()
