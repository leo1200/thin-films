import numpy as np
from matplotlib import pyplot as plt

# read in training_data/example_data.npz
data = np.load('training_data/example_data.npz')

# get the first thickness time series
# and corresponding reflectence time series

thickness = data['thicknesses'][0]
reflectance = data['reflectances'][0]
timepoints = data['x_eval']

# NOTE: our aim is to predict the thickness 
# time series from the reflectance time series

# plot thickness and reflectance time series
fig, axs = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
axs[0].plot(timepoints, thickness, label='Thickness', color='blue')
axs[0].set_ylabel('Thickness in nm')
axs[0].legend()

axs[1].plot(timepoints, reflectance, label='Reflectance', color='orange')
axs[1].set_ylabel('Reflectance')
axs[1].set_xlabel('Time in h')
axs[1].legend()
plt.tight_layout()

plt.savefig('figures/example_data.png')