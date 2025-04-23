import numpy as np
import matplotlib.pyplot as plt

measurement = np.loadtxt("reflectance.txt", skiprows = 1)
time_raw = np.array(measurement[:-100, 0])

# convert the time to hours, numerically ~ nicely between 0 and 1
time_hours = time_raw / 3600
reflectance_raw = np.array(measurement[:-100, 1])

plt.plot(time_hours, reflectance_raw, label = "raw data")
plt.xlabel("time in hours")
plt.ylabel("reflectance")

plt.savefig("figures/raw_data.svg")