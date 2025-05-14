from single_example_optimization import optimize_single_example
from reflax.constants import ONE_LAYER_MODEL, TRANSFER_MATRIX_METHOD
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # Import gridspec
import jax.numpy as jnp
import numpy as np # Keep for potential future use

# A: noise scale 0.01
# B: noise scale 0.05
# C: noise scale 0.1

# load the results from files
data = jnp.load("result_data/noisy_example.npz")
true_reflectanceA = data["true_reflectanceA"]
time_points_true = jnp.linspace(0, 1, true_reflectanceA.shape[0])
true_thicknessA = data["true_thicknessA"]
true_growth_rateA = data["true_growth_rateA"]
predicted_reflectanceA = data["predicted_reflectanceA"]
predicted_thicknessA = data["predicted_thicknessA"]
predicted_growth_rateA = data["predicted_growth_rateA"]
initialized_reflectanceA = data["initialized_reflectanceA"]
initialized_thicknessA = data["initialized_thicknessA"]
initialized_growth_rateA = data["initialized_growth_rateA"]
initialized_time_pointsA = data["initialized_time_pointsA"]
time_points_initialized = jnp.linspace(0, 1, initialized_reflectanceA.shape[0])
snr_dbA = data["snr_dbA"]
true_reflectanceB = data["true_reflectanceB"]
true_thicknessB = data["true_thicknessB"]
true_growth_rateB = data["true_growth_rateB"]
predicted_reflectanceB = data["predicted_reflectanceB"]
predicted_thicknessB = data["predicted_thicknessB"]
predicted_growth_rateB = data["predicted_growth_rateB"]
initialized_reflectanceB = data["initialized_reflectanceB"]
initialized_thicknessB = data["initialized_thicknessB"]
initialized_growth_rateB = data["initialized_growth_rateB"]
initialized_time_pointsB = data["initialized_time_pointsB"]
snr_dbB = data["snr_dbB"]
true_reflectanceC = data["true_reflectanceC"]
true_thicknessC = data["true_thicknessC"]
true_growth_rateC = data["true_growth_rateC"]
predicted_reflectanceC = data["predicted_reflectanceC"]
predicted_thicknessC = data["predicted_thicknessC"]
predicted_growth_rateC = data["predicted_growth_rateC"]
initialized_reflectanceC = data["initialized_reflectanceC"]
initialized_thicknessC = data["initialized_thicknessC"]
initialized_growth_rateC = data["initialized_growth_rateC"]
initialized_time_pointsC = data["initialized_time_pointsC"]
snr_dbC = data["snr_dbC"]

# Calculate ABSOLUTE thickness errors
thickness_abs_errorA = jnp.abs(true_thicknessA - predicted_thicknessA)
thickness_abs_errorB = jnp.abs(true_thicknessB - predicted_thicknessB)
thickness_abs_errorC = jnp.abs(true_thicknessC - predicted_thicknessC)

# --- Create Figure and GridSpec ---
# Define height ratios: Reflectance(2), Thickness(2), Error(1), GrowthRate(2)
height_ratios = [2, 2, 1, 2] # Keep the desired ratios
fig = plt.figure(figsize=(15, 11)) # Adjust figsize if needed
gs = gridspec.GridSpec(4, 3, height_ratios=height_ratios, width_ratios=[1, 1, 1])

# --- Create Axes explicitly using GridSpec based on desired order ---
ax00 = fig.add_subplot(gs[0, 0]) # Row 0, Col 0: Reflectance A
ax10 = fig.add_subplot(gs[1, 0]) # Row 1, Col 0: Thickness A
ax20 = fig.add_subplot(gs[2, 0]) # Row 2, Col 0: Error A (Half Height)
ax30 = fig.add_subplot(gs[3, 0]) # Row 3, Col 0: Growth Rate A

ax01 = fig.add_subplot(gs[0, 1]) # Row 0, Col 1: Reflectance B
ax11 = fig.add_subplot(gs[1, 1]) # Row 1, Col 1: Thickness B
ax21 = fig.add_subplot(gs[2, 1]) # Row 2, Col 1: Error B (Half Height)
ax31 = fig.add_subplot(gs[3, 1]) # Row 3, Col 1: Growth Rate B

ax02 = fig.add_subplot(gs[0, 2]) # Row 0, Col 2: Reflectance C
ax12 = fig.add_subplot(gs[1, 2]) # Row 1, Col 2: Thickness C
ax22 = fig.add_subplot(gs[2, 2]) # Row 2, Col 2: Error C (Half Height)
ax32 = fig.add_subplot(gs[3, 2]) # Row 3, Col 2: Growth Rate C


# --- Column 0: Low Noise (A) ---

# Row 0: Reflectance A
ax00.plot(time_points_true, true_reflectanceA, label = "measured reflectance", color = "blue", alpha = 0.5)
ax00.plot(time_points_true, predicted_reflectanceA, label = "predicted reflectance", color = "red", alpha = 0.5)
ax00.set_ylabel("normalized reflectance")
ax00.legend(loc = "upper right")
ax00.set_title("Reflectance (SNR = " + "{:.2f}".format(snr_dbA) + " dB)")
ax00.tick_params(axis='x', labelbottom=False) # Hide x-axis labels

# Row 1: Thickness A
ax10.plot(time_points_true, true_thicknessA, label = "true thickness", color = "blue", alpha = 0.5)
ax10.plot(time_points_true, predicted_thicknessA, label = "predicted thickness", color = "red", alpha = 0.5)
ax10.plot(initialized_time_pointsA, initialized_thicknessA, label = "initial prediction", linestyle = "--", color = "green", alpha = 0.5)
ax10.set_ylabel("thickness in nm")
ax10.legend(loc = "upper right")
ax10.set_title("Thickness Prediction")
ax10.tick_params(axis='x', labelbottom=False) # Hide x-axis labels

# Row 2: ABSOLUTE Error Plot (A) - Half Height
ax20.plot(time_points_true, thickness_abs_errorA, label="Absolute Error", color="purple", alpha=0.7) # Plot absolute error
ax20.set_ylabel("Abs. Error (nm)") # Update label
# ax20.legend(loc="upper right") # Optional legend
ax20.set_title("Absolute Thickness Prediction Error") # Update title
ax20.set_ylim(bottom=0)
ax20.grid(True, linestyle='--', alpha=0.6)
ax20.tick_params(axis='x', labelbottom=False) # Hide x-axis labels

# Row 3: Growth Rate A
ax30.plot(time_points_true, true_growth_rateA, label = "true growth rate", color = "blue", alpha = 0.5)
ax30.plot(time_points_true, predicted_growth_rateA, label = "predicted growth rate", color = "red", alpha = 0.5)
ax30.plot(initialized_time_pointsA, initialized_growth_rateA, label = "initial prediction", linestyle = "--", color = "green", alpha = 0.5)
ax30.set_xlabel("time in hours") # Add x-label only to the bottom plot
ax30.set_ylabel("growth rate in nm/h")
ax30.legend(loc = "lower right")
ax30.set_title("Growth Rate Prediction")


# --- Column 1: Medium Noise (B) ---

# Row 0: Reflectance B
ax01.plot(time_points_true, true_reflectanceB, label = "measured reflectance", color = "blue", alpha = 0.5)
ax01.plot(time_points_true, predicted_reflectanceB, label = "predicted reflectance", color = "red", alpha = 0.5)
# ax01.set_ylabel("normalized reflectance")
ax01.legend(loc = "upper right")
ax01.set_title("Reflectance (SNR = " + "{:.2f}".format(snr_dbB) + " dB)")
ax01.tick_params(axis='x', labelbottom=False)

# Row 1: Thickness B
ax11.plot(time_points_true, true_thicknessB, label = "true thickness", color = "blue", alpha = 0.5)
ax11.plot(time_points_true, predicted_thicknessB, label = "predicted thickness", color = "red", alpha = 0.5)
ax11.plot(initialized_time_pointsB, initialized_thicknessB, label = "initial prediction", linestyle = "--", color = "green", alpha = 0.5)
# ax11.set_ylabel("thickness in nm")
ax11.legend(loc = "upper right")
ax11.set_title("Thickness Prediction")
ax11.tick_params(axis='x', labelbottom=False)

# Row 2: ABSOLUTE Error Plot (B) - Half Height
ax21.plot(time_points_true, thickness_abs_errorB, label="Absolute Error", color="purple", alpha=0.7) # Plot absolute error
# ax21.set_ylabel("Abs. Error (nm)") # Update label
# ax21.legend(loc="upper right")
ax21.set_title("Absolute Thickness Prediction Error") # Update title
ax21.set_ylim(bottom=0)
ax21.grid(True, linestyle='--', alpha=0.6)
ax21.tick_params(axis='x', labelbottom=False)

# Row 3: Growth Rate B
ax31.plot(time_points_true, true_growth_rateB, label = "true growth rate", color = "blue", alpha = 0.5)
ax31.plot(time_points_true, predicted_growth_rateB, label = "predicted growth rate", color = "red", alpha = 0.5)
ax31.plot(initialized_time_pointsB, initialized_growth_rateB, label = "initial prediction", linestyle = "--", color = "green", alpha = 0.5)
ax31.set_xlabel("time in hours")
# ax31.set_ylabel("growth rate in nm/h")
ax31.legend(loc = "lower right")
ax31.set_title("Growth Rate Prediction")


# --- Column 2: High Noise (C) ---

# Row 0: Reflectance C
ax02.plot(time_points_true, true_reflectanceC, label = "measured reflectance", color = "blue", alpha = 0.5)
ax02.plot(time_points_true, predicted_reflectanceC, label = "predicted reflectance", color = "red", alpha = 0.5)
# ax02.set_ylabel("reflectance")
ax02.legend(loc = "upper right")
ax02.set_title("Reflectance (SNR = " + "{:.2f}".format(snr_dbC) + " dB)")
ax02.tick_params(axis='x', labelbottom=False)

# Row 1: Thickness C
ax12.plot(time_points_true, true_thicknessC, label = "true thickness", color = "blue", alpha = 0.5)
ax12.plot(time_points_true, predicted_thicknessC, label = "predicted thickness", color = "red", alpha = 0.5)
ax12.plot(initialized_time_pointsC, initialized_thicknessC, label = "initial prediction", linestyle = "--", color = "green", alpha = 0.5)
# ax12.set_ylabel("thickness in nm")
ax12.legend(loc = "upper right")
ax12.set_title("Thickness Prediction")
ax12.tick_params(axis='x', labelbottom=False)

# Row 2: ABSOLUTE Error Plot (C) - Half Height
ax22.plot(time_points_true, thickness_abs_errorC, label="Absolute Error", color="purple", alpha=0.7) # Plot absolute error
# ax22.set_ylabel("Abs. Error (nm)") # Update label
# ax22.legend(loc="upper right")
ax22.set_title("Absolute Thickness Prediction Error") # Update title
ax22.set_ylim(bottom=0)
ax22.grid(True, linestyle='--', alpha=0.6)
ax22.tick_params(axis='x', labelbottom=False)

# Row 3: Growth Rate C
ax32.plot(time_points_true, true_growth_rateC, label = "true growth rate", color = "blue", alpha = 0.5)
ax32.plot(time_points_true, predicted_growth_rateC, label = "predicted growth rate", color = "red", alpha = 0.5)
ax32.plot(initialized_time_pointsC, initialized_growth_rateC, label = "initial prediction", linestyle = "--", color = "green", alpha = 0.5)
ax32.set_xlabel("time in hours")
# ax32.set_ylabel("growth rate in nm/h")
ax32.legend(loc = "lower right")
ax32.set_title("Growth Rate Prediction")


# --- Final Adjustments ---
plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust padding if needed
plt.savefig("figures/noisy_example_abs_error_reordered.svg") # Updated filename
# plt.show()