# ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# only use gpu 9
import os

from matplotlib import gridspec # Added for subplot management
# import numpy as np # Not strictly needed if using JAX arrays correctly, but good for conversion
# Numpy is implicitly imported by seaborn and matplotlib for their internal use.
# We will explicitly convert JAX arrays to numpy for plotting.
import numpy as np_for_plotting 


os.environ["CUDA_VISIBLE_DEVICES"] = "4"
# =======================

import jax.numpy as jnp
import jax

from reflax.thickness_modeling.function_sampling import sample_derivative_bound_gp


# NOTE: without 64-bit precision,
# the Cholesky decomposition fails
from jax import config
config.update("jax_enable_x64", True)
# this is not yet properly caught, the error you
# will get is that not all samples met the
# derivative bounds

import matplotlib.pyplot as plt
import seaborn as sns

num_samples = 1000
variance = 15.0
min_slope = 200.0
max_slope = 1800.0

num_eval_points = 100
time_points = jnp.linspace(0, 1, num_eval_points)
master_key = jax.random.PRNGKey(89) # Changed name for clarity with splitting

# --- Figure Setup with GridSpec ---
fig = plt.figure(figsize=(15, 10)) # Adjusted figure size for better layout
gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.25)

ax_thickness_ts = fig.add_subplot(gs[0, 0])
ax_derivative_ts = fig.add_subplot(gs[1, 0])
ax_kde_std_first_deriv = fig.add_subplot(gs[0, 1])
ax_kde_std_second_deriv = fig.add_subplot(gs[1, 1])
# --- End Figure Setup ---

lengthscales_to_plot = [0.5, 0.4, 0.3, 0.2, 0.1]
colors = sns.color_palette("viridis", n_colors=len(lengthscales_to_plot))
kde_common_params = dict(fill=True, alpha=0.4, linewidth=1.5)
num_examples_to_plot_ts = 3 # Number of example time series to plot for each lengthscale

for i, lengthscale in enumerate(lengthscales_to_plot):
    master_key, subkey = jax.random.split(master_key) # Split key for each lengthscale

    thickness_gp, derivatives_gp = sample_derivative_bound_gp(
        subkey, # Use the new subkey for this iteration
        num_samples,
        time_points,
        lengthscale,
        variance,
        min_slope,
        max_slope,
        random_final_values = True,
        min_final_value = 800.0,
        max_final_value = 1200.0,
        convex_samples = True,
    )

    # Convert JAX arrays to NumPy for Matplotlib/Seaborn
    thickness_gp_np = np_for_plotting.array(thickness_gp)
    derivatives_gp_np = np_for_plotting.array(derivatives_gp)
    time_points_np = np_for_plotting.array(time_points)

    # --- Plotting a few example time series ---
    for k in range(min(num_examples_to_plot_ts, num_samples)):
        label_ts = f'L={lengthscale}' if k == 0 else None # Label only first line per group
        ax_thickness_ts.plot(time_points_np, thickness_gp_np[k,:], color=colors[i], alpha=0.6, label=label_ts)
        ax_derivative_ts.plot(time_points_np, derivatives_gp_np[k,:], color=colors[i], alpha=0.6, label=label_ts)

    # --- Calculate data for KDE plots ---
    # 1. Standard deviation of first derivatives
    std_first_deriv_gp = jnp.std(derivatives_gp, axis=1)
    std_first_deriv_gp_np = np_for_plotting.array(std_first_deriv_gp)

    # 2. Second derivative and its standard deviation
    dt = time_points[1] - time_points[0] 
    second_deriv_gp = jnp.gradient(derivatives_gp, dt, axis=1) # dt is JAX scalar, fine here
    std_second_deriv_gp = jnp.std(second_deriv_gp, axis=1)
    std_second_deriv_gp_np = np_for_plotting.array(std_second_deriv_gp)
    
    # --- Plot KDEs ---
    kde_label = f"L={lengthscale}"
    sns.kdeplot(
        std_first_deriv_gp_np,
        ax=ax_kde_std_first_deriv,
        label=kde_label,
        color=colors[i],
        **kde_common_params
    )

    sns.kdeplot(
        std_second_deriv_gp_np,
        ax=ax_kde_std_second_deriv,
        label=kde_label,
        color=colors[i],
        **kde_common_params
    )

# --- Finalize plots with labels, titles, legends ---
ax_thickness_ts.set_title("Example Sampled Thickness Profiles")
ax_thickness_ts.set_xlabel("Time (arbitrary units)")
ax_thickness_ts.set_ylabel("Thickness (arbitrary units)")
ax_thickness_ts.legend(title="Lengthscale")

ax_derivative_ts.set_title("Example Sampled Derivative Profiles")
ax_derivative_ts.set_xlabel("Time (arbitrary units)")
ax_derivative_ts.set_ylabel("Derivative (arbitrary units)")
ax_derivative_ts.legend(title="Lengthscale")

ax_kde_std_first_deriv.set_title("KDE of std(Derivatives)")
ax_kde_std_first_deriv.set_xlabel("Standard Deviation of Derivatives")
ax_kde_std_first_deriv.set_ylabel("Density")
ax_kde_std_first_deriv.legend(title="Lengthscale")

ax_kde_std_second_deriv.set_title("KDE of std(Second Derivatives)")
ax_kde_std_second_deriv.set_xlabel("Standard Deviation of Second Derivatives")
ax_kde_std_second_deriv.set_ylabel("Density")
ax_kde_std_second_deriv.legend(title="Lengthscale")

# plt.tight_layout() # Sometimes problematic with gridspec, hspace/wspace preferred
plt.suptitle("Analysis of Generated GP Samples by Lengthscale", fontsize=16)
fig.subplots_adjust(top=0.92) # Adjust top to make space for suptitle

# If you want to save the figure:
plt.savefig("figures/gp_samples_analysis.png", dpi=300)