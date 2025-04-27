"""

Generate training data consisting of

 - polynomial thicknesses, convex and monotonic, with given max derivative
 - piecewise linear derivative thicknesses, convex and monotonic, with given max derivative
 - linear thicknesses

and their corresponding reflectances.

"""

# GPU selection
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

# argument passing
import argparse

# typing
from typing import Callable, Tuple

# timing
import time as pytime

# numerics
import jax
import numpy as np
import jax.numpy as jnp
from jax import random, vmap, jit

# reflax (our library)
from reflax import polanalyze
from reflax.parameter_classes.parameters import (
    OpticsParams,
    SetupParams,
    LayerParams
)
from reflax.forward_model.variable_layer_size import (
    MIN_MAX_NORMALIZATION,
    ONE_LAYER_INTERNAL_REFLECTIONS,
    TRANSFER_MATRIX_METHOD
)
from reflax.forward_model.variable_layer_size import forward_model

# plotting
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# ======================= ↓ arguments ↓ =======================
# -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Generate Thickness-Reflectance Training Data")
    parser.add_argument('--n_pol', type=int, default=500, help='Number of polynomial profiles')
    parser.add_argument('--n_lin', type=int, default=500, help='Number of piecewise-linear derivative profiles')
    parser.add_argument('--n_const', type=int, default=100, help='Number of constant derivative (linear) profiles')
    parser.add_argument('--num_eval', type=int, default=100, help='Number of evaluation points for each profile')
    parser.add_argument('--min_final_thickness', type=float, default=100.0, help='Minimum final thickness (nm)')
    parser.add_argument('--max_final_thickness', type=float, default=1500.0, help='Maximum final thickness (nm)')
    parser.add_argument('--poly_base_degree', type=int, default=3, help='Base degree for polynomial generator')
    parser.add_argument('--poly_max_deriv', type=float, default=2.0, help='Max derivative constraint for polynomial H\'(x)')
    parser.add_argument('--pwl_max_deriv', type=float, default=2.0, help='Max derivative constraint for PWL H\'(x)')
    parser.add_argument('--output_file', type=str, default='training_data.npz', help='Output file name for saved data')
    parser.add_argument('--plot_file_thickness', type=str, default='generated_thicknesses.png', help='Output file name for thickness plot')
    # --- New argument for reflectance plot ---
    parser.add_argument('--plot_file_reflectance', type=str, default='generated_reflectances.png', help='Output file name for reflectance plot')
    # --- End new argument ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    parser.add_argument('--skip_forward_model', action='store_true', help='Skip reflectance calculation (e.g., if reflax is unavailable)')
    return parser.parse_args()

# -------------------------------------------------------------
# ======================= ↑ arguments ↑ =======================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================= ↓ polynomial thicknesses ↓ ================
# -------------------------------------------------------------

def _calculate_alpha(max_g0: float, integral_g0: float, max_derivative: float, epsilon: float) -> float:
    """Calculates the scaling factor alpha."""
    denominator = jnp.maximum(max_g0 - max_derivative * integral_g0, 1e-9)
    alpha_target = epsilon * (max_derivative - 1.0) / denominator
    alpha = jax.lax.cond(
        max_derivative <= 1.0, # Allow max_derivative=1
        lambda _: 0.0,
        lambda _: jnp.maximum(0.0, alpha_target),
        operand=None
    )
    return alpha

def random_monotonic_convex_polynomial_max_deriv(
    key: random.PRNGKey,
    base_degree: int,
    max_derivative: float,
    n_grid: int = 1000,
    epsilon: float = 1e-6,
    integral_tol: float = 1e-7
) -> jnp.ndarray:
    """
    Generates coefficients of a random monotonic polynomial H(x) on [0, 1]
    such that H(0)=0, H(1)=1, H'(x) is non-decreasing (H''(x)>=0),
    and max(H'(x)) <= max_derivative.
    Final degree is base_degree + 2.
    """
    key_p, _ = random.split(key)
    coeffs_p = random.normal(key_p, shape=(base_degree + 1,))
    x_grid = jnp.linspace(0.0, 1.0, n_grid)
    p_values = jnp.polyval(coeffs_p, x_grid)
    p_min = jnp.min(p_values)
    coeffs_p0 = coeffs_p.at[-1].add(-p_min)
    coeffs_g0 = jnp.polyint(coeffs_p0, k=0.0)
    g0_values = jnp.polyval(coeffs_g0, x_grid)
    max_g0 = jnp.maximum(jnp.max(g0_values), 0.0)
    coeffs_h0 = jnp.polyint(coeffs_g0, k=0.0)
    integral_g0 = jnp.polyval(coeffs_h0, 1.0)
    target_H_shape = (base_degree + 3,)

    def compute_H_nonzero_integral(coeffs_g0_in):
        ig0 = jnp.maximum(integral_g0, integral_tol)
        mg0 = max_g0
        alpha = _calculate_alpha(mg0, ig0, max_derivative, epsilon)
        coeffs_g_final = alpha * coeffs_g0_in
        coeffs_g_final = coeffs_g_final.at[-1].add(epsilon)
        coeffs_h_final = jnp.polyint(coeffs_g_final, k=0.0)
        h_final_at_1 = jnp.polyval(coeffs_h_final, 1.0)
        h_final_at_1_safe = jnp.maximum(h_final_at_1, epsilon * 1e-3)
        coeffs_H = coeffs_h_final / h_final_at_1_safe
        padding = target_H_shape[0] - coeffs_H.shape[0]
        coeffs_H = jnp.pad(coeffs_H, (padding, 0))
        coeffs_H = jnp.reshape(coeffs_H, target_H_shape)
        return coeffs_H

    def compute_H_zero_integral(coeffs_g0_in):
         coeffs_H = jnp.zeros(target_H_shape, dtype=coeffs_g0_in.dtype)
         coeffs_H = coeffs_H.at[-2].set(1.0)
         return coeffs_H

    coeffs_H = jax.lax.cond(
        integral_g0 > integral_tol,
        compute_H_nonzero_integral,
        compute_H_zero_integral,
        operand=coeffs_g0
    )
    return coeffs_H

# @jit
def evaluate_polynomial(coeffs, x):
  """Evaluates a polynomial given its coefficients."""
  coeffs = jnp.atleast_1d(coeffs)
  if coeffs.ndim == 0:
      coeffs = jnp.array([coeffs.item()])
  elif coeffs.size == 0:
      coeffs = jnp.array([0.0])
  return jnp.polyval(coeffs, x)

# Vectorized versions for batch processing
vmap_random_monotonic_convex_polynomial = vmap(
    random_monotonic_convex_polynomial_max_deriv,
    in_axes=(0, None, None, None, None, None)
)
vmap_evaluate_polynomial = vmap(evaluate_polynomial, in_axes=(0, None))

# -------------------------------------------------------------
# ================= ↑ polynomial thicknesses ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======== ↓ piecewise linear derivative thicknesses ↓ ========
# -------------------------------------------------------------

def _generate_and_check_params_nondecreasing(key, xmin, xmax, eps, max_derivative):
    """Generates PWL parameters and checks constraints."""
    key_c, key_y = jax.random.split(key)
    c = jax.random.uniform(key_c, shape=(), minval=jnp.maximum(eps, xmin), maxval=jnp.minimum(1.0 - eps, xmax))
    y_raw = jax.random.uniform(key_y, shape=(3,), minval=eps, maxval=1.0)
    y0_raw, yc_raw, y1_raw = y_raw[0], y_raw[1], y_raw[2]
    integral_raw = 0.5 * c * (y0_raw + yc_raw) + 0.5 * (1.0 - c) * (yc_raw + y1_raw)
    integral_raw = jnp.maximum(integral_raw, eps)
    scale = 1.0 / integral_raw
    y0_cand = y0_raw * scale
    yc_cand = yc_raw * scale
    y1_cand = y1_raw * scale
    y_cand = jnp.array([y0_cand, yc_cand, y1_cand])
    is_within_max_deriv = jnp.max(y_cand) <= max_derivative
    is_non_decreasing = (y0_cand <= yc_cand + eps) & (yc_cand <= y1_cand + eps)
    is_valid = is_within_max_deriv & is_non_decreasing
    return is_valid, (c, y0_cand, yc_cand, y1_cand)

# @functools.partial(jax.jit, static_argnames=['xmin', 'xmax', 'eps', 'max_derivative'])
def generate_params_max_deriv_nondecreasing(key, max_derivative, xmin=0.1, xmax=0.9, eps=1e-6):
    """Generates valid PWL parameters via rejection sampling."""
    if max_derivative < 1.0 - eps:
         raise ValueError(f"max_derivative ({max_derivative}) must be >= 1.0.")
    if not (0 < xmin < xmax < 1):
        raise ValueError("Interval (xmin, xmax) must be within (0, 1).")

    def loop_cond(state):
        _, is_valid, _ = state
        return ~is_valid

    def loop_body(state):
        key, _, _ = state
        key, subkey = jax.random.split(key)
        is_valid, params = _generate_and_check_params_nondecreasing(subkey, xmin, xmax, eps, max_derivative)
        return key, is_valid, params

    key, subkey = jax.random.split(key)
    initial_is_valid, initial_params = _generate_and_check_params_nondecreasing(subkey, xmin, xmax, eps, max_derivative)
    initial_state = (key, initial_is_valid, initial_params)
    final_state = jax.lax.while_loop(loop_cond, loop_body, initial_state)
    _, _, valid_params = final_state
    return valid_params # Returns (c, y0, yc, y1)

# @jit
def evaluate_pwl_single_from_params(params, x):
    """Evaluates a single PWL function H(x) from parameters."""
    c, y0, yc, y1 = params
    eps = 1e-6
    m1 = (yc - y0) / jnp.maximum(c, eps)
    f_seg1_coeff_x = y0
    f_seg1_coeff_x2 = 0.5 * m1
    f_at_c = 0.5 * c * (y0 + yc)
    m2 = (y1 - yc) / jnp.maximum(1.0 - c, eps)
    f_seg2_coeff_xc = yc
    f_seg2_coeff_xc2 = 0.5 * m2
    x = jnp.asarray(x)
    x = jnp.clip(x, 0.0, 1.0)
    f_val1 = f_seg1_coeff_x * x + f_seg1_coeff_x2 * jnp.square(x)
    x_minus_c = x - c
    f_val2 = f_at_c + f_seg2_coeff_xc * x_minus_c + f_seg2_coeff_xc2 * jnp.square(x_minus_c)
    is_segment1 = x < c
    f_val = jnp.where(is_segment1, f_val1, f_val2)
    # Enforce boundary conditions precisely
    f_val = jnp.where(x == 0.0, 0.0, f_val)
    f_val = jnp.where(x == 1.0, 1.0, f_val)
    return f_val

# Vectorized versions for batch processing
vmap_generate_pwl_params = jax.vmap(
    generate_params_max_deriv_nondecreasing,
    in_axes=(0, None, None, None, None)
)
vmap_evaluate_pwl = vmap(evaluate_pwl_single_from_params, in_axes=(0, None))

# -------------------------------------------------------------
# ======== ↑ piecewise linear derivative thicknesses ↑ ========
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ simulator setup ↓ ===================
# -------------------------------------------------------------

def setup_forward_model() -> Tuple[Callable, SetupParams, OpticsParams, LayerParams, LayerParams, int]:
    """Sets up the reflax forward model parameters and returns a callable function."""

    print("Setting up Reflax forward model...")
    interference_model = ONE_LAYER_INTERNAL_REFLECTIONS # Or TRANSFER_MATRIX_METHOD

    # Setup Parameters
    wavelength = 632.8 # nm
    polar_angle = jnp.deg2rad(25)
    azimuthal_angle = jnp.deg2rad(0)
    setup_params = SetupParams(
        wavelength=wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle
    )

    # Optics Parameters
    polarization_state = "Linear TE/perpendicular/s"
    transverse_electric_component, transverse_magnetic_component = polanalyze(polarization_state)
    permeability_reflection = 1.0
    permittivity_reflection = complex(1.0, 0.0)  # Air/Vacuum
    permeability_transmission = 1.0

    # Example substrate: Silicon (using values from refractiveindex.info at 632.8nm)
    # Make sure these are correct for your actual substrate
    n_substrate = 3.8827
    k_substrate = 0.019626
    permittivity_transmission = (n_substrate + 1j * k_substrate)**2

    optics_params = OpticsParams(
        permeability_reflection=permeability_reflection,
        permittivity_reflection=permittivity_reflection,
        permeability_transmission=permeability_transmission,
        permittivity_transmission=permittivity_transmission,
        transverse_electric_component=transverse_electric_component,
        transverse_magnetic_component=transverse_magnetic_component
    )

    # Layer Parameters
    backside_mode = 1

    static_layer_thicknesses = jnp.array([0.0]) # Effectively no static layer
    permeability_static_size_layers = jnp.array([permeability_transmission])
    permittivity_static_size_layers = jnp.array([permittivity_transmission])
    static_layer_params = LayerParams(
        permeabilities=permeability_static_size_layers,
        permittivities=permittivity_static_size_layers,
        thicknesses=static_layer_thicknesses
    )

    n_variable = 1.457
    k_variable = 0.0
    permeability_variable_layer = 1.0
    permittivity_variable_layer = (n_variable + 1j * k_variable)**2
    variable_layer_params = LayerParams(
        permeabilities = permeability_variable_layer,
        permittivities = permittivity_variable_layer,
        thicknesses = None # This will be provided during the call
    )

    # @jit
    def forward_model_func(thickness_profile: jnp.ndarray) -> jnp.ndarray:
        """Calculates reflectance for a given thickness profile."""
        reflectance = forward_model(
            model=interference_model,
            setup_params=setup_params,
            optics_params=optics_params,
            static_layer_params=static_layer_params,
            variable_layer_params=variable_layer_params,
            variable_layer_thicknesses=thickness_profile, # Shape (num_eval,)
            backside_mode=backside_mode,
            normalization=MIN_MAX_NORMALIZATION # Or NONE, or others
        )
        return reflectance

    print("Forward model setup complete.")
    # Return the callable and parameters if needed elsewhere
    return forward_model_func, setup_params, optics_params, static_layer_params, variable_layer_params, backside_mode

# -------------------------------------------------------------
# ==================== ↑ simulator setup ↑ ====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ main generation routine ↓ ================
# -------------------------------------------------------------

def main(args):
    print("Starting training data generation...")
    print(f"Parameters: {args}")

    key = random.PRNGKey(args.seed)
    N_total = args.n_pol + args.n_lin + args.n_const

    if N_total == 0:
        print("Warning: No profiles requested (N_pol=N_lin=N_const=0). Exiting.")
        return

    # --- Evaluation points ---
    x_eval = jnp.linspace(0.0, 1.0, args.num_eval)

    all_thickness_profiles = []
    profile_types = [] # Keep track of the type for plotting

    # --- 1. Generate Polynomial Thicknesses ---
    if args.n_pol > 0:
        print(f"\nGenerating {args.n_pol} polynomial profiles...")
        start_time = pytime.time()
        key, subkey_coeffs, subkey_tf = random.split(key, 3)
        keys_coeffs = random.split(subkey_coeffs, args.n_pol)
        poly_coeffs = vmap_random_monotonic_convex_polynomial(
            keys_coeffs, args.poly_base_degree, args.poly_max_deriv, 1000, 1e-6, 1e-7
        )
        base_poly_profiles = vmap_evaluate_polynomial(poly_coeffs, x_eval)
        final_thicknesses_pol = random.uniform(
            subkey_tf, (args.n_pol, 1), minval=args.min_final_thickness, maxval=args.max_final_thickness
        )
        thickness_profiles_pol = base_poly_profiles * final_thicknesses_pol
        all_thickness_profiles.append(thickness_profiles_pol)
        profile_types.extend(['polynomial'] * args.n_pol)
        print(f"Polynomial generation took {pytime.time() - start_time:.2f}s")

    # --- 2. Generate Piecewise Linear Derivative Thicknesses ---
    if args.n_lin > 0:
        print(f"\nGenerating {args.n_lin} piecewise linear derivative profiles...")
        start_time = pytime.time()
        key, subkey_params, subkey_tf = random.split(key, 3)
        keys_params = random.split(subkey_params, args.n_lin)
        jit_vmap_generate_pwl = jax.jit(vmap_generate_pwl_params, static_argnums=(1, 2, 3, 4))
        params_batch_tuple = jit_vmap_generate_pwl(keys_params, args.pwl_max_deriv, 0.1, 0.9, 1e-6)
        params_batch = jnp.stack(params_batch_tuple, axis=-1)
        base_pwl_profiles = vmap_evaluate_pwl(params_batch, x_eval)
        final_thicknesses_lin = random.uniform(
            subkey_tf, (args.n_lin, 1), minval=args.min_final_thickness, maxval=args.max_final_thickness
        )
        thickness_profiles_lin = base_pwl_profiles * final_thicknesses_lin
        all_thickness_profiles.append(thickness_profiles_lin)
        profile_types.extend(['pwl'] * args.n_lin)
        print(f"PWL generation took {pytime.time() - start_time:.2f}s")

    # --- 3. Generate Constant Derivative (Linear) Thicknesses ---
    if args.n_const > 0:
        print(f"\nGenerating {args.n_const} constant derivative (linear) profiles...")
        start_time = pytime.time()
        base_const_profile = x_eval
        if args.n_const == 1:
             final_thicknesses_const = jnp.array([(args.min_final_thickness + args.max_final_thickness) / 2.0])
        else:
            final_thicknesses_const = jnp.linspace(args.min_final_thickness, args.max_final_thickness, args.n_const)
        final_thicknesses_const = final_thicknesses_const.reshape(args.n_const, 1)
        thickness_profiles_const = base_const_profile * final_thicknesses_const
        all_thickness_profiles.append(thickness_profiles_const)
        profile_types.extend(['constant'] * args.n_const)
        print(f"Constant derivative generation took {pytime.time() - start_time:.2f}s")

    # --- Combine all thickness profiles ---
    if not all_thickness_profiles:
        print("No thickness profiles generated.")
        return

    combined_thicknesses = jnp.concatenate(all_thickness_profiles, axis=0)
    # profile_types list now corresponds row-wise to combined_thicknesses
    print(f"\nTotal generated thickness profiles: {combined_thicknesses.shape[0]}")
    print(f"Thickness profile shape: {combined_thicknesses.shape}")

    # --- Calculate Reflectances using Forward Model ---
    reflectances_calculated = False
    if not args.skip_forward_model:
        print("\nCalculating reflectances using forward model...")
        start_time = pytime.time()
        forward_model_func, *_ = setup_forward_model()
        if forward_model_func is None:
             print("Forward model function setup failed. Skipping reflectance calculation.")
             combined_reflectances = jnp.full_like(combined_thicknesses, jnp.nan)
        else:
            vmap_forward_model = jit(vmap(forward_model_func, in_axes=0))
            combined_reflectances = vmap_forward_model(combined_thicknesses)
            combined_reflectances.block_until_ready()
            reflectances_calculated = True # Mark as calculated
            print(f"Reflectance calculation took {pytime.time() - start_time:.2f}s")
            print(f"Reflectance profile shape: {combined_reflectances.shape}")

    elif args.skip_forward_model:
        print("\nSkipping reflectance calculation as requested.")
        combined_reflectances = jnp.full_like(combined_thicknesses, jnp.nan)
    else:
        print("\nSkipping reflectance calculation because reflax library is not available.")
        combined_reflectances = jnp.full_like(combined_thicknesses, jnp.nan)


    # --- Save Data ---
    print(f"\nSaving data to {args.output_file}...")
    try:
        np.savez_compressed(
            args.output_file,
            thicknesses=np.array(combined_thicknesses),
            reflectances=np.array(combined_reflectances),
            x_eval=np.array(x_eval),
            timepoints=np.array(x_eval),
            n_pol=args.n_pol,
            n_lin=args.n_lin,
            n_const=args.n_const,
            min_final_thickness=args.min_final_thickness,
            max_final_thickness=args.max_final_thickness,
            num_eval=args.num_eval,
            profile_types=np.array(profile_types) # Save profile types too
        )
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {e}")

    # --- Plotting ---
    num_to_plot = min(N_total, 100) # Plot a subset to avoid clutter
    # Ensure we have profiles to plot before sampling
    if N_total > 0:
        indices_to_plot = np.random.choice(N_total, num_to_plot, replace=False)
    else:
        indices_to_plot = np.array([], dtype=int) # Empty array if no profiles

    # --- Plot Thicknesses ---
    if len(indices_to_plot) > 0:
        print(f"\nGenerating plot of thickness profiles ({args.plot_file_thickness})...")
        plt.figure(figsize=(10, 6))

        # Define colors for consistency
        colors = {'polynomial': 'blue', 'pwl': 'green', 'constant': 'red'}
        plotted_labels = set()

        for i in indices_to_plot:
            ptype = profile_types[i]
            color = colors.get(ptype, 'grey') # Default color if type unknown
            label = f'{ptype.capitalize()} (Subset)' if ptype not in plotted_labels else ""
            plt.plot(x_eval, combined_thicknesses[i, :], color=color, alpha=0.3, lw=1, label=label)
            if label:
                plotted_labels.add(ptype)

        plt.axhline(args.min_final_thickness, color='grey', linestyle='--', label=f'Min Final T = {args.min_final_thickness:.0f} nm')
        plt.axhline(args.max_final_thickness, color='grey', linestyle='--', label=f'Max Final T = {args.max_final_thickness:.0f} nm')

        plt.xlabel("Normalized Process Variable (x)")
        plt.ylabel("Layer Thickness (nm)")
        plt.title(f"Generated Thickness Profiles (Subset N = {num_to_plot} of {N_total})")
        plt.ylim(bottom=0)
        plt.grid(True, linestyle=':', alpha=0.6)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize='small')

        try:
            plt.savefig(args.plot_file_thickness, dpi=300)
            print(f"Thickness plot saved successfully to {args.plot_file_thickness}")
        except Exception as e:
            print(f"Error saving thickness plot: {e}")
        # plt.show() # Uncomment to display plot interactively
        plt.close() # Close figure to free memory

    # --- Plot Reflectances ---
    if reflectances_calculated and len(indices_to_plot) > 0:
        print(f"\nGenerating plot of reflectance profiles ({args.plot_file_reflectance})...")
        plt.figure(figsize=(10, 6))

        # Reuse colors and subset indices
        colors = {'polynomial': 'darkblue', 'pwl': 'darkgreen', 'constant': 'darkred'} # Slightly different colors
        plotted_labels = set()

        for i in indices_to_plot:
             ptype = profile_types[i]
             color = colors.get(ptype, 'grey')
             label = f'{ptype.capitalize()} Reflectance (Subset)' if ptype not in plotted_labels else ""
             plt.plot(x_eval, combined_reflectances[i, :], color=color, alpha=0.3, lw=1, label=label)
             if label:
                 plotted_labels.add(ptype)

        plt.xlabel("Normalized Process Variable (x)")
        plt.ylabel("Normalized Reflectance") # Assuming MIN_MAX_NORMALIZATION
        plt.title(f"Generated Reflectance Profiles (Subset N = {num_to_plot} of {N_total})")
        plt.ylim(-1.1, 1.1) # Adjust if using different normalization
        plt.grid(True, linestyle=':', alpha=0.6)

        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize='small')

        try:
            plt.savefig(args.plot_file_reflectance, dpi=300)
            print(f"Reflectance plot saved successfully to {args.plot_file_reflectance}")
        except Exception as e:
            print(f"Error saving reflectance plot: {e}")
        # plt.show() # Uncomment to display plot interactively
        plt.close() # Close figure

    elif len(indices_to_plot) > 0:
        print(f"\nSkipping reflectance plot because reflectances were not calculated.")


    print("\nData generation complete.")


if __name__ == "__main__":
    args = parse_args()
    main(args)

# -------------------------------------------------------------
# ================ ↑ main generation routine ↑ ================
# -------------------------------------------------------------