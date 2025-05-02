# -*- coding: utf-8 -*-
"""

Generate training data consisting of

 - polynomial thicknesses, convex and monotonic, with given min/max derivative constraints
 - piecewise linear derivative thicknesses, convex and monotonic, with given min/max derivative constraints
 - linear thicknesses (constant derivative = 1, if allowed by constraints)

and their corresponding reflectances calculated using the Reflax library.

"""

# GPU selection (Set BEFORE importing JAX)
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # Choose the desired GPU index

# argument passing
import argparse

# typing
from typing import Callable, Optional, Tuple

# timing
import time as pytime

# numerics
import jax
import numpy as np
import jax.numpy as jnp
from jax import random, vmap, jit, lax

# reflax library (ensure it's installed and accessible)
try:
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
    REFLAX_AVAILABLE = True
except ImportError:
    print("WARNING: Reflax library not found. Reflectance calculation will be skipped.")
    REFLAX_AVAILABLE = False

# plotting
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# ======================= ↓ arguments ↓ =======================
# -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate Thickness-Reflectance Training Data with Derivative Constraints",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter # Show default values in help
    )
    # --- Data Generation Parameters ---
    parser.add_argument('--n_pol', type=int, default=500, help='Number of polynomial profiles')
    parser.add_argument('--n_lin', type=int, default=500, help='Number of piecewise-linear derivative profiles')
    parser.add_argument('--n_const', type=int, default=100, help='Number of constant derivative (linear) profiles')
    parser.add_argument('--num_eval', type=int, default=100, help='Number of evaluation points (time steps) for each profile')
    parser.add_argument('--min_final_thickness', type=float, default=800.0, help='Minimum final thickness (nm)')
    parser.add_argument('--max_final_thickness', type=float, default=1000.0, help='Maximum final thickness (nm)')

    # --- Profile Shape Parameters ---
    parser.add_argument('--poly_base_degree', type=int, default=3, help='Base degree for random polynomial p(x) (H(x) will have degree + 2)')
    parser.add_argument('--poly_min_deriv', type=float, default=0.2, help='Min derivative constraint for polynomial H\'(x) [where H(1)=1]')
    parser.add_argument('--poly_max_deriv', type=float, default=1.8, help='Max derivative constraint for polynomial H\'(x) [where H(1)=1]')
    parser.add_argument('--pwl_min_deriv', type=float, default=0.2, help='Min derivative constraint for PWL H\'(x) [where H(1)=1]')
    parser.add_argument('--pwl_max_deriv', type=float, default=1.8, help='Max derivative constraint for PWL H\'(x) [where H(1)=1]')

    # --- Output Files ---
    parser.add_argument('--output_file', type=str, default='training_data.npz', help='Output file name for saved data array')
    parser.add_argument('--plot_file_thickness', type=str, default='generated_thicknesses.png', help='Output file name for thickness plot')
    parser.add_argument('--plot_file_derivative', type=str, default='generated_derivatives.png', help='Output file name for derivative plot')
    parser.add_argument('--plot_file_reflectance', type=str, default='generated_reflectances.png', help='Output file name for reflectance plot')

    # --- Misc ---
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--skip_forward_model', action='store_true', help='Skip reflectance calculation (e.g., if reflax is unavailable or for testing)')

    return parser.parse_args()

# -------------------------------------------------------------
# ======================= ↑ arguments ↑ =======================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================= ↓ polynomial thicknesses ↓ ================
# -------------------------------------------------------------

def _calculate_alpha(max_g0: float, integral_g0: float, max_derivative: float, epsilon: float) -> float:
    """
    Calculates the scaling factor alpha used in polynomial generation.
    This aims to satisfy the max_derivative constraint but doesn't guarantee it.
    Used within the rejection sampling loop.
    """
    denominator = jnp.maximum(max_g0 - max_derivative * integral_g0, epsilon * 1e-3)
    alpha_target = epsilon * (max_derivative - 1.0) / denominator
    # If max_derivative <= 1, we don't need alpha scaling to reduce the derivative.
    # We might still generate candidates with derivative > max_derivative accidentally,
    # but the rejection sampling will catch them. Setting alpha=0 prevents using g0.
    alpha = lax.cond(
        max_derivative <= 1.0 + epsilon, # Allow max_derivative=1 within tolerance
        lambda _: 1.0, # If max_deriv <= 1, just use original g0 (alpha=1) and let rejection sampling work
                      # Previous: lambda _: 0.0, but this forces H(x)=x if max_deriv<=1
        lambda _: jnp.maximum(0.0, alpha_target),
        operand=None
    )
    # Clamp alpha to avoid excessively large values if denominator is tiny
    alpha = jnp.minimum(alpha, 1.0 / (epsilon * 1e-2)) # Heuristic upper bound
    return alpha

# @jit # Jitting the outer function with while_loop can be complex/slow compile time
def random_monotonic_convex_polynomial_deriv_constraint(
    key: random.PRNGKey,
    base_degree: int,
    min_derivative: float,
    max_derivative: float,
    n_grid: int = 1000,
    epsilon: float = 1e-6,
    integral_tol: float = 1e-7
) -> jnp.ndarray:
    """
    Generates coefficients of a random monotonic, convex polynomial H(x) on [0, 1]
    such that H(0)=0, H(1)=1, and min_derivative <= H'(x) <= max_derivative.
    Final degree is base_degree + 2. Uses rejection sampling.
    WARNING: Rejection sampling can be slow if the constraint range is narrow.
    """
    target_H_shape = (base_degree + 3,)
    x_grid = jnp.linspace(0.0, 1.0, n_grid)

    def check_constraints(coeffs_H_candidate):
        """Checks if H'(x) satisfies min/max constraints and H is convex."""
        # Check boundary values first (cheap checks)
        h0 = jnp.polyval(coeffs_H_candidate, 0.0)
        h1 = jnp.polyval(coeffs_H_candidate, 1.0)
        boundaries_ok = jnp.isclose(h0, 0.0, atol=epsilon*10) & jnp.isclose(h1, 1.0, atol=epsilon*10)

        # Check convexity (H''(x) >= 0)
        coeffs_H_prime = jnp.polyder(coeffs_H_candidate, 1)
        coeffs_H_double_prime = jnp.polyder(coeffs_H_prime, 1)
        # If H'' is empty (H was linear) or just a non-negative constant, it's convex.
        # Need to evaluate H''(x) on the grid if degree >= 2.
        is_convex = lax.cond(
            coeffs_H_double_prime.shape[0] == 0, # H was linear or constant
            lambda: True,
            lambda: lax.cond(
                 coeffs_H_double_prime.shape[0] == 1, # H was quadratic
                 lambda: coeffs_H_double_prime[0] >= -epsilon, # Check constant H''
                 lambda: jnp.all(jnp.polyval(coeffs_H_double_prime, x_grid) >= -epsilon) # Check H''(x) >= 0
            )
        )

        # Check derivative bounds H'(x)
        # Handle case where derivative is constant (original poly was linear)
        h_prime_values = jnp.polyval(coeffs_H_prime, x_grid)
        min_h_prime = jnp.min(h_prime_values)
        max_h_prime = jnp.max(h_prime_values)
        # Use tolerance for floating point comparisons
        valid_min = min_h_prime >= min_derivative - epsilon
        valid_max = max_h_prime <= max_derivative + epsilon
        deriv_bounds_ok = valid_min & valid_max

        return boundaries_ok & is_convex & deriv_bounds_ok


    def generate_candidate(key_gen):
        """Generates one candidate polynomial H(x) normalized to H(1)=1."""
        key_p, _ = random.split(key_gen)
        # Generate p(x) with random coefficients
        coeffs_p = random.normal(key_p, shape=(base_degree + 1,))
        p_values = jnp.polyval(coeffs_p, x_grid)
        p_min = jnp.min(p_values)
        # Ensure p(x) >= epsilon (to ensure H''(x) > 0 slightly, helps strict convexity)
        coeffs_p0 = coeffs_p.at[-1].add(-p_min + epsilon)

        # g0(x) = integral(p0(x)) >= 0, g0 is non-decreasing
        coeffs_g0 = jnp.polyint(coeffs_p0, k=0.0)
        g0_values = jnp.polyval(coeffs_g0, x_grid)
        max_g0 = jnp.maximum(jnp.max(g0_values), epsilon) # Ensure > 0

        # h0(x) = integral(g0(x)) >= 0, h0 is non-decreasing & convex
        coeffs_h0 = jnp.polyint(coeffs_g0, k=0.0)
        integral_g0 = jnp.polyval(coeffs_h0, 1.0) # This is h0(1)

        def compute_H_nonzero_integral(coeffs_g0_in):
            ig0 = jnp.maximum(integral_g0, integral_tol)
            mg0 = max_g0
            # Calculate alpha to *potentially* help satisfy max_derivative constraint
            alpha = _calculate_alpha(mg0, ig0, max_derivative, epsilon)

            # Construct g(x) = alpha * g0(x) + epsilon (add epsilon to ensure H'(x) > 0)
            coeffs_g_final = alpha * coeffs_g0_in
            coeffs_g_final = coeffs_g_final.at[-1].add(epsilon) # Ensure base derivative > 0

            # Construct h(x) = integral(g(x))
            coeffs_h_final = jnp.polyint(coeffs_g_final, k=0.0)
            h_final_at_1 = jnp.polyval(coeffs_h_final, 1.0)
            h_final_at_1_safe = jnp.maximum(h_final_at_1, epsilon * 1e-3) # Avoid division by zero

            # Normalize: H(x) = h(x) / h(1)
            coeffs_H = coeffs_h_final / h_final_at_1_safe

            # Pad coefficients to target shape if needed
            padding = target_H_shape[0] - coeffs_H.shape[0]
            coeffs_H = jnp.pad(coeffs_H, (padding, 0))
            coeffs_H = jnp.reshape(coeffs_H, target_H_shape)
            return coeffs_H

        def compute_H_zero_integral(coeffs_g0_in):
            # If integral_g0 is zero => g0=0 => p0=0. Base poly was constant.
            # Leads to H(x) = x.
            coeffs_H = jnp.zeros(target_H_shape, dtype=coeffs_g0_in.dtype)
            coeffs_H = coeffs_H.at[-2].set(1.0) # H(x) = x coeffs [1.0, 0.0] padded
            return coeffs_H

        coeffs_H_cand = lax.cond(
            integral_g0 > integral_tol,
            compute_H_nonzero_integral,
            compute_H_zero_integral,
            operand=coeffs_g0
        )
        return coeffs_H_cand

    # --- Rejection sampling loop ---
    def loop_cond(state):
        """Continue looping if constraints are not met."""
        _, is_valid, _ = state
        return ~is_valid

    def loop_body(state):
        """Generate a new candidate and check its validity."""
        key_in_loop, _, _ = state
        key_in_loop, subkey = random.split(key_in_loop)
        coeffs_H_cand = generate_candidate(subkey)
        is_valid = check_constraints(coeffs_H_cand)
        # Pass the candidate even if invalid, loop condition handles termination
        return key_in_loop, is_valid, coeffs_H_cand

    # Initial state for the loop
    key, subkey_init = random.split(key)
    initial_coeffs = generate_candidate(subkey_init)
    initial_valid = check_constraints(initial_coeffs)
    initial_state = (key, initial_valid, initial_coeffs)

    # Run the while loop using lax.while_loop for JAX compatibility
    final_state = lax.while_loop(loop_cond, loop_body, initial_state)

    # Return the valid coefficients from the final state
    _, _, valid_coeffs_H = final_state
    return valid_coeffs_H


@jit
def evaluate_polynomial(coeffs, x):
  """Evaluates a polynomial given its coefficients using jnp.polyval."""
  coeffs = jnp.atleast_1d(coeffs)
  # Handle case where coeffs might become 0-dimensional (e.g., constant)
  if coeffs.ndim == 0:
      coeffs = jnp.array([coeffs.item()])
  # Handle case where coeffs are empty (e.g., derivative of constant)
  elif coeffs.size == 0:
      coeffs = jnp.array([0.0]) # Represents zero polynomial
  return jnp.polyval(coeffs, x)


# Vectorized versions for batch processing
vmap_random_monotonic_convex_polynomial = vmap(
    random_monotonic_convex_polynomial_deriv_constraint,
    in_axes=(0, None, None, None, None, None, None) # key, base_deg, min_d, max_d, n_grid, eps, int_tol
)
vmap_evaluate_polynomial = vmap(evaluate_polynomial, in_axes=(0, None))

# -------------------------------------------------------------
# ================= ↑ polynomial thicknesses ↑ ================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ======== ↓ piecewise linear derivative thicknesses ↓ ========
# -------------------------------------------------------------

def _generate_and_check_params_deriv_constraint(key, xmin, xmax, eps, min_derivative, max_derivative):
    """Generates PWL parameters (c, y0, yc, y1) and checks constraints."""
    key_c, key_y = jax.random.split(key)

    # 1. Generate random cut position c
    c = random.uniform(key_c, shape=(), minval=jnp.maximum(eps, xmin), maxval=jnp.minimum(1.0 - eps, xmax))

    # 2. Generate 3 random numbers for raw derivatives
    # Sample within a range likely to produce valid results after normalization.
    # Heuristics: If range is narrow, sample closer to it. If wide, sample wider.
    y_range_mid = (min_derivative + max_derivative) / 2.0
    y_range_span = (max_derivative - min_derivative)
    raw_min = jnp.maximum(eps, y_range_mid - y_range_span * 1.5) # Allow going below min_deriv initially
    raw_max = y_range_mid + y_range_span * 1.5 # Allow going above max_deriv initially
    y_raw = random.uniform(key_y, shape=(3,), minval=raw_min, maxval=raw_max)

    # 3. Enforce non-decreasing derivatives physically by sorting
    y_sorted = jnp.sort(y_raw)
    y0_s, yc_s, y1_s = y_sorted[0], y_sorted[1], y_sorted[2]

    # 4. Calculate the integral of the sorted, raw derivatives H'(x)
    integral_raw = 0.5 * c * (y0_s + yc_s) + 0.5 * (1.0 - c) * (yc_s + y1_s)
    integral_safe = jnp.maximum(integral_raw, eps) # Avoid division by zero

    # 5. Calculate scaling factor needed so the final integral H(1) = 1
    scale = 1.0 / integral_safe

    # 6. Calculate final scaled derivative values (candidates H'(0), H'(c), H'(1))
    y0_cand = y0_s * scale
    yc_cand = yc_s * scale
    y1_cand = y1_s * scale

    # 7. Check validity of the *normalized* derivatives
    # Non-decreasing check (should be guaranteed by sorting, but check with tolerance)
    is_non_decreasing = (y0_cand <= yc_cand + eps) & (yc_cand <= y1_cand + eps)
    # Check if min/max derivative constraints are met
    is_within_max_deriv = y1_cand <= max_derivative + eps # Only need to check max value (y1)
    is_within_min_deriv = y0_cand >= min_derivative - eps # Only need to check min value (y0)

    is_valid = is_within_min_deriv & is_within_max_deriv & is_non_decreasing

    # Return the sorted and scaled parameters if valid
    return is_valid, (c, y0_cand, yc_cand, y1_cand)


# @jit # Jitting the outer loop with while_loop can be complex
def generate_params_deriv_constraint_nondecreasing(key, min_derivative, max_derivative, xmin=0.1, xmax=0.9, eps=1e-6):
    """Generates valid PWL parameters (c, y0, yc, y1) via rejection sampling."""
    # Input validation moved to main execution block

    # --- Rejection Sampling Loop ---
    def loop_cond(state):
        """Continue looping if parameters are not valid."""
        _, is_valid, _ = state
        return ~is_valid

    def loop_body(state):
        """Generate a new set of parameters and check validity."""
        key_in_loop, _, _ = state
        key_in_loop, subkey = random.split(key_in_loop)
        # Use the updated checking function
        is_valid, params = _generate_and_check_params_deriv_constraint(subkey, xmin, xmax, eps, min_derivative, max_derivative)
        return key_in_loop, is_valid, params

    # Initial state generation
    key, subkey_init = random.split(key)
    initial_is_valid, initial_params = _generate_and_check_params_deriv_constraint(subkey_init, xmin, xmax, eps, min_derivative, max_derivative)
    initial_state = (key, initial_is_valid, initial_params)

    # Run the while loop until valid parameters are found
    final_state = lax.while_loop(loop_cond, loop_body, initial_state)

    # Extract the valid parameters
    _, _, valid_params = final_state
    return valid_params # Returns tuple (c, y0, yc, y1)


@jit
def evaluate_pwl_single_from_params(params, x):
    """Evaluates a single PWL function H(x) from parameters (c, y0, yc, y1)."""
    c, y0, yc, y1 = params
    eps = 1e-6

    # Calculate the integrated function H(x) = integral H'(t) dt from 0 to x
    # H'(t) = y0 + (yc - y0)/c * t for 0 <= t < c
    # H'(t) = yc + (y1 - yc)/(1-c) * (t - c) for c <= t <= 1

    # Coefficients for integral in segment 1 (0 <= x < c):
    # H(x) = y0*x + 0.5 * (yc - y0)/c * x^2
    m1 = (yc - y0) / jnp.maximum(c, eps)
    f_seg1_coeff_x = y0
    f_seg1_coeff_x2 = 0.5 * m1

    # Value at the knot point x = c: H(c)
    f_at_c = 0.5 * c * (y0 + yc) # Simplified H(c) = integral from 0 to c

    # Coefficients for integral in segment 2 (c <= x <= 1):
    # H(x) = H(c) + yc*(x-c) + 0.5 * (y1 - yc)/(1-c) * (x-c)^2
    m2 = (y1 - yc) / jnp.maximum(1.0 - c, eps)
    f_seg2_coeff_xc = yc # Coefficient for (x-c) term
    f_seg2_coeff_xc2 = 0.5 * m2 # Coefficient for (x-c)^2 term

    # Evaluate based on x
    x = jnp.asarray(x)
    x = jnp.clip(x, 0.0, 1.0) # Ensure x is within [0, 1]

    # Calculate value for segment 1
    f_val1 = f_seg1_coeff_x * x + f_seg1_coeff_x2 * jnp.square(x)

    # Calculate value for segment 2
    x_minus_c = x - c
    f_val2 = f_at_c + f_seg2_coeff_xc * x_minus_c + f_seg2_coeff_xc2 * jnp.square(x_minus_c)

    # Choose value based on which segment x falls into
    is_segment1 = x < c
    f_val = jnp.where(is_segment1, f_val1, f_val2)

    # Enforce boundary conditions precisely due to potential float errors
    f_val = jnp.where(x == 0.0, 0.0, f_val)
    # H(1) should be 1 by construction (due to normalization), but enforce it
    f_val = jnp.where(x == 1.0, 1.0, f_val)

    return f_val

# Vectorized versions for batch processing
vmap_generate_pwl_params = vmap(
    generate_params_deriv_constraint_nondecreasing,
    in_axes=(0, None, None, None, None, None) # key, min_d, max_d, xmin, xmax, eps
)
vmap_evaluate_pwl = vmap(evaluate_pwl_single_from_params, in_axes=(0, None))

# -------------------------------------------------------------
# ======== ↑ piecewise linear derivative thicknesses ↑ ========
# -------------------------------------------------------------


# -------------------------------------------------------------
# ===================== ↓ simulator setup ↓ ===================
# -------------------------------------------------------------

def setup_forward_model() -> Tuple[Optional[Callable], Optional[SetupParams], Optional[OpticsParams], Optional[LayerParams], Optional[LayerParams], Optional[int]]:
    """Sets up the reflax forward model parameters and returns a callable function."""
    if not REFLAX_AVAILABLE:
        print("Skipping forward model setup as reflax is not available.")
        return None, None, None, None, None, None

    print("Setting up Reflax forward model...")
    # Choose the interference model appropriate for the setup
    # ONE_LAYER_INTERNAL_REFLECTIONS is often suitable for single growing films on substrates
    interference_model = ONE_LAYER_INTERNAL_REFLECTIONS # Or TRANSFER_MATRIX_METHOD

    # --- Setup Parameters ---
    wavelength = 632.8 # nm (HeNe laser)
    polar_angle = jnp.deg2rad(25) # Angle of incidence
    azimuthal_angle = jnp.deg2rad(0) # Typically 0 for isotropic samples
    setup_params = SetupParams(
        wavelength=wavelength,
        polar_angle=polar_angle,
        azimuthal_angle=azimuthal_angle
    )

    # --- Optics Parameters ---
    polarization_state = "Linear TE/perpendicular/s" # Common choice, alternatives: "Linear TM/parallel/p", "Linear +45", "RCP", "LCP"
    transverse_electric_component, transverse_magnetic_component = polanalyze(polarization_state)
    # Reflection medium (typically air or vacuum)
    permeability_reflection = 1.0
    permittivity_reflection = complex(1.0, 0.0)
    # Transmission medium (substrate)
    permeability_transmission = 1.0
    # Example: Silicon (Si) substrate at 632.8nm (Values from refractiveindex.info)
    # ** IMPORTANT: Replace with your actual substrate material properties **
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

    # --- Layer Parameters ---
    # Backside mode (1: coherent reflection from substrate backside, 0: incoherent)
    # Typically 1 if substrate is transparent and relatively thin/flat
    backside_mode = 1

    # Static layers (e.g., buffer layers between variable layer and substrate)
    # Assuming direct growth on substrate for simplicity
    static_layer_thicknesses = jnp.array([0.0]) # No static layers
    # These permeabilities/permittivities don't matter if thickness is 0, but should match substrate if thickness > 0
    permeability_static_size_layers = jnp.array([permeability_transmission])
    permittivity_static_size_layers = jnp.array([permittivity_transmission])
    static_layer_params = LayerParams(
        permeabilities=permeability_static_size_layers,
        permittivities=permittivity_static_size_layers,
        thicknesses=static_layer_thicknesses
    )

    # Variable layer parameters (the layer whose thickness profile we generate)
    # Example: Silicon Dioxide (SiO2) at 632.8nm (Values from refractiveindex.info)
    # ** IMPORTANT: Replace with your actual growing film material properties **
    n_variable = 1.457
    k_variable = 0.0 # Assuming non-absorbing film, set k > 0 if it absorbs
    permeability_variable_layer = 1.0
    permittivity_variable_layer = (n_variable + 1j * k_variable)**2
    # Ensure parameters are JAX arrays, even if single values
    variable_layer_params = LayerParams(
        permeabilities = permeability_variable_layer,
        permittivities = permittivity_variable_layer,
        thicknesses = None # This will be provided dynamically during the call
    )

    # Define the forward model function with fixed parameters using jit for performance
    @jit
    def forward_model_func(thickness_profile: jnp.ndarray) -> jnp.ndarray:
        """Calculates reflectance for a given 1D thickness profile."""
        # Ensure thickness profile is 1D as expected by this setup
        thickness_profile = jnp.squeeze(thickness_profile)

        reflectance = forward_model(
            model=interference_model,
            setup_params=setup_params,
            optics_params=optics_params,
            static_layer_params=static_layer_params,
            variable_layer_params=variable_layer_params,
            variable_layer_thicknesses=thickness_profile, # Shape (num_eval,)
            backside_mode=backside_mode,
            normalization=MIN_MAX_NORMALIZATION # Or NONE, depends on desired output range
        )
        return reflectance

    print("Forward model setup complete.")
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

    # --- Evaluation points (normalized time/process variable) ---
    x_eval = jnp.linspace(0.0, 1.0, args.num_eval)

    all_thickness_profiles = []
    profile_types = [] # Keep track of the type for plotting/analysis

    # --- 1. Generate Polynomial Thicknesses ---
    if args.n_pol > 0:
        print(f"\nGenerating {args.n_pol} polynomial profiles...")
        print("(This may take time due to rejection sampling for derivative constraints)")
        start_time = pytime.time()
        key, subkey_coeffs, subkey_tf = random.split(key, 3)
        keys_coeffs = random.split(subkey_coeffs, args.n_pol)

        # Jit the vmapped polynomial generation. Static args identify compile-time constants.
        # Note: If compilation takes too long or fails, remove jit here and accept slower runtime.
        jit_vmap_poly = jit(vmap_random_monotonic_convex_polynomial, static_argnums=(1, 4, 5, 6)) # base_degree, n_grid, epsilon, integral_tol
        try:
            poly_coeffs = jit_vmap_poly(
                keys_coeffs, args.poly_base_degree, args.poly_min_deriv, args.poly_max_deriv, 1000, 1e-6, 1e-7
            )
            poly_coeffs.block_until_ready() # Wait for generation to finish
            print(f"Polynomial coefficient generation took {pytime.time() - start_time:.2f}s")

            # Evaluate generated polynomials H(x)
            eval_start_time = pytime.time()
            base_poly_profiles = vmap_evaluate_polynomial(poly_coeffs, x_eval)
            base_poly_profiles.block_until_ready()
            print(f"Polynomial evaluation took {pytime.time() - eval_start_time:.2f}s")

            # Apply final thickness scaling T(x) = H(x) * FinalThickness
            final_thicknesses_pol = random.uniform(
                subkey_tf, (args.n_pol, 1), minval=args.min_final_thickness, maxval=args.max_final_thickness
            )
            thickness_profiles_pol = base_poly_profiles * final_thicknesses_pol
            all_thickness_profiles.append(thickness_profiles_pol)
            profile_types.extend(['polynomial'] * args.n_pol)

        except Exception as e:
             print(f"\n--- ERROR during Polynomial Generation ---")
             print(f"An error occurred: {e}")
             print("This might be due to very strict derivative constraints leading to excessive rejection sampling.")
             print("Consider widening the [min_deriv, max_deriv] range or reducing n_pol.")
             print("Skipping polynomial profile generation.")
             # Ensure n_pol is set to 0 so subsequent steps know
             args.n_pol = 0


    # --- 2. Generate Piecewise Linear Derivative Thicknesses ---
    if args.n_lin > 0:
        print(f"\nGenerating {args.n_lin} piecewise linear derivative profiles...")
        print("(This may take time due to rejection sampling)")
        start_time = pytime.time()
        key, subkey_params, subkey_tf = random.split(key, 3)
        keys_params = random.split(subkey_params, args.n_lin)

        # Jit the vmapped PWL parameter generation
        jit_vmap_generate_pwl = jit(vmap_generate_pwl_params, static_argnums=(3, 4, 5)) # xmin, xmax, eps are static
        try:
            params_batch_tuple = jit_vmap_generate_pwl(keys_params, args.pwl_min_deriv, args.pwl_max_deriv, 0.1, 0.9, 1e-6)
            # Stack parameters into a single array: (N, 4) where cols are c, y0, yc, y1
            params_batch = jnp.stack(params_batch_tuple, axis=-1)
            params_batch.block_until_ready() # Wait for generation
            print(f"PWL parameter generation took {pytime.time() - start_time:.2f}s")

            # Evaluate the PWL profiles H(x)
            eval_start_time = pytime.time()
            base_pwl_profiles = vmap_evaluate_pwl(params_batch, x_eval)
            base_pwl_profiles.block_until_ready()
            print(f"PWL evaluation took {pytime.time() - eval_start_time:.2f}s")

            # Apply final thickness scaling T(x) = H(x) * FinalThickness
            final_thicknesses_lin = random.uniform(
                subkey_tf, (args.n_lin, 1), minval=args.min_final_thickness, maxval=args.max_final_thickness
            )
            thickness_profiles_lin = base_pwl_profiles * final_thicknesses_lin
            all_thickness_profiles.append(thickness_profiles_lin)
            profile_types.extend(['pwl'] * args.n_lin)

        except Exception as e:
             print(f"\n--- ERROR during PWL Generation ---")
             print(f"An error occurred: {e}")
             print("This might be due to very strict derivative constraints leading to excessive rejection sampling.")
             print("Consider widening the [min_deriv, max_deriv] range or reducing n_lin.")
             print("Skipping PWL profile generation.")
             args.n_lin = 0 # Ensure subsequent steps know

        # --- 3. Generate Constant Derivative (Linear) Thicknesses ---
        # ... (code before generating final thicknesses) ...

        # Sample final thicknesses randomly within the range <--- OLD CODE
        # key, subkey_tf = random.split(key)
        # if args.n_const == 1:
        #      # Use average thickness if only one requested
        #      final_thicknesses_const = jnp.array([(args.min_final_thickness + args.max_final_thickness) / 2.0])
        # else:
        #      final_thicknesses_const = random.uniform(
        #          subkey_tf, (args.n_const,), minval=args.min_final_thickness, maxval=args.max_final_thickness
        #      )
        # final_thicknesses_const = final_thicknesses_const.reshape(args.n_const, 1) # Ensure correct shape for broadcasting

        # --- NEW CODE for regularly spaced final thicknesses ---
        if args.n_const == 1:
            # If only one, place it in the middle of the range
            final_thicknesses_const = jnp.array([(args.min_final_thickness + args.max_final_thickness) / 2.0])
        elif args.n_const > 1:
            # Generate regularly spaced values from min to max (inclusive)
            final_thicknesses_const = jnp.linspace(
                args.min_final_thickness,
                args.max_final_thickness,
                num=args.n_const,
                endpoint=True # Include max_final_thickness
            )
        else: # args.n_const == 0
             final_thicknesses_const = jnp.array([]) # Empty array

        # Reshape for broadcasting, even if empty or single element
        final_thicknesses_const = final_thicknesses_const.reshape(args.n_const, 1)
        # --- END NEW CODE ---

        base_const_profile = x_eval

        # Scale profile T(x) = x * FinalThickness
        thickness_profiles_const = base_const_profile * final_thicknesses_const
        all_thickness_profiles.append(thickness_profiles_const)
        profile_types.extend(['constant'] * args.n_const)
        print(f"Constant derivative generation took {pytime.time() - start_time:.2f}s")

    # --- Combine all thickness profiles ---
    if not all_thickness_profiles:
        print("\nNo thickness profiles were successfully generated. Exiting.")
        return

    combined_thicknesses = jnp.concatenate(all_thickness_profiles, axis=0)
    profile_types = np.array(profile_types) # Convert list to numpy array for indexing
    N_generated_total = combined_thicknesses.shape[0] # Actual number generated
    print(f"\nTotal generated thickness profiles: {N_generated_total}")
    print(f"Thickness profile shape: {combined_thicknesses.shape}") # Should be (N_generated_total, num_eval)

    # --- Calculate First Derivatives T'(x) ---
    print("\nCalculating first derivatives T'(x)...")
    start_time = pytime.time()
    # Use jnp.gradient for numerical differentiation along the x_eval axis (axis=1)
    # Need dx for accurate scaling: T'(x) = dT/dx approx DeltaT/DeltaX
    dx = x_eval[1] - x_eval[0] if len(x_eval) > 1 else 1.0
    combined_derivatives = jnp.gradient(combined_thicknesses, dx, axis=1)
    combined_derivatives.block_until_ready()
    print(f"Derivative calculation took {pytime.time() - start_time:.2f}s")
    print(f"Derivative profile shape: {combined_derivatives.shape}")


    # --- Calculate Reflectances using Forward Model ---
    reflectances_calculated = False
    combined_reflectances = jnp.full_like(combined_thicknesses, jnp.nan) # Initialize with NaN

    if REFLAX_AVAILABLE and not args.skip_forward_model:
        print("\nCalculating reflectances using forward model...")
        start_time = pytime.time()
        forward_model_func, *_ = setup_forward_model() # Get the callable model function

        if forward_model_func is None:
             # This case should not happen if REFLAX_AVAILABLE is True, but check anyway
             print("Forward model function setup failed unexpectedly. Skipping reflectance calculation.")
        else:
            # Vmap the forward model function over the batch of thickness profiles
            vmap_forward_model = jit(vmap(forward_model_func, in_axes=0))
            try:
                # Run calculation (potentially uses GPU if JAX is configured)
                combined_reflectances = vmap_forward_model(combined_thicknesses)
                combined_reflectances.block_until_ready() # Ensure calculation finishes
                reflectances_calculated = True # Mark as successfully calculated
                print(f"Reflectance calculation took {pytime.time() - start_time:.2f}s")
                print(f"Reflectance profile shape: {combined_reflectances.shape}") # Should match thickness shape
            except Exception as e:
                print(f"\n--- ERROR during Reflectance Calculation ---")
                print(f"An error occurred in the forward model: {e}")
                print("Check the forward model setup (optical constants, layer definitions).")
                print("Reflectances will be filled with NaN.")
                # Reflectances remain NaN

    elif args.skip_forward_model:
        print("\nSkipping reflectance calculation as requested via --skip_forward_model.")
    else: # Reflax not available
        print("\nSkipping reflectance calculation because reflax library is not available.")


    # --- Save Data ---
    print(f"\nSaving data to {args.output_file}...")
    try:
        # Convert JAX arrays to NumPy arrays for saving
        save_dict = {
            'thicknesses': np.array(combined_thicknesses),
            'reflectances': np.array(combined_reflectances),
            'derivatives': np.array(combined_derivatives),
            'x_eval': np.array(x_eval),
            'timepoints': np.array(x_eval), # Include 'timepoints' for potential compatibility
            'n_pol': args.n_pol, # Actual number generated might be less if errors occurred
            'n_lin': args.n_lin,
            'n_const': args.n_const,
            'num_eval': args.num_eval,
            'min_final_thickness': args.min_final_thickness,
            'max_final_thickness': args.max_final_thickness,
            'profile_types': profile_types, # Array indicating type of each profile
            'poly_min_deriv': args.poly_min_deriv, # Save constraints used
            'poly_max_deriv': args.poly_max_deriv,
            'pwl_min_deriv': args.pwl_min_deriv,
            'pwl_max_deriv': args.pwl_max_deriv,
            'generation_seed': args.seed,
        }
        # Use compression for potentially large arrays
        np.savez_compressed(args.output_file, **save_dict)
        print("Data saved successfully.")
    except Exception as e:
        print(f"\n--- ERROR saving data to {args.output_file} ---")
        print(f"An error occurred: {e}")

    # --- Plotting ---
    # Plot a random subset to avoid overly dense plots
    num_to_plot = min(N_generated_total, 10000)
    if N_generated_total > 0:
        # Use numpy for choice as JAX random state handling is more complex here
        np.random.seed(args.seed) # Use same seed for reproducible plot subsets
        indices_to_plot = np.random.choice(N_generated_total, num_to_plot, replace=False)
        # Convert JAX arrays to NumPy for plotting compatibility if not already done
        np_thicknesses = np.asarray(combined_thicknesses)
        np_derivatives = np.asarray(combined_derivatives)
        np_reflectances = np.asarray(combined_reflectances)
        np_x_eval = np.asarray(x_eval)
    else:
        indices_to_plot = np.array([], dtype=int) # Empty array if no profiles generated

    # Define colors for consistency across plots
    colors = {'polynomial': 'blue', 'pwl': 'green', 'constant': 'red'}
    alpha_val = 0.3 # Transparency for overlapping lines
    linewidth_val = 1.0

    # --- Plot Thicknesses ---
    if len(indices_to_plot) > 0:
        print(f"\nGenerating plot of thickness profiles ({args.plot_file_thickness})...")
        plt.style.use('seaborn-v0_8-whitegrid') # Use a nice style
        fig_th, ax_th = plt.subplots(figsize=(10, 6))
        plotted_labels_th = set() # Track labels to avoid duplicates in legend

        for i in indices_to_plot:
            ptype = profile_types[i]
            color = colors.get(ptype, 'grey') # Default color if type somehow unknown
            label = f'{str(ptype).capitalize()} (Subset)' if ptype not in plotted_labels_th else ""
            ax_th.plot(np_x_eval, np_thicknesses[i, :], color=color, alpha=alpha_val, lw=linewidth_val, label=label)
            if label:
                plotted_labels_th.add(ptype)

        # Add lines indicating the final thickness range
        ax_th.axhline(args.min_final_thickness, color='grey', linestyle='--', lw=1, label=f'Min Final T ({args.min_final_thickness:.0f} nm)')
        ax_th.axhline(args.max_final_thickness, color='grey', linestyle='--', lw=1, label=f'Max Final T ({args.max_final_thickness:.0f} nm)')

        ax_th.set_xlabel("Normalized Process Variable (x)")
        ax_th.set_ylabel("Layer Thickness (nm)")
        ax_th.set_title(f"Generated Thickness Profiles (Subset N = {num_to_plot} of {N_generated_total})")
        ax_th.set_ylim(bottom=0) # Thickness cannot be negative
        ax_th.grid(True, linestyle=':', alpha=0.6)

        # Consolidate legend by removing duplicate labels
        handles, labels = ax_th.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_th.legend(by_label.values(), by_label.keys(), fontsize='small')

        try:
            fig_th.savefig(args.plot_file_thickness, dpi=300, bbox_inches='tight')
            print(f"Thickness plot saved successfully to {args.plot_file_thickness}")
        except Exception as e:
            print(f"Error saving thickness plot: {e}")
        plt.close(fig_th) # Close figure to free memory


    # --- Plot First Derivatives ---
    if len(indices_to_plot) > 0:
        print(f"\nGenerating plot of first derivative profiles T'(x) ({args.plot_file_derivative})...")
        fig_dr, ax_dr = plt.subplots(figsize=(10, 6))
        plotted_labels_dr = set() # Reset labels for this plot

        # Use slightly darker colors for derivatives if desired, or reuse originals
        deriv_colors = {'polynomial': 'darkblue', 'pwl': 'darkgreen', 'constant': 'darkred'} # Example

        for i in indices_to_plot:
            ptype = profile_types[i]
            color = deriv_colors.get(ptype, 'darkgrey')
            label = f'{str(ptype).capitalize()} Derivative (Subset)' if ptype not in plotted_labels_dr else ""
            ax_dr.plot(np_x_eval, np_derivatives[i, :], color=color, alpha=alpha_val, lw=linewidth_val, label=label)
            if label:
                plotted_labels_dr.add(ptype)

        # Note: Plotting the H'(x) constraints scaled by FinalT might be complex because FinalT varies.
        # Showing the actual calculated T'(x) is usually most informative.

        ax_dr.set_xlabel("Normalized Process Variable (x)")
        ax_dr.set_ylabel("Thickness Growth Rate, T'(x) (nm / unit x)")
        ax_dr.set_title(f"Generated First Derivative Profiles T'(x) (Subset N = {num_to_plot} of {N_generated_total})")
        # Determine appropriate y-limits dynamically or set manually if range is known
        # ax_dr.set_ylim(...)
        ax_dr.grid(True, linestyle=':', alpha=0.6)

        handles, labels = ax_dr.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_dr.legend(by_label.values(), by_label.keys(), fontsize='small')

        try:
            fig_dr.savefig(args.plot_file_derivative, dpi=300, bbox_inches='tight')
            print(f"Derivative plot saved successfully to {args.plot_file_derivative}")
        except Exception as e:
            print(f"Error saving derivative plot: {e}")
        plt.close(fig_dr)


    # --- Plot Reflectances ---
    if reflectances_calculated and len(indices_to_plot) > 0:
        print(f"\nGenerating plot of reflectance profiles ({args.plot_file_reflectance})...")
        fig_rf, ax_rf = plt.subplots(figsize=(10, 6))
        plotted_labels_rf = set() # Reset labels

        # Use different colors for reflectance to distinguish from thickness/derivative plots
        reflect_colors = {'polynomial': 'purple', 'pwl': 'orange', 'constant': 'cyan'}

        for i in indices_to_plot:
             ptype = profile_types[i]
             color = reflect_colors.get(ptype, 'black')
             label = f'{str(ptype).capitalize()} Reflectance (Subset)' if ptype not in plotted_labels_rf else ""
             # Check for NaN values in reflectance before plotting
             if not np.all(np.isnan(np_reflectances[i, :])):
                 ax_rf.plot(np_x_eval, np_reflectances[i, :], color=color, alpha=alpha_val, lw=linewidth_val, label=label)
                 if label:
                     plotted_labels_rf.add(ptype)
             else:
                 # Optionally report if a profile's reflectance was all NaN
                 # print(f"Skipping plot for profile {i} (type {ptype}) due to NaN reflectance.")
                 pass

        ax_rf.set_xlabel("Normalized Process Variable (x)")
        # Label depends on normalization used in forward model
        ylabel = "Normalized Reflectance" if MIN_MAX_NORMALIZATION else "Reflectance" # Check setup_forward_model
        ax_rf.set_ylabel(ylabel)
        ax_rf.set_title(f"Generated Reflectance Profiles (Subset N = {num_to_plot} of {N_generated_total})")

        # Y-limits depend on normalization chosen in forward model
        if MIN_MAX_NORMALIZATION:
            ax_rf.set_ylim(-1.1, 1.1)
        else:
            # Reflectance is typically between 0 and 1, but allow some margin
            ax_rf.set_ylim(-0.05, 1.05)

        ax_rf.grid(True, linestyle=':', alpha=0.6)

        handles, labels = ax_rf.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax_rf.legend(by_label.values(), by_label.keys(), fontsize='small')

        try:
            fig_rf.savefig(args.plot_file_reflectance, dpi=300, bbox_inches='tight')
            print(f"Reflectance plot saved successfully to {args.plot_file_reflectance}")
        except Exception as e:
            print(f"Error saving reflectance plot: {e}")
        plt.close(fig_rf)

    elif len(indices_to_plot) > 0:
        # Condition if reflectances were skipped or failed
        print(f"\nSkipping reflectance plot because reflectances were not calculated or calculation failed.")


    print("\n------------------------------------")
    print("Data generation script finished.")
    print("------------------------------------")


if __name__ == "__main__":
    args = parse_args()

    # --- Input Argument Validation ---
    if args.poly_min_deriv > args.poly_max_deriv:
        raise ValueError("poly_min_deriv cannot be greater than poly_max_deriv")
    if args.pwl_min_deriv > args.pwl_max_deriv:
        raise ValueError("pwl_min_deriv cannot be greater than pwl_max_deriv")
    if args.poly_min_deriv < 0 or args.pwl_min_deriv < 0:
        # Allowing min_derivative = 0 is fine (constant thickness start)
        raise ValueError("Minimum derivatives cannot be negative (must be >= 0)")
    if args.min_final_thickness <= 0:
        raise ValueError("min_final_thickness must be positive")
    if args.max_final_thickness < args.min_final_thickness:
        raise ValueError("max_final_thickness must be >= min_final_thickness")
    if args.num_eval < 2:
        raise ValueError("num_eval must be at least 2 for derivatives")

    # Warnings for potentially problematic derivative constraints for PWL H(1)=1 goal
    # Average derivative must be 1. Need min <= 1 <= max.
    eps_check = 1e-6
    if args.pwl_max_deriv < 1.0 - eps_check:
         print(f"Warning: pwl_max_derivative ({args.pwl_max_deriv}) is < 1.0. May be difficult/impossible to generate PWL profiles satisfying H(1)=1.")
    if args.pwl_min_deriv > 1.0 + eps_check:
         print(f"Warning: pwl_min_derivative ({args.pwl_min_deriv}) is > 1.0. May be difficult/impossible to generate PWL profiles satisfying H(1)=1.")
    # Same logic applies to polynomial constraints
    if args.poly_max_deriv < 1.0 - eps_check:
         print(f"Warning: poly_max_derivative ({args.poly_max_deriv}) is < 1.0. May be difficult/impossible to generate Polynomial profiles satisfying H(1)=1.")
    if args.poly_min_deriv > 1.0 + eps_check:
         print(f"Warning: poly_min_derivative ({args.poly_min_deriv}) is > 1.0. May be difficult/impossible to generate Polynomial profiles satisfying H(1)=1.")


    # --- Run Main Generation Function ---
    main(args)

# -------------------------------------------------------------
# ================ ↑ main generation routine ↑ ================
# -------------------------------------------------------------