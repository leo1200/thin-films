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
    # --- Modified arguments for min/max derivative ---
    parser.add_argument('--poly_min_deriv', type=float, default=0.0, help='Min derivative constraint for polynomial H\'(x)')
    parser.add_argument('--poly_max_deriv', type=float, default=2.0, help='Max derivative constraint for polynomial H\'(x)')
    parser.add_argument('--pwl_min_deriv', type=float, default=0.0, help='Min derivative constraint for PWL H\'(x)')
    parser.add_argument('--pwl_max_deriv', type=float, default=2.0, help='Max derivative constraint for PWL H\'(x)')
    # --- End modified arguments ---
    parser.add_argument('--output_file', type=str, default='training_data.npz', help='Output file name for saved data')
    parser.add_argument('--plot_file_thickness', type=str, default='generated_thicknesses.png', help='Output file name for thickness plot')
    parser.add_argument('--plot_file_reflectance', type=str, default='generated_reflectances.png', help='Output file name for reflectance plot')
    # --- New argument for derivative plot ---
    parser.add_argument('--plot_file_derivative', type=str, default='generated_derivatives.png', help='Output file name for derivative plot')
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
    # Ensure denominator doesn't cause issues if max_derivative <= 1 or if max_g0 is very close to max_derivative * integral_g0
    denominator = jnp.maximum(max_g0 - max_derivative * integral_g0, epsilon * 1e-3)

    alpha_target = epsilon * (max_derivative - 1.0) / denominator
    alpha = jax.lax.cond(
        max_derivative <= 1.0 + epsilon, # Allow max_derivative=1 within tolerance
        lambda _: 0.0,
        lambda _: jnp.maximum(0.0, alpha_target),
        operand=None
    )
    return alpha

# @jit # Jitting the outer loop with while_loop can be complex
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
    Generates coefficients of a random monotonic polynomial H(x) on [0, 1]
    such that H(0)=0, H(1)=1, H'(x) is non-decreasing (H''(x)>=0),
    and min_derivative <= H'(x) <= max_derivative.
    Final degree is base_degree + 2. Uses rejection sampling for constraints.
    """
    if min_derivative < 0.0:
        raise ValueError("min_derivative must be non-negative.")
    if max_derivative < min_derivative:
        raise ValueError("max_derivative must be >= min_derivative.")

    target_H_shape = (base_degree + 3,)
    x_grid = jnp.linspace(0.0, 1.0, n_grid)

    def check_constraints(coeffs_H_candidate):
        """Checks if H'(x) satisfies min/max constraints."""
        # Need to handle edge case where H is linear (degree 1)
        is_linear = jnp.all(coeffs_H_candidate[:-2] == 0.0) & (coeffs_H_candidate[-2] != 0.0)

        def check_poly():
            coeffs_H_prime = jnp.polyder(coeffs_H_candidate)
            # Handle case where derivative is constant (original poly was linear)
            is_const_deriv = coeffs_H_prime.shape[0] == 1
            h_prime_values = jnp.polyval(coeffs_H_prime, x_grid)
            min_h_prime = jnp.min(h_prime_values)
            max_h_prime = jnp.max(h_prime_values)
            # Use tolerance for floating point comparisons
            valid_min = min_h_prime >= min_derivative - epsilon
            valid_max = max_h_prime <= max_derivative + epsilon
            return valid_min & valid_max

        def check_lin():
            # If H(x) = x, H'(x) = 1. Check if 1 is within [min_deriv, max_deriv]
             return (1.0 >= min_derivative - epsilon) & (1.0 <= max_derivative + epsilon)

        return jax.lax.cond(is_linear, check_lin, check_poly)


    def generate_candidate(key_gen):
        """Generates one candidate polynomial H(x)."""
        key_p, _ = random.split(key_gen)
        coeffs_p = random.normal(key_p, shape=(base_degree + 1,))
        p_values = jnp.polyval(coeffs_p, x_grid)
        p_min = jnp.min(p_values)
        # Ensure p(x) >= 0
        coeffs_p0 = coeffs_p.at[-1].add(-p_min + epsilon) # Add epsilon to avoid exactly zero second derivative

        # g0(x) = integral(p0(x)) >= 0, g0'(x) = p0(x) >= 0 => g0 is non-decreasing
        coeffs_g0 = jnp.polyint(coeffs_p0, k=0.0)
        g0_values = jnp.polyval(coeffs_g0, x_grid)
        max_g0 = jnp.maximum(jnp.max(g0_values), 0.0)

        # h0(x) = integral(g0(x)) >= 0, h0'(x) = g0(x) >= 0 => h0 is non-decreasing
        # h0''(x) = g0'(x) = p0(x) >= 0 => h0 is convex
        coeffs_h0 = jnp.polyint(coeffs_g0, k=0.0)
        integral_g0 = jnp.polyval(coeffs_h0, 1.0) # This is h0(1)

        def compute_H_nonzero_integral(coeffs_g0_in):
            ig0 = jnp.maximum(integral_g0, integral_tol)
            mg0 = max_g0
            # Calculate alpha to potentially satisfy max_derivative constraint
            alpha = _calculate_alpha(mg0, ig0, max_derivative, epsilon)

            # Construct g(x) = alpha * g0(x) + epsilon
            coeffs_g_final = alpha * coeffs_g0_in
            coeffs_g_final = coeffs_g_final.at[-1].add(epsilon)

            # Construct h(x) = integral(g(x))
            coeffs_h_final = jnp.polyint(coeffs_g_final, k=0.0)
            h_final_at_1 = jnp.polyval(coeffs_h_final, 1.0)
            h_final_at_1_safe = jnp.maximum(h_final_at_1, epsilon * 1e-3)

            # Normalize: H(x) = h(x) / h(1)
            coeffs_H = coeffs_h_final / h_final_at_1_safe

            # Pad coefficients to target shape if needed (due to integration)
            padding = target_H_shape[0] - coeffs_H.shape[0]
            coeffs_H = jnp.pad(coeffs_H, (padding, 0))
            coeffs_H = jnp.reshape(coeffs_H, target_H_shape)
            return coeffs_H

        def compute_H_zero_integral(coeffs_g0_in):
            # If integral_g0 is zero, means g0 was zero, means p0 was zero.
            # Leads to H(x) = x.
            coeffs_H = jnp.zeros(target_H_shape, dtype=coeffs_g0_in.dtype)
            coeffs_H = coeffs_H.at[-2].set(1.0) # H(x) = x
            return coeffs_H

        coeffs_H_cand = jax.lax.cond(
            integral_g0 > integral_tol,
            compute_H_nonzero_integral,
            compute_H_zero_integral,
            operand=coeffs_g0
        )
        return coeffs_H_cand

    # Rejection sampling loop
    def loop_cond(state):
        _, is_valid = state
        return ~is_valid

    def loop_body(state):
        key, _ = state
        key, subkey = jax.random.split(key)
        coeffs_H_cand = generate_candidate(subkey)
        is_valid = check_constraints(coeffs_H_cand)
        # Ensure H(0)=0 and H(1)=1 (should be guaranteed by construction, but check)
        h0_check = jnp.isclose(jnp.polyval(coeffs_H_cand, 0.0), 0.0, atol=epsilon*10)
        h1_check = jnp.isclose(jnp.polyval(coeffs_H_cand, 1.0), 1.0, atol=epsilon*10)
        is_valid = is_valid & h0_check & h1_check
        return key, is_valid, coeffs_H_cand

    # Initial state
    key, subkey = jax.random.split(key)
    initial_coeffs = generate_candidate(subkey)
    initial_valid = check_constraints(initial_coeffs)
    h0_check_init = jnp.isclose(jnp.polyval(initial_coeffs, 0.0), 0.0, atol=epsilon*10)
    h1_check_init = jnp.isclose(jnp.polyval(initial_coeffs, 1.0), 1.0, atol=epsilon*10)
    initial_valid = initial_valid & h0_check_init & h1_check_init

    initial_state = (key, initial_valid, initial_coeffs)

    # Run the while loop
    final_state = jax.lax.while_loop(loop_cond, loop_body, initial_state)
    _, _, valid_coeffs_H = final_state

    return valid_coeffs_H


# @jit # Jitting causes issues with numpy polyval/polyder inside if not careful
def evaluate_polynomial(coeffs, x):
  """Evaluates a polynomial given its coefficients."""
  coeffs = jnp.atleast_1d(coeffs)
  if coeffs.ndim == 0:
      coeffs = jnp.array([coeffs.item()])
  elif coeffs.size == 0:
      coeffs = jnp.array([0.0])
  # Use numpy for evaluation within JAX - might impact performance/jit compatibility
  # but JAX's polyval can be sensitive to coefficient types/shapes sometimes.
  # For this script's context, np should be fine.
  return np.polyval(np.array(coeffs), np.array(x))


# Vectorized versions for batch processing
# Note: Vmapping the rejection loop might be inefficient. Consider generating more
# candidates than needed initially and filtering, or accept potential slowness.
# For moderate N, vmapping the whole function might be acceptable.
vmap_random_monotonic_convex_polynomial = vmap(
    random_monotonic_convex_polynomial_deriv_constraint,
    in_axes=(0, None, None, None, None, None, None) # key, base_deg, min_d, max_d, n_grid, eps, int_tol
)
# Vmap evaluation still works fine
vmap_evaluate_polynomial = vmap(evaluate_polynomial, in_axes=(0, None))

# -------------------------------------------------------------
# ================= ↑ polynomial thicknesses ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======== ↓ piecewise linear derivative thicknesses ↓ ========
# -------------------------------------------------------------

def _generate_and_check_params_deriv_constraint(key, xmin, xmax, eps, min_derivative, max_derivative):
    """Generates PWL parameters and checks constraints."""
    key_c, key_y = jax.random.split(key)
    # Generate center point
    c = jax.random.uniform(key_c, shape=(), minval=jnp.maximum(eps, xmin), maxval=jnp.minimum(1.0 - eps, xmax))
    # Generate initial y values (derivatives at 0, c, 1 before normalization)
    # Sample within a range that *might* lead to the desired final range after normalization
    y_raw = jax.random.uniform(key_y, shape=(3,), minval=eps, maxval=max_derivative * 1.5) # Heuristic upper bound
    y0_raw, yc_raw, y1_raw = y_raw[0], y_raw[1], y_raw[2]

    # Ensure raw values are non-decreasing before normalization
    # Sort raw y values to enforce non-decreasing nature physically
    y_sorted = jnp.sort(jnp.array([y0_raw, yc_raw, y1_raw]))
    y0_s, yc_s, y1_s = y_sorted[0], y_sorted[1], y_sorted[2]

    # Calculate integral H(1) = integral H'(x) dx
    integral_raw = 0.5 * c * (y0_s + yc_s) + 0.5 * (1.0 - c) * (yc_s + y1_s)
    integral_safe = jnp.maximum(integral_raw, eps) # Avoid division by zero

    # Scale derivatives so that the integral H(1) = 1
    scale = 1.0 / integral_safe
    y0_cand = y0_s * scale
    yc_cand = yc_s * scale
    y1_cand = y1_s * scale

    y_cand = jnp.array([y0_cand, yc_cand, y1_cand])

    # Check constraints on the *normalized* derivatives H'(x)
    # Non-decreasing check (should be guaranteed by sorting, but check with tolerance)
    is_non_decreasing = (y0_cand <= yc_cand + eps) & (yc_cand <= y1_cand + eps)
    # Max derivative check
    is_within_max_deriv = y1_cand <= max_derivative + eps # Only need to check y1 due to non-decreasing
    # Min derivative check
    is_within_min_deriv = y0_cand >= min_derivative - eps # Only need to check y0 due to non-decreasing

    is_valid = is_within_min_deriv & is_within_max_deriv & is_non_decreasing

    # Return the *sorted and scaled* parameters if valid
    return is_valid, (c, y0_cand, yc_cand, y1_cand)


# @functools.partial(jax.jit, static_argnames=['xmin', 'xmax', 'eps', 'min_derivative', 'max_derivative'])
def generate_params_deriv_constraint_nondecreasing(key, min_derivative, max_derivative, xmin=0.1, xmax=0.9, eps=1e-6):
    """Generates valid PWL parameters via rejection sampling."""
    if min_derivative < 0.0:
        raise ValueError("min_derivative must be non-negative.")
    if max_derivative < min_derivative:
        raise ValueError("max_derivative must be >= min_derivative.")
    # Average derivative must be 1. Need min <= 1 <= max (roughly)
    if max_derivative < 1.0 - eps and not jnp.isclose(max_derivative, 1.0, atol=eps):
         print(f"Warning: max_derivative ({max_derivative}) is < 1. May be difficult/impossible to satisfy H(1)=1.")
    if min_derivative > 1.0 + eps and not jnp.isclose(min_derivative, 1.0, atol=eps):
         print(f"Warning: min_derivative ({min_derivative}) is > 1. May be difficult/impossible to satisfy H(1)=1.")

    if not (0 < xmin < xmax < 1):
        raise ValueError("Interval (xmin, xmax) must be within (0, 1).")

    def loop_cond(state):
        _, is_valid, _ = state
        return ~is_valid

    def loop_body(state):
        key, _, _ = state
        key, subkey = jax.random.split(key)
        is_valid, params = _generate_and_check_params_deriv_constraint(subkey, xmin, xmax, eps, min_derivative, max_derivative)
        return key, is_valid, params

    key, subkey = jax.random.split(key)
    initial_is_valid, initial_params = _generate_and_check_params_deriv_constraint(subkey, xmin, xmax, eps, min_derivative, max_derivative)
    initial_state = (key, initial_is_valid, initial_params)

    # Use while_loop for automatic differentiation compatibility if needed,
    # otherwise a standard Python while loop might be conceptually simpler here.
    final_state = jax.lax.while_loop(loop_cond, loop_body, initial_state)
    _, _, valid_params = final_state
    return valid_params # Returns (c, y0, yc, y1)


# @jit
def evaluate_pwl_single_from_params(params, x):
    """Evaluates a single PWL function H(x) from parameters (c, y0, yc, y1)."""
    c, y0, yc, y1 = params
    eps = 1e-6

    # Calculate the integrated function H(x) = integral H'(t) dt from 0 to x
    # H'(t) = y0 + (yc - y0)/c * t for 0 <= t < c
    # H'(t) = yc + (y1 - yc)/(1-c) * (t - c) for c <= t <= 1

    # Integral for 0 <= x < c:
    # H(x) = integral [y0 + (yc - y0)/c * t] dt from 0 to x
    # H(x) = y0*x + 0.5 * (yc - y0)/c * x^2
    m1 = (yc - y0) / jnp.maximum(c, eps)
    f_seg1_coeff_x = y0
    f_seg1_coeff_x2 = 0.5 * m1

    # Value at x = c: H(c)
    f_at_c = y0*c + 0.5 * (yc - y0)/c * c**2
    f_at_c = 0.5 * c * (y0 + yc) # Simplified

    # Integral for c <= x <= 1:
    # H(x) = H(c) + integral [yc + (y1 - yc)/(1-c) * (t - c)] dt from c to x
    # Let u = t - c, du = dt. Limits: 0 to x-c
    # H(x) = H(c) + integral [yc + (y1 - yc)/(1-c) * u] du from 0 to x-c
    # H(x) = H(c) + yc*u + 0.5 * (y1 - yc)/(1-c) * u^2 | from 0 to x-c
    # H(x) = H(c) + yc*(x-c) + 0.5 * (y1 - yc)/(1-c) * (x-c)^2
    m2 = (y1 - yc) / jnp.maximum(1.0 - c, eps)
    f_seg2_coeff_xc = yc
    f_seg2_coeff_xc2 = 0.5 * m2

    # Evaluate based on x
    x = jnp.asarray(x)
    x = jnp.clip(x, 0.0, 1.0) # Ensure x is within [0, 1]

    f_val1 = f_seg1_coeff_x * x + f_seg1_coeff_x2 * jnp.square(x)

    x_minus_c = x - c
    f_val2 = f_at_c + f_seg2_coeff_xc * x_minus_c + f_seg2_coeff_xc2 * jnp.square(x_minus_c)

    is_segment1 = x < c
    f_val = jnp.where(is_segment1, f_val1, f_val2)

    # Enforce boundary conditions precisely due to potential float errors
    f_val = jnp.where(x == 0.0, 0.0, f_val)
    # H(1) should be 1 by construction (due to normalization), but enforce it
    f_val = jnp.where(x == 1.0, 1.0, f_val)

    return f_val

# Vectorized versions for batch processing
vmap_generate_pwl_params = jax.vmap(
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

    # Example variable layer: SiO2 (using values from refractiveindex.info at 632.8nm)
    n_variable = 1.457
    k_variable = 0.0
    permeability_variable_layer = 1.0
    permittivity_variable_layer = (n_variable + 1j * k_variable)**2
    variable_layer_params = LayerParams(
        permeabilities = permeability_variable_layer,
        permittivities = permittivity_variable_layer,
        thicknesses = None # This will be provided during the call
    )

    # @jit # Jitting the forward model itself is usually beneficial
    def forward_model_func(thickness_profile: jnp.ndarray) -> jnp.ndarray:
        """Calculates reflectance for a given thickness profile."""
        # Ensure thickness profile is compatible shape (e.g., 1D for this model)
        if thickness_profile.ndim > 1:
            thickness_profile = jnp.squeeze(thickness_profile)

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

        # Jit the vmapped polynomial generation for potential speedup
        jit_vmap_poly = jit(vmap_random_monotonic_convex_polynomial, static_argnums=(1, 4, 5, 6))
        poly_coeffs = jit_vmap_poly(
            keys_coeffs, args.poly_base_degree, args.poly_min_deriv, args.poly_max_deriv, 1000, 1e-6, 1e-7
        )
        poly_coeffs.block_until_ready() # Wait for generation to finish

        # Evaluate generated polynomials
        base_poly_profiles = vmap_evaluate_polynomial(poly_coeffs, x_eval)
        # Ensure result is JAX array for consistency
        base_poly_profiles = jnp.array(base_poly_profiles)

        # Apply final thickness scaling
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

        # Jit the vmapped PWL parameter generation
        jit_vmap_generate_pwl = jit(vmap_generate_pwl_params, static_argnums=(3, 4, 5)) # xmin, xmax, eps are static
        params_batch_tuple = jit_vmap_generate_pwl(keys_params, args.pwl_min_deriv, args.pwl_max_deriv, 0.1, 0.9, 1e-6)
        # Stack parameters into a single array: (N, 4) where cols are c, y0, yc, y1
        params_batch = jnp.stack(params_batch_tuple, axis=-1)
        params_batch.block_until_ready() # Wait for generation

        # Evaluate the PWL profiles
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
        # Linear profile H(x) = x. H'(x) = 1.
        # Check if min/max derivative constraints allow H'(x)=1
        poly_allows_linear = args.poly_min_deriv <= 1.0 and args.poly_max_deriv >= 1.0
        pwl_allows_linear = args.pwl_min_deriv <= 1.0 and args.pwl_max_deriv >= 1.0
        if not (poly_allows_linear and pwl_allows_linear):
             print("Warning: Specified min/max derivative constraints may exclude linear profiles (H'(x)=1).")

        base_const_profile = x_eval # H(x) = x
        if args.n_const == 1:
             # Use average thickness if only one requested
             final_thicknesses_const = jnp.array([(args.min_final_thickness + args.max_final_thickness) / 2.0])
        else:
             # Sample uniformly between min and max final thickness
             key, subkey_tf = random.split(key)
             final_thicknesses_const = random.uniform(
                 subkey_tf, (args.n_const,), minval=args.min_final_thickness, maxval=args.max_final_thickness
             )
             # Old linspace method:
             # final_thicknesses_const = jnp.linspace(args.min_final_thickness, args.max_final_thickness, args.n_const)

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
    # Ensure profile_types is a numpy array for consistent indexing later
    profile_types = np.array(profile_types)
    print(f"\nTotal generated thickness profiles: {combined_thicknesses.shape[0]}")
    print(f"Thickness profile shape: {combined_thicknesses.shape}")

    # --- Calculate First Derivatives ---
    print("\nCalculating first derivatives...")
    start_time = pytime.time()
    # Use jnp.gradient for numerical differentiation along the x_eval axis
    # Need dx for accurate scaling
    dx = x_eval[1] - x_eval[0] if len(x_eval) > 1 else 1.0
    combined_derivatives = jnp.gradient(combined_thicknesses, dx, axis=1)
    combined_derivatives.block_until_ready()
    print(f"Derivative calculation took {pytime.time() - start_time:.2f}s")
    print(f"Derivative profile shape: {combined_derivatives.shape}")


    # --- Calculate Reflectances using Forward Model ---
    reflectances_calculated = False
    combined_reflectances = jnp.full_like(combined_thicknesses, jnp.nan) # Initialize with NaN

    if not args.skip_forward_model:
        print("\nCalculating reflectances using forward model...")
        start_time = pytime.time()
        forward_model_func, *_ = setup_forward_model()
        if forward_model_func is None:
             print("Forward model function setup failed. Skipping reflectance calculation.")
        else:
            # Vmap the forward model function over the batch of thickness profiles
            vmap_forward_model = jit(vmap(forward_model_func, in_axes=0))
            try:
                combined_reflectances = vmap_forward_model(combined_thicknesses)
                combined_reflectances.block_until_ready() # Wait for calculations
                reflectances_calculated = True # Mark as calculated
                print(f"Reflectance calculation took {pytime.time() - start_time:.2f}s")
                print(f"Reflectance profile shape: {combined_reflectances.shape}")
            except Exception as e:
                print(f"Error during reflectance calculation: {e}")
                print("Reflectances will be filled with NaN.")

    elif args.skip_forward_model:
        print("\nSkipping reflectance calculation as requested.")
    else: # Reflax not available
        print("\nSkipping reflectance calculation because reflax library is not available.")


    # --- Save Data ---
    print(f"\nSaving data to {args.output_file}...")
    try:
        save_dict = {
            'thicknesses': np.array(combined_thicknesses),
            'reflectances': np.array(combined_reflectances),
            'derivatives': np.array(combined_derivatives), # Save derivatives
            'x_eval': np.array(x_eval),
            'timepoints': np.array(x_eval), # Keep 'timepoints' for potential compatibility
            'n_pol': args.n_pol,
            'n_lin': args.n_lin,
            'n_const': args.n_const,
            'min_final_thickness': args.min_final_thickness,
            'max_final_thickness': args.max_final_thickness,
            'num_eval': args.num_eval,
            'profile_types': profile_types, # Save profile types
            'poly_min_deriv': args.poly_min_deriv,
            'poly_max_deriv': args.poly_max_deriv,
            'pwl_min_deriv': args.pwl_min_deriv,
            'pwl_max_deriv': args.pwl_max_deriv,
        }
        np.savez_compressed(args.output_file, **save_dict)
        print("Data saved successfully.")
    except Exception as e:
        print(f"Error saving data: {e}")

    # --- Plotting ---
    num_to_plot = min(N_total, 100) # Plot a subset to avoid clutter
    if N_total > 0:
        # Use numpy for choice as JAX random state handling is verbose for this simple case
        np.random.seed(args.seed) # Use same seed for reproducible plots
        indices_to_plot = np.random.choice(N_total, num_to_plot, replace=False)
    else:
        indices_to_plot = np.array([], dtype=int) # Empty array if no profiles

    # Define colors for consistency across plots
    colors = {'polynomial': 'blue', 'pwl': 'green', 'constant': 'red'}


    # --- Plot Thicknesses ---
    if len(indices_to_plot) > 0:
        print(f"\nGenerating plot of thickness profiles ({args.plot_file_thickness})...")
        plt.figure(figsize=(10, 6))
        plotted_labels = set() # Track labels to avoid duplicates in legend

        for i in indices_to_plot:
            ptype = profile_types[i]
            color = colors.get(ptype, 'grey') # Default color if type unknown
            label = f'{str(ptype).capitalize()} (Subset)' if ptype not in plotted_labels else ""
            plt.plot(x_eval, combined_thicknesses[i, :], color=color, alpha=0.3, lw=1, label=label)
            if label:
                plotted_labels.add(ptype)

        # Add lines for min/max final thickness
        plt.axhline(args.min_final_thickness, color='grey', linestyle='--', lw=1, label=f'Min Final T = {args.min_final_thickness:.0f} nm')
        plt.axhline(args.max_final_thickness, color='grey', linestyle='--', lw=1, label=f'Max Final T = {args.max_final_thickness:.0f} nm')

        plt.xlabel("Normalized Process Variable (x)")
        plt.ylabel("Layer Thickness (nm)")
        plt.title(f"Generated Thickness Profiles (Subset N = {num_to_plot} of {N_total})")
        plt.ylim(bottom=0) # Thickness cannot be negative
        plt.grid(True, linestyle=':', alpha=0.6)

        # Consolidate legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles)) # Remove duplicate labels
        plt.legend(by_label.values(), by_label.keys(), fontsize='small')

        try:
            plt.savefig(args.plot_file_thickness, dpi=300, bbox_inches='tight')
            print(f"Thickness plot saved successfully to {args.plot_file_thickness}")
        except Exception as e:
            print(f"Error saving thickness plot: {e}")
        # plt.show() # Uncomment to display plot interactively
        plt.close() # Close figure to free memory


    # --- Plot First Derivatives ---
    if len(indices_to_plot) > 0:
        print(f"\nGenerating plot of first derivative profiles ({args.plot_file_derivative})...")
        plt.figure(figsize=(10, 6))
        plotted_labels = set() # Reset labels for this plot

        # Use slightly darker colors for derivatives if desired
        deriv_colors = {'polynomial': 'darkblue', 'pwl': 'darkgreen', 'constant': 'darkred'}

        for i in indices_to_plot:
            ptype = profile_types[i]
            color = deriv_colors.get(ptype, 'darkgrey')
            label = f'{str(ptype).capitalize()} Derivative (Subset)' if ptype not in plotted_labels else ""
            plt.plot(x_eval, combined_derivatives[i, :], color=color, alpha=0.3, lw=1, label=label)
            if label:
                plotted_labels.add(ptype)

        # Optional: Add lines for theoretical min/max growth rates IF they are constant
        # Note: The constraints apply to H'(x), not T'(x) = H'(x)*FinalT.
        # Visualizing the actual T'(x) is generally more useful.
        # We could plot lines for min/max possible T'(x) = min/max(H') * min/max(FinalT), but this might clutter the plot.
        # Let's keep it simple and just show the generated T'(x).

        plt.xlabel("Normalized Process Variable (x)")
        plt.ylabel("Thickness Growth Rate (nm / unit x)")
        plt.title(f"Generated First Derivative Profiles (Subset N = {num_to_plot} of {N_total})")
        # Determine appropriate y-limits dynamically or set manually if range is known
        # plt.ylim(...)
        plt.grid(True, linestyle=':', alpha=0.6)

        # Consolidate legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize='small')

        try:
            plt.savefig(args.plot_file_derivative, dpi=300, bbox_inches='tight')
            print(f"Derivative plot saved successfully to {args.plot_file_derivative}")
        except Exception as e:
            print(f"Error saving derivative plot: {e}")
        # plt.show()
        plt.close()


    # --- Plot Reflectances ---
    if reflectances_calculated and len(indices_to_plot) > 0:
        print(f"\nGenerating plot of reflectance profiles ({args.plot_file_reflectance})...")
        plt.figure(figsize=(10, 6))
        plotted_labels = set() # Reset labels

        # Reuse original colors or define new ones
        reflect_colors = {'polynomial': 'purple', 'pwl': 'orange', 'constant': 'cyan'}

        for i in indices_to_plot:
             ptype = profile_types[i]
             color = reflect_colors.get(ptype, 'black') # Use different colors for reflectance
             label = f'{str(ptype).capitalize()} Reflectance (Subset)' if ptype not in plotted_labels else ""
             plt.plot(x_eval, combined_reflectances[i, :], color=color, alpha=0.3, lw=1, label=label)
             if label:
                 plotted_labels.add(ptype)

        plt.xlabel("Normalized Process Variable (x)")
        # Label depends on normalization used in forward model
        ylabel = "Normalized Reflectance" if MIN_MAX_NORMALIZATION else "Reflectance"
        plt.ylabel(ylabel)
        plt.title(f"Generated Reflectance Profiles (Subset N = {num_to_plot} of {N_total})")
        # Y-limits depend on normalization
        if MIN_MAX_NORMALIZATION:
            plt.ylim(-1.1, 1.1)
        else:
            plt.ylim(0, 1.05) # Reflectance typically [0, 1]

        plt.grid(True, linestyle=':', alpha=0.6)

        # Consolidate legend
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys(), fontsize='small')

        try:
            plt.savefig(args.plot_file_reflectance, dpi=300, bbox_inches='tight')
            print(f"Reflectance plot saved successfully to {args.plot_file_reflectance}")
        except Exception as e:
            print(f"Error saving reflectance plot: {e}")
        # plt.show()
        plt.close()

    elif len(indices_to_plot) > 0:
        print(f"\nSkipping reflectance plot because reflectances were not calculated or an error occurred.")


    print("\nData generation complete.")


if __name__ == "__main__":
    args = parse_args()
    # Basic validation for derivative args
    if args.poly_min_deriv > args.poly_max_deriv:
        raise ValueError("poly_min_deriv cannot be greater than poly_max_deriv")
    if args.pwl_min_deriv > args.pwl_max_deriv:
        raise ValueError("pwl_min_deriv cannot be greater than pwl_max_deriv")
    if args.poly_min_deriv < 0 or args.pwl_min_deriv < 0:
        raise ValueError("Minimum derivatives cannot be negative")

    main(args)

# -------------------------------------------------------------
# ================ ↑ main generation routine ↑ ================
# -------------------------------------------------------------