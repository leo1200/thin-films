import jax
import jax.numpy as jnp
import jax.random as random
from jax import jit, lax, nn # Import jax.nn for softplus
# from jax.debug import print as jax_print # Keep commented unless actively debugging
from jax.scipy.linalg import cholesky
import matplotlib.pyplot as plt
import time
import numpy as np # Import numpy for saving and variance calc outside JIT

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # Deactivated for broader compatibility

# <<< --- Ensure Float64 --- >>>
from jax import config
config.update("jax_enable_x64", True)
print(f"JAX using 64-bit precision: {config.jax_enable_x64}")
# <<< --- End Float64 --- >>>


# --- Custom cumulative_trapezoid_jax ---
@jit
def cumulative_trapezoid_jax(y, x=None, dx=1.0, initial=0.0):
    """
    JAX implementation of cumulative trapezoidal integration using float64.
    """
    dtype = jnp.float64
    y = jnp.asarray(y, dtype=dtype)
    if x is not None:
        x = jnp.asarray(x, dtype=dtype)
        if x.ndim != 1: raise ValueError("x must be 1D.")
        if x.shape[0] != y.shape[0]: raise ValueError(f"x ({x.shape}) and y ({y.shape}) must have the same first dimension.")
        d = jnp.diff(x)
    else:
        d = jnp.asarray(dx, dtype=dtype)
    integrands = 0.5 * (y[:-1] + y[1:]) * d
    integrated = jnp.cumsum(integrands)
    initial_shape = (1,) + y.shape[1:]
    initial_arr = jnp.full(initial_shape, initial, dtype=integrated.dtype)
    result = jnp.concatenate([initial_arr, integrated], axis=0)
    if result.shape != y.shape:
        if y.ndim == 1 and result.ndim == 1 and result.shape[0] == y.shape[0]: pass
        else: pass
    return result


# --- GP Helper Functions ---
@jit
def rbf_kernel(X1, X2, lengthscale, variance):
    """ RBF kernel using float64. """
    dtype = jnp.float64
    X1_r = jnp.reshape(X1, (-1, 1)).astype(dtype)
    X2_r = jnp.reshape(X2, (-1, 1)).astype(dtype)
    lengthscale = jnp.asarray(lengthscale, dtype=dtype)
    variance = jnp.asarray(variance, dtype=dtype)
    diff = X1_r - X2_r.T
    sq_dist = diff**2
    return variance * jnp.exp(-0.5 * sq_dist / (lengthscale**2 + jnp.asarray(1e-12, dtype=dtype)))

# Corrected sample_gp_prior function
@jit
def sample_gp_prior(key: random.PRNGKey,
                    x_grid: jnp.ndarray,
                    lengthscale: float,
                    variance: float,
                    jitter: float = 1e-6):
    """ Sample from GP prior using float64 and lax.cond for stability. """
    dtype = jnp.float64
    # n_grid is calculated here and is static within this function's scope
    n_grid = x_grid.shape[0]
    if n_grid <= 1: return jnp.zeros_like(x_grid, dtype=dtype)
    x_grid = x_grid.astype(dtype)

    K = rbf_kernel(x_grid, x_grid, lengthscale, variance)
    K = K.astype(dtype)
    jitter_val = jnp.asarray(jitter, dtype=dtype)
    K_stable = K + jnp.eye(n_grid, dtype=dtype) * jitter_val
    k_stable_ok = ~jnp.any(jnp.isnan(K_stable) | jnp.isinf(K_stable))

    def try_cholesky(matrix):
        try:
            L = cholesky(matrix, lower=True)
            cholesky_ok = ~jnp.any(jnp.isnan(L) | jnp.isinf(L))
            return L.astype(dtype), cholesky_ok
        except Exception as e:
            dummy_L = jnp.zeros_like(matrix, dtype=dtype)
            return dummy_L, False

    L, cholesky_ok = lax.cond(
        k_stable_ok,
        try_cholesky,
        lambda mat: (jnp.zeros_like(mat, dtype=dtype), False),
        operand=K_stable
    )
    is_stable = k_stable_ok & cholesky_ok

    # Inner functions capture n_grid from outer scope
    def stable_sample(args):
        key_in, L_in = args # No n_grid_in here
        # Use n_grid from the outer scope
        z = random.normal(key_in, shape=(n_grid,), dtype=dtype)
        return L_in @ z

    def unstable_sample(args):
        key_in, L_in = args # L_in is dummy here
        # Use n_grid from the outer scope
        return jnp.zeros((n_grid,), dtype=dtype)

    # Pass only dynamic args (key, L) as operand
    f_values = lax.cond(
        is_stable,
        stable_sample,
        unstable_sample,
        operand=(key, L) # Don't pass n_grid here
    )
    return f_values.astype(dtype)


# --- Main Generation Function ---
# Can add @jit back here now if desired, as internal functions use lax.cond

def random_smooth_monotonic_convex_gp_adaptive_scale(
    key: random.PRNGKey,
    min_derivative: float,
    max_derivative: float,
    n_grid: int = 100,
    lengthscale_f_range: tuple[float, float] = (0.15, 0.5),
    variance_f_range: tuple[float, float] = (1.0, 5.0),
    epsilon: float = 1e-7,
    max_attempts: int = 1000,
    h2_min_value: float = 1e-6,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, bool, float, float]: # Return sampled hypers
    """
    Generates H(x) via adaptive scaling, using H''_c = softplus(f) + eps.
    Samples base GP hyperparameters L and Var randomly for each attempt.
    Samples A uniformly from [A_min, A_max]. Enforces float64.
    Returns data, validity flag, and the hyperparameters used for the valid sample.
    """
    dtype = jnp.float64
    if min_derivative > max_derivative:
        raise ValueError(f"min_derivative ({min_derivative}) > max_derivative ({max_derivative})")

    # Ensure n_grid is passed correctly if outer function is JITted
    # If JITting, n_grid needs to be static_argnum or handled differently.
    # Assuming n_grid is constant for now.
    x_grid = jnp.linspace(0.0, 1.0, n_grid, dtype=dtype)
    if n_grid <= 1:
        # print("Warning: n_grid <= 1, returning zeros.") # Cannot print inside JIT
        zeros = jnp.zeros((n_grid,), dtype=dtype) if n_grid == 1 else jnp.zeros((0,), dtype=dtype)
        return x_grid, zeros, zeros, zeros, False, 0.0, 0.0

    def loop_cond(state):
        key_in, is_valid, _, _, _, attempts, _, _ = state
        return (~is_valid) & (attempts < max_attempts)

    def loop_body(state):
        key_in_loop, _, _, _, _, attempts, _, _ = state
        key_L, key_V, key_gp, key_A, key_out_loop = random.split(key_in_loop, 5)
        current_attempt = attempts + 1

        lengthscale_f = random.uniform(key_L, shape=(), minval=lengthscale_f_range[0], maxval=lengthscale_f_range[1], dtype=dtype)
        variance_f = random.uniform(key_V, shape=(), minval=variance_f_range[0], maxval=variance_f_range[1], dtype=dtype)

        f_values_raw = sample_gp_prior(key_gp, x_grid, lengthscale_f, variance_f)
        f_nan_inf = jnp.any(jnp.isnan(f_values_raw) | jnp.isinf(f_values_raw))

        h2_min = jnp.asarray(h2_min_value, dtype=dtype)
        H_double_prime_candidate = nn.softplus(f_values_raw) + h2_min
        h2c_ni = jnp.any(jnp.isnan(H_double_prime_candidate)) | jnp.any(jnp.isinf(H_double_prime_candidate))

        H_prime_base = cumulative_trapezoid_jax(H_double_prime_candidate, x=x_grid, initial=jnp.asarray(0.0, dtype=dtype))
        H_base = cumulative_trapezoid_jax(H_prime_base, x=x_grid, initial=jnp.asarray(0.0, dtype=dtype))
        hpb_ni = jnp.any(jnp.isnan(H_prime_base)) | jnp.any(jnp.isinf(H_prime_base))
        hb_ni = jnp.any(jnp.isnan(H_base)) | jnp.any(jnp.isinf(H_base))

        H_base_at_1 = H_base[-1]
        G = H_prime_base - H_base_at_1
        g_ni = jnp.any(jnp.isnan(G)) | jnp.any(jnp.isinf(G))
        hb1_ni = jnp.isnan(H_base_at_1) | jnp.isinf(H_base_at_1)
        early_nan_inf = f_nan_inf | h2c_ni | hpb_ni | hb_ni | g_ni | hb1_ni

        candidate_is_viable = ~early_nan_inf
        A_min = jnp.asarray(epsilon, dtype=dtype)
        A_max = jnp.asarray(epsilon, dtype=dtype)

        def calculate_bounds_and_viability(_):
            min_deriv_m1 = min_derivative - 1.0
            max_deriv_m1 = max_derivative - 1.0
            div_epsilon = epsilon * 10.0
            valid_mask_g_pos = G > div_epsilon
            safe_denom_g_pos = jnp.where(valid_mask_g_pos, G, 1.0)
            lower1_vals = jnp.where(valid_mask_g_pos, min_deriv_m1 / safe_denom_g_pos, -jnp.inf)
            L1 = jnp.max(lower1_vals)
            valid_mask_g_neg = G < -div_epsilon
            safe_denom_g_neg = jnp.where(valid_mask_g_neg, G, -1.0)
            lower2_vals = jnp.where(valid_mask_g_neg, max_deriv_m1 / safe_denom_g_neg, -jnp.inf)
            L2 = jnp.max(lower2_vals)
            upper1_vals = jnp.where(valid_mask_g_pos, max_deriv_m1 / safe_denom_g_pos, jnp.inf)
            U1 = jnp.min(upper1_vals)
            upper2_vals = jnp.where(valid_mask_g_neg, min_deriv_m1 / safe_denom_g_neg, jnp.inf)
            U2 = jnp.min(upper2_vals)
            A_min_calc_ = jnp.maximum(L1, L2)
            A_min_ = jnp.maximum(A_min_calc_, epsilon)
            A_max_ = jnp.minimum(U1, U2)
            bounds_are_finite_ = jnp.isfinite(A_min_) & jnp.isfinite(A_max_)
            have_valid_interval_ = bounds_are_finite_ & (A_max_ > A_min_ + epsilon)
            candidate_is_viable_ = have_valid_interval_
            return candidate_is_viable_, A_min_, A_max_

        candidate_is_viable, A_min, A_max = lax.cond(
            ~early_nan_inf,
            calculate_bounds_and_viability,
            lambda _: (False, A_min, A_max),
            operand=None
        )

        def compute_valid_outputs(key_A_in, A_min_in, A_max_in, H_double_prime_c, H_prime_b, H_b, H_b_at_1, x_g):
            A = random.uniform(key_A_in, shape=(), minval=A_min_in, maxval=A_max_in, dtype=dtype)
            A = jnp.maximum(A, epsilon)
            a_ok = jnp.isfinite(A) & (A > 0)

            def calc_outputs_if_a_ok(a_val):
                H_double_prime = a_val * H_double_prime_c
                C1 = 1.0 - a_val * H_b_at_1
                H_prime = a_val * H_prime_b + C1
                H_values = a_val * H_b + C1 * x_g
                h_ni = jnp.any(jnp.isnan(H_values)) | jnp.any(jnp.isinf(H_values))
                hp_ni = jnp.any(jnp.isnan(H_prime)) | jnp.any(jnp.isinf(H_prime))
                hpp_ni = jnp.any(jnp.isnan(H_double_prime)) | jnp.any(jnp.isinf(H_double_prime))
                final_step_nan_inf = h_ni | hp_ni | hpp_ni
                return H_values, H_prime, H_double_prime, ~final_step_nan_inf

            def return_zeros_and_invalid():
                 zeros = jnp.zeros_like(x_g, dtype=dtype)
                 return zeros, zeros, zeros, False

            H_values, H_prime, H_double_prime, actual_validity = lax.cond(
                a_ok,
                calc_outputs_if_a_ok,
                lambda _: return_zeros_and_invalid(),
                operand=A
            )
            return H_values, H_prime, H_double_prime, actual_validity

        def compute_invalid_outputs():
            zeros = jnp.zeros_like(x_grid, dtype=dtype)
            return zeros, zeros, zeros, False

        final_H, final_H_prime, final_H_double_prime, actual_is_valid = lax.cond(
            candidate_is_viable,
            lambda k: compute_valid_outputs(k, A_min, A_max, H_double_prime_candidate, H_prime_base, H_base, H_base_at_1, x_grid),
            lambda k: compute_invalid_outputs(),
            operand=key_A
        )

        return key_out_loop, actual_is_valid, final_H, final_H_prime, final_H_double_prime, attempts + 1, lengthscale_f, variance_f

    zeros_shape = (n_grid,) if n_grid > 0 else (0,)
    initial_state = (key, False, jnp.zeros(zeros_shape, dtype=dtype), jnp.zeros(zeros_shape, dtype=dtype), jnp.zeros(zeros_shape, dtype=dtype), 0, 0.0, 0.0)

    if n_grid <= 1:
         final_valid = False
         final_H, final_H_prime, final_H_double_prime = initial_state[2:5]
         final_attempts = 0
         final_L, final_V = 0.0, 0.0
    else:
        final_state = lax.while_loop(loop_cond, loop_body, initial_state)
        final_key, final_valid, final_H, final_H_prime, final_H_double_prime, final_attempts, final_L, final_V = final_state

    # print(f"\nAdaptive scaling attempt finished after {final_attempts} iterations. Valid flag from loop: {final_valid}") # Reduce verbosity

    final_H_np = np.array(final_H)
    final_H_prime_np = np.array(final_H_prime)
    final_H_double_prime_np = np.array(final_H_double_prime)
    nan_inf_present = np.any(np.isnan(final_H_np)) or np.any(np.isinf(final_H_np)) or \
                      np.any(np.isnan(final_H_prime_np)) or np.any(np.isinf(final_H_prime_np)) or \
                      np.any(np.isnan(final_H_double_prime_np)) or np.any(np.isinf(final_H_double_prime_np))

    if final_valid and nan_inf_present:
        # print("!!! WARNING: Loop state is VALID but final arrays contain NaN/Inf !!!") # Reduce verbosity
        final_valid_corrected = False
    else:
        final_valid_corrected = final_valid

    if not final_valid_corrected:
        zeros = jnp.zeros_like(x_grid, dtype=dtype)
        final_H, final_H_prime, final_H_double_prime = zeros, zeros, zeros
        final_L, final_V = 0.0, 0.0

    return x_grid, final_H, final_H_prime, final_H_double_prime, final_valid_corrected, final_L, final_V


# --- Example Usage: Find Sample with Highest H' Variance ---
if __name__ == "__main__":
    key = random.PRNGKey(42)
    n_target_samples = 50
    n_plot_samples = 1

    min_deriv = 0.15
    max_deriv = 1.75
    grid_points = 100
    L_range = (0.15, 0.3)
    V_range = (1.0, 6.0)
    max_generation_attempts = 5000
    epsilon_calc = 1e-8
    h2_floor = 1e-7

    print(f"Attempting to find up to {n_target_samples} valid samples to select best H' variance...")
    print(f"Using Softplus H'', Random L in {L_range}, Random V in {V_range}")

    valid_samples_data = []
    best_hp_var = -1.0
    best_sample_tuple = None

    total_generation_time = 0
    base_keys = random.split(key, n_target_samples)
    globals()['cholesky_warning_printed'] = False # Reset warning flag

    for i in range(n_target_samples):
        if i % 10 == 0:
            print(f"\n--- Generating Candidate {i+1}/{n_target_samples} ---")
        start_time = time.time()
        x_gp, h_gp, h_prime_gp, h_double_prime_gp, is_valid, sample_L, sample_V = \
            random_smooth_monotonic_convex_gp_adaptive_scale(
                base_keys[i],
                min_derivative=min_deriv,
                max_derivative=max_deriv,
                n_grid=grid_points,
                lengthscale_f_range=L_range,
                variance_f_range=V_range,
                epsilon=epsilon_calc,
                max_attempts=max_generation_attempts,
                h2_min_value = h2_floor
            )
        end_time = time.time()
        gen_time = end_time - start_time
        total_generation_time += gen_time

        if is_valid:
            hp_np = np.array(h_prime_gp)
            if not (np.any(np.isnan(hp_np)) or np.any(np.isinf(hp_np))):
                 current_hp_var = np.var(hp_np)
                 if current_hp_var > best_hp_var:
                     print(f"  >>> New best H' variance found: {current_hp_var:.4e} (L={sample_L:.3f}, V={sample_V:.3f}) (Prev: {best_hp_var:.4e})")
                     best_hp_var = current_hp_var
                     best_sample_tuple = (
                         np.array(x_gp), np.array(h_gp), hp_np, np.array(h_double_prime_gp),
                         sample_L, sample_V
                     )


    # --- Process Results ---
    if best_sample_tuple is None:
        print("\n---------------------------------------------------------")
        print(f"FAILURE: No valid samples found after checking {n_target_samples} candidates.")
        print("---------------------------------------------------------")
    else:
        print("\n---------------------------------------------------------")
        print(f"Found best sample with H' variance: {best_hp_var:.4e}")
        best_x, best_h, best_hp, best_hpp, best_L, best_V = best_sample_tuple
        print(f"Generated with L={best_L:.3f}, V={best_V:.3f}")
        print("---------------------------------------------------------")

        # --- Save the Best Sample ---
        save_filename = f"best_hp_var_sample_L{best_L:.2f}_V{best_V:.2f}.npz".replace(" ","")
        np.savez_compressed(
            save_filename,
            x_grid=best_x, H_values=best_h, H_prime_values=best_hp,
            H_double_prime_values=best_hpp, hp_variance=best_hp_var,
            lengthscale_f=best_L, variance_f=best_V,
            min_derivative_constraint=min_deriv, max_derivative_constraint=max_deriv
        )
        print(f"Saved data for the best sample to: {save_filename}")

        # --- Plot the Best Sample ---
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        ax_h, ax_hp, ax_hpp = axes
        ax_hp.axhline(min_deriv, color='r', linestyle='--', lw=1, label=f'Min H\' ({min_deriv})')
        ax_hp.axhline(max_deriv, color='g', linestyle='--', lw=1, label=f'Max H\' ({max_deriv})')
        ax_hpp.axhline(0, color='k', linestyle='--', lw=1, label='Min Convexity (H\'\' >= 0)')

        line_label = f"Best Sample (H' var={best_hp_var:.3e})"
        ax_h.plot(best_x, best_h, label=line_label, color='blue')
        ax_hp.plot(best_x, best_hp, label=line_label, color='blue')
        ax_hpp.plot(best_x, best_hpp, label=line_label, color='blue')

        ax_h.plot([0, 1], [0, 1], 'ko', markersize=5, label='H(0)=0, H(1)=1')
        ax_h.set_title(f'Best Sample H(x) (L={best_L:.2f}, V={best_V:.2f})')
        ax_hp.set_title("Best Sample First Derivative H'(x)")
        ax_hpp.set_title("Best Sample Second Derivative H''(x)")
        ax_h.set_ylabel('H(x)')
        ax_hp.set_ylabel("H'(x)")
        ax_hpp.set_ylabel("H''(x)")
        ax_h.set_ylim(-0.1, 1.1)
        ax_hp.set_ylim(min_deriv - (max_deriv-min_deriv)*0.1 - 0.05, max_deriv + (max_deriv-min_deriv)*0.1 + 0.05)

        hpp_clean = best_hpp[np.isfinite(best_hpp)]
        if len(hpp_clean) > 0:
            max_hpp_val = np.max(hpp_clean)
            min_hpp_val = np.min(hpp_clean)
        else:
             max_hpp_val = 1.0
             min_hpp_val = 0.0
        if max_hpp_val < 1e-6: max_hpp_val = 1.0
        if min_hpp_val < 0: min_hpp_val = 0.0 # H'' should be >= 0
        lower_ylim = min_hpp_val * 0.9 if min_hpp_val > 0 else -0.05 * max(1.0, max_hpp_val)
        upper_ylim = max(max_hpp_val * 1.1, max_hpp_val + 0.1)
        upper_ylim = max(upper_ylim, lower_ylim + 1e-3)
        upper_ylim = max(upper_ylim, 0.1)
        ax_hpp.set_ylim(lower_ylim, upper_ylim)

        ax_hpp.set_xlabel('x')
        for ax in axes:
            ax.grid(True)
            handles, labels = ax.get_legend_handles_labels()
            if handles: ax.legend(fontsize='small')

        fig.suptitle(f"Sample with Highest H' Variance ({n_target_samples} candidates checked)\n"
                     f"Constraints: {min_deriv} <= H'(x) <= {max_deriv}, H''(x) >= 0 (Softplus, Random Hypers)",
                     fontsize=12)
        fig.tight_layout(rect=[0, 0.03, 1, 0.95])

        plot_filename = f"plot_best_hp_var_sample_L{best_L:.2f}_V{best_V:.2f}.png".replace(" ","")
        plt.savefig(plot_filename, dpi=300)
        print(f"Saved plot of the best sample to: {plot_filename}")
        plt.show()

    print(f"\nTotal generation and selection time: {total_generation_time:.2f}s")