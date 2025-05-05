"""
Tools for sampling smooth functions.
"""

# TODO: improve type hints

from functools import partial
from typing import Tuple
import jax.numpy as jnp
import jax

from reflax.thickness_modeling.numerical_integration import cumulative_trapezoid

# -------------------------------------------------------------
# ================ ↓ Linear Function Sampling ↓ ===============
# -------------------------------------------------------------

@jax.jit
def sample_linear_functions(
    random_key,
    num_samples,
    eval_points,
    min_final_value,
    max_final_value,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample linear functions from y = 0.0 at x = 0.0
    to y = final_value at x = 1.0 where final_value
    is uniformly sampled from [min_final_value, max_final_value].

    eval_points[0] = 0.0 and eval_points[-1] = 1.0

    Args:
        random_key: Random key for JAX
        num_samples: Number of samples to generate
        eval_points: Points at which to evaluate the linear functions
        min_final_value: Minimum final value (y at x = 1.0)
        max_final_value: Maximum final value (y at x = 1.0)
    Returns:
        samples, derivatives
    """

    # sample the final values
    final_values = jax.random.uniform(
        random_key,
        shape = (num_samples,),
        minval = min_final_value,
        maxval = max_final_value
    )

    return (
        eval_points * final_values.reshape(-1, 1),
        jnp.ones_like(eval_points) * final_values.reshape(-1, 1)
    )

# -------------------------------------------------------------
# =============== ↑ Linear Function Sampling ↑ ================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ Gaussian Process Sampling ↓ ===============
# -------------------------------------------------------------

"""
Intro to Gaussian Processes
===========================

A Gaussian Process (GP) is a collection of random variables with distribution

    f_pred ~ N(m(X), K(X, X))

where m(X_pred) is the mean function and K(X_pred, X_pred) is the covariance
matrix, based on a pairwise covariance function k(x, x'), e.g. a radial basis
function (RBF) kernel

    k(x, x') = variance * exp(-||x - x'||^2 / (2 * lengthscale^2))

with exponential decay of the covariance with distance between points.

Sampling Smooth Functions using a GP
====================================

Given the kovariance matrix K(X, X) and the mean function m(X), we

1. find a matrix L such that K(X, X) = LL^T (Cholesky decomposition)
2. sample a vector z from a standard normal distribution
3. compute f_pred = m(X) + Lz

We use a zero mean function.

"""

@jax.jit
def rbf_kernel(
    x1,
    x2,
    lengthscale,
    variance
) -> jnp.ndarray:
    """
    Calculates the covariance matrix between two sets
    of points x1 and x2 using the RBF kernel
    with given lengthscale and variance.

    Args:
        x1: First set of points
        x2: Second set of points
        lengthscale: Lengthscale parameter for the RBF kernel
        variance: Variance parameter for the RBF kernel

    Returns:
        Covariance matrix between x1 and x2
    """
    x1 = x1.reshape(-1, 1) if x1.ndim == 1 else x1
    x2 = x2.reshape(-1, 1) if x2.ndim == 1 else x2
    sqdist = jnp.sum(x1**2, 1).reshape(-1, 1) + jnp.sum(x2**2, 1) - 2 * jnp.dot(x1, x2.T)
    return variance * jnp.exp(-0.5 / lengthscale**2 * sqdist)

@partial(jax.jit, static_argnames=["num_samples"])
def sample_gp(
    random_key,
    num_samples,
    eval_points,
    lengthscale,
    variance,
) -> jnp.ndarray:
    """
    Sample from a Gaussian Process with RBF kernel and zero mean.

    Args:
        random_key: Random key for JAX
        num_samples: Number of samples to generate
        eval_points: Points at which to evaluate the GP
        lengthscale: Lengthscale parameter for the RBF kernel
        variance: Variance parameter for the RBF kernel

    Returns:
        f_samples: Samples from the GP at the given points
    """

    num_points = eval_points.shape[0]

    # compute the covariance matrix
    K = rbf_kernel(eval_points, eval_points, lengthscale, variance)

    # Cholesky decomposition
    # one might need to adaptively increase the jitter term
    # (1e-6) to avoid numerical issues
    L = jnp.linalg.cholesky(K + 1e-6 * jnp.eye(num_points))

    # sample from a standard normal distribution
    z = jax.random.normal(random_key, shape=(num_samples, num_points))

    # compute the samples
    f_samples = jnp.dot(z, L.T)

    return f_samples

def sample_monotonic_gp(
    random_key,
    num_samples,
    eval_points,
    lengthscale,
    variance
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Sample a non-decreasing smooth function
    as the numerical integral of the softplus
    of a Gaussian Process.

    Args:
        random_key: Random key for JAX
        num_samples: Number of samples to generate
        eval_points: Points at which to evaluate the GP
        lengthscale: Lengthscale parameter for the RBF kernel
        variance: Variance parameter for the RBF kernel

    Returns:
        samples: samples from the smooth non-decreasing function
        derivatives: derivatives of these samples
    """

    raw_derivatives = sample_gp(random_key, num_samples, eval_points, lengthscale, variance)
    derivatives = jax.nn.softplus(raw_derivatives)
    samples = jnp.array([cumulative_trapezoid(eval_points, derivative) for derivative in derivatives])

    # normalize the samples to go from 0 to 1
    # they start at zero by construction
    samples = samples / jnp.max(samples, axis = 1, keepdims = True)
    derivatives = derivatives / jnp.max(samples, axis = 1, keepdims = True)

    return samples, derivatives


@partial(jax.jit, static_argnames = ["convex_samples"])
def _sample_single_derivative_bound_monotonic_gp(
    key,
    L,
    eval_points,
    min_slope,
    max_slope,
    max_iterations,
    final_value = 1.0,
    convex_samples = False
) -> Tuple[jnp.ndarray, jnp.ndarray, bool]:
    """
    Rejection sampling for smooth monotonic functions with bounded derivatives
    based on a Gaussian Process.

    The lower triangular matrix L is obtained from the Cholesky decomposition
    and passed to it does not need to be recomputed for each sample.

    Args:
        key: Random key for JAX
        L: lower triangular matrix from Cholesky decomposition
        eval_points: Points at which to evaluate the GP
        min_slope: Minimum slope (lower bound)
        max_slope: Maximum slope (upper bound)
        max_iterations: Maximum number of iterations for rejection sampling
        final_value: Final value to normalize the sample
        convex_samples: If True, samples are convex
        (i.e. the second derivative is positive)

    Returns:
        sample: Sampled function values at eval_points
        derivative: Sampled derivative values at eval_points
        condition_met: Boolean indicating if the rejection sampling
                       was sucessful (i.e. all derivatives are within bounds)

    """
    num_points = eval_points.shape[0]

    # initial loop state
    # (key, iteration, condition_met, sample, derivative)
    init_state = (
        key,
        0,
        False,
        jnp.zeros_like(eval_points),
        jnp.zeros_like(eval_points)
    )

    def cond_fun(state):
        """
        Continue as long as the derivative bounds are not met
        and the maximum number of iterations is not reached.
        """
        _, iteration, condition_met, _, _ = state
        return jnp.logical_and(jnp.logical_not(condition_met), iteration < max_iterations)

    def body_fun(state):
        """
        Generate one trial sample and check if the derivative bounds are met.
        """
        current_key, iteration, _, _, _ = state

        # split the key to get a new sample
        iter_key, next_key = jax.random.split(current_key)

        # sample from standard normal
        z = jax.random.normal(iter_key, shape=(num_points,))

        if convex_samples:
            # achive convexity by applying a softplus to the second derivative
            second_derivative = jax.nn.softplus(jnp.dot(z, L.T))

            # add an additive term to the first derivative, otherwise
            # the first derivative would always start at zero
            random_number = jax.random.uniform(next_key, shape=(1,), minval = min_slope, maxval = max_slope)
            first_derivative = cumulative_trapezoid(eval_points, second_derivative) + random_number / final_value
        else:
            first_derivative = jax.nn.softplus(jnp.dot(z, L.T))

        # get the sample by numerical integration
        sample = cumulative_trapezoid(eval_points, first_derivative)

        # normalize appropriately
        max_sample_val = jnp.max(jnp.abs(sample))
        normalized_sample = sample / max_sample_val * final_value
        normalized_first_derivative = first_derivative / max_sample_val * final_value

        # check if the derivative bounds are met
        condition_met = jnp.all(
            jnp.logical_and(
                normalized_first_derivative >= min_slope,
                normalized_first_derivative <= max_slope
            )
        )

        return (
            next_key,
            iteration + 1,
            condition_met,
            normalized_sample,
            normalized_first_derivative
        )

    # run the loop
    final_state = jax.lax.while_loop(cond_fun, body_fun, init_state)

    # unpack final state
    _, _, final_condition_met, final_sample, final_derivative = final_state

    return final_sample, final_derivative, final_condition_met


def sample_derivative_bound_gp(
    random_key,
    num_samples,
    eval_points,
    lengthscale,
    variance,
    min_slope,
    max_slope,
    random_final_values = False,
    min_final_value = 1.0,
    max_final_value = 2.0,
    convex_samples = False,
    max_iterations = 10000,
    jitter = 1e-6
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    """
    Generate smooth monotonic, possibly only convex,
    functions with bounded derivatives based on 
    ramdom samples from a Gaussian Process.

    NOTE: Choosing the variance too small or the
    lengthscale too large will lead to underexploration
    of the possible function space.

    Args:
        random_key: Random key for JAX
        num_samples: Number of samples to generate
        eval_points: Points at which to evaluate the GP
        lengthscale: Lengthscale parameter for the RBF kernel
        variance: Variance parameter for the RBF kernel (on the unscaled baseline GP)
        min_slope: Minimum slope (lower bound) after scaling if applicable
        max_slope: Maximum slope (upper bound) after scaling if applicable
        random_final_values: If True, the final values of the samples are random
        min_final_value: Minimum final value (if random_final_values is True)
        max_final_value: Maximum final value (if random_final_values is True)
        convex_samples: If True, samples are convex
        max_iterations: Maximum number of iterations for rejection sampling
        jitter: Jitter term for numerical stability
        (default: 1e-6)

    Returns:
        samples: Samples from the smooth monotonic function
        derivatives: Derivatives of these samples

    """
    num_points = eval_points.shape[0]

    # compute Covariance Matrix and Cholesky (once for all samples)
    # for efficiency
    K = rbf_kernel(eval_points, eval_points, lengthscale, variance)
    L = jnp.linalg.cholesky(K + jitter * jnp.eye(num_points))

    # random keys for vmapping
    keys = jax.random.split(random_key, num_samples)

    # randomized final values if requested
    if random_final_values:
        final_values = jax.random.uniform(
            random_key,
            shape=(num_samples,),
            minval=min_final_value,
            maxval=max_final_value
        )
    else:
        final_values = jnp.ones(num_samples)

    # vmapping version of the rejection sampling function
    vmapped_sampler = jax.vmap(
        _sample_single_derivative_bound_monotonic_gp,
        in_axes=(0, None, None, None, None, None, 0, None) 
        # key, L, eval_pts, min_s, max_s, max_iter, final_value, convex_samples
    )

    # run the rejection sampling
    samples, derivatives, conditions_med = vmapped_sampler(
        keys,
        L,
        eval_points,
        min_slope,
        max_slope,
        max_iterations,
        final_values,
        convex_samples
    )

    # check if all samples meet the derivative bounds
    success_flags = jnp.array(conditions_med, dtype=jnp.bool_)
    success = jnp.all(success_flags)
    if not success:
        raise ValueError("Not all samples met the derivative bounds.")

    return samples, derivatives


# -------------------------------------------------------------
# =============== ↑ Gaussian Process Sampling ↑ ===============
# -------------------------------------------------------------