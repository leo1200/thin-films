from functools import partial
import jax
import jax.numpy as jnp
from jax import jit

from reflax.constants import S_POLARIZED, P_POLARIZED


def get_polarization_components(polarization_state):
    """
    Returns the S and P components aka Jones vector.
    """
    if polarization_state == S_POLARIZED:
        return 1.0, 0.0
    elif polarization_state == P_POLARIZED:
        return 0.0, 1.0
    else:
        raise ValueError("Unsupported polarization state")

@jit
def snell(n1, n2, theta_incidence):
    """Calculates the angle of refraction using Snell's law.

    Args:
        n1: Refractive index of the incident medium.
        n2: Refractive index of the transmission medium.
        theta_incidence: Angle of incidence (radians).

    Returns:
        Angle of refraction (radians). Handles TIR by returning complex angle.
    """
    # Ensure inputs are complex
    n1 = n1 + 0j
    n2 = n2 + 0j
    
    theta_incidence = jnp.asarray(theta_incidence, dtype=jnp.complex64)

    # Prevent argument to arcsin from exceeding 1 for real cases due to precision
    # For complex n, this ratio can be > 1 anyway
    ratio = n1 / n2 * jnp.sin(theta_incidence)
    # Clip the real part if numbers are real, otherwise proceed
    # ratio = jnp.where(jnp.isreal(ratio), jnp.clip(ratio.real, -1.0, 1.0) + 0j, ratio)
    # The above clipping might hide TIR issues. Relying on complex arcsin is better.
    return jnp.arcsin(ratio)


# Transmission coefficients (optional, not used in baseline_methods below)
@jit
def fresnel_transmission_s(n, theta_incidence):
    """Calculates Fresnel transmission coefficient for s-polarization (TE)."""
    # n = n2 / n1
    n = n + 0j
    cos_theta_i = jnp.cos(theta_incidence)
    sin_theta_i_sq = jnp.sin(theta_incidence)**2
    sqrt_term = jnp.sqrt(n**2 - sin_theta_i_sq)
    return (2 * cos_theta_i) / (cos_theta_i + sqrt_term)

@jit
def fresnel_transmission_p(n, theta_incidence):
    """Calculates Fresnel transmission coefficient for p-polarization (TM)."""
    # n = n2 / n1
    n = n + 0j
    cos_theta_i = jnp.cos(theta_incidence)
    sin_theta_i_sq = jnp.sin(theta_incidence)**2
    sqrt_term = jnp.sqrt(n**2 - sin_theta_i_sq)
    return (2 * n * cos_theta_i) / (n**2 * cos_theta_i + sqrt_term)


@jit
def fresnel_reflection_s(n_rel, theta_incidence):
    """Calculates Fresnel reflection coefficient for s-polarization (TE).

    Args:
        n_rel: Relative refractive index (n2 / n1). Can be complex.
        theta_incidence: Angle of incidence in medium 1 (radians).

    Returns:
        Complex reflection coefficient rs.
    """

    cos_theta_i = jnp.cos(theta_incidence)
    sin_theta_i_sq = jnp.sin(theta_incidence)**2
    # n_rel*cos(theta_t) = sqrt(n_rel^2 - sin^2(theta_i))
    sqrt_term = jnp.sqrt(n_rel**2 - sin_theta_i_sq)

    # Formula: (cos_i - n_rel*cos_t) / (cos_i + n_rel*cos_t)
    numerator = cos_theta_i - sqrt_term
    denominator = cos_theta_i + sqrt_term

    # Avoid explicit division by zero check; JAX handles Inf/NaN appropriately
    # or downstream calculations should handle potential non-finite values.
    return numerator / denominator


@jit
def fresnel_reflection_p(n_rel, theta_incidence):
    """Calculates Fresnel reflection coefficient for p-polarization (TM).

    Args:
        n_rel: Relative refractive index (n2 / n1). Can be complex.
        theta_incidence: Angle of incidence in medium 1 (radians).

    Returns:
        Complex reflection coefficient rp.
    """

    cos_theta_i = jnp.cos(theta_incidence)
    sin_theta_i_sq = jnp.sin(theta_incidence)**2
    # n_rel*cos(theta_t) = sqrt(n_rel^2 - sin^2(theta_i))
    sqrt_term = jnp.sqrt(n_rel**2 - sin_theta_i_sq)

    # Formula: (n_rel^2*cos_i - n_rel*cos_t) / (n_rel^2*cos_i + n_rel*cos_t)
    numerator = n_rel**2 * cos_theta_i - sqrt_term
    denominator = n_rel**2 * cos_theta_i + sqrt_term

    # Avoid explicit division by zero check
    return numerator / denominator


@partial(jax.jit, static_argnames=['polarization_state'])
def calculate_reflection_coeff(n1, n2, theta_incidence, polarization_state):
    """Calculates Fresnel reflection coefficient based on polarization state.

    Args:
        n1: Refractive index of incident medium (can be complex).
        n2: Refractive index of transmission medium (can be complex).
        theta_incidence: Angle of incidence in medium 1 (radians).
        polarization_state: S_POLARIZED or P_POLARIZED.

    Returns:
        Complex Fresnel reflection coefficient (rs or rp).
    """
    # ensure inputs are complex
    n1 = n1 + 0j
    n2 = n2 + 0j

    # calculate relative refractive index
    n_rel = n2 / n1

    # calculate reflection coefficient based on polarization state
    if polarization_state == S_POLARIZED:
        return fresnel_reflection_s(n_rel, theta_incidence)
    elif polarization_state == P_POLARIZED:
        return fresnel_reflection_p(n_rel, theta_incidence)
    else:
         raise ValueError("Unsupported polarization state")