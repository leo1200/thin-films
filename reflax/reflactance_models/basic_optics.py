from functools import partial
import jax
import jax.numpy as jnp
from jax import jit

def polanalyze(polstate):
    if polstate.lower() == "linear te/perpendicular/s":
        return 1.0, 0.0
    elif polstate.lower() == "linear tm/parallel/p":
        return 0.0, 1.0
    else:
        raise ValueError("Unsupported polarization state")

@jit
def snell(n1, n2, theta_incidence):
    return jnp.arcsin(n1 / n2 * jnp.sin(theta_incidence))


@jit
def calculate_ts(n, theta_incidence):
    return (2 * jnp.cos(theta_incidence)) / (jnp.cos(theta_incidence) + jnp.sqrt(n**2 - jnp.sin(theta_incidence)**2))

@jit
def calculate_tp(n, theta_incidence):
    return (2 * n * jnp.cos(theta_incidence)) / (n**2 * jnp.cos(theta_incidence) + jnp.sqrt(n**2 - jnp.sin(theta_incidence)**2))


@jit
def calculate_rs(n, theta_incidence):
    """Calculates Fresnel reflection coefficient for s-polarization (TE)."""
    # n = n2 / n1
    # theta_incidence is the angle in medium 1
    cos_theta_i = jnp.cos(theta_incidence)
    # Calculate cos(theta_t) using Snell's law: n1*sin(theta_i) = n2*sin(theta_t)
    # cos(theta_t) = sqrt(1 - sin^2(theta_t)) = sqrt(1 - (n1/n2 * sin(theta_i))^2)
    # cos(theta_t) = sqrt(1 - (sin(theta_i)/n)^2)
    # Alternatively, use the identity: sqrt(n^2 - sin^2(theta_i)) = n * cos(theta_t)
    sin_theta_i_sq = jnp.sin(theta_incidence)**2
    sqrt_term = jnp.sqrt(n**2 - sin_theta_i_sq + 0j) # Add 0j for complex sqrt if TIR occurs

    # Standard formula: (n1*cos_i - n2*cos_t) / (n1*cos_i + n2*cos_t)
    # Divide by n1: (cos_i - (n2/n1)*cos_t) / (cos_i + (n2/n1)*cos_t)
    # (cos_i - n*cos_t) / (cos_i + n*cos_t)
    # Substitute n*cos_t = sqrt(n^2 - sin^2(theta_i))
    numerator = cos_theta_i - sqrt_term
    denominator = cos_theta_i + sqrt_term
    # Handle potential division by zero (e.g., grazing incidence, although unlikely physically relevant in denominator)
    return jnp.where(jnp.abs(denominator) < 1e-9, jnp.inf, numerator / denominator)


@jit
def calculate_rp(n, theta_incidence):
    """Calculates Fresnel reflection coefficient for p-polarization (TM)."""
    # n = n2 / n1
    # theta_incidence is the angle in medium 1
    cos_theta_i = jnp.cos(theta_incidence)
    sin_theta_i_sq = jnp.sin(theta_incidence)**2
    sqrt_term = jnp.sqrt(n**2 - sin_theta_i_sq + 0j) # Add 0j for complex sqrt if TIR occurs

    # Standard formula: (n2*cos_i - n1*cos_t) / (n2*cos_i + n1*cos_t)
    # Divide by n1: ((n2/n1)*cos_i - cos_t) / ((n2/n1)*cos_i + cos_t)
    # (n*cos_i - cos_t) / (n*cos_i + cos_t)
    # Substitute cos_t = sqrt(n^2 - sin^2(theta_i)) / n
    # (n*cos_i - sqrt(n^2 - sin^2(theta_i))/n) / (n*cos_i + sqrt(n^2 - sin^2(theta_i))/n)
    # Multiply num/den by n: (n^2*cos_i - sqrt(n^2 - sin^2(theta_i))) / (n^2*cos_i + sqrt(n^2 - sin^2(theta_i)))
    numerator = n**2 * cos_theta_i - sqrt_term
    denominator = n**2 * cos_theta_i + sqrt_term
     # Handle potential division by zero (e.g., Brewster's angle if denominator hits zero)
    return jnp.where(jnp.abs(denominator) < 1e-9, jnp.inf, numerator / denominator)


@jax.jit
def calculate_reflection_coeff(n1, n2, theta_incidence, polstate):
    """Calculates Fresnel reflection coefficient based on polarization state."""
    n = n2 / n1 # Relative refractive index
    # Use jax.lax.cond for JIT compatibility
    return calculate_rs(n, theta_incidence)