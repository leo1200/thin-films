import jax.numpy as jnp
from jax import jit

@jit
def snell(n1, n2, theta_incidence):
    """Calculates the angle of refraction using Snell's law.

    Args:
        n1: Refractive index of the incident medium.
        n2: Refractive index of the transmission medium.
        theta_incidence: Angle of incidence (radians).

    Returns:
        Angle of refraction (radians).
    """

    # Ensure inputs are complex
    n1 = n1 + 0j
    n2 = n2 + 0j

    ratio = n1 / n2 * jnp.sin(theta_incidence)

    return jnp.arcsin(ratio)