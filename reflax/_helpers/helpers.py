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
def calculate_rs(n, theta_incidence):
    return (jnp.cos(theta_incidence) - jnp.sqrt(n**2 - jnp.cos(theta_incidence)**2)) / (jnp.cos(theta_incidence) + jnp.sqrt(n**2 - jnp.cos(theta_incidence)**2))

@jit
def calculate_rp(n, theta_incidence):
    return (jnp.sqrt(n**2 - jnp.sin(theta_incidence)**2) - n**2 * jnp.cos(theta_incidence)) / (jnp.sqrt(n**2 - jnp.sin(theta_incidence)**2) + n**2 * jnp.cos(theta_incidence))

@jit
def calculate_ts(n, theta_incidence):
    return (2 * jnp.cos(theta_incidence)) / (jnp.cos(theta_incidence) + jnp.sqrt(n**2 - jnp.sin(theta_incidence)**2))

@jit
def calculate_tp(n, theta_incidence):
    return (2 * n * jnp.cos(theta_incidence)) / (n**2 * jnp.cos(theta_incidence) + jnp.sqrt(n**2 - jnp.sin(theta_incidence)**2))

LINEAR_TE_PERPENDICULAR_S = 0
LINEAR_TM_PARALLEL_P = 1

@jax.jit
def calculate_reflection_coeff(n1, n2, theta_incidence, polstate):
    n = n2 / n1
    return calculate_rs(n, theta_incidence)
    # if polstate == LINEAR_TE_PERPENDICULAR_S:
    #     return calculate_rs(n, theta_incidence)
    # elif polstate == LINEAR_TM_PARALLEL_P:
    #     return calculate_rp(n, theta_incidence)
    # else:
    #     raise ValueError("Unsupported polarization state")