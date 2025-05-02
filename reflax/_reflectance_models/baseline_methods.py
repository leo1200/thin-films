import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial
from typing import Tuple
import jax.lax as lax

from reflax._reflectance_models.basic_optics import calculate_reflection_coeff, snell
from reflax.constants import S_POLARIZED
from reflax.parameter_classes.parameters import LayerParams, OpticsParams, SetupParams

@partial(jax.jit, static_argnames=['polarization_state'])
def one_layer_no_internal_reflections(
    setup_params: SetupParams,
    optics_params: OpticsParams,
    layer_params: LayerParams,
    polarization_state: int = S_POLARIZED,
) -> Tuple[float, float, float]:
    """
    xxx
    """
    n0 = jnp.sqrt(optics_params.permeability_reflection * optics_params.permittivity_reflection)
    n1 = jnp.sqrt(layer_params.permeabilities * layer_params.permittivities)

    n2 = jnp.sqrt(optics_params.permeability_transmission * optics_params.permittivity_transmission)

    # note this is the phase for a cos^2 so we have 2 * delta as the true phase
    delta = 2 * jnp.pi * layer_params.thicknesses * jnp.sqrt(n1 ** 2 - n0 ** 2 * jnp.sin(setup_params.polar_angle) ** 2) / setup_params.wavelength
    theta_transmitted = snell(n0, n1, setup_params.polar_angle)

    r01 = calculate_reflection_coeff(n0, n1, setup_params.polar_angle, polarization_state)
    r12 = calculate_reflection_coeff(n1, n2, theta_transmitted, polarization_state)

    # return jnp.cos(delta) ** 2
    return jnp.abs(r01 + (1 - r01 ** 2) * r12 * jnp.exp(2j * delta)) ** 2


@partial(jax.jit, static_argnames=['polarization_state'])
def one_layer_internal_reflections(
    setup_params: SetupParams,
    optics_params: OpticsParams,
    layer_params: LayerParams,
    polarization_state: int = S_POLARIZED,
) -> float:
    """
    xxx
    """

    n0 = jnp.sqrt(optics_params.permeability_reflection * optics_params.permittivity_reflection)
    n1 = jnp.sqrt(layer_params.permeabilities * layer_params.permittivities)
    n2 = jnp.sqrt(optics_params.permeability_transmission * optics_params.permittivity_transmission)

    delta = 2 * jnp.pi * layer_params.thicknesses * jnp.sqrt(n1 ** 2 - n0 ** 2 * jnp.sin(jnp.pi - setup_params.polar_angle) ** 2) / setup_params.wavelength

    theta_transmitted = snell(n0, n1, setup_params.polar_angle)

    r01 = calculate_reflection_coeff(n0, n1, setup_params.polar_angle, polarization_state)
    r12 = calculate_reflection_coeff(n1, n2, theta_transmitted, polarization_state)
    r10 = -calculate_reflection_coeff(n0, n1, theta_transmitted, polarization_state)

    return jnp.abs((r01 + r12 * jnp.exp(2j * delta)) / (1 - r10 * r12 * jnp.exp(2j * delta))) ** 2


def growth_from_frequencies(
    setup_params: SetupParams,
    optics_params: OpticsParams,
    layer_permeability: float,
    layer_permittivity: float,
    frequencies: Float[Array, "num_measurements"],
) -> Float[Array, "num_measurements"]:
    n0 = jnp.sqrt(optics_params.permeability_reflection * optics_params.permittivity_reflection)
    n1 = jnp.sqrt(layer_permeability * layer_permittivity)

    growth = frequencies / (2 * jnp.sqrt(n1 ** 2 - n0 ** 2 * jnp.sin(jnp.pi - setup_params.polar_angle) ** 2) / setup_params.wavelength)

    return growth

