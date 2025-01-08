import jax
import jax.numpy as jnp
from jaxtyping import Array, Float
from functools import partial
from typing import Tuple
import jax.lax as lax

from reflax._helpers.helpers import calculate_reflection_coeff, snell
from reflax.parameter_classes.parameters import LayerParams, OpticsParams, SetupParams

@jax.jit
def one_layer_no_internal_reflections(
    setup_params: SetupParams,
    optics_params: OpticsParams,
    layer_params: LayerParams,
) -> Tuple[float, float, float]:
    """
    xxx
    """
    n0 = jnp.sqrt(optics_params.permeability_reflection * optics_params.permittivity_reflection)
    n1 = jnp.sqrt(layer_params.permeabilities * layer_params.permittivities)

    n2 = jnp.sqrt(optics_params.permeability_transmission * optics_params.permittivity_transmission)
    delta = 2 * jnp.pi * layer_params.thicknesses * jnp.sqrt(n1 ** 2 - n0 ** 2 * jnp.sin(setup_params.polar_angle) ** 2) / setup_params.wavelength
    theta_transmitted = snell(n0, n1, setup_params.polar_angle)

    r01 = calculate_reflection_coeff(n0, n1, setup_params.polar_angle, setup_params.polstate)
    r12 = calculate_reflection_coeff(n1, n2, theta_transmitted, setup_params.polstate)

    # return jnp.cos(delta) ** 2
    return jnp.abs(r01 + (1 - r01 ** 2) * r12 * jnp.exp(2j * delta)) ** 2


@jax.jit
def one_layer_internal_reflections(
    setup_params: SetupParams,
    optics_params: OpticsParams,
    layer_params: LayerParams,
) -> float:
    """
    xxx
    """

    n0 = jnp.sqrt(optics_params.permeability_reflection * optics_params.permittivity_reflection)
    n1 = jnp.sqrt(layer_params.permeabilities * layer_params.permittivities)
    n2 = jnp.sqrt(optics_params.permeability_transmission * optics_params.permittivity_transmission)

    delta = 2 * jnp.pi * layer_params.thicknesses * jnp.sqrt(n1 ** 2 - n0 ** 2 * jnp.sin(jnp.pi - setup_params.polar_angle) ** 2) / setup_params.wavelength

    theta_transmitted = snell(n0, n1, setup_params.polar_angle)

    r01 = calculate_reflection_coeff(n0, n1, setup_params.polar_angle, setup_params.polstate)
    r12 = calculate_reflection_coeff(n1, n2, theta_transmitted, setup_params.polstate)
    r10 = -calculate_reflection_coeff(n0, n1, theta_transmitted, setup_params.polstate)

    return jnp.abs((r01 + r12 * jnp.exp(2j * delta)) / (1 - r10 * r12 * jnp.exp(2j * delta))) ** 2

@jax.jit
def multiple_layers_no_internal_reflections(
    setup_params: SetupParams,
    optics_params: OpticsParams,
    layer_params: LayerParams
) -> float:
    """
    Reflectance of a multilayer system without internal reflections.
    """

    all_permeabilities = jnp.concatenate((jnp.array([optics_params.permeability_reflection]), 
                                          layer_params.permeabilities, 
                                          jnp.array([optics_params.permeability_transmission])), dtype=jnp.complex128)
    all_permittivities = jnp.concatenate((jnp.array([optics_params.permittivity_reflection]), 
                                          layer_params.permittivities, 
                                          jnp.array([optics_params.permittivity_transmission])), dtype=jnp.complex128)
    all_refractive_indices = jnp.sqrt(all_permeabilities * all_permittivities)

    num_layers = layer_params.thicknesses.shape[0]
    theta_array = jnp.zeros(num_layers + 1)
    theta_array = theta_array.at[0].set(setup_params.polar_angle)

    # Snell's law computation for all layers
    def snell_update(theta_prev, layer_idx):
        next_theta = snell(all_refractive_indices[layer_idx - 1], all_refractive_indices[layer_idx], theta_prev)
        return next_theta, next_theta

    _, theta_array_b = jax.lax.scan(snell_update, setup_params.polar_angle, jnp.arange(1, num_layers + 1))
    theta_array = theta_array.at[1:].set(theta_array_b)

    delta = 2 * jnp.pi * layer_params.thicknesses[0] * jnp.sqrt(all_refractive_indices[1] ** 2 - all_refractive_indices[0] ** 2 * jnp.sin(setup_params.polar_angle) ** 2) / setup_params.wavelength

    # Phase shift delta_array calculation
    delta_array = 2 * jnp.pi * layer_params.thicknesses[1:] * jnp.sqrt(
        all_refractive_indices[2:num_layers+1]**2 - 
        all_refractive_indices[1:num_layers]**2 * jnp.sin(jnp.pi - theta_array[2:])**2
    ) / setup_params.wavelength

    # Initialize reflectivity with first interface
    r01 = calculate_reflection_coeff(
        all_refractive_indices[0],
        all_refractive_indices[1], 
        setup_params.polar_angle, 
        setup_params.polstate
    )

    # Compute contributions from each layer
    def layer_reflectivity_update(layer_idx, reflectivity):
        backside_reflection = calculate_reflection_coeff(
            all_refractive_indices[layer_idx],
            all_refractive_indices[layer_idx + 1],
            theta_array[layer_idx],
            setup_params.polstate,
        )
        intermediate_product = backside_reflection * (1 - r01**2) * jnp.exp(2j * delta)

        def inner_loop_update(k, intermediate_product):
            transmission_reflection = calculate_reflection_coeff(
                all_refractive_indices[k - 1],
                all_refractive_indices[k],
                theta_array[k],
                setup_params.polstate,
            )
            return intermediate_product * (1 - transmission_reflection**2) * jnp.exp(2j * delta_array[k - 2])

        intermediate_product = jax.lax.fori_loop(
            2, layer_idx + 1, inner_loop_update, intermediate_product
        )

        return reflectivity + intermediate_product


    reflectivity = jax.lax.fori_loop(1, num_layers + 1, layer_reflectivity_update, -r01)

    # Reflectance
    reflectance = jnp.abs(reflectivity)**2

    return reflectance


@jax.jit
def multiple_layers_internal_reflections(
    setup_params: SetupParams,
    optics_params: OpticsParams,
    layer_params: LayerParams
) -> float:
    """
    Reflectance of a multilayer system without internal reflections.
    """

    all_permeabilities = jnp.concatenate((jnp.array([optics_params.permeability_reflection]), 
                                          layer_params.permeabilities, 
                                          jnp.array([optics_params.permeability_transmission])), dtype=jnp.complex128)
    all_permittivities = jnp.concatenate((jnp.array([optics_params.permittivity_reflection]), 
                                          layer_params.permittivities, 
                                          jnp.array([optics_params.permittivity_transmission])), dtype=jnp.complex128)
    all_refractive_indices = jnp.sqrt(all_permeabilities * all_permittivities)

    num_layers = layer_params.thicknesses.shape[0]
    theta_array = jnp.zeros(num_layers + 1)
    theta_array = theta_array.at[0].set(setup_params.polar_angle)

    # Snell's law computation for all layers
    def snell_update(theta_prev, layer_idx):
        next_theta = snell(all_refractive_indices[layer_idx - 1], all_refractive_indices[layer_idx], theta_prev)
        return next_theta, next_theta

    _, theta_array_b = jax.lax.scan(snell_update, setup_params.polar_angle, jnp.arange(1, num_layers + 1))
    theta_array = theta_array.at[1:].set(theta_array_b)

    delta = 2 * jnp.pi * layer_params.thicknesses[0] * jnp.sqrt(all_refractive_indices[1] ** 2 - all_refractive_indices[0] ** 2 * jnp.sin(setup_params.polar_angle) ** 2) / setup_params.wavelength

    # Phase shift delta_array calculation
    delta_array = 2 * jnp.pi * layer_params.thicknesses[1:] * jnp.sqrt(
        all_refractive_indices[2:num_layers+1]**2 - 
        all_refractive_indices[1:num_layers]**2 * jnp.sin(jnp.pi - theta_array[2:])**2
    ) / setup_params.wavelength

    # Initialize reflectivity with first interface
    r01 = calculate_reflection_coeff(
        all_refractive_indices[0],
        all_refractive_indices[1], 
        setup_params.polar_angle, 
        setup_params.polstate
    )

    # Compute contributions from each layer
    def layer_reflectivity_update(layer_idx, reflectivity):
        backside_reflection = calculate_reflection_coeff(
            all_refractive_indices[layer_idx],
            all_refractive_indices[layer_idx + 1],
            theta_array[layer_idx],
            setup_params.polstate,
        )
        topside_reflection = calculate_reflection_coeff(
            all_refractive_indices[layer_idx - 1],
            all_refractive_indices[layer_idx],
            theta_array[layer_idx],
            setup_params.polstate,
        )
        intermediate_product = backside_reflection * (1 - r01**2) * jnp.exp(2j * delta)

        def inner_loop_update(k, intermediate_product):
            transmission_reflection = calculate_reflection_coeff(
                all_refractive_indices[k - 1],
                all_refractive_indices[k],
                theta_array[k],
                setup_params.polstate,
            )
            return intermediate_product * (1 - transmission_reflection**2) * jnp.exp(2j * delta_array[k - 2])

        intermediate_product = jax.lax.fori_loop(
            2, layer_idx + 1, inner_loop_update, intermediate_product
        )

        def internal_reflection_update(intermediate_product):
            return intermediate_product / (1 - topside_reflection * backside_reflection * jnp.exp(2j * delta))
        
        def internal_reflection_update2(intermediate_product):
            return intermediate_product / (1 - topside_reflection * backside_reflection * jnp.exp(2j * delta_array[layer_idx - 2]))
        
        intermediate_product = lax.cond(
            layer_idx == 1,
            internal_reflection_update,
            internal_reflection_update2,
            intermediate_product
        )

        return reflectivity + intermediate_product


    reflectivity = jax.lax.fori_loop(1, num_layers + 1, layer_reflectivity_update, -r01)

    # Reflectance
    reflectance = jnp.abs(reflectivity)**2

    return reflectance