from functools import partial
import jax
import jax.numpy as jnp
from reflax.transfer_matrix_method import transfer_matrix_method
from jaxtyping import Array, Float
from functools import partial
from typing import Tuple

@partial(jax.jit, static_argnames=['backside_mode'])
def variable_layer_thickness_simulation(
    wavelength: float,
    polar_angle: float,
    azimuthal_angle: float,
    transverse_electric_component: float,
    transverse_magnetic_component: float,
    permeability_reflection: float,
    permittivity_reflection: float,
    permeability_transmission: float,
    permittivity_transmission: float,
    backside_mode: int,
    permeability_static_size_layers: Float[Array, "num_layers"],
    permittivity_static_size_layers: Float[Array, "num_layers"],
    static_layer_thicknesses: Float[Array, "num_layers"],
    permeability_variable_layer: float,
    permittivity_variable_layer: float,
    variable_layer_thicknesses: Float[Array, "num_thicknesses"]
) -> Tuple[Float[Array, "num_thicknesses"], Float[Array, "num_thicknesses"], Float[Array, "num_thicknesses"]]:
    """
    Compute the reflection, transmission and conservation check for the case of one of the layers
    varying in thickness with computations vectorized over multiple thicknesses of this verying layer.
    """

    # prepend the permeability and permittivity of the layer varying in thickness
    # to these quantities of the other layers
    layer_permeabilities = jnp.concatenate((jnp.array([permeability_variable_layer]), permeability_static_size_layers))
    layer_permittivities = jnp.concatenate((jnp.array([permittivity_variable_layer]), permittivity_static_size_layers))

    # initialize a matrix of thicknesses, we vectorize over
    thickness_matrix = jnp.zeros((variable_layer_thicknesses.shape[0], static_layer_thicknesses.shape[0] + 1))
    thickness_matrix = thickness_matrix.at[:, 1:].set(static_layer_thicknesses)
    thickness_matrix = thickness_matrix.at[:, 0].set(variable_layer_thicknesses)

    reflection_coefficients, transmission_coefficients, conservation_checks = jax.vmap(
        lambda layer_thicknesses: transfer_matrix_method(
            wavelength,
            polar_angle, azimuthal_angle,
            transverse_electric_component,
            transverse_magnetic_component,
            permeability_reflection,
            permittivity_reflection,
            permeability_transmission,
            permittivity_transmission,
            backside_mode,
            layer_permeabilities,
            layer_permittivities,
            layer_thicknesses
        )
    )(thickness_matrix)
    
    return reflection_coefficients, transmission_coefficients, conservation_checks