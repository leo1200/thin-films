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

    
    Args:
        wavelength: Free-space wavelength.
        polar_angle: Polar/zenith angle in radians.
        azimuthal_angle: Azimuthal angle in radians.
        transverse_electric_component: TE polarized component.
        transverse_magnetic_component: TM polarized component.
        permeability_reflection: Relative permeability (reflection side).
        permittivity_reflection: Relative permittivity (reflection side).
        permeability_transmission: Relative permeability (transmission side).
        permittivity_transmission: Relative permittivity (transmission side).
        backside_mode: Decision variable for backside transmission/reflection
                       (-1: reflection with phase inversion, 0: reflection, 1: transmission)
        permeability_static_size_layers: Relative permeabilities of the layers with static thicknesses.
        permittivity_static_size_layers: Relative permittivities of the layers with static thicknesses.
        static_layer_thicknesses: Thicknesses of layers with static sizes.
        permeability_variable_layer: Relative permeability of the layer with varying thickness.
        permittivity_variable_layer: Relative permittivity of the layer with varying thickness.
        variable_layer_thicknesses: Array of the different thicknesses of the layer with varying thickness.

    Returns:
        Reflectance (REF), Transmittance (TRN), Conservation (CON) for all the
        layer thicknesses specified in variable_layer_thicknesses while all
        other options remain constant.
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

@partial(jax.jit, static_argnames=['backside_mode'])
def power_forward_model(
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
    timepoints_measured: Float[Array, "num_timepoints"],
    initial_thickness: float,
    growth_velocity: float,
    growth_acceleration: float,
    power_conversion_factor: float,
    power_consversion_constant: float
) -> Tuple[Float[Array, "num_timepoints"], Float[Array, "num_timepoints"], Float[Array, "num_timepoints"]]:
    """
    TODO: write docstring
    """
    
    variable_layer_thicknesses = initial_thickness + growth_velocity * timepoints_measured + growth_acceleration * timepoints_measured ** 2

    reflection_coefficients, _, _ = variable_layer_thickness_simulation(
                                        wavelength,
                                        polar_angle,
                                        azimuthal_angle,
                                        transverse_electric_component,
                                        transverse_magnetic_component,
                                        permeability_reflection,
                                        permittivity_reflection,
                                        permeability_transmission,
                                        permittivity_transmission,
                                        backside_mode,
                                        permeability_static_size_layers,
                                        permittivity_static_size_layers,
                                        static_layer_thicknesses,
                                        permeability_variable_layer,
                                        permittivity_variable_layer,
                                        variable_layer_thicknesses
                                    ) 
    
    power_output = power_consversion_constant + power_conversion_factor * reflection_coefficients

    return power_output


@partial(jax.jit, static_argnames=['backside_mode'])
def power_forward_residuals(
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
    timepoints_measured: Float[Array, "num_timepoints"],
    power_measured: Float[Array, "num_timepoints"],
    initial_thickness: float,
    growth_velocity: float,
    growth_acceleration: float,
    power_conversion_factor: float,
    power_consversion_constant: float
) -> Tuple[Float[Array, "num_timepoints"], Float[Array, "num_timepoints"], Float[Array, "num_timepoints"]]:
    """
    TODO: write docstring
    """
    
    power_modeled = power_forward_model(
                        wavelength,
                        polar_angle,
                        azimuthal_angle,
                        transverse_electric_component,
                        transverse_magnetic_component,
                        permeability_reflection,
                        permittivity_reflection,
                        permeability_transmission,
                        permittivity_transmission,
                        backside_mode,
                        permeability_static_size_layers,
                        permittivity_static_size_layers,
                        static_layer_thicknesses,
                        permeability_variable_layer,
                        permittivity_variable_layer,
                        timepoints_measured,
                        initial_thickness,
                        growth_velocity,
                        growth_acceleration,
                        power_conversion_factor,
                        power_consversion_constant
                    )
    
    return power_modeled - power_measured