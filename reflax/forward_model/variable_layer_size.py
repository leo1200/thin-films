from functools import partial
import jax
import jax.numpy as jnp
from reflax.parameter_classes.parameters import GrowthModel, LayerParams, OpticsParams, SetupParams
from reflax.reflactance_models import transfer_matrix_method
from jaxtyping import Array, Float
from functools import partial
from typing import Tuple

from reflax.reflactance_models.baseline_methods import one_layer_internal_reflections, one_layer_no_internal_reflections

@partial(jax.jit, static_argnames=['backside_mode'])
def variable_layer_thickness_simulation(
    setup_params: SetupParams,
    optics_params: OpticsParams,
    static_layer_params: LayerParams,
    variable_layer_params: LayerParams,
    variable_layer_thicknesses: Float[Array, "num_thicknesses"],
    backside_mode: int
) -> Tuple[Float[Array, "num_thicknesses"], Float[Array, "num_thicknesses"], Float[Array, "num_thicknesses"]]:
    """
    Compute the reflection, transmission and conservation check for the case of one of the layers
    varying in thickness with computations vectorized over multiple thicknesses of this verying layer.

    
    Args:
        setup_params: parameters of the setup
        optics_params: parameters of the optics
        static_layer_params: parameters of the layer not changing in thickness
        variable_layer_params: parameters of the variable thickness layer, except for the thicknesses
                               which are defined in variable_layer_thicknesses
        variable_layer_thicknesses: Array of the different thicknesses of the layer with varying thicknesses.

    Returns:
        Reflectance (REF), Transmittance (TRN), Conservation (CON) for all the
        layer thicknesses specified in variable_layer_thicknesses while all
        other options remain constant.
    """

    # Construc the overall layer params

    # prepend the permeability and permittivity of the layer varying in thickness
    # to these quantities of the other layers
    layer_permeabilities = jnp.concatenate((jnp.array([variable_layer_params.permeabilities]), static_layer_params.permeabilities))
    layer_permittivities = jnp.concatenate((jnp.array([variable_layer_params.permittivities]), static_layer_params.permittivities))

    # initialize a matrix of thicknesses, we vectorize over
    thickness_matrix = jnp.zeros((variable_layer_thicknesses.shape[0], static_layer_params.thicknesses.shape[0] + 1))
    thickness_matrix = thickness_matrix.at[:, 1:].set(static_layer_params.thicknesses)
    thickness_matrix = thickness_matrix.at[:, 0].set(variable_layer_thicknesses)

    layer_params = LayerParams(permeabilities=layer_permeabilities,permittivities=layer_permittivities)

    reflection_coefficients, transmission_coefficients, conservation_checks = jax.vmap(
        lambda layer_thicknesses: transfer_matrix_method(
            setup_params = setup_params,
            optics_params = optics_params,
            layer_params = layer_params._replace(thicknesses = layer_thicknesses),
            backside_mode = backside_mode
        )
    )(thickness_matrix)
    
    return reflection_coefficients, transmission_coefficients, conservation_checks

@partial(jax.jit, static_argnames=['backside_mode'])
def power_forward_model(
    setup_params: SetupParams,
    optics_params: OpticsParams,
    static_layer_params: LayerParams,
    variable_layer_params: LayerParams,
    timepoints_measured: Float[Array, "num_timepoints"],
    growth_model: GrowthModel,
    power_conversion_factor: float,
    power_conversion_constant: float,
    backside_mode: int
) -> Tuple[Float[Array, "num_timepoints"], Float[Array, "num_timepoints"], Float[Array, "num_timepoints"]]:
    """
    TODO: write docstring
    """
    
    variable_layer_thicknesses = growth_model.initial_thickness + growth_model.growth_velocity * timepoints_measured + 0.5 * growth_model.growth_acceleration * timepoints_measured ** 2

    reflection_coefficients, _, _ = variable_layer_thickness_simulation(
                                        setup_params = setup_params,
                                        optics_params = optics_params,
                                        static_layer_params = static_layer_params,
                                        variable_layer_params = variable_layer_params,
                                        variable_layer_thicknesses = variable_layer_thicknesses,
                                        backside_mode = backside_mode
                                    ) 
    
    power_output = power_conversion_constant + power_conversion_factor * reflection_coefficients

    return power_output


ONE_LAYER_NO_INTERNAL_REFLECTIONS = 0
ONE_LAYER_INTERNAL_REFLECTIONS = 1
TRANSFER_MATRIX_METHOD = 2


@partial(jax.jit, static_argnames=['backside_mode', 'model'])
def normalized_forward_model(
    model: int,
    setup_params: SetupParams,
    optics_params: OpticsParams,
    static_layer_params: LayerParams,
    variable_layer_params: LayerParams,
    timepoints_measured: Float[Array, "num_timepoints"],
    growth_model: GrowthModel,
    backside_mode: int
) -> Tuple[Float[Array, "num_timepoints"], Float[Array, "num_timepoints"], Float[Array, "num_timepoints"]]:
    """
    TODO: write docstring
    """
    
    variable_layer_thicknesses = growth_model.initial_thickness + growth_model.growth_velocity * timepoints_measured + 0.5 * growth_model.growth_acceleration * timepoints_measured ** 2

    if model == ONE_LAYER_NO_INTERNAL_REFLECTIONS:
        variable_layer_params = variable_layer_params._replace(thicknesses = variable_layer_thicknesses)
        out = one_layer_no_internal_reflections(
            setup_params = setup_params,
            optics_params = optics_params,
            layer_params = variable_layer_params
        )
    elif model == ONE_LAYER_INTERNAL_REFLECTIONS:
        variable_layer_params = variable_layer_params._replace(thicknesses = variable_layer_thicknesses)
        out = one_layer_internal_reflections(
            setup_params = setup_params,
            optics_params = optics_params,
            layer_params = variable_layer_params
        )
    elif model == TRANSFER_MATRIX_METHOD:
        out, _, _ = variable_layer_thickness_simulation(
                                    setup_params = setup_params,
                                    optics_params = optics_params,
                                    static_layer_params = static_layer_params,
                                    variable_layer_params = variable_layer_params,
                                    variable_layer_thicknesses = variable_layer_thicknesses,
                                    backside_mode = backside_mode
                                ) 
    else:
        raise ValueError("Invalid model choice")

    return (out - jnp.mean(out)) / jnp.std(out)


@partial(jax.jit, static_argnames=['backside_mode'])
def power_forward_residuals(
    setup_params: SetupParams,
    optics_params: OpticsParams,
    static_layer_params: LayerParams,
    variable_layer_params: LayerParams,
    timepoints_measured: Float[Array, "num_timepoints"],
    growth_model: GrowthModel,
    power_conversion_factor: float,
    power_conversion_constant: float,
    backside_mode: int,
    power_measured: Float[Array, "num_timepoints"],
) -> Tuple[Float[Array, "num_timepoints"], Float[Array, "num_timepoints"], Float[Array, "num_timepoints"]]:
    """
    TODO: write docstring
    """
    
    power_modeled = power_forward_model(
                        setup_params = setup_params,
                        optics_params = optics_params,
                        static_layer_params = static_layer_params,
                        variable_layer_params = variable_layer_params,
                        timepoints_measured = timepoints_measured,
                        growth_model = growth_model,
                        power_conversion_factor = power_conversion_factor,
                        power_conversion_constant = power_conversion_constant,
                        backside_mode = backside_mode
                    )
    
    return power_modeled - power_measured