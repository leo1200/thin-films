# general
import jax
import jax.numpy as jnp
from functools import partial

# typing
from typing import Tuple
from jaxtyping import Array, Float

# reflax parameter classes
from reflax.constants import MIN_MAX_NORMALIZATION, ONE_LAYER_INTERNAL_REFLECTIONS, ONE_LAYER_NO_INTERNAL_REFLECTIONS, S_POLARIZED, TRANSFER_MATRIX_METHOD
from reflax.parameter_classes.parameters import (
    LayerParams,
    OpticsParams,
    SetupParams
)

# reflax interference models
from reflax._reflectance_models.baseline_methods import (
    one_layer_internal_reflections,
    one_layer_no_internal_reflections
)
from reflax._reflectance_models import transfer_matrix_method

@partial(jax.jit, static_argnames=['backside_mode', 'model', 'normalization', 'polarization_state'])
def forward_model(
    model: int,
    setup_params: SetupParams,
    optics_params: OpticsParams,
    static_layer_params: LayerParams,
    variable_layer_params: LayerParams,
    variable_layer_thicknesses: Float[Array, "num_timepoints"],
    backside_mode: int,
    polarization_state: int = S_POLARIZED,
    normalization: int = MIN_MAX_NORMALIZATION,
) -> Float[Array, "num_timepoints"]:
    """
    Forward model: thickness -> reflectance

    NOTE: all one-layer models only 
    consider the variable layer
    """

    if model == ONE_LAYER_NO_INTERNAL_REFLECTIONS:
        variable_layer_params = variable_layer_params._replace(
            thicknesses = variable_layer_thicknesses
        )
        out = one_layer_no_internal_reflections(
            setup_params = setup_params,
            optics_params = optics_params,
            layer_params = variable_layer_params,
            polarization_state = polarization_state
        )
    
    elif model == ONE_LAYER_INTERNAL_REFLECTIONS:
        variable_layer_params = variable_layer_params._replace(
            thicknesses = variable_layer_thicknesses
        )
        out = one_layer_internal_reflections(
            setup_params = setup_params,
            optics_params = optics_params,
            layer_params = variable_layer_params,
            polarization_state = polarization_state
        )
    
    elif model == TRANSFER_MATRIX_METHOD:
        # Construct the overall layer params
        # prepend the permeability and permittivity of the layer varying in thickness
        # to these quantities of the other layers
        layer_permeabilities = jnp.concatenate((
            jnp.array([variable_layer_params.permeabilities]),
            static_layer_params.permeabilities
        ))

        layer_permittivities = jnp.concatenate((
            jnp.array([variable_layer_params.permittivities]),
            static_layer_params.permittivities
        ))

        # initialize a matrix of thicknesses, we vectorize over
        thickness_matrix = jnp.zeros((
            variable_layer_thicknesses.shape[0],
            static_layer_params.thicknesses.shape[0] + 1
        ))
        
        thickness_matrix = thickness_matrix.at[:, 1:].set(static_layer_params.thicknesses)
        thickness_matrix = thickness_matrix.at[:, 0].set(variable_layer_thicknesses)

        layer_params = LayerParams(
            permeabilities = layer_permeabilities,
            permittivities = layer_permittivities
        )

        out, _, _ = jax.vmap(
            lambda layer_thicknesses: transfer_matrix_method(
                setup_params = setup_params,
                optics_params = optics_params,
                layer_params = layer_params._replace(thicknesses = layer_thicknesses),
                backside_mode = backside_mode
            )
        )(thickness_matrix)
    
    else:
        raise ValueError("Invalid model choice")
    
    # the MIN_MAX_NORMALIZATION gets the reflectance 
    # to the range [-1, 1] like a sine function
    if normalization == MIN_MAX_NORMALIZATION:
        out = (out - 0.5 * (jnp.min(out) + jnp.max(out))) / (0.5 * (jnp.max(out) - jnp.min(out)))

    return out