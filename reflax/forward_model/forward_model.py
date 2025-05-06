# general
import jax
import jax.numpy as jnp
from functools import partial

# typing
from typing import Tuple
from jaxtyping import Array, Float

# reflax parameter classes
from reflax.constants import MIN_MAX_NORMALIZATION, ONE_LAYER_MODEL, S_POLARIZED, TRANSFER_MATRIX_METHOD
from reflax.parameter_classes.parameters import (
    IncidentMediumParams,
    LayerParams,
    LightSourceParams,
    SetupParams,
    TransmissionMediumParams
)

# reflax interference models
from reflax._reflectance_models._one_layer_model import (
    one_layer_model,
)
from reflax._reflectance_models import transfer_matrix_method

@partial(jax.jit, static_argnames=['backside_mode', 'model', 'normalization', 'polarization_state'])
def forward_model(
    model: int,
    setup_params: SetupParams,
    light_source_params: LightSourceParams,
    incident_medium_params: IncidentMediumParams,
    transmission_medium_params: TransmissionMediumParams,
    static_layer_params: LayerParams,
    variable_layer_params: LayerParams,
    variable_layer_thicknesses: Float[Array, "num_timepoints"],
    backside_mode: int,
    polarization_state: int = S_POLARIZED,
    normalization: int = MIN_MAX_NORMALIZATION,
) -> Float[Array, "num_timepoints"]:
    """
    Forward model from thickness to reflectance.

    Args:
        model: Model to use for the forward calculation.
            ONE_LAYER_MODEL or TRANSFER_MATRIX_METHOD.
        setup_params: Setup parameters (angle of incidence, etc.).
        light_source_params: Light source parameters (wavelength, etc.).
        incident_medium_params: Incident medium parameters (permittivity, permeability).
        transmission_medium_params: Transmission medium parameters (permittivity, permeability).
        static_layer_params: Static layer parameters (permittivities, permeabilities).
        variable_layer_params: Variable layer parameters (permittivities, permeabilities).
        variable_layer_thicknesses: Thicknesses of the variable layer.
        backside_mode: Backside mode (0 or 1).
        polarization_state: Polarization state (only applies to ONE_LAYER_MODEL).
            S_POLARIZED or P_POLARIZED.
        normalization: Normalization method, either NO_NORMALIZATION or MIN_MAX_NORMALIZATION.
    
    Returns:
        Reflectance values for the given thicknesses.
    """


    if model == ONE_LAYER_MODEL:
        variable_layer_params = variable_layer_params._replace(
            thicknesses = variable_layer_thicknesses
        )
        out = one_layer_model(
            setup_params = setup_params,
            light_source_params = light_source_params,
            incident_medium_params = incident_medium_params,
            transmission_medium_params = transmission_medium_params,
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
                light_source_params = light_source_params,
                incident_medium_params = incident_medium_params,
                transmission_medium_params = transmission_medium_params,
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

@partial(jax.jit, static_argnames=['backside_mode', 'model', 'normalization', 'polarization_state', 'computation_batch_size'])
def batched_forward_model(
    model: int,
    setup_params: SetupParams,
    light_source_params: LightSourceParams,
    incident_medium_params: IncidentMediumParams,
    transmission_medium_params: TransmissionMediumParams,
    static_layer_params: LayerParams,
    variable_layer_params: LayerParams,
    variable_layer_thicknesses: Float[Array, "num_time_series num_timepoints"],
    backside_mode: int,
    polarization_state: int = S_POLARIZED,
    normalization: int = MIN_MAX_NORMALIZATION,
    computation_batch_size = 1000
) -> Float[Array, "num_time_series num_timepoints"]:
    """
    Batched forward model from thickness to reflectance for multiple time series,
    using jax.lax.map with a specified batch size.
    """

    return jax.lax.map(
        lambda variable_layer_thicknesses: forward_model(
            model = model,
            setup_params = setup_params,
            light_source_params = light_source_params,
            incident_medium_params = incident_medium_params,
            transmission_medium_params = transmission_medium_params,
            static_layer_params = static_layer_params,
            variable_layer_params = variable_layer_params,
            variable_layer_thicknesses = variable_layer_thicknesses,
            backside_mode = backside_mode,
            polarization_state = polarization_state,
            normalization = normalization
        ),
        variable_layer_thicknesses,
        batch_size = computation_batch_size
    )