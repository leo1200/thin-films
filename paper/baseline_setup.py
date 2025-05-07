# simulator setup
from reflax._reflectance_models._one_layer_model import get_polarization_components
from reflax.parameter_classes.parameters import (
    ForwardModelParams,
    LayerParams,
    SetupParams,
    LightSourceParams,
    TransmissionMediumParams,
    IncidentMediumParams
)

# constants
from reflax import (
    ONE_LAYER_MODEL,
    TRANSFER_MATRIX_METHOD,
    S_POLARIZED,
    NO_NORMALIZATION,
    MIN_MAX_NORMALIZATION
)

import jax.numpy as jnp

normalization = MIN_MAX_NORMALIZATION

wavelength = 632.8

polar_angle = jnp.deg2rad(75)
azimuthal_angle = jnp.deg2rad(0)

polarization_state = S_POLARIZED
s_component, p_component = get_polarization_components(polarization_state)

light_source_params = LightSourceParams(
    wavelength = wavelength,
    s_component = s_component,
    p_component = p_component
)

setup_params = SetupParams(
    polar_angle = polar_angle,
    azimuthal_angle = azimuthal_angle,
)

permeability_reflection = 1.0
permittivity_reflection = 1.0

incident_medium_params = IncidentMediumParams(
    permeability_reflection = permeability_reflection,
    permittivity_reflection = permittivity_reflection
)

permeability_transmission = 1.0
permittivity_transmission = (3.8827 + 0.019626j)**2

transmission_medium_params = TransmissionMediumParams(
    permeability_transmission = permeability_transmission,
    permittivity_transmission = permittivity_transmission
)

backside_mode = 1


static_layer_params = LayerParams(
    permeabilities = jnp.array([1.0]),
    permittivities = jnp.array([1.457**2]),
    thicknesses = jnp.array([0.0])
)

n_variable = 1.457
k_variable = 0.0
permeability_variable_layer = 1.0
permittivity_variable_layer = (n_variable + 1j * k_variable)**2
variable_layer_params = LayerParams(
    permeabilities = permeability_variable_layer,
    permittivities = permittivity_variable_layer,
)

forward_model_params = ForwardModelParams(
    model = TRANSFER_MATRIX_METHOD,
    setup_params = setup_params,
    light_source_params = light_source_params,
    incident_medium_params = incident_medium_params,
    transmission_medium_params = transmission_medium_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_params,
    backside_mode = backside_mode,
    polarization_state = polarization_state,
    normalization = normalization,
)
