"""
Here we will compare the different models of interference.

The models compared will be:
 - one layer, no internal reflections
 - one layer, with internal reflections
 - multi-layer transfer matrix method

"""

# ==== GPU selection ====
# from autocvd import autocvd

# from reflax.constants import S_POLARIZED
# autocvd(num_gpus = 1)
# only use gpu 9
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
# =======================

import jax.numpy as jnp

import matplotlib.pyplot as plt

# simulator setup
from reflax.parameter_classes.parameters import (
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

# forward model
from reflax import (
    forward_model
)

from reflax import get_polarization_components

# -------------------------------------------------------------
# ===================== ↓ Simulator Setup ↓ ===================
# -------------------------------------------------------------

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


static_layer_paramsA = LayerParams(
    permeabilities = jnp.array([1.0]),
    permittivities = jnp.array([1.457**2]),
    thicknesses = jnp.array([0.0])
)

static_layer_paramsB = LayerParams(
    permeabilities = jnp.array([1.0, 1.0]),
    permittivities = jnp.array([1.5**2, 1.6**2]),
    thicknesses = jnp.array([100.0, 50.0])
)

n_variable = 1.457
k_variable = 0.0
permeability_variable_layer = 1.0
permittivity_variable_layer = (n_variable + 1j * k_variable)**2
variable_layer_params = LayerParams(
    permeabilities = permeability_variable_layer,
    permittivities = permittivity_variable_layer,
)

# -------------------------------------------------------------
# ===================== ↑ Simulator Setup ↑ ===================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ===================== ↓ data generation ↓ ===================
# -------------------------------------------------------------

# one hour of data
time = jnp.linspace(0, 1, 1000)

# assume a linear growth of 1000 nm / hour
growth_rate = 1000

# calculate the thickness of the variable layer
variable_layer_thicknesses = time * growth_rate

# nomalization
normalization = NO_NORMALIZATION

reflectanceII_A = forward_model(
    model = ONE_LAYER_MODEL,
    setup_params = setup_params,
    light_source_params = light_source_params,
    incident_medium_params = incident_medium_params,
    transmission_medium_params = transmission_medium_params,
    static_layer_params = static_layer_paramsA,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = variable_layer_thicknesses,
    backside_mode = backside_mode,
    polarization_state = polarization_state,
    normalization = normalization
)

reflectanceTMM_A = forward_model(
    model = TRANSFER_MATRIX_METHOD,
    setup_params = setup_params,
    light_source_params = light_source_params,
    incident_medium_params = incident_medium_params,
    transmission_medium_params = transmission_medium_params,
    static_layer_params = static_layer_paramsA,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = variable_layer_thicknesses,
    backside_mode = backside_mode,
    polarization_state = polarization_state,
    normalization = normalization
)

reflectanceTMM_B = forward_model(
    model = TRANSFER_MATRIX_METHOD,
    setup_params = setup_params,
    light_source_params = light_source_params,
    incident_medium_params = incident_medium_params,
    transmission_medium_params = transmission_medium_params,
    static_layer_params = static_layer_paramsB,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = variable_layer_thicknesses,
    backside_mode = backside_mode,
    polarization_state = polarization_state,
    normalization = normalization
)

# -------------------------------------------------------------
# ===================== ↑ data generation ↑ ===================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ======================== ↓ plotting ↓ =======================
# -------------------------------------------------------------

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

ax1.plot(time, reflectanceII_A, "b--", label = "one layer, with internal reflections", alpha = 0.5)
ax1.plot(time, reflectanceTMM_A, "r--", label = "transfer matrix method", alpha = 0.5)
ax1.set_xlabel("time in hours")
ax1.set_ylabel("reflectance")
ax1.set_title("single growing layer")
ax1.legend(loc="upper right")
ax1.set_ylim(0, 1)

ax2.plot(time, reflectanceTMM_B, "r--", label = "transfer matrix method", alpha = 0.5)
ax2.set_xlabel("time in hours")
ax2.set_ylabel("reflectance")
ax2.set_title("three layers, one growing")
ax2.legend(loc="upper right")
ax2.set_ylim(0, 1)

plt.tight_layout()
plt.savefig("figures/interference_model_comparison.svg")

# -------------------------------------------------------------
# ======================== ↑ plotting ↑ =======================
# -------------------------------------------------------------
