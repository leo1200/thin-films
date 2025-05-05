"""
The frequency of the reflectance signal scales with
2 pi d / \lambda * sqrt(n_1 ^ 2 - n_0 ^ 2 * sin^2(theta))
where n_0 is the refractive index of the incident medium
and n_1 is the refractive index of the thin film.

define v_tilde = v * sqrt(n_1 ^ 2 - n_0 ^ 2 * sin^2(theta))

"""

v_tilde = 1000.0

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

polar_angle = jnp.deg2rad(25)
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

n_incident = jnp.sqrt(incident_medium_params.permittivity_reflection * incident_medium_params.permeability_reflection)

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

n_variable_A = 1.5
v_A = v_tilde / jnp.sqrt(n_variable_A**2 - n_incident**2 * jnp.sin(polar_angle)**2)

variable_layer_paramsA = LayerParams(
    permeabilities = 1.0,
    permittivities = n_variable_A**2,
)

n_variable_B = 1.25
v_B = v_tilde / jnp.sqrt(n_variable_B**2 - n_incident**2 * jnp.sin(polar_angle)**2)

variable_layer_paramsB = LayerParams(
    permeabilities = 1.0,
    permittivities = n_variable_B**2,
)

# -------------------------------------------------------------
# ===================== ↑ Simulator Setup ↑ ===================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ===================== ↓ data generation ↓ ===================
# -------------------------------------------------------------

# one hour of data
time = jnp.linspace(0, 1, 1000)

# calculate the thickness of the variable layer
variable_layer_thicknessesA = time * v_A
variable_layer_thicknessesB = time * v_B

# nomalization
normalization = NO_NORMALIZATION

reflectanceA = forward_model(
    model = ONE_LAYER_MODEL,
    setup_params = setup_params,
    light_source_params = light_source_params,
    incident_medium_params = incident_medium_params,
    transmission_medium_params = transmission_medium_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_paramsA,
    variable_layer_thicknesses = variable_layer_thicknessesA,
    backside_mode = backside_mode,
    polarization_state = polarization_state,
    normalization = normalization
)

reflectanceB = forward_model(
    model = ONE_LAYER_MODEL,
    setup_params = setup_params,
    light_source_params = light_source_params,
    incident_medium_params = incident_medium_params,
    transmission_medium_params = transmission_medium_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_paramsB,
    variable_layer_thicknesses = variable_layer_thicknessesB,
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

fig, ax1 = plt.subplots(1, 1, figsize=(5, 5))

ax1.plot(time, reflectanceA, "b--", label = "n = {}, v = {} nm/h".format(n_variable_A, v_A))
ax1.plot(time, reflectanceB, "r--", label = "n = {}, v = {} nm/h".format(n_variable_B, v_B))
ax1.set_xlabel("time in hours")
ax1.set_ylabel("reflectance")
ax1.set_title("degeneracy test")
ax1.legend(loc="upper right")

plt.savefig("figures/degeneracy_test.svg")

# -------------------------------------------------------------
# ======================== ↑ plotting ↑ =======================
# -------------------------------------------------------------
