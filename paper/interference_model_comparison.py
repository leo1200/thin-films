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
    OpticsParams,
    SetupParams
)

# constants
from reflax import (
    ONE_LAYER_INTERNAL_REFLECTIONS,
    TRANSFER_MATRIX_METHOD,
    S_POLARIZED,
    NO_NORMALIZATION,
    MIN_MAX_NORMALIZATION
)

# forward model
from reflax import (
    forward_model
)

from reflax._reflectance_models.basic_optics import get_polarization_components


# -------------------------------------------------------------
# ===================== ↓ Simulator Setup ↓ ===================
# -------------------------------------------------------------

wavelength = 632.8

polar_angle = jnp.deg2rad(75)
azimuthal_angle = jnp.deg2rad(0)

polarization_state = S_POLARIZED
s_component, p_component = get_polarization_components(polarization_state)

setup_params = SetupParams(
    wavelength = wavelength,
    polar_angle = polar_angle,
    azimuthal_angle = azimuthal_angle,
)

permeability_reflection = 1
permittivity_reflection = 1

permeability_transmission = 1
permittivity_transmission = (3.8827 + 0.019626j)**2

optics_params = OpticsParams(
    permeability_reflection = permeability_reflection,
    permittivity_reflection = permittivity_reflection,
    permeability_transmission = permeability_transmission,
    permittivity_transmission = permittivity_transmission,
    s_component = s_component,
    p_component = p_component
)

backside_mode = 1

static_layer_thicknesses = jnp.array([0.0])
permeability_static_size_layers = jnp.array([1.0])
permittivity_static_size_layers = jnp.array([1.45704**2])

static_layer_params = LayerParams(
    permeabilities = permeability_static_size_layers,
    permittivities = permittivity_static_size_layers,
    thicknesses = static_layer_thicknesses
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

reflectanceII = forward_model(
    model = ONE_LAYER_INTERNAL_REFLECTIONS,
    setup_params = setup_params,
    optics_params = optics_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = variable_layer_thicknesses,
    backside_mode = backside_mode,
    polarization_state = polarization_state,
    normalization = normalization
)

reflectanceTMM = forward_model(
    model = TRANSFER_MATRIX_METHOD,
    setup_params = setup_params,
    optics_params = optics_params,
    static_layer_params = static_layer_params,
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

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(time, reflectanceII, "b--", label = "one layer, with internal reflections", alpha = 0.5)
ax.plot(time, reflectanceTMM, "r--", label = "transfer matrix method", alpha = 0.5)
# ax.plot(time, reflectanceTMM, label = "transfer matrix method")
ax.set_xlabel("time in hours")
ax.set_ylabel("reflectance")
ax.set_title("model comparison for linear thin-film growth")
ax.legend(loc="upper right")
plt.savefig("figures/interference_model_comparison.svg")

# -------------------------------------------------------------
# ======================== ↑ plotting ↑ =======================
# -------------------------------------------------------------
