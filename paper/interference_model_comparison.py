"""
Here we will compare the different models of interference.

The models compared will be:
 - one layer, no internal reflections
 - one layer, with internal reflections
 - multi-layer transfer matrix method

"""

# ==== GPU selection ====
from autocvd import autocvd
autocvd(num_gpus = 1)
# =======================

import jax.numpy as jnp

import matplotlib.pyplot as plt

from reflax.reflactance_models.basic_optics import polanalyze

# simulator setup
from reflax.parameter_classes.parameters import (
    LayerParams,
    OpticsParams,
    SetupParams
)

# different interference models
from reflax.forward_model.variable_layer_size import (
    ONE_LAYER_NO_INTERNAL_REFLECTIONS,
    ONE_LAYER_INTERNAL_REFLECTIONS,
    TRANSFER_MATRIX_METHOD,
    forward_model
)

# normalization of the model output
from reflax.forward_model.variable_layer_size import (
    NO_NORMALIZATION,
    MIN_MAX_NORMALIZATION
)

# -------------------------------------------------------------
# ===================== ↓ Simulator Setup ↓ ===================
# -------------------------------------------------------------

polar_angle = jnp.deg2rad(75)
azimuthal_angle = jnp.deg2rad(0)

setup_params = SetupParams(
    polar_angle = polar_angle,
    azimuthal_angle = azimuthal_angle
)

polarization_state = "Linear TE/perpendicular/s"
transverse_electric_component, transverse_magnetic_component = polanalyze(polarization_state)

permeability_reflection = 1
permittivity_reflection = 1

permeability_transmission = 1
permittivity_transmission = (3.8827 + 0.019626j)**2

optics_params = OpticsParams(
    permeability_reflection = permeability_reflection,
    permittivity_reflection = permittivity_reflection,
    permeability_transmission = permeability_transmission,
    permittivity_transmission = permittivity_transmission,
    transverse_electric_component = transverse_electric_component,
    transverse_magnetic_component = transverse_magnetic_component
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

reflectanceI = forward_model(
    model = ONE_LAYER_NO_INTERNAL_REFLECTIONS,
    setup_params = setup_params,
    optics_params = optics_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = variable_layer_thicknesses,
    backside_mode = backside_mode,
    normalization = normalization
)

reflectanceII = forward_model(
    model = ONE_LAYER_INTERNAL_REFLECTIONS,
    setup_params = setup_params,
    optics_params = optics_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = variable_layer_thicknesses,
    backside_mode = backside_mode,
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
    normalization = normalization
)

# -------------------------------------------------------------
# ===================== ↑ data generation ↑ ===================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ======================== ↓ plotting ↓ =======================
# -------------------------------------------------------------

fig, ax = plt.subplots(1, 1, figsize=(10, 5))
ax.plot(time, reflectanceI, label = "one layer, no internal reflections")
ax.plot(time, reflectanceII, label = "one layer, with internal reflections")
ax.plot(time, reflectanceTMM, label = "transfer matrix method")
ax.set_xlabel("time in hours")
ax.set_ylabel("reflectance")
ax.set_title("model comparison for linear thin-film growth")
ax.legend(loc="upper right")
plt.savefig("figures/interference_model_comparison.svg")

# -------------------------------------------------------------
# ======================== ↑ plotting ↑ =======================
# -------------------------------------------------------------
