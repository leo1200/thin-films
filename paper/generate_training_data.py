# ==== GPU selection ====
# from autocvd import autocvd

# from reflax.constants import S_POLARIZED
# autocvd(num_gpus = 1)
# only use gpu 9
import os

from reflax.forward_model.forward_model import batched_forward_model
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
# =======================

import jax.numpy as jnp
import jax

from flax import nnx

# NOTE: without 64-bit precision,
# the Cholesky decomposition fails
from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

# simulator setup
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

# forward model
from reflax import (
    forward_model
)

from reflax import get_polarization_components

from reflax.thickness_modeling.function_sampling import sample_derivative_bound_gp, sample_linear_functions

# -------------------------------------------------------------
# ===================== ↓ Simulator Setup ↓ ===================
# -------------------------------------------------------------

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

# -------------------------------------------------------------
# ===================== ↑ Simulator Setup ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ data generation ↓ ===================
# -------------------------------------------------------------

num_eval_points = 100
time_points = jnp.linspace(0, 1, num_eval_points)

num_samples_linear = 100
num_samplesA = 1000
num_samplesB = 1000

random_key_linear = jax.random.PRNGKey(123)

thicknesses_linear, derivatives_linear = sample_linear_functions(
    random_key_linear,
    num_samples_linear,
    time_points,
    800.0,
    1200.0,
)

random_keyA = jax.random.PRNGKey(89)

lengthscaleA = 0.4
variance = 15.0
min_slope = 200.0
max_slope = 1800.0

thickness_gpA, derivatives_gpA = sample_derivative_bound_gp(
    random_keyA,
    num_samplesA,
    time_points,
    lengthscaleA,
    variance,
    min_slope,
    max_slope,
    random_final_values = True,
    min_final_value = 800.0,
    max_final_value = 1200.0,
    convex_samples = True,
)

random_keyB = jax.random.PRNGKey(69)
lengthscaleB = 0.2

# THERE WAS AN ERROR HERE SO WE ACTUALLY USED LENGTHSCALE A
# TWICE!!!!, so 2k with lengthscale 0.4!
thickness_gpB, derivatives_gpB = sample_derivative_bound_gp(
    random_keyB,
    num_samplesB,
    time_points,
    lengthscaleB,
    variance,
    min_slope,
    max_slope,
    random_final_values = True,
    min_final_value = 800.0,
    max_final_value = 1200.0,
    convex_samples = True,
)

# concatenate all thicknesses (linear and GP) and derivatives
thicknesses = jnp.concatenate((thicknesses_linear, thickness_gpA, thickness_gpB), axis=0)
derivatives = jnp.concatenate((derivatives_linear, derivatives_gpA, derivatives_gpB), axis=0)

# generate reflectance data
reflectances = batched_forward_model(
    model = ONE_LAYER_MODEL,
    setup_params = setup_params,
    light_source_params = light_source_params,
    incident_medium_params = incident_medium_params,
    transmission_medium_params = transmission_medium_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = thicknesses,
    backside_mode = backside_mode,
    polarization_state = polarization_state,
    normalization = normalization,
    computation_batch_size = 100
)

# print the shapes of the generated data
print(f"Thicknesses shape: {thicknesses.shape}")
print(f"Derivatives shape: {derivatives.shape}")
print(f"Reflectances shape: {reflectances.shape}")
print(f"Time points shape: {time_points.shape}")

# save the training data
jnp.savez(
    "training_data/training_data.npz",
    thicknesses = thicknesses,
    derivatives = derivatives,
    reflectances = reflectances,
    time_points = time_points,
)

# -------------------------------------------------------------
# ===================== ↑ data generation ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================== ↓ plotting ↓ =======================
# -------------------------------------------------------------

# plot all the generated data
fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

ax1.plot(time_points, reflectances.T, alpha=0.5)
ax1.set_title("Reflectance")
ax1.set_xlabel("Time")
ax1.set_ylabel("Reflectance")

ax2.plot(time_points, thicknesses.T, alpha=0.5)
ax2.set_title("Thicknesses")
ax2.set_xlabel("Time")
ax2.set_ylabel("Thickness")

ax3.plot(time_points, derivatives.T, alpha=0.5)
ax3.set_title("Derivatives")
ax3.set_xlabel("Time")
ax3.set_ylabel("Derivative")

plt.tight_layout()

plt.savefig("figures/training_data.png", dpi=300)

# -------------------------------------------------------------
# ======================== ↑ plotting ↑ =======================
# -------------------------------------------------------------
