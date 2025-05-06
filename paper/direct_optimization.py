
# ==== GPU selection ====
# from autocvd import autocvd

# from reflax.constants import S_POLARIZED
# autocvd(num_gpus = 1)
# only use gpu 9
import os
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

from reflax.thickness_modeling.function_sampling import sample_derivative_bound_gp

from reflax.thickness_modeling.nn_modeling import (
    RawGrowthNN,
    linear_output_initialization,
    predict_growth_rate,
    predict_thickness,
    train_nn_model
)

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
# ================ ↓ example data generation ↓ ================
# -------------------------------------------------------------

time_points = jnp.linspace(0, 1, 400)

random_key = jax.random.PRNGKey(42)
num_samples = 1

lengthscale = 0.4
variance = 10.0
min_slope = 200.0
max_slope = 1800.0

thickness_sample, derivatives = sample_derivative_bound_gp(
    random_key,
    num_samples,
    time_points,
    lengthscale,
    variance,
    min_slope,
    max_slope,
    random_final_values = True,
    min_final_value = 800.0,
    max_final_value = 1200.0,
    convex_samples = True,
)

# squeeze the output, as it is generate with
# shape num_samples x num_time_steps
true_thickness = jnp.squeeze(thickness_sample)
true_growth_rate = jnp.squeeze(derivatives)

true_reflectance = forward_model(
    model = ONE_LAYER_MODEL,
    setup_params = setup_params,
    light_source_params = light_source_params,
    incident_medium_params = incident_medium_params,
    transmission_medium_params = transmission_medium_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = thickness_sample,
    backside_mode = backside_mode,
    polarization_state = polarization_state,
    normalization = normalization
)

true_reflectance = jnp.squeeze(true_reflectance)

# -------------------------------------------------------------
# ================ ↑ example data generation ↑ ================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================= ↓ neural network fitting ↓ ================
# -------------------------------------------------------------

# initialize the neural network
growth_nn = RawGrowthNN(dmid = 1024, rngs = nnx.Rngs(0))

# initialize the neural network to a growth rate of 1000 nm/h
init_growth_rate = 1000.0
initial_thickness_guess = init_growth_rate * time_points
growth_nn = linear_output_initialization(
    growth_nn,
    init_growth_rate
)

# train the neural network on the generated data
growth_nn = train_nn_model(
    growth_nn,
    forward_model_params,
    time_points,
    true_reflectance,
    learning_rate = 1e-4,
    num_epochs = 12000,
    print_interval = 500,
    patience = 4000
)

# predict the thickness and derivative
predicted_thickness = predict_thickness(
    growth_nn,
    time_points
)
predicted_growth_rate = predict_growth_rate(
    growth_nn,
    time_points
)

# calculate the reflectance from the predicted thickness
predicted_reflectance = forward_model(
    model = ONE_LAYER_MODEL,
    setup_params = setup_params,
    light_source_params = light_source_params,
    incident_medium_params = incident_medium_params,
    transmission_medium_params = transmission_medium_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = predicted_thickness,
    backside_mode = backside_mode,
    polarization_state = polarization_state,
    normalization = normalization
)

# -------------------------------------------------------------
# ================= ↑ neural network fitting ↑ ================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ======================== ↓ plotting ↓ =======================
# -------------------------------------------------------------

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

# plot true reflectance in the first subplot
ax1.plot(time_points, true_reflectance, label = "measured reflectance")
ax1.plot(time_points, predicted_reflectance, label = "predicted reflectance")
ax1.set_xlabel("time in hours")
ax1.set_ylabel("reflectance")
ax1.legend(loc = "upper right")
ax1.set_title("Reflectance")

# plot thickness sample in the second subplot
ax2.plot(time_points, true_thickness, label = "true thickness")
ax2.plot(time_points, predicted_thickness, label = "predicted thickness")
ax2.plot(time_points, initial_thickness_guess, label = "initial guess")
ax2.set_xlabel("time in hours")
ax2.set_ylabel("thickness in nm")
ax2.legend(loc = "upper right")
ax2.set_title("Thickness")

# plot derivative in the third subplot
ax3.plot(time_points, true_growth_rate, label = "true growth rate")
ax3.plot(time_points, predicted_growth_rate, label = "predicted growth rate")
ax3.set_xlabel("time in hours")
ax3.set_ylabel("growth rate in nm/h")
ax3.legend(loc = "lower right")
ax3.set_title("Growth Rate")

plt.tight_layout()

plt.savefig("figures/direct_optimization_example.svg")

# -------------------------------------------------------------
# ======================== ↑ plotting ↑ =======================
# -------------------------------------------------------------
