# ==== GPU selection ====
# from autocvd import autocvd

# from reflax.constants import S_POLARIZED
# autocvd(num_gpus = 1)
# only use gpu 9
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# =======================

from reflax.forward_model.forward_model import batched_forward_model
from reflax.thickness_modeling.operator_learning import NeuralOperatorMLP, load_model

import jax.numpy as jnp
import jax

import time as pytime

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
    pretrained_initialization,
    train_nn_model
)


# -------------------------------------------------------------
# ===================== ↓ Simulator Setup ↓ ===================
# -------------------------------------------------------------

from baseline_setup import forward_model_params
model = forward_model_params.model
setup_params = forward_model_params.setup_params
light_source_params = forward_model_params.light_source_params
incident_medium_params = forward_model_params.incident_medium_params
transmission_medium_params = forward_model_params.transmission_medium_params
static_layer_params = forward_model_params.static_layer_params
variable_layer_params = forward_model_params.variable_layer_params
backside_mode = forward_model_params.backside_mode
polarization_state = forward_model_params.polarization_state
normalization = forward_model_params.normalization

# -------------------------------------------------------------
# ===================== ↑ Simulator Setup ↑ ===================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ example data generation ↓ ================
# -------------------------------------------------------------

time_points = jnp.linspace(0, 1, 400)

random_key = jax.random.PRNGKey(69)
num_samples = 1

lengthscale = 0.1
variance = 10.0
min_slope = 200.0
max_slope = 1800.0

true_thickness, true_derivative = sample_derivative_bound_gp(
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

true_thickness = jnp.squeeze(true_thickness)

true_reflectance = forward_model(
    model = ONE_LAYER_MODEL,
    setup_params = setup_params,
    light_source_params = light_source_params,
    incident_medium_params = incident_medium_params,
    transmission_medium_params = transmission_medium_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = true_thickness,
    backside_mode = backside_mode,
    polarization_state = polarization_state,
    normalization = normalization
)

fig, ax = plt.subplots(1, 1, figsize=(3, 3))
# only plot the reflectance
ax.plot(true_thickness[:150], true_reflectance[:150], label="reflectance")
ax.set_xlabel("thickness in nm")
ax.set_ylabel("reflectance")
plt.tight_layout()
plt.savefig("reflectance.svg")