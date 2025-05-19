# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================

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

# constants
from reflax import (
    ONE_LAYER_MODEL
)

from reflax.thickness_modeling.function_sampling import sample_derivative_bound_gp

def generate_validation_data(validation_data_path = "simulated_data/validation_data.npz"):

    # -------------------------------------------------------------
    # ===================== ↓ Simulator Setup ↓ ===================
    # -------------------------------------------------------------

    from baseline_forward_model_setup import forward_model_params
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
    # ===================== ↓ Simulator Setup ↓ ===================
    # -------------------------------------------------------------

    from baseline_forward_model_setup import forward_model_params
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
    num_samples = 200

    lengthscale = 0.1
    variance = 10.0
    min_slope = 200.0
    max_slope = 1800.0

    true_thicknesses, true_derivatives = sample_derivative_bound_gp(
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

    true_reflectances = batched_forward_model(
        model = ONE_LAYER_MODEL,
        setup_params = setup_params,
        light_source_params = light_source_params,
        incident_medium_params = incident_medium_params,
        transmission_medium_params = transmission_medium_params,
        static_layer_params = static_layer_params,
        variable_layer_params = variable_layer_params,
        variable_layer_thicknesses = true_thicknesses,
        backside_mode = backside_mode,
        polarization_state = polarization_state,
        normalization = normalization,
        computation_batch_size = 100
    )

    jnp.savez(
        validation_data_path,
        thicknesses = true_thicknesses,
        derivatives = true_derivatives,
        reflectances = true_reflectances,
        time_points = time_points,
    )