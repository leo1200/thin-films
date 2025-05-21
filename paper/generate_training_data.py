# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================

import jax
import jax.numpy as jnp
# NOTE: without 64-bit precision,
# the Cholesky decomposition fails
from jax import config

config.update("jax_enable_x64", True)

# constants
from reflax import ONE_LAYER_MODEL
from reflax.forward_model.forward_model import batched_forward_model
from reflax.thickness_modeling.function_sampling import (
    sample_derivative_bound_gp, sample_linear_functions)


def generate_training_data(training_data_path = "simulated_data/training_data.npz"):

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
    # one might choose a smaller lengthscale for the second GP
    lengthscaleB = 0.4

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
        training_data_path,
        thicknesses = thicknesses,
        derivatives = derivatives,
        reflectances = reflectances,
        time_points = time_points,
    )

    # -------------------------------------------------------------
    # ===================== ↑ data generation ↑ ===================
    # -------------------------------------------------------------