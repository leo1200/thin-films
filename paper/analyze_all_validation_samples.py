# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================

from reflax.forward_model.forward_model import batched_forward_model
from reflax.thickness_modeling.operator_learning import NeuralOperatorMLP, load_model

import jax.numpy as jnp
import jax

from flax import nnx

# NOTE: without 64-bit precision,
# the Cholesky decomposition fails
from jax import config
config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

# constants
from reflax import (
    ONE_LAYER_MODEL,
)

# forward model
from reflax import (
    forward_model
)

from reflax.thickness_modeling.function_sampling import sample_derivative_bound_gp

from reflax.thickness_modeling.nn_modeling import (
    RawGrowthNN,
    linear_output_initialization,
    predict_growth_rate,
    predict_thickness,
    pretrained_initialization,
    train_nn_model
)

def analyze_all_validation_samples():

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


    # plot all the generated data
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 12))

    ax1.plot(time_points, true_reflectances.T, alpha=0.5)
    ax1.set_title("Reflectance")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Reflectance")

    ax2.plot(time_points, true_thicknesses.T, alpha=0.5)
    ax2.set_title("Thicknesses")
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Thickness")

    ax3.plot(time_points, true_derivatives.T, alpha=0.5)
    ax3.set_title("Derivatives")
    ax3.set_xlabel("Time")
    ax3.set_ylabel("Derivative")

    plt.tight_layout()

    plt.savefig("figures/validation_data.png", dpi=300)


    # -------------------------------------------------------------
    # ================ ↑ example data generation ↑ ================
    # -------------------------------------------------------------


    # -------------------------------------------------------------
    # ================= ↓ neural network fitting ↓ ================
    # -------------------------------------------------------------

    RANDOM_INITIALIZATION = 0
    LINEAR_INITIALIZATION_SET = 1
    LINEAR_INITIALIZATION_TRAINED = 2
    NEURAL_OPERATOR_INITIALIZATION = 3

    def estimate_thickness(
        time_points,
        true_reflectance,
        true_thickness,
        true_growth_rate,
        learning_rate = 4e-4,
        num_epochs = 15000,
        print_interval = 500,
        patience = 8000,
        nn_initialization = LINEAR_INITIALIZATION_TRAINED,
        initial_linear_growth_rate = 1000.0,
        seed_for_random_initialization = 0,
        pretrain_learning_rate = 4e-4,
        pretrain_num_epochs = 20000,
    ):

        # initialize the neural network
        growth_nn = RawGrowthNN(dmid = 1024, rngs = nnx.Rngs(seed_for_random_initialization))

        if nn_initialization == RANDOM_INITIALIZATION:

            initialized_thickness = predict_thickness(
                growth_nn,
                time_points
            )

            initialized_growth_rate = predict_growth_rate(
                growth_nn,
                time_points
            )

            initialized_time_points = time_points
            true_reflectance_initialized = true_reflectance
            true_thickness_initialized = true_thickness
            true_growth_rate_initialized = true_growth_rate


        elif nn_initialization == LINEAR_INITIALIZATION_SET:

            # initialize the neural network
            initialized_growth_rate = jnp.ones(time_points.shape) * initial_linear_growth_rate
            initialized_thickness = initial_linear_growth_rate * time_points
            initialized_time_points = time_points
            true_reflectance_initialized = true_reflectance
            true_thickness_initialized = true_thickness
            true_growth_rate_initialized = true_growth_rate

            growth_nn = linear_output_initialization(
                growth_nn,
                initial_linear_growth_rate
            )

        elif nn_initialization == LINEAR_INITIALIZATION_TRAINED:

            initialized_growth_rate = jnp.ones(time_points.shape) * initial_linear_growth_rate
            initialized_thickness = initial_linear_growth_rate * time_points
            initialized_time_points = time_points
            true_reflectance_initialized = true_reflectance
            true_thickness_initialized = true_thickness
            true_growth_rate_initialized = true_growth_rate
            growth_nn = pretrained_initialization(
                growth_nn,
                initialized_thickness,
                initialized_time_points,
                pretrain_num_epochs,
                pretrain_learning_rate,
            )

        elif nn_initialization == NEURAL_OPERATOR_INITIALIZATION:
            training_data = jnp.load("simulated_data/training_data.npz")
            initialized_time_points = training_data["time_points"]
            num_points_no = initialized_time_points.shape[0]
            neural_operator = NeuralOperatorMLP(
                hidden_dims = [512, 512],
                num_eval_points = num_points_no,
                rngs = nnx.Rngs(42),
            )
            # load the neural operator model
            neural_operator = load_model(
                filepath = "saved_models/neural_operator.pickle",
                abstract_model = neural_operator,
            )
            # downsample the time points to the number of points in the neural operator
            indices = jnp.linspace(0, time_points.shape[0] - 1, num_points_no).astype(int)
            initialized_time_points = time_points[indices]
            true_reflectance_initialized = true_reflectance[indices]
            true_thickness_initialized = true_thickness[indices]
            true_growth_rate_initialized = true_growth_rate[indices]
            # predict the thickness from the reflectance
            initialized_thickness = neural_operator(true_reflectance_initialized)
            # calculate the derivative from the predicted thickness
            initialized_growth_rate = jnp.gradient(initialized_thickness, initialized_time_points)

            # initialize the neural network to the neural operator result
            growth_nn = pretrained_initialization(
                growth_nn,
                initialized_thickness,
                initialized_time_points,
                pretrain_num_epochs,
                pretrain_learning_rate,
            )


        # train the neural network on the generated data
        growth_nn, reflectance_losses, thickness_losses, growth_rate_losses = train_nn_model(
            growth_nn,
            forward_model_params,
            time_points,
            true_reflectance,
            learning_rate = learning_rate,
            num_epochs = num_epochs,
            print_interval = print_interval,
            patience = patience,
            true_thickness = true_thickness,
            true_growth_rate = true_growth_rate,
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

        # calculate the reflectance of the initialized thickness
        initialized_reflectance = forward_model(
            model = model,
            setup_params = setup_params,
            light_source_params = light_source_params,
            incident_medium_params = incident_medium_params,
            transmission_medium_params = transmission_medium_params,
            static_layer_params = static_layer_params,
            variable_layer_params = variable_layer_params,
            variable_layer_thicknesses = initialized_thickness,
            backside_mode = backside_mode,
            polarization_state = polarization_state,
            normalization = normalization
        )

        # calculate the reflectance from the predicted thickness
        predicted_reflectance = forward_model(
            model = model,
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

        # initial guess losses
        initial_reflectance_loss = jnp.mean((true_reflectance_initialized - initialized_reflectance)**2)
        initial_thickness_loss = jnp.mean((true_thickness_initialized - initialized_thickness)**2)
        initial_growth_rate_loss = jnp.mean((true_growth_rate_initialized - initialized_growth_rate)**2)

        print(f"initial reflectance loss: {initial_reflectance_loss:.2e}")
        print(f"initial thickness loss: {initial_thickness_loss:.2e}")
        print(f"initial growth rate loss: {initial_growth_rate_loss:.2e}")

        # final result losses
        reflectance_loss = jnp.mean((true_reflectance - predicted_reflectance)**2)
        thickness_loss = jnp.mean((true_thickness - predicted_thickness)**2)
        growth_rate_loss = jnp.mean((true_growth_rate - predicted_growth_rate)**2)

        print(f"reflectance loss: {reflectance_loss:.2e}")
        print(f"thickness loss: {thickness_loss:.2e}")
        print(f"growth rate loss: {growth_rate_loss:.2e}")

        return (
            reflectance_losses,
            thickness_losses,
            growth_rate_losses,
            predicted_reflectance,
            predicted_thickness,
            predicted_growth_rate,
            initialized_thickness,
            initialized_reflectance,
            initialized_growth_rate,
            initialized_time_points,
            initial_reflectance_loss,
            initial_thickness_loss,
            initial_growth_rate_loss,
            reflectance_loss,
            thickness_loss,
            growth_rate_loss,
        )

    initializations = [
        RANDOM_INITIALIZATION,
        LINEAR_INITIALIZATION_SET,
        LINEAR_INITIALIZATION_TRAINED,
        NEURAL_OPERATOR_INITIALIZATION,
    ]

    def initialization_to_string(initialization):
        if initialization == RANDOM_INITIALIZATION:
            return "RANDOM_INITIALIZATION"
        elif initialization == LINEAR_INITIALIZATION_SET:
            return "LINEAR_INITIALIZATION_SET"
        elif initialization == LINEAR_INITIALIZATION_TRAINED:
            return "LINEAR_INITIALIZATION_TRAINED"
        elif initialization == NEURAL_OPERATOR_INITIALIZATION:
            return "NEURAL_OPERATOR_INITIALIZATION"
        else:
            raise ValueError("Invalid initialization type.")
        
    # loop over all samples
    for i in range(true_thicknesses.shape[0]):
        print(f"Sample {i + 1}/{true_thicknesses.shape[0]}")

        # get the true reflectance, thickness and growth rate
        true_reflectance = true_reflectances[i]
        true_thickness = true_thicknesses[i]
        true_growth_rate = true_derivatives[i]

        # loop over all initializations
        for initialization in initializations:
            print(f"Initialization: {initialization_to_string(initialization)}")

            # estimate the thickness
            (
                reflectance_losses,
                thickness_losses,
                growth_rate_losses,
                predicted_reflectance,
                predicted_thickness,
                predicted_growth_rate,
                initialized_thickness,
                initialized_reflectance,
                initialized_growth_rate,
                initialized_time_points,
                initial_reflectance_loss,
                initial_thickness_loss,
                initial_growth_rate_loss,
                reflectance_loss,
                thickness_loss,
                growth_rate_loss,
            ) = estimate_thickness(
                time_points,
                true_reflectance,
                true_thickness,
                true_growth_rate,
                learning_rate = 4e-4,
                num_epochs = 14000,
                print_interval = 500,
                patience = 8000,
                nn_initialization = initialization,
                initial_linear_growth_rate = 1000.0,
                seed_for_random_initialization = 0,
                pretrain_learning_rate = 4e-3,
                pretrain_num_epochs = 30000,
            )

            # save the results
            jnp.savez(
                f"validation_results_data/{initialization_to_string(initialization)}/{initialization_to_string(initialization)}_{i + 1}.npz",
                sample_index = i,
                initialization = initialization,
                time_points = time_points,
                true_reflectance = true_reflectance,
                true_thickness = true_thickness,
                true_growth_rate = true_growth_rate,
                initialized_time_points = initialized_time_points,
                initialized_reflectance = initialized_reflectance,
                initialized_thickness = initialized_thickness,
                initialized_growth_rate = initialized_growth_rate,
                predicted_reflectance = predicted_reflectance,
                predicted_thickness = predicted_thickness,
                predicted_growth_rate = predicted_growth_rate,
                reflectance_losses = reflectance_losses,
                thickness_losses = thickness_losses,
                growth_rate_losses = growth_rate_losses,
                initial_reflectance_loss = initial_reflectance_loss,
                initial_thickness_loss = initial_thickness_loss,
                initial_growth_rate_loss = initial_growth_rate_loss,
                reflectance_loss = reflectance_loss,
                thickness_loss = thickness_loss,
                growth_rate_loss = growth_rate_loss
            )

    # -------------------------------------------------------------
    # ================= ↑ neural network fitting ↑ ================
    # -------------------------------------------------------------