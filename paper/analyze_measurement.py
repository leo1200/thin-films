# # ==== GPU selection ====
# from autocvd import autocvd
# autocvd(num_gpus = 1)
# # =======================

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from flax import nnx

# NOTE: without 64-bit precision,
# the Cholesky decomposition fails
from jax import config
from matplotlib import gridspec
from train_neural_operator import neural_operator_training

from reflax.thickness_modeling.operator_learning import NeuralOperatorMLP, load_model

config.update("jax_enable_x64", True)

import matplotlib.pyplot as plt

# forward model
# constants
from reflax import ONE_LAYER_MODEL, forward_model
from reflax.forward_model.forward_model import batched_forward_model
from reflax.thickness_modeling.function_sampling import (
    sample_derivative_bound_gp,
    sample_linear_functions,
)
from reflax.thickness_modeling.nn_modeling import (
    RawGrowthNN,
    predict_growth_rate,
    predict_thickness,
    pretrained_initialization,
    train_nn_model,
)


def analyze_measurement():

    train_data_path = "simulated_data/training_data_setup2.npz"
    neural_operator_path = "saved_models/neural_operator_setup2.pickle"
    figpath = "figures/measurement_analysis.svg"
    analysis_save_path = "measurements/analysis_results.npz"
    model = ONE_LAYER_MODEL
    learning_rate = 4e-4
    num_epochs = 15000
    print_interval = 500
    patience = 8000
    initial_linear_growth_rate = 1000.0
    seed_for_random_initialization = 0
    pretrain_learning_rate = 4e-4
    pretrain_num_epochs = 20000

    # -------------------------------------------------------------
    # ==================== ↓ load measurement ↓ ===================
    # -------------------------------------------------------------

    measurement = np.loadtxt("measurements/reflectance.txt", skiprows=1)
    time_raw = jnp.array(measurement[:-100, 0])

    # convert the time to hours, numerically ~ nicely between 0 and 1
    time_points_measured = time_raw / 3600
    final_time = time_points_measured[-1]
    reflectance_raw = jnp.array(measurement[:-100, 1])

    # normalize and center the reflectance
    measured_reflectance = (
        reflectance_raw - 0.5 * (jnp.min(reflectance_raw) + jnp.max(reflectance_raw))
    ) / (0.5 * (jnp.max(reflectance_raw) - jnp.min(reflectance_raw)))

    # -------------------------------------------------------------
    # ==================== ↑ load measurement ↑ ===================
    # -------------------------------------------------------------

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

    # the polar angle is different between the simulation
    # and the measurement test, this happened by accident
    # but should not influence the results
    setup_params = setup_params._replace(polar_angle=jnp.deg2rad(25))
    forward_model_params = forward_model_params._replace(setup_params=setup_params)

    # -------------------------------------------------------------
    # ===================== ↑ Simulator Setup ↑ ===================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ===================== ↓ data generation ↓ ===================
    # -------------------------------------------------------------

    num_eval_points = 100
    time_points_neural_operator = jnp.linspace(0, final_time, num_eval_points)

    num_samples_linear = 100
    num_samplesA = 1000
    num_samplesB = 1000

    random_key_linear = jax.random.PRNGKey(123)

    thicknesses_linear, derivatives_linear = sample_linear_functions(
        random_key_linear,
        num_samples_linear,
        time_points_neural_operator,
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
        time_points_neural_operator,
        lengthscaleA,
        variance,
        min_slope,
        max_slope,
        random_final_values=True,
        min_final_value=800.0,
        max_final_value=1200.0,
        convex_samples=True,
    )

    random_keyB = jax.random.PRNGKey(69)
    lengthscaleB = 0.2

    thickness_gpB, derivatives_gpB = sample_derivative_bound_gp(
        random_keyB,
        num_samplesB,
        time_points_neural_operator,
        lengthscaleA,
        variance,
        min_slope,
        max_slope,
        random_final_values=True,
        min_final_value=800.0,
        max_final_value=1200.0,
        convex_samples=True,
    )

    # concatenate all thicknesses (linear and GP) and derivatives
    thicknesses = jnp.concatenate(
        (thicknesses_linear, thickness_gpA, thickness_gpB), axis=0
    )
    derivatives = jnp.concatenate(
        (derivatives_linear, derivatives_gpA, derivatives_gpB), axis=0
    )

    # generate reflectance data
    reflectances = batched_forward_model(
        model=model,
        setup_params=setup_params,
        light_source_params=light_source_params,
        incident_medium_params=incident_medium_params,
        transmission_medium_params=transmission_medium_params,
        static_layer_params=static_layer_params,
        variable_layer_params=variable_layer_params,
        variable_layer_thicknesses=thicknesses,
        backside_mode=backside_mode,
        polarization_state=polarization_state,
        normalization=normalization,
        computation_batch_size=100,
    )

    # print the shapes of the generated data
    print(f"Thicknesses shape: {thicknesses.shape}")
    print(f"Derivatives shape: {derivatives.shape}")
    print(f"Reflectances shape: {reflectances.shape}")
    print(f"Time points shape: {time_points_neural_operator.shape}")

    # save the training data
    jnp.savez(
        train_data_path,
        thicknesses=thicknesses,
        derivatives=derivatives,
        reflectances=reflectances,
        time_points=time_points_neural_operator,
    )

    # -------------------------------------------------------------
    # ===================== ↑ data generation ↑ ===================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================= ↓ train neural operator ↓ =================
    # -------------------------------------------------------------

    neural_operator_training(
        training_data_path=train_data_path, model_save_path=neural_operator_path
    )

    # -------------------------------------------------------------
    # ================= ↑ train neural operator ↑ =================
    # -------------------------------------------------------------

    # -------------------------------------------------------------
    # ================ ↓ neural network fitting ↓ =================
    # -------------------------------------------------------------

    # initialize the neural network
    growth_nn = RawGrowthNN(dmid=1024, rngs=nnx.Rngs(seed_for_random_initialization))

    training_data = jnp.load(train_data_path)
    initialized_time_points = training_data["time_points"]
    num_points_no = initialized_time_points.shape[0]
    neural_operator = NeuralOperatorMLP(
        hidden_dims=[512, 512],
        num_eval_points=num_points_no,
        rngs=nnx.Rngs(42),
    )
    # load the neural operator model
    neural_operator = load_model(
        filepath=neural_operator_path,
        abstract_model=neural_operator,
    )
    # downsample the time points to the number of points in the neural operator
    indices = jnp.linspace(0, time_points_measured.shape[0] - 1, num_points_no).astype(
        int
    )
    initialized_time_points = time_points_measured[indices]
    measured_reflectance_initialized = measured_reflectance[indices]
    initialized_thickness = neural_operator(measured_reflectance_initialized)
    # calculate the derivative from the predicted thickness
    initialized_growth_rate = jnp.gradient(
        initialized_thickness, initialized_time_points
    )

    # initialize the neural network to the neural operator result
    growth_nn = pretrained_initialization(
        growth_nn,
        initialized_thickness,
        initialized_time_points,
        pretrain_num_epochs,
        pretrain_learning_rate,
    )

    # train the neural network on the generated data
    growth_nn, reflectance_losses = train_nn_model(
        growth_nn,
        forward_model_params,
        time_points_measured,
        measured_reflectance,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        print_interval=print_interval,
        patience=patience,
    )

    # predict the thickness and derivative
    predicted_thickness = predict_thickness(growth_nn, time_points_measured)
    predicted_growth_rate = predict_growth_rate(growth_nn, time_points_measured)

    # calculate the reflectance of the initialized thickness
    initialized_reflectance = forward_model(
        model=model,
        setup_params=setup_params,
        light_source_params=light_source_params,
        incident_medium_params=incident_medium_params,
        transmission_medium_params=transmission_medium_params,
        static_layer_params=static_layer_params,
        variable_layer_params=variable_layer_params,
        variable_layer_thicknesses=initialized_thickness,
        backside_mode=backside_mode,
        polarization_state=polarization_state,
        normalization=normalization,
    )

    # calculate the reflectance from the predicted thickness
    predicted_reflectance = forward_model(
        model=model,
        setup_params=setup_params,
        light_source_params=light_source_params,
        incident_medium_params=incident_medium_params,
        transmission_medium_params=transmission_medium_params,
        static_layer_params=static_layer_params,
        variable_layer_params=variable_layer_params,
        variable_layer_thicknesses=predicted_thickness,
        backside_mode=backside_mode,
        polarization_state=polarization_state,
        normalization=normalization,
    )

    # -------------------------------------------------------------
    # ================ ↑ neural network fitting ↑ =================
    # -------------------------------------------------------------

    # save all data needed for plotting into a npz file
    jnp.savez(
        analysis_save_path,
        time_points_measured=time_points_measured,
        measured_reflectance=measured_reflectance,
        initialized_time_points=initialized_time_points,
        initialized_thickness=initialized_thickness,
        initialized_growth_rate=initialized_growth_rate,
        predicted_reflectance=predicted_reflectance,
        predicted_thickness=predicted_thickness,
        predicted_growth_rate=predicted_growth_rate,
    )

    # load the data for plotting
    data = jnp.load(analysis_save_path)
    time_points_measured = data["time_points_measured"]
    measured_reflectance = data["measured_reflectance"]
    initialized_time_points = data["initialized_time_points"]
    initialized_thickness = data["initialized_thickness"]
    initialized_growth_rate = data["initialized_growth_rate"]
    predicted_reflectance = data["predicted_reflectance"]
    predicted_thickness = data["predicted_thickness"]
    predicted_growth_rate = data["predicted_growth_rate"]

    # print the rms reflectance error
    rms_reflectance_error = jnp.sqrt(
        jnp.mean((predicted_reflectance - measured_reflectance) ** 2)
    )
    print(f"RMS reflectance error: {rms_reflectance_error}")

    # -------------------------------------------------------------
    # ======================= ↓ plotting ↓ ========================
    # -------------------------------------------------------------

    fig = plt.figure(
        figsize=(15, 5)
    )  # Adjusted figsize slightly for the new subplot layout

    # Define the main GridSpec: 3 rows for 3 main blocks of plots
    # Block 1: Reflectance + Error (total height ratio 3, e.g., 2 for Reflectance, 1 for Error)
    # Block 2: Thickness (height ratio 2)
    # Block 3: Growth Rate (height ratio 2)
    gs_main = gridspec.GridSpec(1, 3, figure=fig)

    # Create a nested GridSpec for the first block (Reflectance and its Error)
    # This nested GridSpec will live inside gs_main[0]
    # It will have 2 rows: Reflectance (height 2) and Error (height 1)
    # hspace is reduced to make them closer.
    gs_reflectance_error = gridspec.GridSpecFromSubplotSpec(
        2, 1, subplot_spec=gs_main[0], height_ratios=[2, 1], hspace=0.05
    )

    # Create axes based on the nested gridspec structure
    ax1 = fig.add_subplot(gs_reflectance_error[0])  # Reflectance plot
    ax_error = fig.add_subplot(gs_reflectance_error[1])  # Absolute error plot
    ax2 = fig.add_subplot(gs_main[1])  # Thickness plot
    ax3 = fig.add_subplot(gs_main[2])  # Growth rate plot

    # plot true reflectance in the first subplot (ax1)
    ax1.plot(time_points_measured, measured_reflectance, label="measured reflectance")
    ax1.plot(time_points_measured, predicted_reflectance, label="predicted reflectance")
    ax1.set_ylabel("normalized reflectance")
    ax1.legend(loc="lower left")
    ax1.set_title("Reflectance")
    # ax1.set_xlabel("time in hours")

    # plot absolute error in reflectance in the new subplot (ax_error)
    absolute_error = jnp.abs(measured_reflectance - predicted_reflectance)
    ax_error.plot(
        time_points_measured, absolute_error, label="absolute error", color="red"
    )
    ax_error.set_ylabel("abs. error")
    ax_error.legend(loc="upper right")
    ax_error.set_xlabel("time in hours")

    # plot thickness sample in the second main subplot (ax2)
    ax2.plot(time_points_measured, predicted_thickness, label="predicted thickness")
    ax2.plot(
        initialized_time_points,
        initialized_thickness,
        label="initial guess",
        linestyle="--",
    )
    ax2.set_ylabel("thickness in nm")
    ax2.legend(loc="upper left")
    ax2.set_title("Thickness")
    ax2.set_xlabel("time in hours")

    # plot derivative in the third main subplot (ax3)
    ax3.plot(time_points_measured, predicted_growth_rate, label="predicted growth rate")
    ax3.plot(
        initialized_time_points,
        initialized_growth_rate,
        label="initial prediction",
        linestyle="--",
    )
    ax3.set_xlabel("time in hours")
    ax3.set_ylabel("growth rate in nm/h")
    ax3.legend(loc="lower right")
    ax3.set_title("Growth Rate")
    ax3.set_xlabel("time in hours")

    plt.tight_layout()
    # Potentially adjust overall spacing if tight_layout isn't perfect with nested grids
    # fig.subplots_adjust(hspace=0.3) # Example: Adjust main vertical spacing if needed

    plt.savefig(figpath)

    # -------------------------------------------------------------
    # ======================= ↑ plotting ↑ ========================
    # -------------------------------------------------------------
