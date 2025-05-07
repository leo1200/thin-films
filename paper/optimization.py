# ==== GPU selection ====
# from autocvd import autocvd

# from reflax.constants import S_POLARIZED
# autocvd(num_gpus = 1)
# only use gpu 9
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "9"
# =======================

from reflax.thickness_modeling.operator_learning import NeuralOperatorMLP, load_model

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

# add random noise to the reflectance
noise = jax.random.normal(random_key, true_reflectance.shape) * 0.01
# true_reflectance = true_reflectance + noise

true_reflectance = jnp.squeeze(true_reflectance)

# -------------------------------------------------------------
# ================ ↑ example data generation ↑ ================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ================= ↓ neural network fitting ↓ ================
# -------------------------------------------------------------

RANDOM_INITIALIZATION = 0
LINEAR_INITIALIZATION = 1
NEURAL_OPERATOR_INITIALIZATION = 2

def estimate_thickness(
    time_points,
    true_reflectance,
    true_thickness,
    true_growth_rate,
    learning_rate = 4e-4,
    num_epochs = 14000,
    print_interval = 500,
    patience = 8000,
    nn_initialization = LINEAR_INITIALIZATION,
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


    elif nn_initialization == LINEAR_INITIALIZATION:

        # initialize the neural network to a growth rate of 1000 nm/h
        init_growth_rate = 1000.0
        initialized_growth_rate = jnp.ones(time_points.shape) * init_growth_rate
        initialized_thickness = init_growth_rate * time_points
        initialized_time_points = time_points
        true_reflectance_initialized = true_reflectance
        true_thickness_initialized = true_thickness
        true_growth_rate_initialized = true_growth_rate

        growth_nn = linear_output_initialization(
            growth_nn,
            init_growth_rate
        )

    elif nn_initialization == NEURAL_OPERATOR_INITIALIZATION:
        training_data = jnp.load("training_data/training_data.npz")
        initialized_time_points = training_data["time_points"]
        num_points_no = initialized_time_points.shape[0]
        neural_operator = NeuralOperatorMLP(
            hidden_dims = [512, 512],
            num_eval_points = num_points_no,
            rngs = nnx.Rngs(42),
        )
        # load the neural operator model
        neural_operator = load_model(
            filepath = "models/neural_operator.pickle",
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
    growth_nn, losses = train_nn_model(
        growth_nn,
        forward_model_params,
        time_points,
        true_reflectance,
        learning_rate = learning_rate,
        num_epochs = num_epochs,
        print_interval = print_interval,
        patience = patience
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
        variable_layer_thicknesses = predicted_thickness,
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

    print(f"initial reflectance loss: {initial_reflectance_loss}")
    print(f"initial thickness loss: {initial_thickness_loss}")
    print(f"initial growth rate loss: {initial_growth_rate_loss}")

    # final result losses
    reflectance_loss = jnp.mean((true_reflectance - predicted_reflectance)**2)
    thickness_loss = jnp.mean((true_thickness - predicted_thickness)**2)
    growth_rate_loss = jnp.mean((true_growth_rate - predicted_growth_rate)**2)

    print(f"reflectance loss: {reflectance_loss}")
    print(f"thickness loss: {thickness_loss}")
    print(f"growth rate loss: {growth_rate_loss}")

    return (
        losses,
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

(
    losses,
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
    nn_initialization = NEURAL_OPERATOR_INITIALIZATION,
    seed_for_random_initialization = 0,
    pretrain_learning_rate = 4e-4,
    pretrain_num_epochs = 20000,
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
# ax2.plot(time_points, initial_thickness_guess, label = "initial guess")
ax2.plot(
    initialized_time_points,
    initialized_thickness,
    label = "neural operator prediction",
    linestyle = "--"
)
ax2.set_xlabel("time in hours")
ax2.set_ylabel("thickness in nm")
ax2.legend(loc = "upper right")
ax2.set_title("Thickness")

# plot derivative in the third subplot
ax3.plot(time_points, true_growth_rate, label = "true growth rate")
ax3.plot(time_points, predicted_growth_rate, label = "predicted growth rate")
ax3.plot(
    initialized_time_points,
    initialized_growth_rate,
    label = "neural operator prediction",
    linestyle = "--"
)
ax3.set_xlabel("time in hours")
ax3.set_ylabel("growth rate in nm/h")
ax3.legend(loc = "lower right")
ax3.set_title("Growth Rate")

plt.tight_layout()

plt.savefig("figures/optimization_example.svg")

# -------------------------------------------------------------
# ======================== ↑ plotting ↑ =======================
# -------------------------------------------------------------
