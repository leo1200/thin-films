# GPU selection
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# typing
from typing import Tuple

# timing
import time as pytime

# numerics
import numpy as np
import jax.numpy as jnp

# neural nets 
from flax import nnx
import optax

# reflax (our library)
from reflax import polanalyze
from reflax.parameter_classes.parameters import (
    OpticsParams,
    SetupParams,
    LayerParams
)
from reflax.forward_model.forward_model import (
    MIN_MAX_NORMALIZATION,
    ONE_LAYER_INTERNAL_REFLECTIONS,
    TRANSFER_MATRIX_METHOD
)
from reflax.forward_model.forward_model import forward_model

# plotting
import matplotlib.pyplot as plt

# -------------------------------------------------------------
# ======================== ↓ settings ↓ =======================
# -------------------------------------------------------------

interference_model = ONE_LAYER_INTERNAL_REFLECTIONS
use_simulated_data = True

if use_simulated_data:
    print("Using simulated data.")

# -------------------------------------------------------------
# ======================== ↑ settings ↑ =======================
# -------------------------------------------------------------


# -------------------------------------------------------------
# =================== ↓ data preprocessing ↓ ==================
# -------------------------------------------------------------

measurement = np.loadtxt("measurements/reflectance.txt", skiprows = 1)
time_raw = jnp.array(measurement[:-100, 0])

# convert the time to hours, numerically ~ nicely between 0 and 1
time_hours = time_raw / 3600
reflectance_raw = jnp.array(measurement[:-100, 1])

# normalize and center the reflectance
reflectance_norm = (
    reflectance_raw - 0.5 * (jnp.min(reflectance_raw) + jnp.max(reflectance_raw))
) / (0.5 * (jnp.max(reflectance_raw) - jnp.min(reflectance_raw)))

# calculate the time differences
# we need these to get from growth rate to thickness
# predicting growth rate it is easier to ensure monotonicity
# and starting at ~ zero thickness
dt_hours = jnp.diff(time_hours)
dt_hours = jnp.concatenate((jnp.array([dt_hours[0]]), dt_hours))
dt_hours = dt_hours.reshape(-1, 1)

# reshaping the time, so it is a column vector
time_nn_input = time_hours[:, None] # Shape becomes (N, 1)

# we target reproducing our measured reflectance
reflectance_target = reflectance_norm

print(f"Time input shape for NN: {time_nn_input.shape}")
print(f"Reflectance target shape: {reflectance_target.shape}")
print(f"dt_hours shape: {dt_hours.shape}")

# -------------------------------------------------------------
# =================== ↑ data preprocessing ↑ ==================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ===================== ↓ simulator setup ↓ ===================
# -------------------------------------------------------------

wavelength = 632.8
polar_angle = jnp.deg2rad(25)
azimuthal_angle = jnp.deg2rad(0)
setup_params = SetupParams(
    wavelength=wavelength,
    polar_angle=polar_angle,
    azimuthal_angle=azimuthal_angle
)

polarization_state = "Linear TE/perpendicular/s"
s_component, p_component = polanalyze(polarization_state)
permeability_reflection = 1.0
permittivity_reflection = complex(1.0, 0.0)
permeability_transmission = 1.0
permittivity_transmission = (3.8827 + 0.019626j)**2
optics_params = OpticsParams(
    permeability_reflection=permeability_reflection,
    permittivity_reflection=permittivity_reflection,
    permeability_transmission=permeability_transmission,
    permittivity_transmission=permittivity_transmission,
    s_component=s_component,
    p_component=p_component
)

backside_mode = 1

static_layer_thicknesses = jnp.array([0.0])
permeability_static_size_layers = jnp.array([permeability_transmission])
permittivity_static_size_layers = jnp.array([permittivity_transmission])
static_layer_params = LayerParams(
    permeabilities=permeability_static_size_layers,
    permittivities=permittivity_static_size_layers,
    thicknesses=static_layer_thicknesses
)

permeability_variable_layer = 1.0
permittivity_variable_layer = complex(1.57, 0.0)**2
variable_layer_params = LayerParams(
    permeabilities = permeability_variable_layer,
    permittivities = permittivity_variable_layer,
    thicknesses = None
)

# -------------------------------------------------------------
# ==================== ↑ simulator setup ↑ ====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# =============== ↓ simulated data generation ↓ ===============
# -------------------------------------------------------------

# for verification purposes
if use_simulated_data:

    def smooth_sqrt_growth(t, A=850, epsilon=0.1):
        return A * jnp.sqrt(t + epsilon) - A * jnp.sqrt(epsilon)

    def d_smooth_sqrt_growth(t, A=850, epsilon=0.1):
        return A / (2 * jnp.sqrt(t + epsilon))

    # Usage:
    true_growth_rate = d_smooth_sqrt_growth(time_hours)
    true_thicknesses = smooth_sqrt_growth(time_hours)
    
    reflectance_target = forward_model(
        model = interference_model,
        setup_params = setup_params,
        optics_params = optics_params,
        static_layer_params = static_layer_params,
        variable_layer_params = variable_layer_params,
        variable_layer_thicknesses = true_thicknesses,
        backside_mode = backside_mode,
        normalization = MIN_MAX_NORMALIZATION
    )


# -------------------------------------------------------------
# ================ ↓ neural network definition ↓ ==============
# -------------------------------------------------------------


class RawGrowthNN(nnx.Module):
    """
    We use a simple neural network to predict a quantity
    raw_growth which after application of a softplus 
    (to ensure non-negativity of the growth rate) 
    and scaling estimates the layer growth rate and 
    after integration the layer thickness.

    time -> raw_growth -> softplus -> rate -> scale_factor 
    -> growth_rate -> integration over multiple rates -> thickness

    The network therefore goes from a scalar input 
    (time) to a scalar output (raw_growth).
    """

    def __init__(self, dmid: int, *, rngs: nnx.Rngs):

        # input layer: 1 neuron -> hidden layer: dmid neurons
        self.linear1 = nnx.Linear(1, dmid, rngs = rngs)

        # dropout generally helps against overfitting
        # in our case to possibly make the model more robust
        # to experimental artifacts
        self.dropout = nnx.Dropout(0.1, deterministic = False, rngs = rngs)

        # middle layer: dmid neurons -> hidden layer: dmid neurons
        self.linear2 = nnx.Linear(dmid, dmid, rngs = rngs)

        # output layer: dmid neurons -> 1 neuron
        self.linear_out = nnx.Linear(dmid, 1, rngs = rngs)

    def __call__(self, x, *, train: bool = True):
        self.dropout.deterministic = not train
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        x = nnx.relu(self.linear2(x))
        raw_output = self.linear_out(x)
        return raw_output
    
def calculate_growth_rate(
        raw_nn_output: jnp.ndarray,
        scale_factor: float = 100.0
    ) -> jnp.ndarray:

    # ensure growth rate positivity
    rate = nnx.softplus(raw_nn_output) # shape (N, 1)

    # we scale here so the network output must not
    # be too large
    scaled_rate = rate * scale_factor

    return scaled_rate.squeeze(axis=-1) # Return shape (N,)

def calculate_monotonic_thickness(
        raw_nn_output: jnp.ndarray,
        dt: jnp.ndarray,
        scale_factor: float = 100.0
    ) -> jnp.ndarray:

    """
    Given the raw growth rate over all time steps,
    the thickness profile is calculated by approximating
    the integral by a cumulative sum.

    NOTE: As this numerical integration is not exact,
    we do not truly learn the growth rate, as our
    objective is the thickness profile.

    """

    # ensure growth rate positivity
    rate = nnx.softplus(raw_nn_output) # shape (N, 1)

    # we scale here so the network output must not
    # be too large
    scaled_rate = rate * scale_factor

    # approximate the thickness profile resulting
    # from the growth rates
    thickness = jnp.cumsum(scaled_rate * dt, axis = 0)

    # ensure non-negativity of the thickness
    thickness = thickness + 1e-7

    # reshape
    return thickness.squeeze(axis=-1) # Return shape (N,)

# -------------------------------------------------------------
# ================ ↑ neural network definition ↑ ==============
# -------------------------------------------------------------

# -------------------------------------------------------------
# ============= ↓ neural network initialization ↓ =============
# -------------------------------------------------------------

"""
We initialize the neural network to
predict a given constant initial growth rate.

This is achieved using a bias in the output layer
and setting the output layer weights to zero.
"""

# start with a standard initialization
model = RawGrowthNN(dmid = 512, rngs = nnx.Rngs(0))

# set the target rate
target_rate_init = 0.25 * 3600.0
scale_factor_init = 100.0
target_softplus_value_init = target_rate_init / scale_factor_init

# Calculate the required bias for the output layer
target_raw_output_init = jnp.log(jnp.exp(target_softplus_value_init) - 1.0)
target_raw_output_init = jnp.astype(
    target_raw_output_init,
    model.linear_out.bias.value.dtype
)

# zero output layer weights to zero
model.linear_out.kernel.value = jnp.zeros_like(model.linear_out.kernel.value)

# bias appropriately
model.linear_out.bias.value = jnp.full_like(
    model.linear_out.bias.value,
    target_raw_output_init
)

# -------------------------------------------------------------
# ============= ↑ neural network initialization ↑ =============
# -------------------------------------------------------------

# -------------------------------------------------------------
# ===================== ↓ training setup ↓ ====================
# -------------------------------------------------------------

# set learning parameters
learning_rate = 4e-4
num_epochs = 1000000
print_interval = 500
patience = 4000

# learning schedule
schedule = optax.exponential_decay(
    init_value = learning_rate,
    transition_steps = 5000,
    decay_rate = 0.95
)

# optimizer
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=schedule))

# loss function
def calculate_loss(
        model: RawGrowthNN,
        time: jnp.ndarray,
        target_reflectance: jnp.ndarray,
        dt: jnp.ndarray
    ) -> jnp.ndarray:

    # evaluate the model
    raw_nn_output = model(time, train = True)

    # obtain the predicted thicknesses
    predicted_thicknesses = calculate_monotonic_thickness(raw_nn_output, dt)

    # forward model through our differentiable simulator
    predicted_reflectance = forward_model(
        model = interference_model,
        setup_params = setup_params,
        optics_params = optics_params,
        static_layer_params = static_layer_params,
        variable_layer_params = variable_layer_params,
        variable_layer_thicknesses = predicted_thicknesses,
        backside_mode = backside_mode,
        normalization = MIN_MAX_NORMALIZATION
    )

    # L2 loss
    loss = jnp.mean((predicted_reflectance - target_reflectance)**2)

    return loss

# training step, updating the model
@nnx.jit
def train_step(
    model: RawGrowthNN,
    optimizer: nnx.Optimizer,
    time: jnp.ndarray,
    target_reflectance: jnp.ndarray,
    dt: jnp.ndarray
) -> Tuple[jnp.ndarray, RawGrowthNN]:
    loss_fn = lambda m: calculate_loss(m, time, target_reflectance, dt)
    loss_val, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss_val, model

# thickness prediction given the model
# and the time input
@nnx.jit
def predict_thickness(
    model: RawGrowthNN,
    time: jnp.ndarray,
    dt: jnp.ndarray
) -> jnp.ndarray:
    raw_nn_output = model(time, train=False)
    final_thickness = calculate_monotonic_thickness(raw_nn_output, dt)
    return final_thickness

initial_predicted_thicknesses = predict_thickness(model, time_nn_input, dt_hours)

# growth rate prediction given the model
# and the time input
@nnx.jit
def predict_growth_rate(
    model: RawGrowthNN,
    time: jnp.ndarray
) -> jnp.ndarray:
    raw_nn_output = model(time, train=False)
    growth_rate = calculate_growth_rate(raw_nn_output)
    return growth_rate


# -------------------------------------------------------------
# ===================== ↑ training setup ↑ ====================
# -------------------------------------------------------------


# -------------------------------------------------------------
# ===================== ↓ training loop ↓ =====================
# -------------------------------------------------------------

print("starting training...")
start_time = pytime.time()

losses = []
best_loss = float('inf')
epochs_no_improve = 0
static_dt_hours = dt_hours # Use this in the loop

for epoch in range(num_epochs):

    loss_val, model = train_step(
        model,
        optimizer,
        time_nn_input,
        reflectance_target,
        static_dt_hours
    )

    losses.append(loss_val)

    if jnp.isnan(loss_val):
        print(f"NaN loss encountered at epoch {epoch}! Stopping training.")
        break

    if epoch % print_interval == 0 or epoch == num_epochs - 1:
        print(f"Epoch {epoch}/{num_epochs}, Loss: {loss_val:.6f}")

    if loss_val < best_loss:
        best_loss = loss_val
        epochs_no_improve = 0
    else:
        epochs_no_improve += 1

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered at epoch {epoch} with loss {loss_val:.6f}")
        break

training_time = pytime.time() - start_time
print(f"training finished in {training_time:.2f} seconds.")

# -------------------------------------------------------------
# ===================== ↑ training loop ↑ =====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ======================= ↓ evaluation ↓ ======================
# -------------------------------------------------------------

## plot the training loss

fig, ax = plt.subplots(1, 1, figsize=(8, 5))
ax.plot(range(len(losses)), losses, label = "Training Loss (MSE)", color = "purple")
ax.set_xlabel("Epoch")
ax.set_ylabel("Loss")
ax.set_yscale('log')
ax.legend()
ax.set_title("Training Loss")
plt.savefig("figures/neural_network_thickness_model_training_loss.svg")

## plot the results

# get thickness predictions of our final model
final_predicted_thicknesses = predict_thickness(model, time_nn_input, static_dt_hours)

# check monotonicity
is_monotonic = jnp.all(jnp.diff(final_predicted_thicknesses) >= -1e-6)
print(f"Is final thickness profile monotonic? {is_monotonic}")
print(f"Final thickness starts at: {final_predicted_thicknesses[0]:.4f} nm")

# get the growth rate predictions of our final model
# NOTE: using AD we might also differentiate through our model
# to get the true growth rate
final_growth_rate = predict_growth_rate(model, time_nn_input)

# get the reflectance predictions of our final model
final_reflection_coefficients = forward_model(
    model = interference_model,
    setup_params = setup_params,
    optics_params = optics_params,
    static_layer_params = static_layer_params,
    variable_layer_params = variable_layer_params,
    variable_layer_thicknesses = final_predicted_thicknesses,
    backside_mode = backside_mode,
    normalization = MIN_MAX_NORMALIZATION # Match target normalization
)

fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Reflectance Comparison
axs[0].plot(time_hours, reflectance_target, '-', label = "measurement (normalized)", color = "black")
axs[0].plot(time_hours, final_reflection_coefficients, '-', label = "NN Model Prediction (Monotonic)", color = "red")
axs[0].set_ylabel("normalized reflectance")
axs[0].legend(loc = "lower right")
axs[0].set_title("Reflectance Fit with Initialized Monotonic NN Thickness")
axs[0].set_xlabel("time in hours")

# Plot 2: Learned Thickness Profile
axs[1].plot(time_hours, initial_predicted_thicknesses, '--', label = "Initial Thickness Guess", color = "grey")
axs[1].plot(time_hours, final_predicted_thicknesses, '-', label = "Learned Thickness (NN, Monotonic)", color = "red")

if use_simulated_data:
    axs[1].plot(time_hours, true_thicknesses, '-', label = "True Thickness (Simulated)", color = "black")

axs[1].set_ylabel("layer thickness in nm")
axs[1].set_xlabel("time in hours")
axs[1].legend(loc = "lower right")
axs[1].set_title("Predicted Monotonic Thickness Profile")

# Plot 3: Growth Rate
axs[2].plot(time_hours, final_growth_rate, '-', label = "Learned Growth Rate (NN)", color = "red")

if use_simulated_data:
    axs[2].plot(time_hours, true_growth_rate, '-', label = "True Growth Rate (Simulated)", color = "black")

axs[2].set_ylabel("growth rate in nm/h")
axs[2].set_xlabel("time in hours")
axs[2].legend(loc = "lower right")
axs[2].set_title("Predicted Growth Rate Profile")

plt.tight_layout()

plt.savefig("figures/neural_network_thickness_model_results.svg")

# -------------------------------------------------------------
# ======================= ↑ evaluation ↑ ======================
# -------------------------------------------------------------