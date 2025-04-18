import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3" # Set GPU if available, adjust if needed
os.makedirs("figures", exist_ok=True) # Create figures directory if it doesn't exist

import numpy as np
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
from flax import nnx
import optax
import time as pytime # Avoid conflict with jax time variable

from reflax import polanalyze
from reflax.parameter_classes.parameters import OpticsParams, SetupParams, LayerParams
from reflax.forward_model.variable_layer_size import MIN_MAX_NORMALIZATION, ONE_LAYER_INTERNAL_REFLECTIONS, forward_model

# --- Data Loading and Preprocessing ---
measurement = np.loadtxt("reflectance.txt", skiprows=1)
time_raw = jnp.array(measurement[:-100, 0])
time_hours = time_raw / 3600  # convert to hours
reflectance_raw = jnp.array(measurement[:-100, 1])

# Normalize reflectance (MinMax scaling to [0, 1])
reflectance_norm = (reflectance_raw - 0.5 * (jnp.min(reflectance_raw) + jnp.max(reflectance_raw))) / (0.5 * (jnp.max(reflectance_raw) - jnp.min(reflectance_raw)))


# Calculate dt in original time units (hours)
dt_hours = jnp.diff(time_hours)
if len(dt_hours) > 0:
    dt_hours = jnp.concatenate((jnp.array([dt_hours[0]]), dt_hours))
else:
    dt_hours = jnp.array([1.0])
dt_hours = dt_hours.reshape(-1, 1)

# Time input for NN (Using original hours, reshaped)
time_nn_input = time_hours[:, None] # Shape becomes (N, 1)

# Target data for training
reflectance_target = reflectance_norm # Use normalized reflectance as target

print(f"Time input shape for NN: {time_nn_input.shape}")
print(f"Reflectance target shape: {reflectance_target.shape}")
print(f"dt_hours shape: {dt_hours.shape}")


# --- Setup reflax Forward Model Parameters ---
# (Keep this section exactly as before)
wavelength = 632.8
polar_angle = jnp.deg2rad(25)
azimuthal_angle = jnp.deg2rad(0)
setup_params = SetupParams(
    wavelength=wavelength,
    polar_angle=polar_angle,
    azimuthal_angle=azimuthal_angle
)
polarization_state = "Linear TE/perpendicular/s"
transverse_electric_component, transverse_magnetic_component = polanalyze(polarization_state)
permeability_reflection = 1.0
permittivity_reflection = complex(1.0, 0.0)
permeability_transmission = 1.0
permittivity_transmission = (3.8827 + 0.019626j)**2
optics_params = OpticsParams(
    permeability_reflection=permeability_reflection,
    permittivity_reflection=permittivity_reflection,
    permeability_transmission=permeability_transmission,
    permittivity_transmission=permittivity_transmission,
    transverse_electric_component=transverse_electric_component,
    transverse_magnetic_component=transverse_magnetic_component
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

# --- Neural Network Definition ---
# NN now outputs RAW rate value, softplus is applied later
class ThicknessNN(nnx.Module):
  def __init__(self, dmid: int, dout: int, *, rngs: nnx.Rngs):
    self.linear1 = nnx.Linear(1, dmid, rngs=rngs)
    self.dropout = nnx.Dropout(0.1, deterministic=False, rngs=rngs)
    self.linear2 = nnx.Linear(dmid, dmid, rngs=rngs)
    self.linear_out = nnx.Linear(dmid, dout, rngs=rngs) # dout = 1 for raw rate value

  def __call__(self, x, *, train: bool = True):
    self.dropout.deterministic = not train
    x = self.linear1(x)
    x = nnx.relu(x)
    x = self.dropout(x)
    x = nnx.relu(self.linear2(x))
    raw_output = self.linear_out(x) # Output raw value
    return raw_output

# --- Helper function to calculate monotonic thickness from raw NN output ---
def calculate_monotonic_thickness(raw_nn_output: jnp.ndarray, dt: jnp.ndarray, scale_factor: float = 100.0) -> jnp.ndarray:
    """Calculates monotonic thickness via integrating a non-negative rate."""
    rate = nnx.softplus(raw_nn_output) # Shape (N, 1)
    scaled_rate = rate * scale_factor
    thickness = jnp.cumsum(scaled_rate * dt, axis=0)
    thickness = thickness + 1e-7
    return thickness.squeeze(axis=-1) # Return shape (N,)

# --- Model Initialization ---
# Initialize NN first with standard initializers
model = ThicknessNN(dmid=512, dout=1, rngs=nnx.Rngs(0))

# --- Custom Initialization for Constant Initial Rate ---
target_rate_init = 0.25 * 3600.0
scale_factor_init = 100.0
target_softplus_value_init = target_rate_init / scale_factor_init

# Calculate the required bias for the output layer
target_raw_output_init = jnp.log(jnp.exp(jnp.astype(target_softplus_value_init, jnp.float64)) - 1.0)
target_raw_output_init = jnp.astype(target_raw_output_init, model.linear_out.bias.value.dtype) # Match original bias dtype

print(f"Target initial rate: {target_rate_init:.2f} nm/hr")
print(f"Target initial softplus value (rate/scale): {target_softplus_value_init:.4f}")
print(f"Target initial raw NN output (bias_out): {target_raw_output_init:.4f}")

# --- CHANGE HERE: Update .value attribute directly ---
# Modify the final layer's weights and biases
# Set weights to zero by updating the .value attribute
model.linear_out.kernel.value = jnp.zeros_like(model.linear_out.kernel.value)
# Set bias to the calculated target raw output value by updating the .value attribute
model.linear_out.bias.value = jnp.full_like(model.linear_out.bias.value, target_raw_output_init)
# --- End CHANGE ---

print("Model output layer weights zeroed and bias set for initial constant rate.")

# --- Training Setup ---
learning_rate = 2e-4
num_epochs = 100000
print_interval = 500
patience = 2000

# Initialize Optimizer *after* modifying the model parameters
schedule = optax.exponential_decay(
    init_value=learning_rate,
    transition_steps=5000,
    decay_rate=0.95
)
optimizer = nnx.Optimizer(model, optax.adam(learning_rate=schedule))

# --- Define the loss function incorporating monotonic thickness calculation ---
def calculate_loss(model: ThicknessNN, x: jnp.ndarray, y_target: jnp.ndarray, dt: jnp.ndarray):
    raw_nn_output = model(x, train=True)
    predicted_thicknesses = calculate_monotonic_thickness(raw_nn_output, dt) # Uses scale_factor=100 internally
    predicted_reflectance = forward_model(
        model=ONE_LAYER_INTERNAL_REFLECTIONS,
        setup_params=setup_params,
        optics_params=optics_params,
        static_layer_params=static_layer_params,
        variable_layer_params=variable_layer_params,
        variable_layer_thicknesses=predicted_thicknesses,
        backside_mode=backside_mode,
        normalization=MIN_MAX_NORMALIZATION # Match target normalization
    )
    loss = jnp.mean((predicted_reflectance - y_target)**2)
    return loss

# Define the training step
@nnx.jit
def train_step(model: ThicknessNN, optimizer: nnx.Optimizer, x: jnp.ndarray, y_target: jnp.ndarray, dt: jnp.ndarray):
    loss_fn = lambda m: calculate_loss(m, x, y_target, dt)
    loss_val, grads = nnx.value_and_grad(loss_fn)(model)
    optimizer.update(grads)
    return loss_val, model

# --- Prediction Function ---
@nnx.jit
def predict_final_thickness(model: ThicknessNN, x: jnp.ndarray, dt: jnp.ndarray):
    raw_nn_output = model(x, train=False)
    final_thickness = calculate_monotonic_thickness(raw_nn_output, dt) # Uses scale_factor=100 internally
    return final_thickness

# --- Check Initial Thickness Profile ---
print("Checking initial thickness profile based on initialization...")
# Ensure this check happens BEFORE training loop starts
initial_predicted_thicknesses = predict_final_thickness(model, time_nn_input, dt_hours)
# Check if time_hours has more than one element before calculating diff
if time_hours.shape[0] > 1:
    initial_rate = jnp.diff(initial_predicted_thicknesses) / jnp.diff(time_hours) # Approx rate nm/hr
    print(f"Approx initial rate (nm/hr) from first few steps: {initial_rate[:5]}")
else:
    print("Cannot calculate initial rate (only one time point).")
print(f"Initial thickness starts at: {initial_predicted_thicknesses[0]:.4f} nm")

# --- Training Loop ---
print("Starting training with monotonic constraint and custom initialization...")
start_time = pytime.time()

losses = []
best_loss = float('inf')
epochs_no_improve = 0
static_dt_hours = dt_hours # Use this in the loop

for epoch in range(num_epochs):
    loss_val, model = train_step(model, optimizer, time_nn_input, reflectance_target, static_dt_hours)
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
print(f"Training finished in {training_time:.2f} seconds.")

# --- Evaluation and Plotting ---
# Get final predictions using the trained model
final_predicted_thicknesses = predict_final_thickness(model, time_nn_input, static_dt_hours)

# Calculate final reflectance based on NN predictions
final_reflection_coefficients = forward_model(
    model=ONE_LAYER_INTERNAL_REFLECTIONS,
    setup_params=setup_params,
    optics_params=optics_params,
    static_layer_params=static_layer_params,
    variable_layer_params=variable_layer_params,
    variable_layer_thicknesses=final_predicted_thicknesses,
    backside_mode=backside_mode,
    normalization=MIN_MAX_NORMALIZATION # Match target normalization
)

# --- Plotting ---
fig, axs = plt.subplots(3, 1, figsize=(10, 12))

# Plot 1: Reflectance Comparison
axs[0].plot(time_hours, reflectance_target, '-', label="Measurement (Normalized)", color="black", alpha=0.7, lw=1)
axs[0].plot(time_hours, final_reflection_coefficients, '-', label="NN Model Prediction (Monotonic)", color="red", lw=1.5)
axs[0].set_ylabel("Normalized Reflectance")
axs[0].legend()
axs[0].set_title("Reflectance Fit with Initialized Monotonic NN Thickness")
axs[0].set_xlabel("Time (hours)")

# Plot 2: Learned Thickness Profile
axs[1].plot(time_hours, initial_predicted_thicknesses, '--', label="Initial Thickness Guess", color="grey", alpha=0.7) # Plot initial guess
axs[1].plot(time_hours, final_predicted_thicknesses, '-', label="Learned Thickness (NN, Monotonic)", color="red")
axs[1].set_ylabel("Layer Thickness (nm)")
axs[1].set_xlabel("Time (hours)")
axs[1].legend()
axs[1].set_title("Predicted Monotonic Thickness Profile")
is_monotonic = jnp.all(jnp.diff(final_predicted_thicknesses) >= -1e-6)
print(f"Is final thickness profile monotonic? {is_monotonic}")
print(f"Final thickness starts at: {final_predicted_thicknesses[0]:.4f} nm")


# Plot 3: Loss Curve
axs[2].plot(range(len(losses)), losses, label="Training Loss (MSE)", color="purple")
axs[2].set_xlabel("Epoch")
axs[2].set_ylabel("Loss")
axs[2].set_yscale('log')
axs[2].legend()
axs[2].set_title("Training Loss")

plt.tight_layout()
plt.savefig("figures/nn_monotonic_cumsum_initialized_fit.png")
print("Saved plots to figures/nn_monotonic_cumsum_initialized_fit.png")
plt.show()