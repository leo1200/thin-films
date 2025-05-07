from typing import Tuple
import jax
import jax.numpy as jnp

# neural nets 
from flax import nnx

# timing
import time as pytime

import optax

from reflax import forward_model
from reflax.parameter_classes.parameters import ForwardModelParams

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

        # middle layer: dmid neurons -> hidden layer: dmid neurons
        self.linear2 = nnx.Linear(dmid, dmid, rngs = rngs)

        # output layer: dmid neurons -> 1 neuron
        self.linear_out = nnx.Linear(dmid, 1, rngs = rngs)

    def __call__(self, x):
        x = self.linear1(x)
        x = nnx.relu(x)
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

    return scaled_rate.squeeze(axis = -1) # Return shape (N,)

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
    # from the growth rates using trapezoidal integration
    
    # first order numerical integration
    # thickness = jnp.cumsum(scaled_rate * dt, axis = 0)

    # trapezoidal integration
    thickness = jnp.cumsum(
        (scaled_rate[:-1] + scaled_rate[1:]) / 2 * dt[1:], axis = 0
    )
    # start the integral at 0
    thickness = jnp.insert(thickness, 0, 0.0)

    # ensure non-negativity of the thickness
    thickness = thickness + 1e-7

    return thickness

# -------------------------------------------------------------
# ================ ↑ neural network definition ↑ ==============
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============== ↓ neural network initialization ↓ ============
# -------------------------------------------------------------

def linear_output_initialization(
    model: RawGrowthNN,
    target_growth_rate: float,
    scale_factor: float = 100.0
) -> RawGrowthNN:
    """
    Initialize the output layer of the neural network
    such that the output of the network is equal to
    the target growth rate after application of the
    softplus function and scaling.

    Args:
        model (RawGrowthNN): The neural network model.
        target_growth_rate (float): The target growth rate.
        scale_factor (float): The scaling factor for the output used 
                              for numerical favorability.
    
    Returns:
        RawGrowthNN: The initialized neural network model.
    """

    # calculate the scaled target growth rate
    target_softplus_value_init = target_growth_rate / scale_factor

    # calculate the required bias for the output layer
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

    return model

@nnx.jit
def pretrain_loss_fn(
    growth_model: RawGrowthNN,
    time_input: jnp.ndarray,
    target_thickness: jnp.ndarray,
    dt: jnp.ndarray
) -> jnp.ndarray:
    raw_output = growth_model(time_input)
    predicted_thickness = calculate_monotonic_thickness(raw_output, dt)
    loss = jnp.mean((predicted_thickness - target_thickness)**2)
    return loss

@nnx.jit
def pretrain_step(
    growth_model: RawGrowthNN,
    optimizer: nnx.Optimizer,
    time_input: jnp.ndarray,
    target_thickness: jnp.ndarray,
    dt: jnp.ndarray
) -> Tuple[jnp.ndarray, RawGrowthNN, nnx.Optimizer]:
    grad_fn = nnx.value_and_grad(pretrain_loss_fn)
    loss, grads = grad_fn(growth_model, time_input, target_thickness, dt)
    optimizer.update(grads)
    return loss, growth_model, optimizer

def pretrained_initialization(
    growth_model: RawGrowthNN,
    target_thickness: jnp.ndarray,
    time_points: jnp.ndarray,
    pretrain_epochs: int,
    pretrain_lr: float,
) -> RawGrowthNN:
    """
    Initialize the neural network to reproduce the target
    thickness profile at the given time points.
    """

    time_nn_input = time_points[:, None]

    dt = jnp.diff(time_points)
    dt = jnp.concatenate((jnp.array([dt[0]]), dt))
    dt = dt.reshape(-1, 1)

    # adam optimizer
    pretrain_optimizer = nnx.Optimizer(growth_model, optax.adam(learning_rate = pretrain_lr))

    # start pre-training
    print("👷 pre-training neural network...")
    pretrain_start_time = pytime.time()

    for epoch in range(pretrain_epochs):
        loss, growth_model, pretrain_optimizer = pretrain_step(
            growth_model, pretrain_optimizer, time_nn_input, target_thickness, dt
        )
        if epoch % 200 == 0 or epoch == pretrain_epochs - 1:
            print(f"Pre-train Epoch {epoch}/{pretrain_epochs}, Loss (vs NO guess): {loss:.4e}")
            if jnp.isnan(loss):
                print("NaN loss during pre-training. Stopping.")
                return
    
    print(f"Pre-training finished in {pytime.time() - pretrain_start_time:.2f}s")

    return growth_model


# -------------------------------------------------------------
# ============== ↑ neural network initialization ↑ ============
# -------------------------------------------------------------


# -------------------------------------------------------------
# ============= ↓ optimization through simulator ↓ ============
# -------------------------------------------------------------

def train_nn_model(
    model: RawGrowthNN,
    forward_model_params: ForwardModelParams,
    time_points: jnp.ndarray,
    target_reflectance: jnp.ndarray,
    learning_rate: float = 1e-4,
    num_epochs: int = 1000000,
    print_interval: int = 500,
    patience: int = 4000
) -> Tuple[RawGrowthNN, jnp.ndarray]:
    """
    Train the neural network model using the forward model.
    """

    # define the loss function
    def calculate_loss(
        model: RawGrowthNN,
        time_points: jnp.ndarray,
        dt: jnp.ndarray,
        target_reflectance: jnp.ndarray
    ) -> jnp.ndarray:
    
        # evaluate the model
        raw_nn_output = model(time_points)

        # obtain the predicted thicknesses
        predicted_thicknesses = calculate_monotonic_thickness(raw_nn_output, dt)

        # forward model through our differentiable simulator
        predicted_reflectance = forward_model(
            model = forward_model_params.model,
            setup_params = forward_model_params.setup_params,
            light_source_params = forward_model_params.light_source_params,
            incident_medium_params = forward_model_params.incident_medium_params,
            transmission_medium_params = forward_model_params.transmission_medium_params,
            static_layer_params = forward_model_params.static_layer_params,
            variable_layer_params = forward_model_params.variable_layer_params,
            variable_layer_thicknesses = predicted_thicknesses,
            backside_mode = forward_model_params.backside_mode,
            polarization_state = forward_model_params.polarization_state,
            normalization = forward_model_params.normalization,
        )

        # L2 loss
        loss = jnp.mean((predicted_reflectance - target_reflectance)**2)

        return loss

    # training step, updating the model
    @nnx.jit
    def train_step(
        model: RawGrowthNN,
        optimizer: nnx.Optimizer,
        time_points: jnp.ndarray,
        dt: jnp.ndarray,
        target_reflectance: jnp.ndarray
    ) -> Tuple[jnp.ndarray, RawGrowthNN]:
        
        # define the loss function
        loss_fn = lambda m: calculate_loss(m, time_points, dt, target_reflectance)
        loss_val, grads = nnx.value_and_grad(loss_fn)(model)

        # update the model parameters
        optimizer.update(grads)

        return loss_val, model
    
    # optimizer setup

    # learning schedule
    schedule = optax.exponential_decay(
        init_value = learning_rate,
        transition_steps = 5000,
        decay_rate = 0.95
    )

    # optimizer
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=schedule))    
    
    print("starting training...")
    start_time = pytime.time()

    losses = []
    best_loss = float('inf')
    epochs_no_improve = 0
    time_nn = time_points[:, None]

    dt = jnp.diff(time_points)
    dt = jnp.concatenate((jnp.array([dt[0]]), dt))
    dt = dt.reshape(-1, 1)

    # Get initial model graph definition and state.
    # Graph definition is assumed to be static throughout training.
    # State will be updated. We store the 'best' state found.
    model_graphdef, current_model_state = nnx.split(model)
    best_model_state = current_model_state # Initialize with the starting model state

    for epoch in range(num_epochs):

        loss_val, model = train_step(
            model,
            optimizer,
            time_nn,
            dt,
            target_reflectance
        )

        losses.append(loss_val)

        if jnp.isnan(loss_val):
            print(f"NaN loss encountered at epoch {epoch}! Stopping training.")
            break

        if epoch % print_interval == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch}/{num_epochs}, Loss: {loss_val:.4e}")

        if loss_val < best_loss:
            best_loss = loss_val
            epochs_no_improve = 0
            _, current_model_state = nnx.split(model)
            best_model_state = current_model_state
        else:
            epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(f"Early stopping triggered at epoch {epoch} with loss {loss_val:.4e}")
            break

    training_time = pytime.time() - start_time
    print(f"training finished in {training_time:.2f} seconds.")
    print(f"Best loss: {best_loss:.4e}")

    final_model = nnx.merge(model_graphdef, best_model_state)

    return final_model, jnp.array(losses)


# thickness prediction given the model
# and the time input
@nnx.jit
def predict_thickness(
    model: RawGrowthNN,
    time_points: jnp.ndarray
) -> jnp.ndarray:
    
    time_nn = time_points[:, None]

    dt = jnp.diff(time_points)
    dt = jnp.concatenate((jnp.array([dt[0]]), dt))
    dt = dt.reshape(-1, 1)

    raw_nn_output = model(time_nn)
    final_thickness = calculate_monotonic_thickness(raw_nn_output, dt)
    return final_thickness

# growth rate prediction given the model
# and the time input
@nnx.jit
def predict_growth_rate(
    model: RawGrowthNN,
    time_points: jnp.ndarray
) -> jnp.ndarray:
    time_nn = time_points[:, None]
    raw_nn_output = model(time_nn)
    growth_rate = calculate_growth_rate(raw_nn_output)
    return growth_rate

# -------------------------------------------------------------
# ============= ↑ optimization through simulator ↑ ============
# -------------------------------------------------------------