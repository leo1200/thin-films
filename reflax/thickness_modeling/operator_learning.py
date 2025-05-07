from typing import Sequence, Tuple
import jax
import jax.numpy as jnp

import optax

# neural nets 
from flax import nnx

import flax

# Data Loading
from sklearn.model_selection import train_test_split

import pickle

# -------------------------------------------------------------
# ================ ↓ neural network definition ↓ ==============
# -------------------------------------------------------------

class NeuralOperatorMLP(nnx.Module):
    def __init__(self, hidden_dims: Sequence[int], num_eval_points: int, *, rngs: nnx.Rngs):
        self.hidden_layers = []
        in_dim = num_eval_points
        for i, h_dim in enumerate(hidden_dims):
            layer = nnx.Linear(in_dim, h_dim, rngs=rngs)
            setattr(self, f'linear_{i}', layer)
            self.hidden_layers.append(layer)
            in_dim = h_dim
        self.output_layer = nnx.Linear(in_dim, num_eval_points, rngs=rngs)

    def __call__(self, x: jnp.ndarray) -> jnp.ndarray:
        for layer in self.hidden_layers:
            x = layer(x)
            x = nnx.relu(x)
        x = self.output_layer(x)
        return x
    
# -------------------------------------------------------------
# ================ ↑ neural network definition ↑ ==============
# -------------------------------------------------------------


# -------------------------------------------------------------
# ===================== ↓ training setup ↓ ====================
# -------------------------------------------------------------

def calculate_loss(
    model: NeuralOperatorMLP,
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray
) -> jnp.ndarray:
    y_pred = model(batch_x)
    loss = jnp.mean((y_pred - batch_y)**2)
    return loss

@nnx.jit
def train_step(
    model: NeuralOperatorMLP,
    optimizer: nnx.Optimizer,
    batch_x: jnp.ndarray,
    batch_y: jnp.ndarray
) -> Tuple[jnp.ndarray, NeuralOperatorMLP, nnx.Optimizer]: # Still return optimizer
    grad_fn = nnx.value_and_grad(calculate_loss)
    loss_val, grads = grad_fn(model, batch_x, batch_y)
    optimizer.update(grads)
    return loss_val, model, optimizer

@nnx.jit
def evaluate_model(
    model: NeuralOperatorMLP,
    x_data: jnp.ndarray,
    y_data: jnp.ndarray
) -> jnp.ndarray:
    y_pred = model(x_data)
    loss = jnp.mean((y_pred - y_data)**2)
    return loss

def train_neural_operator(
    model: NeuralOperatorMLP,
    reflectance_data: jnp.ndarray, # x, shape (num_samples, num_eval_points)
    thickness_data: jnp.ndarray, # y, shape (num_samples, num_eval_points)
    learning_rate: float = 1e-4,
    test_set_size: float = 0.2, # Proportion of data for validation
    num_epochs: int = 10000,
    print_interval: int = 500, # Log every N epochs
    patience: int = 4000, # Early stopping patience: num of checks (1 check per print_interval)
    random_seed_split: int = 42 
) -> NeuralOperatorMLP:
    
    # train-test split
    x_train, x_test, y_train, y_test = train_test_split(
        reflectance_data, thickness_data, test_size=test_set_size, random_state=random_seed_split
    )

    # optimizer
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate = learning_rate))


    # early stopping
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    # Get initial model graph definition and state.
    # we store the best state
    model_graphdef, current_model_state = nnx.split(model)
    best_model_state = current_model_state 

    print(f"Starting training for {num_epochs} epochs...")

    print(f"Train data: {x_train.shape[0]} samples, Validation data: {x_test.shape[0]} samples.")
    print(f"Validation & logging every {print_interval} epochs. Early stopping patience: {patience} checks.")

    # training loop
    for epoch in range(num_epochs):

        # in-place training step
        loss_val, model, optimizer = train_step(
            model, optimizer, x_train, y_train
        )

        # Log training loss and perform validation periodically
        perform_logging_and_validation = ((epoch + 1) % print_interval == 0) or (epoch == num_epochs - 1)

        if perform_logging_and_validation:

            train_loss_scalar = float(loss_val.item()) # Convert JAX array to Python float

            val_loss_array = evaluate_model(model, x_test, y_test)
            val_loss_scalar = float(val_loss_array.item())
            
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss_scalar:.6f} | Val Loss: {val_loss_scalar:.6f}")

            # Early stopping check
            if val_loss_scalar < best_val_loss:
                best_val_loss = val_loss_scalar
                epochs_no_improve = 0
                # save best model state
                _, best_model_state = nnx.split(model)
            else:
                epochs_no_improve += 1

            if epochs_no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch+1}. "
                        f"No improvement in validation loss for {patience} validation checks.")
                break
    
    print("Training finished.")
    
    # get best model state
    final_model = nnx.merge(model_graphdef, best_model_state)

    if best_val_loss != float('inf'): # Check if any validation improvement was recorded
        print(f"Returning model with best validation loss: {best_val_loss:.6f}")
    else:
        # This case can occur if num_epochs < print_interval, so no validation check happened
        # or if all validation losses were NaN/inf.
        print("Validation was enabled, but no validation step recorded an improved finite loss. "
                "Returning model from last trained state (or initial if no training/validation occurred).")

    return final_model

# -------------------------------------------------------------
# ===================== ↑ training setup ↑ ====================
# -------------------------------------------------------------

# -------------------------------------------------------------
# ================ ↓ model saving & loading ↓ =================
# -------------------------------------------------------------

def save_model(model: NeuralOperatorMLP, filepath: str) -> None:
    # split the model into graph definition and state
    _, model_state = nnx.split(model)
    
    # just save the state using pickle
    with open(filepath, 'wb') as f:
        pickle.dump(model_state, f)

def load_model(filepath: str, abstract_model: NeuralOperatorMLP) -> NeuralOperatorMLP:

    # NOTE: the abstract model must have the same structure as the saved model
    
    model_structure, _ = nnx.split(abstract_model)

    with open(filepath, 'rb') as f:
        # load the model state
        model_state = pickle.load(f)
    
    # Merge the model structure with the loaded state
    loaded_model = nnx.merge(model_structure, model_state)

    return loaded_model

# -------------------------------------------------------------
# ================ ↑ model saving & loading ↑ =================
# -------------------------------------------------------------
