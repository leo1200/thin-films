# GPU selection (adjust if needed)
import os # Ensure os is imported
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # Or your preferred GPU ID

# Typing
from typing import Tuple, Sequence, Dict, Any

# Argument Parsing
import argparse

# Timing
import time as pytime

# Numerics
import jax
import jax.numpy as jnp
import numpy as np
from jax import random

# Neural Nets
from flax import nnx # Import nnx itself
import optax

# Plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper') # Paper-ready style
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": [8, 5],
    "savefig.dpi": 300
})

# Data Loading
from sklearn.model_selection import train_test_split

# --- Orbax Checkpointing ---
import orbax.checkpoint as ocp # Import Orbax

# -------------------------------------------------------------
# ======================= ↓ arguments ↓ =======================
# -------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(description="Train Neural Operator (Reflectance -> Thickness) using Orbax (Model State Only)")
    parser.add_argument('--data_file', type=str, default='training_data.npz',
                        help='Path to the generated training data (.npz file)')
    parser.add_argument('--epochs', type=int, default=10000, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=128, help='Training batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='Optimizer learning rate')
    parser.add_argument('--hidden_dims', nargs='+', type=int, default=[512, 512],
                        help='Hidden layer dimensions for the MLP')
    parser.add_argument('--test_split', type=float, default=0.15, help='Fraction of data for testing')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--checkpoint_dir', type=str, default='neural_operator_model_checkpoints',
                        help='Directory to save Orbax model state checkpoints')
    parser.add_argument('--save_interval_epochs', type=int, default=50,
                        help='How often to save a model state checkpoint (in epochs)')
    parser.add_argument('--max_checkpoints_to_keep', type=int, default=3,
                        help='Maximum number of model state checkpoints to keep (-1 for all)')
    parser.add_argument('--results_prefix', type=str, default='neural_operator',
                        help='Prefix for saving plots')
    return parser.parse_args()

# -------------------------------------------------------------
# ======================== ↓ NN Model ↓ =======================
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

    def __call__(self, x: jnp.ndarray, train: bool = True) -> jnp.ndarray:
        for layer in self.hidden_layers:
            x = layer(x)
            x = nnx.relu(x)
        x = self.output_layer(x)
        return x

# -------------------------------------------------------------
# ===================== ↓ Training Setup ↓ ====================
# -------------------------------------------------------------

def calculate_loss(model: NeuralOperatorMLP, batch_x: jnp.ndarray, batch_y: jnp.ndarray) -> jnp.ndarray:
    y_pred = model(batch_x, train=True)
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
def evaluate_model(model: NeuralOperatorMLP, x_data: jnp.ndarray, y_data: jnp.ndarray) -> jnp.ndarray:
    y_pred = model(x_data, train=False)
    loss = jnp.mean((y_pred - y_data)**2)
    return loss

# -------------------------------------------------------------
# ======================= ↓ Main Logic ↓ ======================
# -------------------------------------------------------------
def main(args):
    print("--- Neural Operator Training (Model State Checkpointing) ---")
    print(f"Args: {args}")

    key = random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    # --- Create Absolute Checkpoint Path ---
    checkpoint_dir_abs = os.path.abspath(args.checkpoint_dir) # Convert to absolute path
    if not os.path.exists(checkpoint_dir_abs):
        os.makedirs(checkpoint_dir_abs)
        print(f"Created checkpoint directory: {checkpoint_dir_abs}")
    else:
        print(f"Using checkpoint directory: {checkpoint_dir_abs}")


    print(f"\nLoading data from {args.data_file}...")
    try:
        data = np.load(args.data_file)
        reflectances = jnp.array(data['reflectances'])
        thicknesses = jnp.array(data['thicknesses'])
        timepoints = jnp.array(data['timepoints'])
        num_eval = thicknesses.shape[1]
        print(f"Data shapes: Reflectance {reflectances.shape}, Thickness {thicknesses.shape}")
        if jnp.isnan(reflectances).any():
             print("Warning: NaN values found in reflectances. Replacing with 0.")
             reflectances = jnp.nan_to_num(reflectances, nan=0.0)
        if jnp.isnan(thicknesses).any():
             print("Error: NaN values found in thicknesses. Cannot train.")
             return
    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_file}")
        return
    except KeyError as e:
        print(f"Error: Missing key in data file: {e}")
        return

    X_train, X_test, Y_train, Y_test = train_test_split(
        reflectances, thicknesses, test_size=args.test_split, random_state=args.seed
    )
    print(f"Train shapes: X={X_train.shape}, Y={Y_train.shape}")
    print(f"Test shapes: X={X_test.shape}, Y={Y_test.shape}")

    key, abs_key = random.split(key)
    print("\nCreating abstract model structure...")
    abstract_model = nnx.eval_shape(
        lambda: NeuralOperatorMLP(args.hidden_dims, num_eval, rngs=nnx.Rngs(abs_key))
    )
    graphdef, abstract_model_state = nnx.split(abstract_model)
    print("Abstract model state structure created.")

    key, init_key = random.split(key)
    model = NeuralOperatorMLP(
        hidden_dims=args.hidden_dims,
        num_eval_points=num_eval,
        rngs=nnx.Rngs(init_key)
    )
    optimizer = nnx.Optimizer(model, optax.adam(learning_rate=args.learning_rate))

    print("\nInitial Concrete Model Structure:")
    print(model)

    # --- Setup Orbax CheckpointManager with Absolute Path ---
    mngr_options = ocp.CheckpointManagerOptions(
        max_to_keep=args.max_checkpoints_to_keep,
        create=True # Ensure directory is created if needed
    )
    ckpt_manager = ocp.CheckpointManager(
        checkpoint_dir_abs, # Use the absolute path here
        options=mngr_options
    )

    # --- Restore Checkpoint if available (Model State Only) ---
    start_epoch = 0
    latest_step = ckpt_manager.latest_step()
    best_test_loss = float('inf')

    if latest_step is not None:
        print(f"Restoring model state checkpoint from step {latest_step}...")
        target_restore_structure = abstract_model_state
        try:
            restored_model_state = ckpt_manager.restore(
                latest_step,
                args=ocp.args.StandardRestore(target_restore_structure)
            )
            model = nnx.merge(graphdef, restored_model_state)
            optimizer = nnx.Optimizer(model, optax.adam(learning_rate=args.learning_rate))
            print("Optimizer re-initialized.")
            start_epoch = latest_step + 1
            print(f"Successfully restored model state. Resuming from epoch {start_epoch}.")
            best_test_loss = evaluate_model(model, X_test, Y_test)
            print(f"Initial best test loss after restore: {best_test_loss:.4e}")
        except Exception as e:
            print(f"Warning: Failed to restore model state checkpoint {latest_step}. Starting from scratch. Error: {e}")
            key, init_key = random.split(key)
            model = NeuralOperatorMLP(
                hidden_dims=args.hidden_dims,
                num_eval_points=num_eval,
                rngs=nnx.Rngs(init_key)
            )
            optimizer = nnx.Optimizer(model, optax.adam(learning_rate=args.learning_rate))
            start_epoch = 0
            best_test_loss = float('inf')
    else:
        print("No checkpoint found. Starting training from scratch.")

    # --- Training Loop ---
    print("\nStarting training...")
    start_time = pytime.time()
    train_losses = []
    test_losses_at_save = {}

    n_train = X_train.shape[0]
    steps_per_epoch = n_train // args.batch_size

    for epoch in range(start_epoch, args.epochs):
        key, perm_key = random.split(key)
        perm = random.permutation(perm_key, n_train)
        X_train_perm = X_train[perm]
        Y_train_perm = Y_train[perm]

        epoch_loss = 0.0
        for step in range(steps_per_epoch):
            start_idx = step * args.batch_size
            end_idx = start_idx + args.batch_size
            batch_x = X_train_perm[start_idx:end_idx]
            batch_y = Y_train_perm[start_idx:end_idx]

            loss_val, model, optimizer = train_step(model, optimizer, batch_x, batch_y)
            epoch_loss += loss_val

        avg_epoch_loss = epoch_loss / steps_per_epoch
        train_losses.append(avg_epoch_loss)

        if epoch % 10 == 0 or epoch == args.epochs - 1:
             current_test_loss = evaluate_model(model, X_test, Y_test)
             print(f"Epoch {epoch}/{args.epochs}, Train Loss: {avg_epoch_loss:.4e}, Test Loss: {current_test_loss:.4e}")
             if current_test_loss < best_test_loss:
                 best_test_loss = current_test_loss
                 print(f"  -> New best test loss: {best_test_loss:.4e}")

        if epoch % args.save_interval_epochs == 0 or epoch == args.epochs - 1:
            current_test_loss_for_save = evaluate_model(model, X_test, Y_test)
            test_losses_at_save[epoch] = float(current_test_loss_for_save)

            print(f"Saving model state checkpoint at epoch {epoch}...")
            _, model_state = nnx.split(model)
            try:
                ckpt_manager.save(
                    epoch,
                    args=ocp.args.StandardSave(model_state)
                )
                print(f"Checkpoint {epoch} saved successfully.")
            except Exception as e:
                # Catch potential errors during save
                print(f"Error saving checkpoint {epoch}: {e}")
                # Optionally break or continue depending on desired behavior
                # break

    ckpt_manager.wait_until_finished()

    training_time = pytime.time() - start_time
    print(f"\nTraining finished in {training_time:.2f} seconds.")
    print(f"Best Test Loss (MSE) recorded during run: {best_test_loss:.4e}")

    print("\nGenerating plots...")

    # --- Plot Loss Curves --- (Logic remains the same)
    fig_loss, ax_loss = plt.subplots()
    epochs_run = range(start_epoch, epoch + 1)
    actual_train_losses = train_losses[:len(epochs_run)]
    ax_loss.plot(epochs_run, actual_train_losses, label="Training Loss", alpha=0.6)
    saved_epochs = sorted(test_losses_at_save.keys())
    if saved_epochs:
        saved_test_losses = [test_losses_at_save[ep] for ep in saved_epochs]
        ax_loss.plot(saved_epochs, saved_test_losses, label="Test Loss (at saves)", marker='.', linestyle='--', alpha=0.9)
    ax_loss.set_xlabel("Epoch")
    ax_loss.set_ylabel("Mean Squared Error (MSE)")
    ax_loss.set_yscale('log')
    ax_loss.set_title("Neural Operator Training Loss")
    ax_loss.legend()
    ax_loss.grid(True, which='both', linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{args.results_prefix}_loss_curve.png")
    plt.savefig(f"{args.results_prefix}_loss_curve.svg")
    print(f"Saved loss curve plot to {args.results_prefix}_loss_curve.png/svg")
    plt.close(fig_loss)

    # --- Plot Example Prediction using the LATEST checkpoint state ---
    print("\nLoading LATEST checkpoint state for example plot...")
    latest_step_to_load = None
    model_to_use_for_pred = None
    try:
        latest_step_to_load = ckpt_manager.latest_step()
        if latest_step_to_load is None:
             raise FileNotFoundError("No checkpoints found to load for plotting.")

        target_plot_restore = abstract_model_state
        restored_model_state = ckpt_manager.restore(
            latest_step_to_load,
            args=ocp.args.StandardRestore(target_plot_restore)
        )
        model_to_use_for_pred = nnx.merge(graphdef, restored_model_state)
        print(f"Loaded model state from checkpoint {latest_step_to_load} and merged for plotting.")

    except Exception as e:
        print(f"Warning: Could not load and merge latest checkpoint state. Error: {e}")
        print("Attempting to use final model state from training loop (may not be the best/latest saved).")
        model_to_use_for_pred = model
        if 'model' not in locals() or model is None:
             print("Error: Final model state is not available.")
             return

    if model_to_use_for_pred is None:
        print("Error: No valid model available for plotting.")
        return

    # (Rest of plotting logic remains the same)
    example_idx = 0
    example_x = X_test[example_idx:example_idx+1]
    example_y_true = Y_test[example_idx]
    example_y_pred = model_to_use_for_pred(example_x, train=False).squeeze()
    fig_pred, axs_pred = plt.subplots(2, 1, sharex=True, figsize=(8, 8))
    axs_pred[0].plot(timepoints, example_x.squeeze(), label="Input Reflectance (Normalized)", color='orange')
    axs_pred[0].set_ylabel("Reflectance")
    axs_pred[0].set_title(f"Neural Operator Example Prediction (Test Sample {example_idx} from Ckpt {latest_step_to_load if latest_step_to_load is not None else 'N/A'})")
    axs_pred[0].legend()
    axs_pred[0].grid(True, linestyle=':', linewidth=0.5)
    axs_pred[1].plot(timepoints, example_y_true, label="Ground Truth Thickness", color='black')
    axs_pred[1].plot(timepoints, example_y_pred, '--', label="Neural Operator Prediction", color='blue')
    axs_pred[1].set_xlabel("Time (Normalized or hours)")
    axs_pred[1].set_ylabel("Thickness (nm)")
    axs_pred[1].legend()
    axs_pred[1].grid(True, linestyle=':', linewidth=0.5)
    plt.tight_layout()
    plt.savefig(f"{args.results_prefix}_example_prediction.png")
    plt.savefig(f"{args.results_prefix}_example_prediction.svg")
    print(f"Saved example prediction plot to {args.results_prefix}_example_prediction.png/svg")
    plt.close(fig_pred)

    print("\n--- Neural Operator Training Complete ---")

if __name__ == "__main__":
    args = parse_args()
    main(args)