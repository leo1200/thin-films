# GPU selection (adjust if needed)
import os

from sklearn.model_selection import train_test_split
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # Or your preferred GPU ID

# Typing
from typing import Tuple, Sequence

# Argument Parsing
import argparse

# Timing
import time as pytime

# Numerics
import jax
import jax.numpy as jnp
import numpy as np
from jax import random, jit

# Neural Nets
from flax import nnx
import optax

# Plotting
import matplotlib.pyplot as plt
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": [10, 8], # Adjust figure size for 2 subplots
    "savefig.dpi": 300
})

# Reflax (Assuming it's importable)
try:
    from reflax import polanalyze
    from reflax.parameter_classes.parameters import OpticsParams, SetupParams, LayerParams
    from reflax.forward_model.variable_layer_size import (
        MIN_MAX_NORMALIZATION, ONE_LAYER_INTERNAL_REFLECTIONS, forward_model
    )
    REFLAX_AVAILABLE = True
except ImportError:
    print("Warning: Reflax library not found. Simulator refinement will not work.")
    REFLAX_AVAILABLE = False

# Neural Operator Model Definition (must match the one used for training)
class NeuralOperatorMLP(nnx.Module):
    """ Definition needs to be available to load the saved model """
    def __init__(self, hidden_dims: Sequence[int], num_eval_points: int, *, rngs: nnx.Rngs):
        self.layers = []
        in_dim = num_eval_points
        for i, h_dim in enumerate(hidden_dims):
            self.layers.append(nnx.Linear(in_dim, h_dim, rngs=rngs, name=f'linear_{i}'))
        self.output_layer = nnx.Linear(in_dim, num_eval_points, rngs=rngs, name='linear_out')

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray: # Default to eval mode
        for layer in self.layers:
            x = layer(x)
            x = nnx.relu(x)
        x = self.output_layer(x)
        return x

# Refinement Network Definition (similar to your RawGrowthNN)
class RefinementNN(nnx.Module):
    """ NN to parameterize the growth rate for refinement. """
    def __init__(self, dmid: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(1, dmid, rngs=rngs)
        # Using fixed dropout rate during refinement might be okay, or make it deterministic
        self.dropout = nnx.Dropout(0.1, deterministic=True, rngs=rngs) # Usually deterministic for refinement
        self.linear2 = nnx.Linear(dmid, dmid, rngs=rngs)
        self.linear_out = nnx.Linear(dmid, 1, rngs=rngs)

    def __call__(self, x, *, train: bool = False): # Default to eval/deterministic
        # self.dropout.deterministic = not train # Can enable if needed
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.dropout(x) # Apply dropout even if deterministic
        x = nnx.relu(self.linear2(x))
        raw_output = self.linear_out(x)
        return raw_output

# Helper functions from your script (adapted for direct use)
def calculate_growth_rate(
        raw_nn_output: jnp.ndarray,
        scale_factor: float = 100.0 # Keep consistent or make configurable
    ) -> jnp.ndarray:
    rate = nnx.softplus(raw_nn_output)
    scaled_rate = rate * scale_factor
    return scaled_rate.squeeze(axis=-1)

def calculate_monotonic_thickness(
        raw_nn_output: jnp.ndarray,
        dt: jnp.ndarray,
        scale_factor: float = 100.0 # Keep consistent or make configurable
    ) -> jnp.ndarray:
    rate = nnx.softplus(raw_nn_output) # shape (N, 1)
    scaled_rate = rate * scale_factor
    # Ensure dt has the correct shape for broadcasting (N, 1)
    dt = dt.reshape(-1, 1) if dt.ndim == 1 else dt
    thickness = jnp.cumsum(scaled_rate * dt, axis=0)
    thickness = thickness + 1e-7 # Ensure positivity
    return thickness.squeeze(axis=-1) # Return shape (N,)

# -------------------------------------------------------------
# ======================= ↓ arguments ↓ =======================
# -------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Refine Neural Operator prediction with Simulator")
    parser.add_argument('--neural_operator_model', type=str, default='neural_operator_model.nnx',
                        help='Path to the saved Neural Operator model')
    parser.add_argument('--data_file', type=str, default='training_data.npz',
                        help='Path to the data file (for test set or specific measurement)')
    parser.add_argument('--measurement_file', type=str, default=None, # Default to using test data
                        help='Optional: Path to a specific measurement file (e.g., reflectance.txt)')
    parser.add_argument('--test_data_index', type=int, default=0,
                        help='Index of the test sample from data_file to use if measurement_file is None')
    parser.add_argument('--refinement_epochs', type=int, default=5000, help='Number of refinement epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=1000, help='Number of pre-training epochs for RefinementNN')
    parser.add_argument('--refinement_lr', type=float, default=5e-5, help='Learning rate for refinement')
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='Learning rate for pre-training')
    parser.add_argument('--hidden_dims_op', nargs='+', type=int, default=[512, 512],
                        help='Hidden layer dimensions for the loaded Neural Operator')
    parser.add_argument('--hidden_dim_refine', type=int, default=256, help='Hidden layer dimension for RefinementNN')
    parser.add_argument('--seed', type=int, default=43, help='Random seed for refinement')
    parser.add_argument('--results_prefix', type=str, default='refinement', help='Prefix for saving plots')
    return parser.parse_args()

# -------------------------------------------------------------
# ==================== ↓ Simulator Setup ↓ ====================
# (Copied and adapted from your second script - ensure parameters match your data generation)
# -------------------------------------------------------------

def setup_simulator_params():
    if not REFLAX_AVAILABLE:
        return None, None, None, None, None, None

    print("Setting up Reflax simulator parameters...")
    interference_model = ONE_LAYER_INTERNAL_REFLECTIONS # Or TRANSFER_MATRIX_METHOD

    # --- Values should match data generation script ---
    wavelength = 632.8 # nm
    polar_angle = jnp.deg2rad(25)
    azimuthal_angle = jnp.deg2rad(0)
    setup_params = SetupParams(
        wavelength=wavelength, polar_angle=polar_angle, azimuthal_angle=azimuthal_angle
    )

    polarization_state = "Linear TE/perpendicular/s"
    transverse_electric_component, transverse_magnetic_component = polanalyze(polarization_state)
    permeability_reflection = 1.0
    permittivity_reflection = complex(1.0, 0.0)  # Air/Vacuum
    permeability_transmission = 1.0
    # Substrate: Silicon (match data generation)
    n_substrate = 3.8827
    k_substrate = 0.019626
    permittivity_transmission = (n_substrate + 1j * k_substrate)**2
    optics_params = OpticsParams(
        permeability_reflection=permeability_reflection, permittivity_reflection=permittivity_reflection,
        permeability_transmission=permeability_transmission, permittivity_transmission=permittivity_transmission,
        transverse_electric_component=transverse_electric_component, transverse_magnetic_component=transverse_magnetic_component
    )

    backside_mode = 1
    static_layer_thicknesses = jnp.array([0.0])
    permeability_static_size_layers = jnp.array([permeability_transmission])
    permittivity_static_size_layers = jnp.array([permittivity_transmission])
    static_layer_params = LayerParams(
        permeabilities=permeability_static_size_layers, permittivities=permittivity_static_size_layers,
        thicknesses=static_layer_thicknesses
    )

    # Variable Layer: SiO2 (match data generation)
    n_variable = 1.457
    k_variable = 0.0
    permeability_variable_layer = 1.0
    permittivity_variable_layer = (n_variable + 1j * k_variable)**2
    variable_layer_params = LayerParams(
        permeabilities=permeability_variable_layer, permittivities=permittivity_variable_layer,
        thicknesses=None # Provided during call
    )
    print("Simulator setup complete.")
    return (interference_model, setup_params, optics_params, static_layer_params,
            variable_layer_params, backside_mode)

# -------------------------------------------------------------
# ==================== ↓ Refinement Logic ↓ ===================
# -------------------------------------------------------------

# --- Pre-training Step (Match Neural Operator Output) ---
@jit
def pretrain_loss_fn(
    refine_model: RefinementNN,
    time_input: jnp.ndarray,
    target_thickness: jnp.ndarray,
    dt: jnp.ndarray
) -> jnp.ndarray:
    raw_output = refine_model(time_input, train=True)
    predicted_thickness = calculate_monotonic_thickness(raw_output, dt)
    loss = jnp.mean((predicted_thickness - target_thickness)**2)
    return loss

@jit
def pretrain_step(
    refine_model: RefinementNN,
    optimizer: nnx.Optimizer,
    time_input: jnp.ndarray,
    target_thickness: jnp.ndarray,
    dt: jnp.ndarray
) -> Tuple[jnp.ndarray, RefinementNN]:
    loss, grads = nnx.value_and_grad(pretrain_loss_fn)(refine_model, time_input, target_thickness, dt)
    optimizer.update(grads)
    return loss, refine_model

# --- Refinement Step (Match Measured Reflectance via Simulator) ---
@jit
def refinement_loss_fn(
    refine_model: RefinementNN,
    time_input: jnp.ndarray,
    target_reflectance: jnp.ndarray,
    dt: jnp.ndarray,
    sim_params: tuple # Contains simulator settings
) -> jnp.ndarray:
    if not REFLAX_AVAILABLE: return jnp.array(jnp.nan) # Cannot compute loss

    interference_model, setup_params, optics_params, static_layer_params, \
        variable_layer_params, backside_mode = sim_params

    raw_output = refine_model(time_input, train=True)
    predicted_thickness = calculate_monotonic_thickness(raw_output, dt)

    # Run forward model
    predicted_reflectance = forward_model(
        model=interference_model,
        setup_params=setup_params,
        optics_params=optics_params,
        static_layer_params=static_layer_params,
        variable_layer_params=variable_layer_params,
        variable_layer_thicknesses=predicted_thickness,
        backside_mode=backside_mode,
        normalization=MIN_MAX_NORMALIZATION # Crucial: Match target normalization
    )

    loss = jnp.mean((predicted_reflectance - target_reflectance)**2)
    return loss

@jit
def refinement_step(
    refine_model: RefinementNN,
    optimizer: nnx.Optimizer,
    time_input: jnp.ndarray,
    target_reflectance: jnp.ndarray,
    dt: jnp.ndarray,
    sim_params: tuple
) -> Tuple[jnp.ndarray, RefinementNN]:
    loss, grads = nnx.value_and_grad(refinement_loss_fn)(
        refine_model, time_input, target_reflectance, dt, sim_params
    )
    optimizer.update(grads)
    return loss, refine_model


# -------------------------------------------------------------
# ======================= ↓ Main Logic ↓ ======================
# -------------------------------------------------------------

def main(args):
    print("--- Neural Operator Refinement ---")
    print(f"Args: {args}")

    if not REFLAX_AVAILABLE and args.refinement_epochs > 0:
        print("Error: Reflax library not found, cannot perform simulator refinement.")
        return

    key = random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    # --- Load Target Data ---
    print("\nLoading target data...")
    try:
        if args.measurement_file:
            # Load from specific measurement file
            measurement = np.loadtxt(args.measurement_file, skiprows=1)
            time_raw = jnp.array(measurement[:, 0])
            reflectance_raw = jnp.array(measurement[:, 1])
            # Convert time to hours (assuming input is seconds)
            time_hours = time_raw / 3600.0
            # Normalize reflectance as done in your example script
            min_r, max_r = jnp.min(reflectance_raw), jnp.max(reflectance_raw)
            reflectance_target = (reflectance_raw - 0.5 * (min_r + max_r)) / (0.5 * (max_r - min_r))
            time_input_nn = time_hours[:, None]
            num_eval = len(time_hours)
            # Ground truth thickness is unknown for real measurements
            thickness_gt = jnp.full_like(time_hours, jnp.nan)
            print(f"Loaded data from measurement file: {args.measurement_file}")
            print(f"  Time points: {num_eval}, Reflectance range: [{min_r:.3f}, {max_r:.3f}] -> [-1, 1]")
        else:
            # Load from the NPZ data file (test set)
            data = np.load(args.data_file)
            reflectances = jnp.array(data['reflectances'])
            thicknesses = jnp.array(data['thicknesses'])
            timepoints = jnp.array(data['timepoints']) # or x_eval

            # Split to get the same test set as in training
            _, X_test, _, Y_test = train_test_split(
                reflectances, thicknesses, test_size=0.15, random_state=42 # Use same split params
            )

            idx = args.test_data_index
            if idx >= len(X_test):
                print(f"Error: test_data_index {idx} is out of bounds for test set size {len(X_test)}")
                return

            reflectance_target = X_test[idx]
            thickness_gt = Y_test[idx] # Ground truth available for test data
            time_hours = timepoints # Assuming timepoints are the desired scale
            time_input_nn = time_hours[:, None]
            num_eval = len(time_hours)
            print(f"Loaded test sample {idx} from data file: {args.data_file}")

        # Calculate dt (time differences) - essential for thickness integration
        dt_hours = jnp.diff(time_hours)
        dt_hours = jnp.concatenate((jnp.array([dt_hours[0]]), dt_hours)) # Pad start

    except Exception as e:
        print(f"Error loading data: {e}")
        return

    # --- Load Neural Operator ---
    print("\nLoading pre-trained Neural Operator...")
    try:
        key, op_key = random.split(key)
        neural_op_model = NeuralOperatorMLP(
            hidden_dims=args.hidden_dims_op,
            num_eval_points=num_eval,
            rngs=nnx.Rngs(op_key) # Provide RNGs for structure init
        )
        neural_op_model.load(args.neural_operator_model)
        print(f"Successfully loaded model from {args.neural_operator_model}")
    except Exception as e:
        print(f"Error loading Neural Operator model: {e}")
        return

    # --- Get Neural Operator Initial Guess ---
    thickness_no_guess = neural_op_model(reflectance_target[None, :], train=False).squeeze() # Add batch dim

    # --- Initialize Refinement Network ---
    print("\nInitializing Refinement Network...")
    key, refine_key = random.split(key)
    refine_model = RefinementNN(dmid=args.hidden_dim_refine, rngs=nnx.Rngs(refine_key))

    # --- Pre-train Refinement Network to Match Neural Operator ---
    print(f"\nPre-training Refinement Network for {args.pretrain_epochs} epochs...")
    pretrain_optimizer = nnx.Optimizer(refine_model, optax.adam(learning_rate=args.pretrain_lr))
    pretrain_start_time = pytime.time()

    for epoch in range(args.pretrain_epochs):
        loss, refine_model = pretrain_step(
            refine_model, pretrain_optimizer, time_input_nn, thickness_no_guess, dt_hours
        )
        if epoch % 200 == 0 or epoch == args.pretrain_epochs - 1:
            print(f"  Pre-train Epoch {epoch}/{args.pretrain_epochs}, Loss (vs NO): {loss:.4e}")
            if jnp.isnan(loss):
                print("NaN loss during pre-training. Stopping.")
                return

    print(f"Pre-training finished in {pytime.time() - pretrain_start_time:.2f}s")

    # --- Refine using Simulator ---
    if args.refinement_epochs > 0 and REFLAX_AVAILABLE:
        print(f"\nRefining using simulator for {args.refinement_epochs} epochs...")
        sim_params = setup_simulator_params()
        if sim_params is None:
             print("Simulator setup failed. Skipping refinement.")
             thickness_refined = calculate_monotonic_thickness(
                 refine_model(time_input_nn, train=False), dt_hours
             ) # Use pre-trained result
        else:
            refinement_optimizer = nnx.Optimizer(refine_model, optax.adam(learning_rate=args.refinement_lr))
            refine_start_time = pytime.time()
            best_refine_loss = float('inf')
            epochs_no_improve = 0
            patience = 500 # Early stopping for refinement

            for epoch in range(args.refinement_epochs):
                loss, refine_model = refinement_step(
                    refine_model, refinement_optimizer, time_input_nn, reflectance_target, dt_hours, sim_params
                )

                if epoch % 200 == 0 or epoch == args.refinement_epochs - 1:
                    print(f"  Refinement Epoch {epoch}/{args.refinement_epochs}, Loss (vs Reflectance): {loss:.4e}")
                    if jnp.isnan(loss):
                         print("NaN loss during refinement. Stopping.")
                         break # Keep last non-NaN state

                    # Early stopping check
                    if loss < best_refine_loss:
                         best_refine_loss = loss
                         epochs_no_improve = 0
                         # Could save best refinement model state here if needed
                    else:
                         epochs_no_improve += 1

                    if epochs_no_improve * 200 >= patience * 200: # Check based on reporting interval
                         print(f"  -> Early stopping triggered at epoch {epoch}")
                         break

            print(f"Refinement finished in {pytime.time() - refine_start_time:.2f}s")
            # Get final prediction from the refined model
            thickness_refined = calculate_monotonic_thickness(
                refine_model(time_input_nn, train=False), dt_hours
            )
    else:
        print("\nSkipping simulator refinement step.")
        # Use the result after pre-training as the "refined" result for plotting
        thickness_refined = calculate_monotonic_thickness(
            refine_model(time_input_nn, train=False), dt_hours
        )

    # --- Calculate Final Results for Plotting ---
    print("\nCalculating final results for plotting...")
    # Neural Operator prediction already calculated: thickness_no_guess
    # Refined prediction already calculated: thickness_refined

    # Calculate Growth Rates (using gradient)
    dx = time_hours[1] - time_hours[0] # Assumes uniform spacing
    growth_rate_gt = jnp.gradient(thickness_gt, dx) if not jnp.isnan(thickness_gt).any() else jnp.full_like(time_hours, jnp.nan)
    growth_rate_no = jnp.gradient(thickness_no_guess, dx)
    growth_rate_refined_nn = calculate_growth_rate(refine_model(time_input_nn, train=False)) # From NN directly
    growth_rate_refined_grad = jnp.gradient(thickness_refined, dx) # From final thickness gradient


    # Calculate Final Refined Reflectance
    final_reflectance_refined = jnp.full_like(reflectance_target, jnp.nan)
    if REFLAX_AVAILABLE and 'sim_params' in locals() and sim_params is not None:
        interference_model, setup_params, optics_params, static_layer_params, \
            variable_layer_params, backside_mode = sim_params
        final_reflectance_refined = forward_model(
            model=interference_model, setup_params=setup_params, optics_params=optics_params,
            static_layer_params=static_layer_params, variable_layer_params=variable_layer_params,
            variable_layer_thicknesses=thickness_refined, backside_mode=backside_mode,
            normalization=MIN_MAX_NORMALIZATION
        )

    # --- Plotting Comparison ---
    print("\nGenerating comparison plot...")
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 12)) # Add subplot for thickness

    # Plot 1: Reflectance
    axs[0].plot(time_hours, reflectance_target, '-', label="Target Reflectance", color='black', linewidth=1.5)
    if not jnp.isnan(final_reflectance_refined).any():
        axs[0].plot(time_hours, final_reflectance_refined, '--', label="Refined Model Reflectance", color='red', linewidth=1.5)
    axs[0].set_ylabel("Norm. Reflectance")
    axs[0].set_title("Comparison of Neural Operator and Refined Prediction")
    axs[0].legend(fontsize='small')
    axs[0].grid(True, linestyle=':', linewidth=0.5)

    # Plot 2: Thickness
    if not jnp.isnan(thickness_gt).any():
        axs[1].plot(time_hours, thickness_gt, '-', label="Ground Truth Thickness", color='black', linewidth=1.5)
    axs[1].plot(time_hours, thickness_no_guess, ':', label="Neural Operator Guess", color='blue', linewidth=1.5)
    axs[1].plot(time_hours, thickness_refined, '--', label="Refined Thickness", color='red', linewidth=1.5)
    axs[1].set_ylabel("Thickness (nm)")
    axs[1].legend(fontsize='small')
    axs[1].grid(True, linestyle=':', linewidth=0.5)

    # Plot 3: Growth Rate
    if not jnp.isnan(growth_rate_gt).any():
         axs[2].plot(time_hours, growth_rate_gt, '-', label="Ground Truth Growth Rate", color='black', linewidth=1.5)
    # Plotting NO rate can be noisy due to gradient, maybe smooth or omit
    # axs[2].plot(time_hours, growth_rate_no, ':', label="Neural Operator Rate (Grad)", color='blue', linewidth=1.5, alpha=0.7)
    axs[2].plot(time_hours, growth_rate_refined_nn, '--', label="Refined Rate (NN)", color='red', linewidth=1.5)
    # axs[2].plot(time_hours, growth_rate_refined_grad, '-.', label="Refined Rate (Grad)", color='magenta', linewidth=1.0, alpha=0.7) # Optional: compare NN rate vs gradient rate
    axs[2].set_xlabel("Time (hours)")
    axs[2].set_ylabel("Growth Rate (nm/h)")
    axs[2].legend(fontsize='small')
    axs[2].grid(True, linestyle=':', linewidth=0.5)
    # Optionally set ylim for growth rate if needed
    # rate_min = min(jnp.nanmin(growth_rate_gt) if not np.isnan(growth_rate_gt).all() else 0, jnp.min(growth_rate_refined_nn))
    # rate_max = max(jnp.nanmax(growth_rate_gt) if not np.isnan(growth_rate_gt).all() else 1000, jnp.max(growth_rate_refined_nn))
    # axs[2].set_ylim(max(0, rate_min - 50), rate_max + 50)


    plt.tight_layout(rect=[0, 0.03, 1, 0.98]) # Adjust layout slightly
    plt.savefig(f"{args.results_prefix}_comparison.png")
    plt.savefig(f"{args.results_prefix}_comparison.svg")
    print(f"Saved comparison plot to {args.results_prefix}_comparison.png/svg")
    plt.close(fig)

    # --- Calculate Test Errors ---
    print("\nCalculating Final Errors:")
    mse_no = jnp.mean((thickness_no_guess - thickness_gt)**2) if not jnp.isnan(thickness_gt).any() else jnp.nan
    mse_refined = jnp.mean((thickness_refined - thickness_gt)**2) if not jnp.isnan(thickness_gt).any() else jnp.nan
    mse_reflectance = jnp.mean((final_reflectance_refined - reflectance_target)**2) if not jnp.isnan(final_reflectance_refined).any() else jnp.nan

    print(f"  Thickness MSE (vs GT) - Neural Operator: {mse_no:.4e}")
    print(f"  Thickness MSE (vs GT) - Refined Model:   {mse_refined:.4e}")
    print(f"  Reflectance MSE (vs Target) - Refined Model: {mse_reflectance:.4e}")


    print("\n--- Refinement Complete ---")

if __name__ == "__main__":
    args = parse_args()
    main(args)