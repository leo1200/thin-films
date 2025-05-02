# GPU selection (adjust if needed)
import os # Ensure os is imported
os.environ["CUDA_VISIBLE_DEVICES"] = "4" # Or your preferred GPU ID

# Typing
from typing import Tuple, Sequence, Optional

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
plt.style.use('seaborn-v0_8-paper')
plt.rcParams.update({
    "font.size": 12,
    "figure.figsize": [10, 8], # Adjust figure size for subplots
    "savefig.dpi": 300
})

# Data loading utility
from sklearn.model_selection import train_test_split


# Reflax (Assuming it's importable)
try:
    from reflax import polanalyze
    from reflax.parameter_classes.parameters import OpticsParams, SetupParams, LayerParams
    from reflax.forward_model.forward_model import (
        MIN_MAX_NORMALIZATION, ONE_LAYER_INTERNAL_REFLECTIONS, forward_model
    )
    REFLAX_AVAILABLE = True
    # --- Define simulator params globally after import ---
    # These will be assigned values later in main() if REFLAX_AVAILABLE
    # Using placeholders initially
    SIM_INTERFERENCE_MODEL = None
    SIM_SETUP_PARAMS = None
    SIM_OPTICS_PARAMS = None
    SIM_STATIC_LAYER_PARAMS = None
    SIM_VARIABLE_LAYER_PARAMS = None
    SIM_BACKSIDE_MODE = None
    SIM_NORMALIZATION = None

except ImportError:
    print("Warning: Reflax library not found. Simulator refinement will not work.")
    REFLAX_AVAILABLE = False
    SIM_INTERFERENCE_MODEL = None # Ensure these are None if import fails
    SIM_SETUP_PARAMS = None
    SIM_OPTICS_PARAMS = None
    SIM_STATIC_LAYER_PARAMS = None
    SIM_VARIABLE_LAYER_PARAMS = None
    SIM_BACKSIDE_MODE = None
    SIM_NORMALIZATION = None


# --- Orbax Checkpointing ---
import orbax.checkpoint as ocp # Import Orbax

# -------------------------------------------------------------
# ===================== ↓ NN Definitions ↓ ====================
# -------------------------------------------------------------

# Neural Operator Model Definition (MUST match the one used for training)
class NeuralOperatorMLP(nnx.Module):
    """ Definition needs to match the training script for loading checkpoints """
    def __init__(self, hidden_dims: Sequence[int], num_eval_points: int, *, rngs: nnx.Rngs):
        self.hidden_layers = []
        in_dim = num_eval_points
        for i, h_dim in enumerate(hidden_dims):
            layer = nnx.Linear(in_dim, h_dim, rngs=rngs)
            setattr(self, f'linear_{i}', layer)
            self.hidden_layers.append(layer)
            in_dim = h_dim
        self.output_layer = nnx.Linear(in_dim, num_eval_points, rngs=rngs)

    def __call__(self, x: jnp.ndarray, train: bool = False) -> jnp.ndarray: # Default to eval mode
        for layer in self.hidden_layers:
            x = layer(x)
            x = nnx.relu(x)
        x = self.output_layer(x)
        return x


# Refinement Network Definition (similar to your RawGrowthNN)
class RefinementNN(nnx.Module):
    """ NN to parameterize the growth rate for refinement. """
    def __init__(self, dmid: int, *, rngs: nnx.Rngs):
        self.linear1 = nnx.Linear(1, dmid, rngs=rngs)
        self.dropout = nnx.Dropout(0.1, deterministic=True, rngs=rngs)
        self.linear2 = nnx.Linear(dmid, dmid, rngs=rngs)
        self.linear_out = nnx.Linear(dmid, 1, rngs=rngs)

    def __call__(self, x, *, train: bool = False):
        x = self.linear1(x)
        x = nnx.relu(x)
        x = self.dropout(x)
        x = nnx.relu(self.linear2(x))
        raw_output = self.linear_out(x)
        return raw_output

# -------------------------------------------------------------
# ===================== ↓ Helper Funcs ↓ ======================
# -------------------------------------------------------------

def calculate_growth_rate(
        raw_nn_output: jnp.ndarray,
        scale_factor: float = 100.0
    ) -> jnp.ndarray:
    rate = nnx.softplus(raw_nn_output)
    scaled_rate = rate * scale_factor
    return scaled_rate.squeeze(axis=-1)

def calculate_monotonic_thickness(
        raw_nn_output: jnp.ndarray,
        dt: jnp.ndarray,
        scale_factor: float = 100.0
    ) -> jnp.ndarray:
    rate = nnx.softplus(raw_nn_output)
    scaled_rate = rate * scale_factor
    dt = dt.reshape(-1, 1) if dt.ndim == 1 else dt
    thickness = jnp.cumsum(scaled_rate * dt, axis=0)
    thickness = thickness + 1e-7
    return thickness.squeeze(axis=-1)

# -------------------------------------------------------------
# ======================= ↓ arguments ↓ =======================
# -------------------------------------------------------------
def parse_args():
    # ... (Argument parsing remains the same) ...
    parser = argparse.ArgumentParser(description="Refine Neural Operator prediction with Simulator using Orbax Checkpoints")
    parser.add_argument('--neural_operator_checkpoint_dir', type=str, required=True,
                        help='Path to the directory containing Neural Operator Orbax checkpoints')
    parser.add_argument('--neural_operator_step', type=int, default=None,
                        help='Specific checkpoint step (epoch) to load. If None, loads the latest.')
    # Add argument for the expected dimension of the Neural Operator input/output
    parser.add_argument('--neural_operator_eval_points', type=int, required=True,
                        help='The number of evaluation points (input/output dimension) the Neural Operator was trained with.')
    parser.add_argument('--data_file', type=str, default='training_data.npz',
                        help='Path to the data file (for test set or specific measurement)')
    parser.add_argument('--measurement_file', type=str, default=None,
                        help='Optional: Path to a specific measurement file (e.g., reflectance.txt)')
    parser.add_argument('--test_data_index', type=int, default=0,
                        help='Index of the test sample from data_file to use if measurement_file is None')
    parser.add_argument('--refinement_epochs', type=int, default=5000, help='Number of refinement epochs')
    parser.add_argument('--pretrain_epochs', type=int, default=1000, help='Number of pre-training epochs for RefinementNN')
    parser.add_argument('--refinement_lr', type=float, default=1e-4, help='Learning rate for refinement')
    parser.add_argument('--pretrain_lr', type=float, default=1e-4, help='Learning rate for pre-training')
    parser.add_argument('--hidden_dims_op', nargs='+', type=int, default=[512, 512],
                        help='Hidden layer dimensions for the loaded Neural Operator (MUST match training)')
    parser.add_argument('--hidden_dim_refine', type=int, default=512, help='Hidden layer dimension for RefinementNN')
    parser.add_argument('--seed', type=int, default=43, help='Random seed for refinement')
    parser.add_argument('--results_prefix', type=str, default='refinement', help='Prefix for saving plots')
    parser.add_argument('--split_seed', type=int, default=42, help='Random seed used for train/test split in training')
    return parser.parse_args()

# -------------------------------------------------------------
# ==================== ↓ Simulator Setup Func ↓ ===============
# (This function *returns* params, but we'll assign them globally)
# -------------------------------------------------------------
def setup_simulator_params():
    if not REFLAX_AVAILABLE:
        return None, None, None, None, None, None, None # Added one None for normalization

    print("Setting up Reflax simulator parameters...")
    interference_model = ONE_LAYER_INTERNAL_REFLECTIONS
    wavelength = 632.8
    polar_angle = jnp.deg2rad(25)
    azimuthal_angle = jnp.deg2rad(0)
    setup_params = SetupParams(
        wavelength=wavelength, polar_angle=polar_angle, azimuthal_angle=azimuthal_angle
    )
    polarization_state = "Linear TE/perpendicular/s"
    s_component, p_component = polanalyze(polarization_state)
    permeability_reflection = 1.0
    permittivity_reflection = complex(1.0, 0.0)
    permeability_transmission = 1.0
    n_substrate = 3.8827
    k_substrate = 0.019626
    permittivity_transmission = (n_substrate + 1j * k_substrate)**2
    optics_params = OpticsParams(
        permeability_reflection=permeability_reflection, permittivity_reflection=permittivity_reflection,
        permeability_transmission=permeability_transmission, permittivity_transmission=permittivity_transmission,
        s_component=s_component, p_component=p_component
    )
    backside_mode = 1
    static_layer_thicknesses = jnp.array([0.0])
    permeability_static_size_layers = jnp.array([permeability_transmission])
    permittivity_static_size_layers = jnp.array([permittivity_transmission])
    static_layer_params = LayerParams(
        permeabilities=permeability_static_size_layers, permittivities=permittivity_static_size_layers,
        thicknesses=static_layer_thicknesses
    )
    n_variable = 1.457
    k_variable = 0.0
    permeability_variable_layer = 1.0
    permittivity_variable_layer = (n_variable + 1j * k_variable)**2
    variable_layer_params = LayerParams(
        permeabilities=permeability_variable_layer, permittivities=permittivity_variable_layer,
        thicknesses=None
    )
    # IMPORTANT: Get normalization constant as well
    normalization = MIN_MAX_NORMALIZATION
    print("Simulator setup complete.")
    return (interference_model, setup_params, optics_params, static_layer_params,
            variable_layer_params, backside_mode, normalization)

# -------------------------------------------------------------
# ==================== ↓ Refinement Logic ↓ ===================
# -------------------------------------------------------------

# --- Pre-training Step (Match Neural Operator Output) ---
# (No change needed here as it doesn't use the simulator)
@nnx.jit
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

@nnx.jit
def pretrain_step(
    refine_model: RefinementNN,
    optimizer: nnx.Optimizer,
    time_input: jnp.ndarray,
    target_thickness: jnp.ndarray,
    dt: jnp.ndarray
) -> Tuple[jnp.ndarray, RefinementNN, nnx.Optimizer]:
    grad_fn = nnx.value_and_grad(pretrain_loss_fn)
    loss, grads = grad_fn(refine_model, time_input, target_thickness, dt)
    optimizer.update(grads)
    return loss, refine_model, optimizer

# --- Refinement Step (Match Measured Reflectance via Simulator) ---
# --- Loss function now accesses global simulator params ---
# @nnx.jit # Jitting the loss directly might still cause issues if forward_model is complex
def refinement_loss_fn_global_access( # Renamed for clarity
    refine_model: RefinementNN,
    time_input: jnp.ndarray,
    target_reflectance: jnp.ndarray, # This will be the downsampled target
    dt: jnp.ndarray
    # No sim_params argument needed
) -> jnp.ndarray:
    # Access simulator params from the global-like scope
    # Add checks to ensure they have been set
    if not REFLAX_AVAILABLE or SIM_INTERFERENCE_MODEL is None:
         raise RuntimeError("Reflax or simulator parameters not initialized properly.")

    raw_output = refine_model(time_input, train=True)
    predicted_thickness = calculate_monotonic_thickness(raw_output, dt)

    # Run forward model using globally accessed parameters
    # The output reflectance will have the same length as predicted_thickness (downsampled length)
    predicted_reflectance = forward_model(
        model=SIM_INTERFERENCE_MODEL,
        setup_params=SIM_SETUP_PARAMS,
        optics_params=SIM_OPTICS_PARAMS,
        static_layer_params=SIM_STATIC_LAYER_PARAMS,
        variable_layer_params=SIM_VARIABLE_LAYER_PARAMS,
        variable_layer_thicknesses=predicted_thickness,
        backside_mode=SIM_BACKSIDE_MODE,
        normalization=SIM_NORMALIZATION # Use the global normalization
    )

    # Loss compares predicted (downsampled length) vs target (also downsampled length)
    loss = jnp.mean((predicted_reflectance - target_reflectance)**2)
    return loss

# --- Training step still JITted, calls the non-JITted loss ---
@nnx.jit
def refinement_step(
    refine_model: RefinementNN,
    optimizer: nnx.Optimizer,
    time_input: jnp.ndarray,
    target_reflectance: jnp.ndarray, # This will be the downsampled target
    dt: jnp.ndarray
    # No sim_params argument needed
) -> Tuple[jnp.ndarray, RefinementNN, nnx.Optimizer]:
    # Use the loss function that accesses global params
    grad_fn = nnx.value_and_grad(refinement_loss_fn_global_access)
    loss, grads = grad_fn(
        refine_model, time_input, target_reflectance, dt
    )
    optimizer.update(grads)
    return loss, refine_model, optimizer

# -------------------------------------------------------------
# ======================= ↓ Main Logic ↓ ======================
# -------------------------------------------------------------
def main(args):
    print("--- Neural Operator Refinement (Loading from Orbax Checkpoints) ---")
    print(f"Args: {args}")

    if not REFLAX_AVAILABLE and args.refinement_epochs > 0:
        print("Error: Reflax library not found, cannot perform simulator refinement.")
        return

    key = random.PRNGKey(args.seed)
    np.random.seed(args.seed)

    # --- Assign Global Simulator Params ---
    # Call setup_simulator_params here and assign to the global-like variables
    # This happens *before* any JIT compilation of functions using them
    global SIM_INTERFERENCE_MODEL, SIM_SETUP_PARAMS, SIM_OPTICS_PARAMS, \
           SIM_STATIC_LAYER_PARAMS, SIM_VARIABLE_LAYER_PARAMS, SIM_BACKSIDE_MODE, \
           SIM_NORMALIZATION
    if REFLAX_AVAILABLE:
        sim_params_tuple = setup_simulator_params()
        if sim_params_tuple[0] is not None: # Check if setup was successful
            SIM_INTERFERENCE_MODEL, SIM_SETUP_PARAMS, SIM_OPTICS_PARAMS, \
            SIM_STATIC_LAYER_PARAMS, SIM_VARIABLE_LAYER_PARAMS, SIM_BACKSIDE_MODE, \
            SIM_NORMALIZATION = sim_params_tuple
        else:
            print("Error: Simulator parameter setup failed.")
            return

    # --- Load Target Data ---
    print("\nLoading target data...")
    try:
        # --- Load original data first ---
        time_hours_original = None
        reflectance_raw_original = None
        thickness_gt_original = None # Ground truth thickness might not exist or be at original res

        if args.measurement_file:
            measurement = np.loadtxt(args.measurement_file, skiprows=1)
            time_raw_original = jnp.array(measurement[:, 0])
            reflectance_raw_original = jnp.array(measurement[:, 1])
            time_hours_original = time_raw_original / 3600.0
            # Normalize original reflectance for plotting reference
            min_r_orig, max_r_orig = jnp.min(reflectance_raw_original), jnp.max(reflectance_raw_original)
            if max_r_orig - min_r_orig < 1e-6:
                 reflectance_target_original_norm = jnp.zeros_like(reflectance_raw_original)
            else:
                 reflectance_target_original_norm = (reflectance_raw_original - 0.5 * (min_r_orig + max_r_orig)) / (0.5 * (max_r_orig - min_r_orig))
            thickness_gt_original = jnp.full_like(time_hours_original, jnp.nan) # No GT from measurement file
            print(f"Loaded original data from measurement file: {args.measurement_file}")
            print(f"  Original time points: {len(time_hours_original)}, Reflectance range: [{min_r_orig:.3f}, {max_r_orig:.3f}]")
        else:
            # Assuming the .npz file case provides GT thickness
            # data = np.load(args.data_file)
            # reflectances_all = jnp.array(data['reflectances'])
            # thicknesses_all = jnp.array(data['thicknesses'])
            # timepoints_all = jnp.array(data['timepoints'])
            # print(f"Using split seed: {args.split_seed}")
            # from sklearn.model_selection import train_test_split
            # X_train, X_test, Y_train, Y_test = train_test_split(
            #     reflectances_all, thicknesses_all, test_size=0.15, random_state=args.split_seed
            # )
            # idx = args.test_data_index
            # if idx >= len(X_test):
            #     print(f"Error: test_data_index {idx} is out of bounds for test set size {len(X_test)}")
            #     return
            # reflectance_target_original_norm = X_test[idx] # Assume already normalized if from training data
            # thickness_gt_original = Y_test[idx]
            # time_hours_original = timepoints_all
            # print(f"Loaded original test sample {idx} from data file: {args.data_file}")
            # print(f"  Original time points: {len(time_hours_original)}")

            # load first_h_convex_function.npz (as per original code)
            data = np.load("best_hp_var_sample_L0.19_V5.06.npz")
            thickness_gt_original = jnp.array(data['H_values']) * 874.5
            time_hours_original = jnp.array(data['x_grid'])
            # Calculate the 'original' reflectance target from the GT thickness
            if REFLAX_AVAILABLE and SIM_INTERFERENCE_MODEL:
                reflectance_target_original_norm = forward_model(
                    model=SIM_INTERFERENCE_MODEL, setup_params=SIM_SETUP_PARAMS, optics_params=SIM_OPTICS_PARAMS,
                    static_layer_params=SIM_STATIC_LAYER_PARAMS, variable_layer_params=SIM_VARIABLE_LAYER_PARAMS,
                    variable_layer_thicknesses=thickness_gt_original, backside_mode=SIM_BACKSIDE_MODE,
                    normalization=SIM_NORMALIZATION
                )
                print(f"Generated original reflectance from GT thickness using simulator.")
            else:
                 print("Warning: Cannot generate original reflectance from GT without Reflax/simulator.")
                 reflectance_target_original_norm = jnp.full_like(time_hours_original, jnp.nan) # Placeholder
            print(f"Loaded original data from generated GT: best_hp_var_sample_L0.19_V5.06.npz")
            print(f"  Original time points: {len(time_hours_original)}")


        # --- Downsample the data to match neural_operator_eval_points ---
        num_original_points = len(time_hours_original)
        num_eval_target = args.neural_operator_eval_points # Target dimension for NO

        if num_original_points == num_eval_target:
            print(f"Data already has the target number of points ({num_eval_target}). No downsampling needed.")
            time_hours = time_hours_original
            reflectance_target = reflectance_target_original_norm # Use the normalized version
            thickness_gt = thickness_gt_original # Use original GT for final comparison
        elif num_original_points > num_eval_target:
            print(f"Downsampling data from {num_original_points} to {num_eval_target} points using linear spacing.")
            indices = jnp.linspace(0, num_original_points - 1, num_eval_target, dtype=int)
            time_hours = time_hours_original[indices]
            reflectance_target = reflectance_target_original_norm[indices]
            # Downsample GT thickness if it exists, for potential intermediate checks (optional)
            if thickness_gt_original is not None and not jnp.isnan(thickness_gt_original).all():
                 thickness_gt = thickness_gt_original[indices] # Downsampled GT
            else:
                 thickness_gt = jnp.full_like(time_hours, jnp.nan) # Keep as NaN if no original GT
            # Keep thickness_gt_original for final plotting comparison against original resolution data
        else: # num_original_points < num_eval_target
            print(f"Error: Original data has {num_original_points} points, which is fewer than the required "
                  f"neural_operator_eval_points ({num_eval_target}). Upsampling not implemented.")
            return

        num_eval = len(time_hours) # This should now be num_eval_target
        print(f"Using {num_eval} evaluation points for Neural Operator and Refinement.")

        # Calculate dt for the potentially downsampled time axis
        if len(time_hours) < 2:
             print("Error: Need at least two time points after downsampling to calculate dt.")
             return
        dt_hours = jnp.diff(time_hours)
        dt_hours = jnp.concatenate((jnp.array([dt_hours[0]]), dt_hours)) # Simple forward difference for first point
        time_input_nn = time_hours[:, None] # Input shape for RefinementNN

    except Exception as e:
        print(f"Error loading or processing data: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Load Neural Operator from Orbax Checkpoint ---
    print("\nLoading pre-trained Neural Operator state from checkpoint...")
    # Now, use the *target* num_eval for initialization
    neural_op_model = None
    step_loaded = -1
    try:
        checkpoint_dir_abs = os.path.abspath(args.neural_operator_checkpoint_dir)
        if not os.path.isdir(checkpoint_dir_abs):
            raise FileNotFoundError(f"Checkpoint directory not found: {checkpoint_dir_abs}")

        mngr_options = ocp.CheckpointManagerOptions(max_to_keep=1, create=False)
        ckpt_manager = ocp.CheckpointManager(checkpoint_dir_abs, options=mngr_options)

        step_to_load = args.neural_operator_step
        if step_to_load is None:
            step_to_load = ckpt_manager.latest_step()
            if step_to_load is None:
                 raise FileNotFoundError(f"No checkpoints found in {checkpoint_dir_abs}")
            print(f"Loading latest step: {step_to_load}")
        else:
             all_steps = ckpt_manager.all_steps(read=True) # Ensure steps are read
             if step_to_load not in all_steps:
                      raise ValueError(f"Specified step {step_to_load} not found in checkpoints: {all_steps}")
             print(f"Loading specified step: {step_to_load}")

        key, abs_key = random.split(key)
        print(f"Creating abstract Neural Operator structure ({num_eval} eval points) for restore...")
        # Use num_eval (the potentially downsampled dimension) here
        abstract_neural_op = nnx.eval_shape(
            lambda: NeuralOperatorMLP(args.hidden_dims_op, num_eval, rngs=nnx.Rngs(abs_key))
        )
        graphdef_op, abstract_state_op = nnx.split(abstract_neural_op)

        restored_state = ckpt_manager.restore(
            step_to_load,
            args=ocp.args.StandardRestore(abstract_state_op)
        )

        key, merge_key = random.split(key)
        # Use num_eval again for the actual instance
        neural_op_model_instance = NeuralOperatorMLP(
            hidden_dims=args.hidden_dims_op,
            num_eval_points=num_eval, # Use the potentially downsampled number
            rngs=nnx.Rngs(merge_key)
        )
        # Merge the loaded state into the graph definition
        # Ensure graphdef_op came from the correct abstract model size
        neural_op_model = nnx.merge(graphdef_op, restored_state)
        step_loaded = step_to_load
        print(f"Successfully loaded and merged model state from step {step_loaded}.")

    except Exception as e:
        print(f"Error loading Neural Operator checkpoint: {e}")
        import traceback
        traceback.print_exc()
        return

    # --- Get Neural Operator Initial Guess ---
    if neural_op_model is None:
         print("Cannot proceed without a loaded Neural Operator model.")
         return
    # Input reflectance_target is now potentially downsampled
    thickness_no_guess = neural_op_model(reflectance_target[None, :], train=False).squeeze(axis=0)
    print(f"Generated initial thickness guess using Neural Operator (shape: {thickness_no_guess.shape})") # Shape should match num_eval

    # --- Initialize Refinement Network ---
    print("\nInitializing Refinement Network...")
    key, refine_key = random.split(key)
    refine_model = RefinementNN(dmid=args.hidden_dim_refine, rngs=nnx.Rngs(refine_key))

    # --- Pre-train Refinement Network ---
    print(f"\nPre-training Refinement Network for {args.pretrain_epochs} epochs...")
    # Inputs (time_input_nn, thickness_no_guess, dt_hours) are all based on the downsampled resolution
    pretrain_optimizer = nnx.Optimizer(refine_model, optax.adam(learning_rate=args.pretrain_lr))
    pretrain_start_time = pytime.time()
    for epoch in range(args.pretrain_epochs):
        loss, refine_model, pretrain_optimizer = pretrain_step(
            refine_model, pretrain_optimizer, time_input_nn, thickness_no_guess, dt_hours
        )
        if epoch % 200 == 0 or epoch == args.pretrain_epochs - 1:
            print(f"  Pre-train Epoch {epoch}/{args.pretrain_epochs}, Loss (vs NO guess): {loss:.4e}")
            if jnp.isnan(loss):
                print("NaN loss during pre-training. Stopping.")
                return
    print(f"Pre-training finished in {pytime.time() - pretrain_start_time:.2f}s")


    # --- Refine using Simulator ---
    thickness_refined = None
    if args.refinement_epochs > 0 and REFLAX_AVAILABLE:
        print(f"\nRefining using simulator for {args.refinement_epochs} epochs...")
        # No need to get sim_params here, they are global
        refinement_optimizer = nnx.Optimizer(refine_model, optax.adam(learning_rate=args.refinement_lr))
        refine_start_time = pytime.time()
        best_refine_loss = float('inf')
        epochs_no_improve = 0
        patience_reports = 5

        for epoch in range(args.refinement_epochs):
            # Call the JITted refinement_step which calls the non-JITted loss
            # Inputs (time_input_nn, reflectance_target, dt_hours) are based on downsampled resolution
            loss, refine_model, refinement_optimizer = refinement_step(
                refine_model, refinement_optimizer, time_input_nn, reflectance_target, dt_hours
                # No sim_params passed here
            )

            if epoch % 200 == 0 or epoch == args.refinement_epochs - 1:
                print(f"  Refinement Epoch {epoch}/{args.refinement_epochs}, Loss (vs Downsampled Reflectance): {loss:.4e}")

                if jnp.isnan(loss):
                     print("NaN loss during refinement. Stopping.")
                     break

                if loss < best_refine_loss - 1e-7:
                     best_refine_loss = loss
                     epochs_no_improve = 0
                else:
                     epochs_no_improve += 1

                if epochs_no_improve >= patience_reports:
                     print(f"  -> Early stopping triggered at epoch {epoch}")
                     break

        print(f"Refinement finished in {pytime.time() - refine_start_time:.2f}s")
        # Output thickness will have the downsampled resolution
        raw_output_post_refine = refine_model(time_input_nn, train=False)
        thickness_refined = calculate_monotonic_thickness(raw_output_post_refine, dt_hours)

    elif thickness_refined is None:
        print("\nSkipping simulator refinement step or it failed.")
        # Use the result after pre-training
        raw_output_post_pretrain = refine_model(time_input_nn, train=False)
        thickness_refined = calculate_monotonic_thickness(raw_output_post_pretrain, dt_hours)


    # --- Calculate Final Results for Plotting ---
    # Model outputs (thickness_no_guess, thickness_refined) are at the downsampled resolution `time_hours`
    # Ground truth (thickness_gt_original) and target reflectance (reflectance_target_original_norm) are at `time_hours_original`
    print("\nCalculating final results for plotting...")
    if thickness_refined is None:
        print("Error: Refined thickness calculation failed.")
        return

    # Calculate growth rates based on the available resolutions
    # GT rate from original thickness and time
    growth_rate_gt = jnp.full_like(time_hours_original, jnp.nan) # Default to NaN
    if not jnp.isnan(thickness_gt_original).any():
        if len(time_hours_original) > 1:
            dx_orig = time_hours_original[1] - time_hours_original[0] # Assumes uniform original spacing for gradient
            growth_rate_gt = jnp.gradient(thickness_gt_original, dx_orig)
        elif len(time_hours_original) == 1:
            growth_rate_gt = jnp.zeros_like(time_hours_original)

    # NO rate from NO guess thickness and downsampled time
    growth_rate_no = jnp.full_like(time_hours, jnp.nan) # Default to NaN
    if len(time_hours) > 1 :
        dx = time_hours[1] - time_hours[0] # Assumes uniform downsampled spacing for gradient
        growth_rate_no = jnp.gradient(thickness_no_guess, dx)
    elif len(time_hours) == 1:
         growth_rate_no = jnp.zeros_like(time_hours)

    # Refined rate calculated directly from NN output (at downsampled resolution)
    growth_rate_refined_nn = calculate_growth_rate(refine_model(time_input_nn, train=False))

    # Calculate final reflectance from the refined thickness (at downsampled resolution)
    final_reflectance_refined = jnp.full_like(reflectance_target, jnp.nan) # Will have downsampled length
    if REFLAX_AVAILABLE and SIM_INTERFERENCE_MODEL is not None:
        try:
            final_reflectance_refined = forward_model(
                model=SIM_INTERFERENCE_MODEL, setup_params=SIM_SETUP_PARAMS, optics_params=SIM_OPTICS_PARAMS,
                static_layer_params=SIM_STATIC_LAYER_PARAMS, variable_layer_params=SIM_VARIABLE_LAYER_PARAMS,
                variable_layer_thicknesses=thickness_refined, # Input is downsampled length
                backside_mode=SIM_BACKSIDE_MODE,
                normalization=SIM_NORMALIZATION
            ) # Output will be downsampled length
        except Exception as e:
            print(f"Warning: Error during final reflectance calculation: {e}")


    # --- Plotting Comparison ---
    # Plot original data against original time axis
    # Plot model predictions against the downsampled time axis
    print("\nGenerating comparison plot...")
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(10, 12))

    # Plot 1: Reflectance
    axs[0].plot(time_hours_original, reflectance_target_original_norm, '-', label="Target Reflectance (Original)", color='black', linewidth=1.5, alpha=0.7)
    if not jnp.isnan(final_reflectance_refined).any():
        # Plot refined reflectance against the time points it was calculated at (downsampled)
        axs[0].plot(time_hours, final_reflectance_refined, '--', label="Refined Model Reflectance", color='red', linewidth=1.5)
    axs[0].set_ylabel("Norm. Reflectance")
    axs[0].set_title(f"Comparison: NO (Ckpt {step_loaded}, {num_eval} pts) vs Refined Prediction vs Original Data ({num_original_points} pts)")
    axs[0].legend(fontsize='small')
    axs[0].grid(True, linestyle=':', linewidth=0.5)

    # Plot 2: Thickness
    if not jnp.isnan(thickness_gt_original).any():
        axs[1].plot(time_hours_original, thickness_gt_original, '-', label="Ground Truth Thickness (Original)", color='black', linewidth=1.5, alpha=0.7)
    # Plot model thicknesses against the time points they correspond to (downsampled)
    axs[1].plot(time_hours, thickness_no_guess, ':', label="Neural Operator Guess", color='blue', linewidth=1.5)
    axs[1].plot(time_hours, thickness_refined, '--', label="Refined Thickness", color='red', linewidth=1.5)
    axs[1].set_ylabel("Thickness (nm)")
    axs[1].legend(fontsize='small')
    axs[1].grid(True, linestyle=':', linewidth=0.5)

    # Plot 3: Growth Rate
    if not jnp.isnan(growth_rate_gt).any():
         axs[2].plot(time_hours_original, growth_rate_gt, '-', label="Ground Truth Growth Rate (Original)", color='black', linewidth=1.5, alpha=0.7)
    # Plot model growth rates against the time points they correspond to (downsampled)
    if not jnp.isnan(growth_rate_no).any():
        axs[2].plot(time_hours, growth_rate_no, ':', label="Neural Operator Growth Rate (Grad)", color='blue', linewidth=1.5)
    axs[2].plot(time_hours, growth_rate_refined_nn, '--', label="Refined Rate (NN Output)", color='red', linewidth=1.5)
    axs[2].set_xlabel("Time (hours)")
    axs[2].set_ylabel("Growth Rate (nm/h)")
    axs[2].legend(fontsize='small')
    axs[2].grid(True, linestyle=':', linewidth=0.5)

    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig(f"{args.results_prefix}_comparison.png")
    plt.savefig(f"{args.results_prefix}_comparison.svg")
    print(f"Saved comparison plot to {args.results_prefix}_comparison.png/svg")
    plt.close(fig)


    # --- Calculate Test Errors ---
    # Compare model outputs (downsampled res) against ground truth (original res)
    # Need to interpolate model outputs to original time grid for fair comparison
    print("\nCalculating Final Errors (interpolated to original time grid):")
    mse_no = jnp.nan
    mse_refined = jnp.nan
    mse_reflectance = jnp.nan

    if not jnp.isnan(thickness_gt_original).any():
        # Interpolate model thicknesses to the original time grid
        thickness_no_guess_interp = jnp.interp(time_hours_original, time_hours, thickness_no_guess)
        thickness_refined_interp = jnp.interp(time_hours_original, time_hours, thickness_refined)

        mse_no = jnp.mean((thickness_no_guess_interp - thickness_gt_original)**2)
        mse_refined = jnp.mean((thickness_refined_interp - thickness_gt_original)**2)
        print(f"  Thickness MSE (vs GT Original) - Neural Operator: {mse_no:.4e}")
        print(f"  Thickness MSE (vs GT Original) - Refined Model:   {mse_refined:.4e}")
    else:
        print("  Skipping Thickness MSE calculation (no Ground Truth available).")

    # Compare final refined reflectance (downsampled res) against original target reflectance
    if not jnp.isnan(final_reflectance_refined).any() and not jnp.isnan(reflectance_target_original_norm).any():
        # Interpolate refined reflectance to the original time grid
        final_reflectance_refined_interp = jnp.interp(time_hours_original, time_hours, final_reflectance_refined)
        mse_reflectance = jnp.mean((final_reflectance_refined_interp - reflectance_target_original_norm)**2)
        print(f"  Reflectance MSE (vs Target Original) - Refined Model: {mse_reflectance:.4e}")
    else:
        print("  Skipping Reflectance MSE calculation (refined or target reflectance unavailable/NaN).")


    print("\n--- Refinement Complete ---")

if __name__ == "__main__":
    args = parse_args()
    main(args)