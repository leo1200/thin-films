import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # Import gridspec
import os # For checking directory existence

# Constants for initialization types (assuming these are used elsewhere or for clarity)
RANDOM_INITIALIZATION = 0
LINEAR_INITIALIZATION_SET = 1
LINEAR_INITIALIZATION_TRAINED = 2
NEURAL_OPERATOR_INITIALIZATION = 3

# Helper functions for converting initialization type to string
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

def initialization_to_label_string(initialization):
    # This function seems unused in the plotting script, but kept for completeness
    if initialization == RANDOM_INITIALIZATION:
        return "random initialization"
    elif initialization == LINEAR_INITIALIZATION_SET:
        return "linear initialization (set)"
    elif initialization == LINEAR_INITIALIZATION_TRAINED:
        return "linear initialization (trained)"
    elif initialization == NEURAL_OPERATOR_INITIALIZATION:
        return "neural operator initialization"
    else:
        raise ValueError("Invalid initialization type.")

# Function to load data from .npz files
def get_sample(initialization, sample_number):
    """Loads relevant data arrays for a given initialization type and sample number."""
    file_path = f"result_data/{initialization_to_string(initialization)}/{initialization_to_string(initialization)}_{sample_number}.npz"
    try:
        data = np.load(file_path)
        # Extract only the necessary arrays for plotting
        time_points = data["time_points"]
        true_reflectance = data["true_reflectance"]
        true_thickness = data["true_thickness"]
        true_growth_rate = data["true_growth_rate"]
        initialized_time_points = data["initialized_time_points"]
        initialized_reflectance = data["initialized_reflectance"]
        initialized_thickness = data["initialized_thickness"]
        initialized_growth_rate = data["initialized_growth_rate"]
        predicted_reflectance = data["predicted_reflectance"]
        predicted_thickness = data["predicted_thickness"]
        predicted_growth_rate = data["predicted_growth_rate"]

        # Ensure all required keys were present
        required_keys = [
            "time_points", "true_reflectance", "true_thickness", "true_growth_rate",
            "initialized_time_points", "initialized_reflectance", "initialized_thickness",
            "initialized_growth_rate", "predicted_reflectance", "predicted_thickness",
            "predicted_growth_rate"
        ]
        if not all(key in data for key in required_keys):
             missing = [key for key in required_keys if key not in data]
             raise KeyError(f"Missing keys in {file_path}: {missing}")


    except FileNotFoundError:
        print(f"ERROR: File not found at {file_path}")
        raise # Reraise the exception to stop execution if data is missing
    except KeyError as e:
        print(f"ERROR: {e}")
        raise # Reraise the exception

    return (
        time_points,
        true_reflectance,
        true_thickness,
        true_growth_rate,
        initialized_reflectance,
        initialized_thickness,
        initialized_growth_rate,
        initialized_time_points, # Return the specific time points for this initialization
        predicted_reflectance,
        predicted_thickness,
        predicted_growth_rate,
    )

def plot_single_sample_initialization_comparison():
    sample_num = 15

    # --- Load Data ---
    # Load data for Linear Initialization (Trained)
    (
        time_points_linear, # Note: time_points should be the same across loads if from same base sample
        true_reflectance,
        true_thickness,
        true_growth_rate,
        initialized_reflectance_linear,
        initialized_thickness_linear,
        initialized_growth_rate_linear,
        initialized_time_points_linear, # Specific init times for linear
        predicted_reflectance_linear,
        predicted_thickness_linear,
        predicted_growth_rate_linear,
    ) = get_sample(
        LINEAR_INITIALIZATION_TRAINED,
        sample_num,
    )

    # Load data for Neural Operator Initialization
    (
        time_points_no, # Should be same as time_points_linear
        _, # True values are the same, no need to reload
        _,
        _,
        initialized_reflectance_neural_operator,
        initialized_thickness_neural_operator,
        initialized_growth_rate_neural_operator,
        initialized_time_points_neural_operator, # Specific init times for NO
        predicted_reflectance_neural_operator,
        predicted_thickness_neural_operator,
        predicted_growth_rate_neural_operator,
    ) = get_sample(
        NEURAL_OPERATOR_INITIALIZATION,
        sample_num,
    )

    # Verify that the main time axes are consistent
    if not np.array_equal(time_points_linear, time_points_no):
        print("Warning: time_points arrays differ between loaded samples. Using the one from LINEAR_INITIALIZATION_TRAINED.")
    time_points = time_points_linear # Use one consistent time axis for predictions and truth

    # --- Calculate Squared Errors ---
    # Errors are calculated between the *final prediction* and the *true* data,
    # both of which should be defined on the main `time_points` array.
    squared_error_reflectance_linear = (predicted_reflectance_linear - true_reflectance)**2
    squared_error_reflectance_neural = (predicted_reflectance_neural_operator - true_reflectance)**2

    squared_error_thickness_linear = (predicted_thickness_linear - true_thickness)**2
    squared_error_thickness_neural = (predicted_thickness_neural_operator - true_thickness)**2

    squared_error_growth_rate_linear = (predicted_growth_rate_linear - true_growth_rate)**2
    squared_error_growth_rate_neural = (predicted_growth_rate_neural_operator - true_growth_rate)**2

    # Add a small epsilon to prevent log(0) issues if errors are exactly zero
    epsilon = np.finfo(float).eps
    squared_error_reflectance_linear += epsilon
    squared_error_reflectance_neural += epsilon
    squared_error_thickness_linear += epsilon
    squared_error_thickness_neural += epsilon
    squared_error_growth_rate_linear += epsilon
    squared_error_growth_rate_neural += epsilon


    # --- Plotting Setup ---
    fig = plt.figure(figsize=(8, 14)) # Adjusted height for error plots

    # Outer GridSpec: 3 rows for sections, 1 column. Larger hspace.
    # Adjust hspace here to control spacing BETWEEN sections.
    gs_outer = gridspec.GridSpec(3, 1, figure=fig, hspace=0.15) # <--- Increase this for more space BETWEEN sections

    # --- Section 1: Reflectance ---
    # Inner GridSpec for Reflectance (nested inside outer grid's 1st row)
    # 2 rows (main, error), 1 column. Smaller hspace.
    # Adjust hspace here to control spacing WITHIN the section (main vs error).
    gs_inner1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[0],
                                                height_ratios=[3, 1], hspace=0.1) # <-- Minimal space WITHIN section
    ax1_main = fig.add_subplot(gs_inner1[0])
    ax1_err = fig.add_subplot(gs_inner1[1], sharex=ax1_main)

    # --- Section 2: Thickness ---
    # Inner GridSpec for Thickness (nested inside outer grid's 2nd row)
    gs_inner2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[1],
                                                height_ratios=[3, 1], hspace=0.1) # <-- Minimal space WITHIN section
    ax2_main = fig.add_subplot(gs_inner2[0])
    ax2_err = fig.add_subplot(gs_inner2[1], sharex=ax2_main)
    # --- Section 3: Growth Rate ---
    # Inner GridSpec for Growth Rate (nested inside outer grid's 3rd row)
    gs_inner3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[2],
                                                height_ratios=[3, 1], hspace=0.1) # <-- Minimal space WITHIN section
    ax3_main = fig.add_subplot(gs_inner3[0])
    ax3_err = fig.add_subplot(gs_inner3[1], sharex=ax3_main)

    # --- Plotting Reflectance ---
    ax1_main.plot(time_points, true_reflectance, label="true", color="black", linewidth=2)
    # Plot initializations using their specific time points
    ax1_main.plot(initialized_time_points_linear, initialized_reflectance_linear, label="init linear (trained)", color="blue", linestyle="--")
    ax1_main.plot(initialized_time_points_neural_operator, initialized_reflectance_neural_operator, label="init neural op", color="red", linestyle="--")
    # Plot predictions using the main time points
    ax1_main.plot(time_points, predicted_reflectance_linear, label="pred linear (trained)", color="blue")
    ax1_main.plot(time_points, predicted_reflectance_neural_operator, label="pred neural op", color="red")
    ax1_main.set_title("Reflectance")
    ax1_main.set_ylabel("normalized reflectance")
    ax1_main.legend(loc="upper left", fontsize='small')
    ax1_main.tick_params(axis='x', labelbottom=False) # Hide x-tick labels on main plot

    # Plotting Reflectance Squared Error (Log Scale)
    ax1_err.plot(time_points, squared_error_reflectance_linear, color="blue", label="SE linear (trained)")
    ax1_err.plot(time_points, squared_error_reflectance_neural, color="red", label="SE neural op")
    ax1_err.set_yscale('log') # Set y-axis to log scale
    ax1_err.set_ylabel("squared error")
    ax1_err.grid(True, which='both', linestyle=':', linewidth=0.5) # Grid useful on log scale
    ax1_err.tick_params(axis='x', labelbottom=False) # Hide x-tick labels on error plot (except last)
    # Optional: Add legend to error plot if needed
    # ax1_err.legend(loc="upper left", fontsize='x-small')


    # --- Plotting Thickness ---
    ax2_main.plot(time_points, true_thickness, label="true", color="black", linewidth=2)
    ax2_main.plot(initialized_time_points_linear, initialized_thickness_linear, label="init linear (trained)", color="blue", linestyle="--")
    ax2_main.plot(initialized_time_points_neural_operator, initialized_thickness_neural_operator, label="init neural op", color="red", linestyle="--")
    ax2_main.plot(time_points, predicted_thickness_linear, label="pred linear (trained)", color="blue")
    ax2_main.plot(time_points, predicted_thickness_neural_operator, label="pred neural op", color="red")
    ax2_main.set_title("Thickness")
    ax2_main.set_ylabel("thickness in nm")
    ax2_main.legend(loc="upper left", fontsize='small')
    ax2_main.tick_params(axis='x', labelbottom=False)

    # Plotting Thickness Squared Error (Log Scale)
    ax2_err.plot(time_points, squared_error_thickness_linear, color="blue", label="SE linear (trained)")
    ax2_err.plot(time_points, squared_error_thickness_neural, color="red", label="SE neural op")
    ax2_err.set_yscale('log')
    ax2_err.set_ylabel("squared error\nin nm²")
    ax2_err.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax2_err.tick_params(axis='x', labelbottom=False)
    # ax2_err.legend(loc="upper left", fontsize='x-small')


    # --- Plotting Growth Rate ---
    ax3_main.plot(time_points, true_growth_rate, label="true", color="black", linewidth=2)
    ax3_main.plot(initialized_time_points_linear, initialized_growth_rate_linear, label="init linear (trained)", color="blue", linestyle="--")
    ax3_main.plot(initialized_time_points_neural_operator, initialized_growth_rate_neural_operator, label="init neural op", color="red", linestyle="--")
    ax3_main.plot(time_points, predicted_growth_rate_linear, label="pred linear (trained)", color="blue")
    ax3_main.plot(time_points, predicted_growth_rate_neural_operator, label="pred neural op", color="red")
    ax3_main.set_title("Growth Rate")
    ax3_main.set_ylabel("growth rate in nm/h")
    ax3_main.legend(loc="upper left", fontsize='small')
    ax3_main.set_ylim(200.0, 1900.0) # Keep original ylim if desired for main plot
    ax3_main.tick_params(axis='x', labelbottom=False)


    # Plotting Growth Rate Squared Error (Log Scale)
    ax3_err.plot(time_points, squared_error_growth_rate_linear, color="blue", label="SE linear (trained)")
    ax3_err.plot(time_points, squared_error_growth_rate_neural, color="red", label="SE neural op")
    ax3_err.set_yscale('log')
    ax3_err.set_ylabel("squared error\nin nm²/h²")
    ax3_err.set_xlabel("time in hours") # Set x-label only on the bottom-most plot
    ax3_err.grid(True, which='both', linestyle=':', linewidth=0.5)
    # ax3_err.legend(loc="upper left", fontsize='x-small')

    # Ensure the output directory exists
    output_dir = "figures"

    # Save and show the figure
    output_filename = os.path.join(output_dir, f"initialization_comparison_sample_{sample_num}_log_sq_error.svg")
    plt.savefig(
        output_filename,
        bbox_inches='tight',  # Calculate bounding box to fit contents
        pad_inches=0.02       # Add minimal padding (adjust as needed, e.g., 0.01 or 0)
    )
    print(f"Plot saved to {output_filename}")
    plt.show()

# --- Run the plotting function ---
if __name__ == "__main__":
    # Example check for data directory (optional but good practice)
    if not os.path.exists("result_data"):
         print("ERROR: 'result_data' directory not found. Please ensure data exists.")
    else:
        plot_single_sample_initialization_comparison()