import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec # Import gridspec
import matplotlib.lines as mlines # For manual legend
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
    file_path = f"validation_results_data/{initialization_to_string(initialization)}/{initialization_to_string(initialization)}_{sample_number}.npz"
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
    (
        time_points_linear, 
        true_reflectance,
        true_thickness,
        true_growth_rate,
        initialized_reflectance_linear,
        initialized_thickness_linear,
        initialized_growth_rate_linear,
        initialized_time_points_linear, 
        predicted_reflectance_linear,
        predicted_thickness_linear,
        predicted_growth_rate_linear,
    ) = get_sample(
        LINEAR_INITIALIZATION_TRAINED,
        sample_num,
    )

    (
        time_points_no, 
        _, 
        _,
        _,
        initialized_reflectance_neural_operator,
        initialized_thickness_neural_operator,
        initialized_growth_rate_neural_operator,
        initialized_time_points_neural_operator, 
        predicted_reflectance_neural_operator,
        predicted_thickness_neural_operator,
        predicted_growth_rate_neural_operator,
    ) = get_sample(
        NEURAL_OPERATOR_INITIALIZATION,
        sample_num,
    )

    if not np.array_equal(time_points_linear, time_points_no):
        print("Warning: time_points arrays differ between loaded samples. Using the one from LINEAR_INITIALIZATION_TRAINED.")
    time_points = time_points_linear 

    # --- Calculate Squared Errors ---
    squared_error_reflectance_linear = (predicted_reflectance_linear - true_reflectance)**2
    squared_error_reflectance_neural = (predicted_reflectance_neural_operator - true_reflectance)**2

    squared_error_thickness_linear = (predicted_thickness_linear - true_thickness)**2
    squared_error_thickness_neural = (predicted_thickness_neural_operator - true_thickness)**2

    squared_error_growth_rate_linear = (predicted_growth_rate_linear - true_growth_rate)**2
    squared_error_growth_rate_neural = (predicted_growth_rate_neural_operator - true_growth_rate)**2

    epsilon = np.finfo(float).eps
    squared_error_reflectance_linear += epsilon
    squared_error_reflectance_neural += epsilon
    squared_error_thickness_linear += epsilon
    squared_error_thickness_neural += epsilon
    squared_error_growth_rate_linear += epsilon
    squared_error_growth_rate_neural += epsilon

    # --- Plotting Setup ---
    fig = plt.figure(figsize=(20, 7)) 

    gs_outer = gridspec.GridSpec(1, 3, figure=fig, wspace=0.25) # Increased wspace

    # --- Section 1: Reflectance ---
    gs_inner1 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[0, 0],
                                                height_ratios=[2, 1], hspace=0.1)
    ax1_main = fig.add_subplot(gs_inner1[0])
    ax1_err = fig.add_subplot(gs_inner1[1], sharex=ax1_main)

    # --- Section 2: Thickness ---
    gs_inner2 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[0, 1],
                                                height_ratios=[2, 1], hspace=0.1)
    ax2_main = fig.add_subplot(gs_inner2[0])
    ax2_err = fig.add_subplot(gs_inner2[1], sharex=ax2_main)

    # --- Section 3: Growth Rate ---
    gs_inner3 = gridspec.GridSpecFromSubplotSpec(2, 1, subplot_spec=gs_outer[0, 2],
                                                height_ratios=[2, 1], hspace=0.1)
    ax3_main = fig.add_subplot(gs_inner3[0])
    ax3_err = fig.add_subplot(gs_inner3[1], sharex=ax3_main)

    # --- Plotting Reflectance ---
    ax1_main.plot(time_points, true_reflectance, label="ground truth", color="black", linewidth=4)
    ax1_main.plot(initialized_time_points_linear, initialized_reflectance_linear, label="linear initialization (pre-trained)", color="blue", linestyle="--")
    ax1_main.plot(initialized_time_points_neural_operator, initialized_reflectance_neural_operator, label="neural operator initialization", color="red", linestyle="--")
    ax1_main.plot(time_points, predicted_reflectance_linear, label="optimization result after linear initialization (pre-trained)", color="blue")
    ax1_main.plot(time_points, predicted_reflectance_neural_operator, label="optimization result after neural operator initialization", color="red")
    ax1_main.set_title("Reflectance")
    ax1_main.set_ylabel("normalized reflectance")
    ax1_main.tick_params(axis='x', labelbottom=False)

    ax1_err.plot(time_points, squared_error_reflectance_linear, color="blue") 
    ax1_err.plot(time_points, squared_error_reflectance_neural, color="red")  
    ax1_err.set_yscale('log')
    ax1_err.set_ylabel("squared error")
    ax1_err.set_xlabel("time in hours") 
    ax1_err.grid(True, which='both', linestyle=':', linewidth=0.5)

    # --- Plotting Thickness ---
    ax2_main.plot(time_points, true_thickness, label="ground truth", color="black", linewidth=4)
    ax2_main.plot(initialized_time_points_linear, initialized_thickness_linear, label="linear initialization (pre-trained)", color="blue", linestyle="--")
    ax2_main.plot(initialized_time_points_neural_operator, initialized_thickness_neural_operator, label="neural operator initialization", color="red", linestyle="--")
    ax2_main.plot(time_points, predicted_thickness_linear, label="optimization result after linear initialization (pre-trained)", color="blue")
    ax2_main.plot(time_points, predicted_thickness_neural_operator, label="optimization result after neural operator initialization", color="red")
    ax2_main.set_title("Thickness")
    ax2_main.set_ylabel("thickness in nm")
    ax2_main.tick_params(axis='x', labelbottom=False)

    ax2_err.plot(time_points, squared_error_thickness_linear, color="blue") 
    ax2_err.plot(time_points, squared_error_thickness_neural, color="red")  
    ax2_err.set_yscale('log')
    ax2_err.set_ylabel("squared error\nin nm²")
    ax2_err.set_xlabel("time in hours") 
    ax2_err.grid(True, which='both', linestyle=':', linewidth=0.5)

    # --- Plotting Growth Rate ---
    ax3_main.plot(time_points, true_growth_rate, label="ground truth", color="black", linewidth=4)
    ax3_main.plot(initialized_time_points_linear, initialized_growth_rate_linear, label="linear initialization (pre-trained)", color="blue", linestyle="--")
    ax3_main.plot(initialized_time_points_neural_operator, initialized_growth_rate_neural_operator, label="neural operator initialization", color="red", linestyle="--")
    ax3_main.plot(time_points, predicted_growth_rate_linear, label="optimization result after linear initialization (pre-trained)", color="blue")
    ax3_main.plot(time_points, predicted_growth_rate_neural_operator, label="optimization result after neural operator initialization", color="red")
    ax3_main.set_title("Growth Rate")
    ax3_main.set_ylabel("growth rate in nm/h")
    ax3_main.set_ylim(200.0, 1900.0) 
    ax3_main.tick_params(axis='x', labelbottom=False)

    ax3_err.plot(time_points, squared_error_growth_rate_linear, color="blue") 
    ax3_err.plot(time_points, squared_error_growth_rate_neural, color="red")  
    ax3_err.set_yscale('log')
    ax3_err.set_ylabel("squared error\nin nm²/h²")
    ax3_err.set_xlabel("time in hours") 
    ax3_err.grid(True, which='both', linestyle=':', linewidth=0.5)

    # --- Common Legend (Manually Created for 1,2,2 column structure) ---
    h_gt = mlines.Line2D([], [], color='black', linewidth=2, label='ground truth')
    h_lin_init = mlines.Line2D([], [], color='blue', linestyle='--', label='linear initialization (pre-trained)')
    h_no_init = mlines.Line2D([], [], color='red', linestyle='--', label='neural operator initialization')
    h_opt_lin = mlines.Line2D([], [], color='blue', label='optimization result after linear initialization (pre-trained)')
    h_opt_no = mlines.Line2D([], [], color='red', label='optimization result after neural operator initialization')
    
    # Dummy handle for layout: invisible line, non-empty but invisible label string to ensure it takes a slot.
    # Using a single space or an empty string '' for the label should also work typically.
    h_dummy = mlines.Line2D([], [], color='none', marker='None', linestyle='None', label=' ') 

    # Order of handles to achieve the desired 1,2,2 column layout with ncol=3 and row-major fill:
    # Row 1: h_gt (Col1), h_lin_init (Col2), h_opt_lin (Col3)
    # Row 2: h_dummy (Col1), h_no_init (Col2), h_opt_no (Col3)
    legend_handles_ordered = [
        h_gt,   
        h_dummy,       
        h_lin_init,  
        h_no_init,  
        h_opt_lin,   
        h_opt_no     
    ]
    
    fig.subplots_adjust(bottom=0.22) 

    fig.legend(handles=legend_handles_ordered, 
               loc='lower center',
               bbox_to_anchor=(0.5, 0.03), 
               ncol=3, 
               fontsize='large', # Larger font size
               # Optional: Adjust column spacing if labels are too close/far within the legend
               # columnspacing=1.0 
              )

    output_dir = "figures"
    output_filename = os.path.join(output_dir, f"single_analysis_result.svg")
    plt.savefig(
        output_filename,
        bbox_inches='tight',
        pad_inches=0.02
    )
    print(f"Plot saved to {output_filename}")
    plt.show()