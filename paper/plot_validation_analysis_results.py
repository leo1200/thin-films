from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

import matplotlib.lines as mlines # For manual legend

RANDOM_INITIALIZATION = 0
LINEAR_INITIALIZATION_SET = 1
LINEAR_INITIALIZATION_TRAINED = 2
NEURAL_OPERATOR_INITIALIZATION = 3


initializations = [
    RANDOM_INITIALIZATION,
    LINEAR_INITIALIZATION_SET,
    LINEAR_INITIALIZATION_TRAINED,
    NEURAL_OPERATOR_INITIALIZATION,
]

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
    


def plot_single_validation_result(initialization, sample_number):

    data = np.load(f"validation_results_data/{initialization_to_string(initialization)}/{initialization_to_string(initialization)}_{sample_number}.npz")

    sample_index = data["sample_index"]
    initialization = data["initialization"]
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
    reflectance_losses = data["reflectance_losses"]
    thickness_losses = data["thickness_losses"]
    growth_rate_losses = data["growth_rate_losses"]
    initial_reflectance_loss = data["initial_reflectance_loss"]
    initial_thickness_loss = data["initial_thickness_loss"]
    initial_growth_rate_loss = data["initial_growth_rate_loss"]
    reflectance_loss = data["reflectance_loss"]
    thickness_loss = data["thickness_loss"]
    growth_rate_loss = data["growth_rate_loss"]

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
        label = "initial prediction",
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
        label = "initial prediction",
        linestyle = "--"
    )
    ax3.set_xlabel("time in hours")
    ax3.set_ylabel("growth rate in nm/h")
    ax3.legend(loc = "lower right")
    ax3.set_title("Growth Rate")

    plt.tight_layout()

    plt.savefig(f"validation_results_figures/{initialization_to_string(initialization)}/predictions_{initialization_to_string(initialization)}_{sample_number}.svg")
    plt.close("all")

    # in a second plot, plot the losses
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

    # plot the reflectance loss
    ax1.plot(reflectance_losses, label = "reflectance loss")
    # set the y axis to log scale
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("loss")
    ax1.legend(loc = "upper right")
    ax1.set_title("Reflectance Loss")

    # plot the thickness loss
    ax2.plot(thickness_losses, label = "thickness loss")
    # set the y axis to log scale
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("loss")
    ax2.legend(loc = "upper right")
    ax2.set_title("Thickness Loss")

    # plot the growth rate loss
    ax3.plot(growth_rate_losses, label = "growth rate loss")
    # set the y axis to log scale
    ax3.set_yscale("log")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("loss")
    ax3.legend(loc = "upper right")
    ax3.set_title("Growth Rate Loss")

    plt.tight_layout()
    plt.savefig(f"validation_results_figures/{initialization_to_string(initialization)}/losses_{initialization_to_string(initialization)}_{sample_number}.svg")

    plt.close("all")

def plot_all_validation_results(total_samples = 200):
    for initialization in initializations:
        for sample_number in range(1, total_samples + 1):
            plot_single_validation_result(initialization, sample_number)
            print(f"produced plot for {initialization_to_string(initialization)}_{sample_number}")

def plot_validation_loss_curves(total_samples = 200, figpath = "figures/loss_curves.svg"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5)) 
    axes = [ax1, ax2, ax3] 

    initialization_colors = {
        LINEAR_INITIALIZATION_TRAINED: "green",
        NEURAL_OPERATOR_INITIALIZATION: "red",
    }

    initialization_name_map = {
        LINEAR_INITIALIZATION_TRAINED: "linear initialization (pre-trained)",
        NEURAL_OPERATOR_INITIALIZATION: "neural operator initialization",
    }

    def find_strictly_smaller_than_previous_vectorized(arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return np.array([]), np.array([], dtype=int)
        cummin = np.minimum.accumulate(arr)
        shifted_cummin = np.roll(cummin, 1)
        shifted_cummin[0] = np.inf
        mask = arr < shifted_cummin
        return arr[mask], np.nonzero(mask)[0]

    losses_by_init_and_epoch = {
        init: {
            "reflectance": defaultdict(list),
            "thickness": defaultdict(list),
            "growth_rate": defaultdict(list),
        } for init in [LINEAR_INITIALIZATION_TRAINED, NEURAL_OPERATOR_INITIALIZATION]
    }

    for initialization in [LINEAR_INITIALIZATION_TRAINED, NEURAL_OPERATOR_INITIALIZATION]:
        for sample_number in range(1, total_samples + 1):
            file_path = f"validation_results_data/{initialization_to_string(initialization)}/{initialization_to_string(initialization)}_{sample_number}.npz"
            data = np.load(file_path) # Removed file existence check
            
            reflectance_losses = data["reflectance_losses"]
            thickness_losses = data["thickness_losses"]
            growth_rate_losses = data["growth_rate_losses"]

            ref_dec, epochs_ref_dec = find_strictly_smaller_than_previous_vectorized(reflectance_losses)
            th_dec, epochs_th_dec = find_strictly_smaller_than_previous_vectorized(thickness_losses)
            gr_dec, epochs_gr_dec = find_strictly_smaller_than_previous_vectorized(growth_rate_losses)

            failure_thickness = 100
            if np.min(thickness_losses) < failure_thickness: 
                for epoch, val in zip(epochs_ref_dec, ref_dec):
                    losses_by_init_and_epoch[initialization]["reflectance"][epoch].append(val)
                for epoch, val in zip(epochs_th_dec, th_dec):
                    losses_by_init_and_epoch[initialization]["thickness"][epoch].append(val)
                for epoch, val in zip(epochs_gr_dec, gr_dec):
                    losses_by_init_and_epoch[initialization]["growth_rate"][epoch].append(val)

            ax1.plot(epochs_ref_dec, ref_dec, color = initialization_colors[initialization], alpha = 0.03)
            ax2.plot(epochs_th_dec, th_dec, color = initialization_colors[initialization], alpha = 0.03)
            ax3.plot(epochs_gr_dec, gr_dec, color = initialization_colors[initialization], alpha = 0.03)

    for initialization in [LINEAR_INITIALIZATION_TRAINED, NEURAL_OPERATOR_INITIALIZATION]:
        color = initialization_colors[initialization]
        plot_label_for_median = initialization_to_label_string(initialization) 
        
        for loss_type_idx, (loss_type, ax) in enumerate(zip(["reflectance", "thickness", "growth_rate"], axes)):
            loss_dict = losses_by_init_and_epoch[initialization][loss_type]
            epochs = sorted(loss_dict.keys())
            # Removed 'if not epochs: continue'

            medians = [np.median(loss_dict[epoch]) for epoch in epochs]
            smoothing_window = 100
            # Removed conditional smoothing; always attempt convolve
            # If len(medians) < smoothing_window, convolve with mode='valid' returns empty, plot does nothing.
            smoothed_medians = np.convolve(medians, np.ones(smoothing_window)/smoothing_window, mode='valid')
            adjusted_epochs = epochs[smoothing_window-1:] # This will also be empty if smoothed_medians is empty

            ax.plot(adjusted_epochs, smoothed_medians, 
                    label=plot_label_for_median, 
                    color=color, linewidth=2)

    titles = ["Monotonized Reflectance Loss", "Monotonized Thickness Loss", "Monotonized Growth Rate Loss"]
    for i, ax in enumerate(axes):
        ax.set_yscale("log")
        ax.set_xlabel("epoch")
        ax.set_title(titles[i])
        ax.set_ylabel("mean squared error")

    legend_handles = []
    for init_type in [LINEAR_INITIALIZATION_TRAINED, NEURAL_OPERATOR_INITIALIZATION]:
        handle = mlines.Line2D([], [], 
                               color=initialization_colors[init_type], 
                               linewidth=2, 
                               label=f"loss curves after {initialization_name_map[init_type]}, smoothed median highlighted")
        legend_handles.append(handle)
    
    fig.subplots_adjust(bottom=0.20, wspace=0.20) 

    fig.legend(handles=legend_handles, 
               loc='lower center',
               bbox_to_anchor=(0.5, 0.03), 
               ncol=2, 
               fontsize='medium')

    # Removed output directory check/creation
    plt.savefig(
        figpath,
        bbox_inches='tight',
        pad_inches=0.02
    )
    print(f"Validation loss curves plot saved to {figpath}")
    # plt.show()

def loss_analysis_plot(total_samples = 200, figpath = "figures/losses_kde.svg"):
    # Define the sequence of initializations (was missing in the provided snippet)
    initializations_list = [
        RANDOM_INITIALIZATION,
        LINEAR_INITIALIZATION_SET,
        LINEAR_INITIALIZATION_TRAINED,
        NEURAL_OPERATOR_INITIALIZATION,
    ]

    reflectance_losses = {
        RANDOM_INITIALIZATION: [],
        LINEAR_INITIALIZATION_SET: [],
        LINEAR_INITIALIZATION_TRAINED: [],
        NEURAL_OPERATOR_INITIALIZATION: [],
    }

    thickness_losses = {
        RANDOM_INITIALIZATION: [],
        LINEAR_INITIALIZATION_SET: [],
        LINEAR_INITIALIZATION_TRAINED: [],
        NEURAL_OPERATOR_INITIALIZATION: [],
    }

    growth_rate_losses = {
        RANDOM_INITIALIZATION: [],
        LINEAR_INITIALIZATION_SET: [],
        LINEAR_INITIALIZATION_TRAINED: [],
        NEURAL_OPERATOR_INITIALIZATION: [],
    }

    # This dict seems unused in the plotting part of the snippet, but kept as is
    first_epoch_with_growth_rate_loss_below = {
        RANDOM_INITIALIZATION: [],
        LINEAR_INITIALIZATION_SET: [],
        LINEAR_INITIALIZATION_TRAINED: [],
        NEURAL_OPERATOR_INITIALIZATION: [],
    }

    neural_operator_initial_losses_reflectance = []
    neural_operator_initial_losses_thickness = []
    neural_operator_initial_losses_growth_rate = []

    growth_rate_loss_barrier = 5e2 # Unused in plotting part, kept as is

    for initialization in initializations_list: # Use the defined list
        for sample_number in range(1, total_samples + 1):
            # Ensure validation_results_data directory exists or handle FileNotFoundError
            file_path = f"validation_results_data/{initialization_to_string(initialization)}/{initialization_to_string(initialization)}_{sample_number}.npz"
            try:
                data = np.load(file_path)
            except FileNotFoundError:
                print(f"Warning: File not found {file_path}, skipping sample.")
                continue


            reflectance_losses[initialization].append(data["reflectance_loss"])
            thickness_losses[initialization].append(data["thickness_loss"])
            growth_rate_losses[initialization].append(data["growth_rate_loss"])

            if initialization == NEURAL_OPERATOR_INITIALIZATION:
                neural_operator_initial_losses_reflectance.append(data["initial_reflectance_loss"])
                neural_operator_initial_losses_thickness.append(data["initial_thickness_loss"])
                neural_operator_initial_losses_growth_rate.append(data["initial_growth_rate_loss"])
            
            # Logic for first_epoch_with_growth_rate_loss_below (kept as is)
            if "growth_rate_losses" in data: # Check if key exists
                lower_than_barrier = data["growth_rate_losses"] < growth_rate_loss_barrier
                if np.any(lower_than_barrier):
                    first_epoch = np.argmax(lower_than_barrier)
                else:
                    first_epoch = len(data["growth_rate_losses"])
                first_epoch_with_growth_rate_loss_below[initialization].append(first_epoch)
            else: # Handle cases where 'growth_rate_losses' might be missing
                first_epoch_with_growth_rate_loss_below[initialization].append(np.nan) # Or some other placeholder


        reflectance_losses[initialization] = np.array(reflectance_losses[initialization])
        thickness_losses[initialization] = np.array(thickness_losses[initialization])
        growth_rate_losses[initialization] = np.array(growth_rate_losses[initialization])
        
        num_samples_above_100 = np.sum(thickness_losses[initialization] > 100)
        if total_samples > 0 : # Avoid division by zero if no samples loaded
            fraction_samples_above_100 = num_samples_above_100 / len(thickness_losses[initialization]) if len(thickness_losses[initialization]) > 0 else 0
        else:
            fraction_samples_above_100 = 0
        print(f"Initialization {initialization_to_string(initialization)}: {num_samples_above_100} samples above 100 thicknes MSE, fraction: {fraction_samples_above_100:.2f}")

    initialization_colors = {
        RANDOM_INITIALIZATION: "blue",
        LINEAR_INITIALIZATION_SET: "orange",
        LINEAR_INITIALIZATION_TRAINED: "green",
        NEURAL_OPERATOR_INITIALIZATION: "red",
    }
    
    # Labels for the legend
    legend_labels = {
        RANDOM_INITIALIZATION: "optimization result after random initialization",
        LINEAR_INITIALIZATION_SET: "optimization result after linear initialization (set)",
        LINEAR_INITIALIZATION_TRAINED: "optimization result after linear initialization (pre-trained)",
        NEURAL_OPERATOR_INITIALIZATION: "optimization result after neural operator initialization",
        "NEURAL_OPERATOR_GUESS": "neural operator initial guess"
    }

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))  # 1 row, 3 columns

    loss_dicts = [reflectance_losses, thickness_losses, growth_rate_losses]
    initial_loss_lists = [
        neural_operator_initial_losses_reflectance,
        neural_operator_initial_losses_thickness,
        neural_operator_initial_losses_growth_rate
    ]
    titles = ["Reflectance Loss", "Thickness Loss", "Growth Rate Loss"]
    x_labels_list = ["mean squared reflectance error (unitless)",
                     "mean squared thickness error in nm²",
                     "mean squared growth rate error in nm²/h²"]

    for i, ax in enumerate(axes):
        loss_dict = loss_dicts[i]
        title = titles[i]
        initial_losses_for_plot = initial_loss_lists[i]

        for init_type in initializations_list:
            data = np.array(loss_dict[init_type])
            data = data[data > 0]  # log scale requires positive values
            if len(data) > 0: # Check if data is not empty after filtering
                sns.kdeplot(data, ax=ax, label=legend_labels[init_type], bw_adjust = 0.5, log_scale = True, color = initialization_colors[init_type])
                sns.rugplot(data, ax=ax, height = 0.03, lw = 0.1, alpha = 0.5, color = initialization_colors[init_type])

        # Add the neural operator initial losses
        initial_data = np.array(initial_losses_for_plot)
        initial_data = initial_data[initial_data > 0]
        if len(initial_data) > 0:
            sns.kdeplot(
                initial_data,
                ax=ax,
                label=legend_labels["NEURAL_OPERATOR_GUESS"],
                bw_adjust=0.5,
                log_scale=True,
                color="black",
            )
            sns.rugplot(
                initial_data,
                ax=ax,
                height=0.03,
                lw=0.1,
                alpha=0.5,
                color="black",
            )

        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_xlabel(x_labels_list[i])
        # if i == 0:
        ax.set_ylabel("probability density")
        # else:
        #     ax.set_yticklabels([]) # Hide y-tick labels for middle and right plots

    # Set specific xlims (as in original)
    axes[1].set_xlim(right=1e8)
    axes[2].set_xlim(right=1e8)

    axes[0].set_ylim(top=1.1)
    axes[1].set_ylim(top=1.0)
    axes[2].set_ylim(top=1.5)

    # --- Common Legend (Manually Created for 1,2,2 column structure) ---
    h_no_guess = mlines.Line2D([], [], color='black', label=legend_labels["NEURAL_OPERATOR_GUESS"])
    h_opt_random = mlines.Line2D([], [], color=initialization_colors[RANDOM_INITIALIZATION], label=legend_labels[RANDOM_INITIALIZATION])
    h_opt_linear_set = mlines.Line2D([], [], color=initialization_colors[LINEAR_INITIALIZATION_SET], label=legend_labels[LINEAR_INITIALIZATION_SET])
    h_opt_linear_trained = mlines.Line2D([], [], color=initialization_colors[LINEAR_INITIALIZATION_TRAINED], label=legend_labels[LINEAR_INITIALIZATION_TRAINED])
    h_opt_no_init = mlines.Line2D([], [], color=initialization_colors[NEURAL_OPERATOR_INITIALIZATION], label=legend_labels[NEURAL_OPERATOR_INITIALIZATION])
    
    h_dummy = mlines.Line2D([], [], color='none', marker='None', linestyle='None', label=' ') 

    legend_handles_ordered = [
        h_no_guess,
        h_dummy, 
        h_opt_random,   
        h_opt_linear_set,
        h_opt_linear_trained,
        h_opt_no_init
    ]
    
    fig.subplots_adjust(bottom=0.25, wspace=0.25) # Adjust bottom for legend and wspace

    fig.legend(handles=legend_handles_ordered, 
               loc='lower center',
               bbox_to_anchor=(0.5, 0.03), 
               ncol=3, 
               fontsize='medium')

        
    plt.savefig(
        figpath,
        bbox_inches='tight',
        pad_inches=0.02
    )
    print(f"KDE plot saved to {figpath}")