from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt

import seaborn as sns

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
    


def produce_plot(initialization, sample_number):

    data = np.load(f"result_data/{initialization_to_string(initialization)}/{initialization_to_string(initialization)}_{sample_number}.npz")

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

    plt.savefig(f"result_figures/{initialization_to_string(initialization)}/predictions_{initialization_to_string(initialization)}_{sample_number}.svg")
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
    plt.savefig(f"result_figures/{initialization_to_string(initialization)}/losses_{initialization_to_string(initialization)}_{sample_number}.svg")

    plt.close("all")

def plot_all_validation_results(total_samples = 200):
    for initialization in initializations:
        for sample_number in range(1, total_samples + 1):
            produce_plot(initialization, sample_number)
            print(f"produced plot for {initialization_to_string(initialization)}_{sample_number}")


def plot_validation_loss_curves(total_samples = 200, figpath = "figures/all_losses.svg"):
    # create a figure for all losses
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(8, 10))

    initialization_colors = {
        RANDOM_INITIALIZATION: "blue",
        LINEAR_INITIALIZATION_SET: "orange",
        LINEAR_INITIALIZATION_TRAINED: "green",
        NEURAL_OPERATOR_INITIALIZATION: "red",
    }

    def find_strictly_smaller_than_previous_vectorized(arr):
        arr = np.asarray(arr)
        if arr.size == 0:
            return np.array([]), np.array([], dtype=int)

        # Compute the cumulative minimum from the left
        cummin = np.minimum.accumulate(arr)

        # Create a shifted version of the cumulative min to compare against
        shifted_cummin = np.roll(cummin, 1)
        shifted_cummin[0] = np.inf  # First element always qualifies

        # Compare each element to the minimum of all previous elements
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
            data = np.load(f"result_data/{initialization_to_string(initialization)}/{initialization_to_string(initialization)}_{sample_number}.npz")
            
            reflectance_losses = data["reflectance_losses"]
            thickness_losses = data["thickness_losses"]
            growth_rate_losses = data["growth_rate_losses"]

            ref_dec, epochs_ref_dec = find_strictly_smaller_than_previous_vectorized(reflectance_losses)
            th_dec, epochs_th_dec = find_strictly_smaller_than_previous_vectorized(thickness_losses)
            gr_dec, epochs_gr_dec = find_strictly_smaller_than_previous_vectorized(growth_rate_losses)

            for epoch, val in zip(epochs_ref_dec, ref_dec):
                losses_by_init_and_epoch[initialization]["reflectance"][epoch].append(val)
            for epoch, val in zip(epochs_th_dec, th_dec):
                losses_by_init_and_epoch[initialization]["thickness"][epoch].append(val)
            for epoch, val in zip(epochs_gr_dec, gr_dec):
                losses_by_init_and_epoch[initialization]["growth_rate"][epoch].append(val)


            # only plot the epochs where the loss is better than all previous epochs

            ax1.plot(epochs_ref_dec, ref_dec, color = initialization_colors[initialization], alpha = 0.03)
            ax2.plot(epochs_th_dec, th_dec, color = initialization_colors[initialization], alpha = 0.03)
            ax3.plot(epochs_gr_dec, gr_dec, color = initialization_colors[initialization], alpha = 0.03)


    for initialization in [LINEAR_INITIALIZATION_TRAINED, NEURAL_OPERATOR_INITIALIZATION]:
        color = initialization_colors[initialization]
        
        for loss_type, ax in zip(["reflectance", "thickness", "growth_rate"], [ax1, ax2, ax3]):
            loss_dict = losses_by_init_and_epoch[initialization][loss_type]
            epochs = sorted(loss_dict.keys())
            medians = [np.median(loss_dict[epoch]) for epoch in epochs]
            # apply a moving average to the medians
            smoothing_window = 100
            smoothed_medians = np.convolve(medians, np.ones(smoothing_window)/smoothing_window, mode='valid')
            # adjust the epochs to match the length of the smoothed medians
            adjusted_epochs = epochs[smoothing_window-1:]
            # plot the smoothed medians
            ax.plot(adjusted_epochs, smoothed_medians, label=f"smoothed median ({initialization_to_label_string(initialization)})", color=color, linewidth=2)

    # manually add the legend
    ax1.plot([], [], color = initialization_colors[LINEAR_INITIALIZATION_TRAINED], label = initialization_to_label_string(LINEAR_INITIALIZATION_TRAINED))
    ax1.plot([], [], color = initialization_colors[NEURAL_OPERATOR_INITIALIZATION], label = initialization_to_label_string(NEURAL_OPERATOR_INITIALIZATION))

    ax2.plot([], [], color = initialization_colors[LINEAR_INITIALIZATION_TRAINED], label = initialization_to_label_string(LINEAR_INITIALIZATION_TRAINED))
    ax2.plot([], [], color = initialization_colors[NEURAL_OPERATOR_INITIALIZATION], label = initialization_to_label_string(NEURAL_OPERATOR_INITIALIZATION))

    ax3.plot([], [], color = initialization_colors[LINEAR_INITIALIZATION_TRAINED], label = initialization_to_label_string(LINEAR_INITIALIZATION_TRAINED))
    ax3.plot([], [], color = initialization_colors[NEURAL_OPERATOR_INITIALIZATION], label = initialization_to_label_string(NEURAL_OPERATOR_INITIALIZATION))

    # set the y axis to log scale
    ax1.set_yscale("log")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("mean squared error")
    ax1.legend(loc = "upper right")
    ax1.set_title("monotonized reflectance loss")


    # set the y axis to log scale
    ax2.set_yscale("log")
    ax2.set_xlabel("epoch")
    ax2.set_ylabel("mean squared error")
    ax2.legend(loc = "upper right")
    ax2.set_title("monotonized thickness loss")

    # set the y axis to log scale
    ax3.set_yscale("log")
    ax3.set_xlabel("epoch")
    ax3.set_ylabel("mean squared error")
    ax3.legend(loc = "upper right")
    ax3.set_title("monotonized growth rate loss")

    plt.tight_layout()

    plt.savefig(figpath)

def kde_loss_plot(total_samples = 200, figpath = "figures/losses_kde.svg"):
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

    first_epoch_with_growth_rate_loss_below = {
        RANDOM_INITIALIZATION: [],
        LINEAR_INITIALIZATION_SET: [],
        LINEAR_INITIALIZATION_TRAINED: [],
        NEURAL_OPERATOR_INITIALIZATION: [],
    }

    neural_operator_initial_losses_reflectance = []
    neural_operator_initial_losses_thickness = []
    neural_operator_initial_losses_growth_rate = []

    growth_rate_loss_barrier = 5e2

    for initialization in initializations:
        for sample_number in range(1, total_samples + 1):
            data = np.load(f"result_data/{initialization_to_string(initialization)}/{initialization_to_string(initialization)}_{sample_number}.npz")

            reflectance_losses[initialization].append(data["reflectance_loss"])
            thickness_losses[initialization].append(data["thickness_loss"])
            growth_rate_losses[initialization].append(data["growth_rate_loss"])

            if initialization == NEURAL_OPERATOR_INITIALIZATION:
                neural_operator_initial_losses_reflectance.append(data["initial_reflectance_loss"])
                neural_operator_initial_losses_thickness.append(data["initial_thickness_loss"])
                neural_operator_initial_losses_growth_rate.append(data["initial_growth_rate_loss"])

            # find the first epoch (so the first index) with growth rate loss below the barrier
            lower_than_barrier = data["growth_rate_losses"] < growth_rate_loss_barrier

            if np.any(lower_than_barrier):
                first_epoch = np.argmax(lower_than_barrier)
            else:
                first_epoch = len(data["growth_rate_losses"])
                    
            first_epoch_with_growth_rate_loss_below[initialization].append(
                first_epoch
            )
        
        # for the initialization, plot the number of MSE thickness losses above 100
        # and their fraction of the total number of samples
        reflectance_losses[initialization] = np.array(reflectance_losses[initialization])
        thickness_losses[initialization] = np.array(thickness_losses[initialization])
        growth_rate_losses[initialization] = np.array(growth_rate_losses[initialization])
        # print the number of samples with thickness loss above 100
        num_samples_above_100 = np.sum(thickness_losses[initialization] > 100)
        # print the fraction of samples with thickness loss above 100
        fraction_samples_above_100 = num_samples_above_100 / total_samples
        print(f"Initialization {initialization_to_string(initialization)}: {num_samples_above_100} samples above 100 thicknes MSE, fraction: {fraction_samples_above_100:.2f}")


    # # Example: KDE plot of growth_rate_losses with log scale on x-axis
    # plt.figure(figsize=(6, 4))

    # for initialization in initializations:
    #     losses = np.array(growth_rate_losses[initialization])
    #     # Filter out zeros or negative values since log-scale can't handle them
    #     losses = losses[losses > 0]
    #     sns.kdeplot(losses, label=initialization_to_string(initialization), log_scale=True)

    # plt.xlabel("Growth Rate Loss (log scale)")
    # plt.ylabel("Density")
    # plt.title("KDE of Growth Rate Losses")
    # plt.legend()
    # plt.tight_layout()
    # plt.savefig("figures/growth_rate_losses_kde.svg")

    initialization_colors = {
        RANDOM_INITIALIZATION: "blue",
        LINEAR_INITIALIZATION_SET: "orange",
        LINEAR_INITIALIZATION_TRAINED: "green",
        NEURAL_OPERATOR_INITIALIZATION: "red",
    }


    fig, axes = plt.subplots(3, 1, figsize=(6, 9))  # 3 rows, 1 column

    loss_dicts = [reflectance_losses, thickness_losses, growth_rate_losses]
    titles = ["Reflectance Loss", "Thickness Loss", "Growth Rate Loss"]

    for ax, loss_dict, title in zip(axes, loss_dicts, titles):
        for init in initializations:
            data = np.array(loss_dict[init])
            data = data[data > 0]  # log scale requires positive values
            sns.kdeplot(data, ax=ax, label=initialization_to_label_string(init), bw_adjust = 0.5, log_scale = True, color = initialization_colors[init])
            sns.rugplot(data, ax=ax, height = 0.03, lw = 0.1, alpha = 0.5, color = initialization_colors[init])

    # add the neural operator initial losses
    sns.kdeplot(
        np.array(neural_operator_initial_losses_reflectance),
        ax=axes[0],
        label="neural operator guess loss",
        bw_adjust=0.5,
        log_scale=True,
        color="black",
    )
    sns.kdeplot(
        np.array(neural_operator_initial_losses_thickness),
        ax=axes[1],
        label="neural operator guess loss",
        bw_adjust=0.5,
        log_scale=True,
        color="black",
    )
    sns.kdeplot(
        np.array(neural_operator_initial_losses_growth_rate),
        ax=axes[2],
        label="neural operator guess loss",
        bw_adjust=0.5,
        log_scale=True,
        color="black",
    )
    # add the rugplot for the neural operator initial losses
    sns.rugplot(
        np.array(neural_operator_initial_losses_reflectance),
        ax=axes[0],
        height=0.03,
        lw=0.1,
        alpha=0.5,
        color="black",
    )
    sns.rugplot(
        np.array(neural_operator_initial_losses_thickness),
        ax=axes[1],
        height=0.03,
        lw=0.1,
        alpha=0.5,
        color="black",
    )
    sns.rugplot(
        np.array(neural_operator_initial_losses_growth_rate),
        ax=axes[2],
        height=0.03,
        lw=0.1,
        alpha=0.5,
        color="black",
    )

    for ax, loss_dict, title in zip(axes, loss_dicts, titles):
        ax.set_xscale("log")
        ax.set_title(title)
        ax.set_ylabel("probability density")
        ax.legend()

    axes[0].set_xlabel("mean squared reflectance error (unitless)")
    axes[1].set_xlabel("mean squared thickness error in nm²")
    axes[2].set_xlabel("mean squared growth rate error in nm²/h²")


    # set upper x limit of ax 1 and 2 to 10^8
    axes[1].set_xlim(right=1e8)
    axes[2].set_xlim(right=1e8)

    plt.tight_layout()
    plt.savefig(figpath)