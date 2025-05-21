import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib import gridspec


def plot_validation_and_train_data(figpath="figures/train_val_data.png"):
    # Load the training data
    data_train = np.load("simulated_data/training_data.npz")
    thicknesses_train = data_train["thicknesses"]  # [100:, :]
    derivatives_train = data_train["derivatives"]  # [100:, :]
    reflectances_train = data_train["reflectances"]  # [100:, :]
    time_points_train = data_train["time_points"]

    # Load the validation data
    data_val = np.load("simulated_data/validation_data.npz")
    thicknesses_val = data_val["thicknesses"]
    derivatives_val = data_val["derivatives"]
    reflectances_val = data_val["reflectances"]
    time_points_val = data_val["time_points"]

    fig = plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(6, 2, figure=fig)

    ax1 = fig.add_subplot(gs[0:2, 0])  # top third
    ax2 = fig.add_subplot(gs[2:4, 0])  # middle third
    ax3 = fig.add_subplot(gs[4:6, 0])  # bottom third

    ax4 = fig.add_subplot(gs[0:3, 1])
    ax5 = fig.add_subplot(gs[3:6, 1])

    example_index_train = 1024
    example_index_val = 69
    train_color = "blue"
    val_color = "orange"

    # Plot reflectance
    ax1.plot(time_points_train, reflectances_train.T, alpha=0.002, color=train_color)
    ax1.plot(time_points_val, reflectances_val.T, alpha=0.03, color=val_color)
    ax1.plot(
        time_points_train,
        reflectances_train[example_index_train],
        alpha=1.0,
        label="training data",
        color=train_color,
    )
    ax1.plot(
        time_points_val,
        reflectances_val[example_index_val],
        alpha=1.0,
        label="validation data",
        color=val_color,
    )
    ax1.legend(loc="upper right")
    ax1.set_title("reflectance")
    ax1.set_ylabel("normalized reflectance")

    # Plot thickness
    ax2.plot(time_points_train, thicknesses_train.T, alpha=0.002, color=train_color)
    ax2.plot(time_points_val, thicknesses_val.T, alpha=0.03, color=val_color)
    ax2.plot(
        time_points_train,
        thicknesses_train[example_index_train],
        alpha=1.0,
        label="training data",
        color=train_color,
    )
    ax2.plot(
        time_points_val,
        thicknesses_val[example_index_val],
        alpha=1.0,
        label="validation data",
        color=val_color,
    )
    ax2.legend(loc="lower right")
    ax2.set_title("thicknesses")
    ax2.set_ylabel("thickness in nm")

    # Plot first derivative
    ax3.plot(time_points_train, derivatives_train.T, alpha=0.002, color=train_color)
    ax3.plot(time_points_val, derivatives_val.T, alpha=0.03, color=val_color)
    ax3.plot(
        time_points_train,
        derivatives_train[example_index_train],
        alpha=1.0,
        label="training data",
        color=train_color,
    )
    ax3.plot(
        time_points_val,
        derivatives_val[example_index_val],
        alpha=1.0,
        label="validation data",
        color=val_color,
    )
    ax3.legend(loc="lower right")
    ax3.set_title("growth rate")
    ax3.set_xlabel("time in hours")
    ax3.set_ylabel("growth rate in nm/h")

    # Compute second derivative numerically
    dt_train = np.mean(np.diff(time_points_train))
    dt_val = np.mean(np.diff(time_points_val))
    second_deriv_train = np.gradient(derivatives_train, dt_train, axis=1)
    second_deriv_val = np.gradient(derivatives_val, dt_val, axis=1)

    # --- ax4: KDE of first derivatives ---
    sns.kdeplot(
        np.std(derivatives_train, axis=1),
        ax=ax4,
        label="training",
        color=train_color,
        fill=True,
        alpha=0.4,
        linewidth=1,
    )
    sns.kdeplot(
        np.std(derivatives_val, axis=1),
        ax=ax4,
        label="validation",
        color=val_color,
        fill=True,
        alpha=0.4,
        linewidth=1,
    )
    ax4.set_title("KDE of std(growth rates)")
    ax4.set_xlabel("growth rate std in nm/h")
    ax4.set_ylabel("density")
    ax4.legend(loc="lower right")

    # --- ax5: KDE of second derivatives with log y-scale ---
    dt_train = np.mean(np.diff(time_points_train))
    dt_val = np.mean(np.diff(time_points_val))
    step = int(round(dt_train / dt_val))
    sns.kdeplot(
        np.std(second_deriv_train, axis=1),
        ax=ax5,
        label="training",
        color=train_color,
        fill=True,
        alpha=0.4,
        linewidth=1,
    )
    sns.kdeplot(
        np.std(second_deriv_val, axis=1),
        ax=ax5,
        label="validation",
        color=val_color,
        fill=True,
        alpha=0.4,
        linewidth=1,
    )

    ax5.set_title("KDE of std(growth acceleration)")
    # # xlim 0 to 5000
    # ax5.set_xlim(0, 5000)
    # # y lim til 0.0005
    # ax5.set_ylim(0, 0.0005)
    ax5.set_xlabel("acceleration std in nm/h²")
    ax5.set_ylabel("density")
    ax5.legend(loc="upper right")

    # fig.subplots_adjust(left=0.08, right=0.97, top=0.95, bottom=0.08, hspace=0.6, wspace=0.35)
    plt.tight_layout()

    plt.savefig(figpath, dpi=300)
