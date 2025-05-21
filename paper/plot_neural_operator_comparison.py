import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.lines import Line2D

sns.set_theme(style="whitegrid", context="paper", font_scale=1.1)

LABEL_MAP = {
    "direct_fcnn": "FCNN (tuned)",
    "direct_fcnn_plain": "FCNN",
    "deeponet": "DeepONet",
    "operator_fcnn": "Operator-FCNN",
}


def get_label(name: str) -> str:
    return LABEL_MAP.get(name, name)


def plot_combined_loss_and_mse_kde(
    x_all: dict[str, list],
    histories: dict[str, list],
    preds_dict: dict[str, np.ndarray],
    targets_dict: dict[str, np.ndarray],
    fig_path: str,
):

    sns.set_theme(style="ticks", context="paper", font_scale=1.1)

    models = list(histories.keys())
    fig, (ax0, ax1) = plt.subplots(
        1,
        2,
        figsize=(10, 4),
        gridspec_kw={"width_ratios": [1, 1]},
        constrained_layout=True,
    )

    # Left: loss curves
    for i, name in enumerate(models):
        hist = histories[name]
        xs = x_all.get(name, [])
        if hist and xs:
            ax0.plot(xs, hist, lw=2, label=get_label(name), color=f"C{i}")
    ax0.set_yscale("log")
    ax0.set_xlabel("epoch")
    ax0.set_ylabel("test MSE")
    ax0.set_title("Neural Operator MSE Loss")
    ax0.grid(False)
    if ax0.get_legend():
        ax0.get_legend().remove()
    ax0.minorticks_off()
    means = {}

    # Right: MSE KDE + rug + mean lines
    for i, name in enumerate(models):
        preds = preds_dict[name]
        targs = targets_dict[name]
        mse = ((preds - targs) ** 2).mean(axis=1)
        data = mse[mse > 0]
        # KDE
        sns.kdeplot(
            data=data,
            ax=ax1,
            bw_adjust=0.5,
            common_norm=False,
            fill=False,
            lw=2,
            label=get_label(name),
            log_scale=True,
            color=f"C{i}",
        )
        # rug
        sns.rugplot(
            data=data,
            ax=ax1,
            height=0.02,
            lw=0.5,
            alpha=0.3,
            color=f"C{i}",
        )
        # vertical mean with value in legend
        m = data.mean()
        means[name] = m
        ax1.axvline(
            m,
            color=f"C{i}",
            linestyle="--",
            linewidth=2,
            label=f"{get_label(name)} mean = {m:.1e}",
        )

    ax1.set_xscale("log")
    ax1.set_xlabel("per-sample MSE (nm²)")
    ax1.set_ylabel("probability density")
    ax1.set_title("KDE of sample-wise MSE")
    ax1.tick_params(
        axis="both",  # apply to both x and y axes
        direction="out",  # draw ticks outside the plot
        length=6,  # tick length in points
        width=1,  # tick width in points
        pad=4,  # distance to label
        top=False,  # disable ticks on top edge
        right=False,  # disable ticks on right edge
    )
    ax1.grid(False)
    ax1.minorticks_off()
    ax1.set_xlim(3e-1, 5e3)
    ax1.set_ylim(0, 1.08)

    if ax1.get_legend():
        ax1.get_legend().remove()

    # --- Shared legend: first all model curves, then all mean lines ---
    legend_handles = []

    for i, name in enumerate(models):
        color = f"C{i}"
        # solid line for the model
        legend_handles.append(
            Line2D([], [], color=color, linewidth=2, label=get_label(name))
        )
        # dashed line for that model's mean
        legend_handles.append(
            Line2D(
                [],
                [],
                color=color,
                linestyle="--",
                linewidth=1.5,
                label=f"mean = {means[name]:.2e}",
            )
        )

    # And labels come along from each handle
    legend_labels = [h.get_label() for h in legend_handles]

    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        ncol=4,
        frameon=False,
        bbox_to_anchor=(0.5, -0.15),
    )

    fig.savefig(fig_path, bbox_inches="tight", dpi=300)
    plt.close(fig)
