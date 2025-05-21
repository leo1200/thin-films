import os

import torch
from plot_neural_operator_comparison import plot_combined_loss_and_mse_kde

from reflax.neural_operators.models import AbstractSurrogate
from reflax.neural_operators.training import load_dataset

CFG = {
    "archs": [
        "direct_fcnn",
        "direct_fcnn_plain",
        "deeponet",
        "operator_fcnn",
    ],
    "batch_sizes": [256, 256, 16384, 16384],
}


def neural_operator_comparison():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    archs = CFG["archs"]
    batches = CFG["batch_sizes"]

    # We use the full validation set for evaluation
    _, _, _, _, val_in, val_out = load_dataset(
        filepath="simulated_data/validation_data.npz", splits=(0.0, 0.0, 1.0)
    )

    # Downsample the validation set to the 100 timestepds used for training
    val_in = val_in[:, ::4]
    val_out = val_out[:, ::4]

    histories = {}
    report_evers = {}
    preds_dict = {}
    targets_dict = {}

    n_evals = val_in.shape[1]

    for name, batch in zip(archs, batches):
        print(f"→ Evaluating {name}s")

        model = AbstractSurrogate.load(
            surrogate_name=name,
            device=device,
            n_evals=n_evals,
            path=os.path.join(os.getcwd(), "neural_operator_models", f"{name}.pth"),
        )

        histories[name] = getattr(model, "test_hist", [])
        report_evers[name] = getattr(model, "report_every", 1)

        val_in_n = model.normalize(val_in, model.in_min, model.in_max)
        val_out_n = model.normalize(val_out, model.out_min, model.out_max)
        _, _, val_loader = model.prepare_data(
            dataset_train=(val_in_n, val_out_n),
            dataset_test=(val_in_n, val_out_n),
            dataset_val=(val_in_n, val_out_n),
            batch_size=batch,
            shuffle=False,
        )

        preds_n, targets_n = model.predict(val_loader)
        preds, targets = model.denormalize(
            (preds_n, targets_n), model.out_min, model.out_max
        )

        preds_dict[name] = preds
        targets_dict[name] = targets

    figpath = "figures/neural_operator_comparison.svg"

    x_all = {
        name: [(i + 1) * report_evers[name] for i in range(len(histories[name]))]
        for name in archs
    }

    plot_combined_loss_and_mse_kde(x_all, histories, preds_dict, targets_dict, figpath)
    print("→ Evaluation complete!")
