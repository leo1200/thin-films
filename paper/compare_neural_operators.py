import os

import yaml

from reflax.neural_operators.models import AbstractSurrogate
from reflax.neural_operators.training import load_dataset

from .plot_neural_operator_comparison import plot_combined_loss_and_mse_kde


def neural_operator_comparison():
    cfg = yaml.safe_load(open("config.yaml"))
    ds = cfg["dataset_name"]
    tid = cfg.get("training_id", ds)
    device = cfg.get("device", "cpu")
    archs = cfg["architectures"]
    batches = cfg["batch_sizes"]

    _, _, _, _, val_in, val_out = load_dataset(ds)

    histories = {}
    report_evers = {}
    preds_dict = {}
    targets_dict = {}

    for name, batch in zip(archs, batches):
        print(f"→ Evaluating {name}")

        model = AbstractSurrogate.load(
            surrogate_name=name,
            dataset_name=ds,
            training_id=tid,
            device=device,
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

        out_dir = os.path.join("results", tid, name)
        os.makedirs(out_dir, exist_ok=True)

    comp_dir = os.path.join("results", tid, "comparative")
    os.makedirs(comp_dir, exist_ok=True)

    x_all = {
        name: [(i + 1) * report_evers[name] for i in range(len(histories[name]))]
        for name in archs
    }

    plot_combined_loss_and_mse_kde(x_all, histories, preds_dict, targets_dict, comp_dir)
    print("→ Evaluation complete!")
