import os

import numpy as np
import torch
import torch.nn as nn


class AbstractSurrogate(nn.Module):
    @staticmethod
    def get_activation(name: str) -> nn.Module:
        """
        Return an activation layer by name from torch.nn.
        """
        try:
            act_cls = getattr(nn, name)
            return act_cls()
        except AttributeError:
            raise ValueError(f"Unknown activation: {name}")

    def fit_normalization(
        self, train_inputs: np.ndarray, train_outputs: np.ndarray
    ) -> None:
        """
        Compute and store global min/max for inputs and outputs,
        using only the training data.
        """
        self.in_min, self.in_max = float(train_inputs.min()), float(train_inputs.max())
        self.out_min, self.out_max = float(train_outputs.min()), float(
            train_outputs.max()
        )
        if self.in_max == self.in_min:
            self.in_max = self.in_min + 1.0
        if self.out_max == self.out_min:
            self.out_max = self.out_min + 1.0

    def normalize(self, data, min_val: float, max_val: float):
        """
        Min-max normalize a single array or tuple of arrays using provided bounds.

        Args:
            data: np.ndarray, torch.Tensor, or tuple of these
            min_val: scalar minimum
            max_val: scalar maximum
        Returns:
            same type as input, normalized to [0,1]
        """

        def _norm(x):
            span = max_val - min_val
            if span == 0.0:
                span = 1.0
            return (x - min_val) / span

        if isinstance(data, tuple):
            return tuple(_norm(x) for x in data)
        else:
            return _norm(data)

    def denormalize(self, data, min_val: float, max_val: float):
        """
        Invert min-max scaling for a single array or tuple of arrays.

        Args:
            data: np.ndarray, torch.Tensor, or tuple of these
            min_val: scalar minimum used during normalization
            max_val: scalar maximum used during normalization
        Returns:
            same type as input, rescaled to original bounds
        """

        def _denorm(x):
            span = max_val - min_val
            if span == 0.0:
                span = 1.0
            return x * span + min_val

        if isinstance(data, tuple):
            return tuple(_denorm(x) for x in data)
        else:
            return _denorm(data)

    def save(self, model_name: str, training_id: str, base_dir: str):
        """
        Save model weights + normalization coeffs to:
            <base_dir>/<training_id>/<model_name>.pth
        """
        save_dir = os.path.join(base_dir, training_id)
        os.makedirs(save_dir, exist_ok=True)
        path = os.path.join(save_dir, f"{model_name}.pth")

        payload = {
            "model_state": self.state_dict(),
            "in_min": self.in_min,
            "in_max": self.in_max,
            "out_min": self.out_min,
            "out_max": self.out_max,
            "n_train": self.n_train_samples,
            "report_every": self.report_every,
        }

        # include histories if they were set
        if hasattr(self, "train_hist"):
            payload["train_hist"] = self.train_hist
        if hasattr(self, "test_hist"):
            payload["test_hist"] = self.test_hist

        torch.save(payload, path)

    @classmethod
    def load(
        cls,
        surrogate_name: str,
        dataset_name: str,
        training_id: str,
        device: str = "cpu",
    ):
        """
        Instantiate, load weights + norm-coeffs, and return a ready-to-use model.
        """
        import numpy as np

        from .registry import get_model_config, get_surrogate

        # infer n_evals from the training split
        arr = np.load(f"data/{dataset_name}/train.npz")
        n_evals = arr["reflectances"].shape[1]
        arr.close()

        # build empty model
        SurClass = get_surrogate(surrogate_name)
        cfg = get_model_config(surrogate_name)
        model = SurClass(device=device, n_evals=n_evals, config=cfg)

        # load everything
        path = os.path.join(
            "trained", training_id, f"{surrogate_name.lower()}_{dataset_name}.pth"
        )
        ckpt = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(ckpt["model_state"])

        # restore normalization
        model.in_min = ckpt["in_min"]
        model.in_max = ckpt["in_max"]
        model.out_min = ckpt["out_min"]
        model.out_max = ckpt["out_max"]
        model.n_train_samples = ckpt["n_train"]
        model.report_every = ckpt["report_every"]
        if "train_hist" in ckpt:
            model.train_hist = ckpt["train_hist"]
        if "test_hist" in ckpt:
            model.test_hist = ckpt["test_hist"]

        # set to eval mode
        model.eval()
        model.to(device)

        return model
