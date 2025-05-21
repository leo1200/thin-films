# import os

import numpy as np
import torch
import torch.nn as nn
from optuna.exceptions import TrialPruned
from schedulefree import AdamWScheduleFree
from torch.utils.data import IterableDataset
from tqdm import tqdm

from .abstract import AbstractSurrogate
from .hyperparameters import OperatorFCNNConfig


class PreBatchedDataset(IterableDataset):
    """
    Iterable dataset that precomputes batches of (input, target) tensors.
    """

    def __init__(
        self,
        inputs: torch.Tensor,
        targets: torch.Tensor,
        batch_size: int,
        shuffle: bool = True,
    ):
        super().__init__()
        self.inputs = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._make_batches()

    def _make_batches(self):
        N = self.inputs.size(0)
        if self.shuffle:
            perm = torch.randperm(N)
        else:
            perm = torch.arange(N)
        inputs = self.inputs[perm]
        targets = self.targets[perm]
        self.batches = []
        for i in range(0, N, self.batch_size):
            self.batches.append(
                (
                    inputs[i : i + self.batch_size],
                    targets[i : i + self.batch_size],
                )
            )

    def __iter__(self):
        for xb, yb in self.batches:
            yield xb, yb

    def __len__(self):
        return len(self.batches)


class OperatorFCNN(AbstractSurrogate):
    """
    Learns a mapping from (reflectances, coordinate) -> scalar thickness.
    To predict n_evals outputs, performs n_evals forward passes.
    """

    def __init__(self, device: str, n_evals: int, config: OperatorFCNNConfig):
        super().__init__()
        self.device = torch.device(device)
        self.n_evals = n_evals
        self.config = config

        in_dim = n_evals + 1
        layers = []
        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_size))
            layers.append(self.get_activation(config.activation))
            in_dim = config.hidden_size
        layers.append(nn.Linear(in_dim, 1))

        self.net = nn.Sequential(*layers).to(self.device)
        self.optimizer = None

    def setup_optimizer_and_scheduler(self) -> torch.optim.Optimizer:
        optim = AdamWScheduleFree(self.parameters(), lr=self.config.learning_rate)
        self.optimizer = optim
        return optim

    def prepare_data(
        self,
        dataset_train: tuple[np.ndarray, np.ndarray],
        dataset_test: tuple[np.ndarray, np.ndarray],
        dataset_val: tuple[np.ndarray, np.ndarray] | None,
        batch_size: int,
        shuffle: bool = True,
    ):
        X_train, Y_train = dataset_train
        X_test, Y_test = dataset_test
        coords = torch.linspace(0.0, 1.0, steps=self.n_evals)

        def expand(X: np.ndarray, Y: np.ndarray):
            N = X.shape[0]
            X_rep = np.repeat(X, self.n_evals, axis=0)
            coords_rep = coords.repeat(N).unsqueeze(1).numpy()
            inp = np.concatenate([X_rep, coords_rep], axis=1)
            out = Y.reshape(-1, 1)
            return (
                torch.tensor(inp, dtype=torch.float32).to(self.device),
                torch.tensor(out, dtype=torch.float32).to(self.device),
            )

        # train loader
        tx, ty = expand(X_train, Y_train)
        train_loader = PreBatchedDataset(tx, ty, batch_size=batch_size, shuffle=shuffle)

        # test loader
        tx2, ty2 = expand(X_test, Y_test)
        test_loader = PreBatchedDataset(tx2, ty2, batch_size=batch_size, shuffle=False)

        # optional val loader
        if dataset_val is not None:
            X_val, Y_val = dataset_val
            tx3, ty3 = expand(X_val, Y_val)
            val_loader = PreBatchedDataset(
                tx3, ty3, batch_size=batch_size, shuffle=False
            )
        else:
            val_loader = None

        return train_loader, test_loader, val_loader

    def fit(
        self,
        train_loader: IterableDataset,
        test_loader: IterableDataset,
        epochs: int,
        trial: object | None = None,
        report_every: int = 1,
    ):
        criterion = nn.MSELoss()
        if self.optimizer is None:
            self.setup_optimizer_and_scheduler()

        pbar = tqdm(total=epochs, desc="Training OperatorFCNN")
        n_train = sum(batch[0].size(0) for batch in train_loader.batches)
        n_test = sum(batch[0].size(0) for batch in test_loader.batches)
        self.n_train_samples = n_train
        self.report_every = report_every

        # prepare history if not using Optuna
        store_history = trial is None
        if store_history:
            self.train_hist = []
            self.test_hist = []

        for epoch in range(1, epochs + 1):
            # training
            self.train()
            self.optimizer.train()
            train_loss_sum = 0.0
            for xb, yb in train_loader:
                xb, yb = xb, yb
                self.optimizer.zero_grad()
                pred = self.net(xb)
                loss = criterion(pred, yb)
                l2 = sum((p**2).sum() for p in self.parameters())
                loss = loss + self.config.regularisation_factor * l2
                loss.backward()
                self.optimizer.step()
                train_loss_sum += loss.item() * xb.size(0)

            avg_train = train_loss_sum / n_train

            # evaluation
            self.eval()
            self.optimizer.eval()
            test_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in test_loader:
                    test_loss_sum += criterion(self.net(xb), yb).item() * xb.size(0)
            avg_test = test_loss_sum / n_test

            # Report or store loss
            if epoch % report_every == 0:
                if store_history:
                    self.train_hist.append(avg_train)
                    self.test_hist.append(avg_test)
                else:
                    trial.report(avg_test, epoch)
                    if trial.should_prune():
                        raise TrialPruned()

            pbar.update(1)
            pbar.set_postfix(
                epoch=f"{epoch}/{epochs}",
                train_loss=f"{avg_train:.4f}",
                test_loss=f"{avg_test:.4f}",
            )
        pbar.close()

    def predict(self, dataloader: IterableDataset):
        self.eval()
        if self.optimizer is not None:
            self.optimizer.eval()

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in dataloader:
                out = self.net(xb)
                all_preds.append(out.cpu())
                all_targets.append(yb.cpu())

        preds = torch.cat(all_preds, dim=0).view(-1, self.n_evals)
        targets = torch.cat(all_targets, dim=0).view(-1, self.n_evals)
        return preds, targets

    # def save(self, model_name: str, training_id: str, base_dir: str):
    #     save_dir = os.path.join(base_dir, training_id)
    #     os.makedirs(save_dir, exist_ok=True)
    #     torch.save(self.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
