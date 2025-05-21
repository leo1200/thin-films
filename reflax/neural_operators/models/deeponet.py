import numpy as np
import torch
import torch.nn as nn
from optuna.exceptions import TrialPruned
from schedulefree import AdamWScheduleFree
from torch.utils.data import IterableDataset
from tqdm import tqdm

from .abstract import AbstractSurrogate
from .hyperparameters import DeepOperatorConfig


class PreBatchedDualInputDataset(IterableDataset):
    """
    Iterable dataset that precomputes batches for two input tensors and one target tensor.
    """

    def __init__(
        self,
        inputs: tuple[torch.Tensor, torch.Tensor],
        targets: torch.Tensor,
        batch_size: int,
        shuffle: bool = True,
    ):
        super().__init__()
        self.input1, self.input2 = inputs
        self.targets = targets
        self.batch_size = batch_size
        self.shuffle = shuffle
        self._make_batches()

    def _make_batches(self):
        N = self.targets.size(0)
        if self.shuffle:
            perm = torch.randperm(N)
        else:
            perm = torch.arange(N)
        inp1 = self.input1[perm]
        inp2 = self.input2[perm]
        tgt = self.targets[perm]
        self.batches = []
        for i in range(0, N, self.batch_size):
            self.batches.append(
                (
                    inp1[i : i + self.batch_size],
                    inp2[i : i + self.batch_size],
                    tgt[i : i + self.batch_size],
                )
            )

    def __iter__(self):
        for b in self.batches:
            yield b

    def __len__(self):
        return len(self.batches)


class DeepOperatorNetwork(AbstractSurrogate):
    """
    Deep operator network with separate branch and trunk MLPs.
    Branch: R^n_evals -> R^latent_size
    Trunk: [0,1] -> R^latent_size
    Final: dot(branch(x), trunk(coord)) -> scalar
    """

    def __init__(self, device: str, n_evals: int, config: DeepOperatorConfig):
        super().__init__()
        self.device = torch.device(device)
        self.n_evals = n_evals
        self.config = config

        # Build branch network
        layers = []
        in_dim = n_evals
        for _ in range(config.branch_layers):
            layers.append(nn.Linear(in_dim, config.hidden_width))
            layers.append(self.get_activation(config.activation))
            in_dim = config.hidden_width
        layers.append(nn.Linear(in_dim, config.latent_size))
        self.branch_net = nn.Sequential(*layers).to(self.device)

        # Build trunk network
        layers = []
        in_dim = 1
        for _ in range(config.trunk_layers):
            layers.append(nn.Linear(in_dim, config.hidden_width))
            layers.append(self.get_activation(config.activation))
            in_dim = config.hidden_width
        layers.append(nn.Linear(in_dim, config.latent_size))
        self.trunk_net = nn.Sequential(*layers).to(self.device)

        self.optimizer = None

    def setup_optimizer_and_scheduler(self) -> torch.optim.Optimizer:
        params = list(self.branch_net.parameters()) + list(self.trunk_net.parameters())
        optim = AdamWScheduleFree(params, lr=self.config.learning_rate)
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
            inp1 = torch.tensor(X_rep, dtype=torch.float32).to(self.device)
            inp2 = torch.tensor(coords_rep, dtype=torch.float32).to(self.device)
            tgt = torch.tensor(Y.reshape(-1, 1), dtype=torch.float32).to(self.device)
            return inp1, inp2, tgt

        # train loader
        inp1, inp2, tgt = expand(X_train, Y_train)
        self._n_train = tgt.size(0)
        train_loader = PreBatchedDualInputDataset(
            (inp1, inp2), tgt, batch_size=batch_size, shuffle=shuffle
        )

        # test loader
        inp1, inp2, tgt = expand(X_test, Y_test)
        self._n_test = tgt.size(0)
        test_loader = PreBatchedDualInputDataset(
            (inp1, inp2), tgt, batch_size=batch_size, shuffle=False
        )

        # val loader
        if dataset_val is not None:
            X_val, Y_val = dataset_val
            inp1, inp2, tgt = expand(X_val, Y_val)
            self._n_val = tgt.size(0)
            val_loader = PreBatchedDualInputDataset(
                (inp1, inp2), tgt, batch_size=batch_size, shuffle=False
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

        pbar = tqdm(total=epochs, desc="Training DeepOperator")
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
            for xb1, xb2, yb in train_loader:
                self.optimizer.zero_grad()
                # forward
                z_branch = self.branch_net(xb1)
                z_trunk = self.trunk_net(xb2)
                pred = torch.sum(z_branch * z_trunk, dim=1, keepdim=True)
                loss = criterion(pred, yb)
                # L2 reg
                l2 = sum(
                    (p**2).sum()
                    for p in list(self.branch_net.parameters())
                    + list(self.trunk_net.parameters())
                )
                loss = loss + self.config.regularisation_factor * l2
                loss.backward()
                self.optimizer.step()
                train_loss_sum += loss.item() * xb1.size(0)

            avg_train = train_loss_sum / n_train

            # evaluation
            self.eval()
            self.optimizer.eval()
            test_loss_sum = 0.0
            with torch.no_grad():
                for xb1, xb2, yb in test_loader:
                    z_branch = self.branch_net(xb1)
                    z_trunk = self.trunk_net(xb2)
                    pred = torch.sum(z_branch * z_trunk, dim=1, keepdim=True)
                    test_loss_sum += criterion(pred, yb).item() * xb1.size(0)
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
            for xb1, xb2, yb in dataloader:
                z_branch = self.branch_net(xb1)
                z_trunk = self.trunk_net(xb2)
                pred = torch.sum(z_branch * z_trunk, dim=1, keepdim=True)
                all_preds.append(pred.cpu())
                all_targets.append(yb.cpu())

        preds = torch.cat(all_preds, dim=0).view(-1, self.n_evals)
        targets = torch.cat(all_targets, dim=0).view(-1, self.n_evals)
        return preds, targets

    # def save(self, model_name: str, training_id: str, base_dir: str):
    #     save_dir = os.path.join(base_dir, training_id)
    #     os.makedirs(save_dir, exist_ok=True)
    #     torch.save(
    #         {
    #             "branch": self.branch_net.state_dict(),
    #             "trunk": self.trunk_net.state_dict(),
    #         },
    #         os.path.join(save_dir, f"{model_name}.pth"),
    #     )
