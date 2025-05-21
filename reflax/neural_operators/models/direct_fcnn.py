import torch
import torch.nn as nn
from optuna.exceptions import TrialPruned
from schedulefree import AdamWScheduleFree
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from .abstract import AbstractSurrogate
from .hyperparameters import DirectFCNNConfig


class DirectFCNN(AbstractSurrogate):
    def __init__(self, device: str, n_evals: int, config: DirectFCNNConfig):
        super().__init__()
        self.device = torch.device(device)
        layers, in_dim = [], n_evals

        for _ in range(config.num_hidden_layers):
            layers.append(nn.Linear(in_dim, config.hidden_size))
            layers.append(self.get_activation(config.activation))
            in_dim = config.hidden_size
        layers.append(nn.Linear(in_dim, n_evals))

        self.net = nn.Sequential(*layers).to(self.device)
        self.config = config
        self.optimizer = None

    def setup_optimizer_and_scheduler(self):
        optim = AdamWScheduleFree(self.parameters(), lr=self.config.learning_rate)
        self.optimizer = optim
        return optim

    def prepare_data(
        self, dataset_train, dataset_test, dataset_val, batch_size, shuffle=True
    ):
        def make_loader(X, Y, do_shuffle):
            tx = torch.tensor(X, dtype=torch.float32)
            ty = torch.tensor(Y, dtype=torch.float32)
            ds = TensorDataset(tx, ty)
            return DataLoader(ds, batch_size=batch_size, shuffle=do_shuffle)

        train_loader = make_loader(*dataset_train, shuffle)
        test_loader = make_loader(*dataset_test, False)
        val_loader = (
            make_loader(*dataset_val, False) if dataset_val is not None else None
        )
        return train_loader, test_loader, val_loader

    def fit(
        self,
        train_loader: DataLoader,
        test_loader: DataLoader,
        epochs: int,
        trial: object | None = None,
        report_every: int = 1,
    ):
        criterion = nn.MSELoss()
        if self.optimizer is None:
            self.setup_optimizer_and_scheduler()

        pbar = tqdm(total=epochs, desc="Training")
        n_train = len(train_loader.dataset)
        n_test = len(test_loader.dataset)
        self.n_train_samples = n_train
        self.report_every = report_every

        # prepare history if not using Optuna
        store_history = trial is None
        if store_history:
            self.train_hist = []
            self.test_hist = []

        for epoch in range(1, epochs + 1):
            # --- Training phase ---
            self.train()
            self.optimizer.train()
            train_loss_sum = 0.0
            for xb, yb in train_loader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                self.optimizer.zero_grad()
                pred = self.net(xb)
                loss = criterion(pred, yb)
                # L2 regularization
                l2 = sum((p**2).sum() for p in self.parameters())
                loss = loss + self.config.regularisation_factor * l2
                loss.backward()
                self.optimizer.step()
                train_loss_sum += loss.item() * xb.size(0)

            avg_train = train_loss_sum / n_train

            # --- Evaluation phase ---
            self.eval()
            self.optimizer.eval()
            test_loss_sum = 0.0
            with torch.no_grad():
                for xb, yb in test_loader:
                    xb, yb = xb.to(self.device), yb.to(self.device)
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

            # update progress bar once per epoch
            pbar.update(1)
            pbar.set_postfix(
                epoch=f"{epoch}/{epochs}",
                train_loss=f"{avg_train:.4f}",
                test_loss=f"{avg_test:.4f}",
            )

        pbar.close()

    def predict(self, dataloader: torch.utils.data.DataLoader):
        """
        Returns:
            preds: (N, n_evals) tensor
            targets: (N, n_evals) tensor
        """
        self.eval()
        # if using AdamWScheduleFree or similar:
        if self.optimizer is not None:
            self.optimizer.eval()

        all_preds = []
        all_targets = []
        with torch.no_grad():
            for xb, yb in dataloader:
                xb, yb = xb.to(self.device), yb.to(self.device)
                out = self.net(xb)  # [batch, n_evals]
                all_preds.append(out.cpu())
                all_targets.append(yb.cpu())

        preds = torch.cat(all_preds, dim=0)
        targets = torch.cat(all_targets, dim=0)
        return preds, targets

    # def save(self, model_name: str, training_id: str, base_dir: str):
    #     save_dir = os.path.join(base_dir, training_id)
    #     os.makedirs(save_dir, exist_ok=True)
    #     torch.save(self.state_dict(), os.path.join(save_dir, f"{model_name}.pth"))
