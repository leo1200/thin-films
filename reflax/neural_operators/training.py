import os
import random
from typing import Optional, Tuple

import numpy as np
import torch

from .models import get_model_config, get_surrogate


def set_random_seeds(seed: int, device: str) -> None:
    if "cuda" in device:
        torch.cuda.device(torch.device(device))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def load_dataset(
    filepath: str,
    splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
    seed: Optional[int] = None,
    epsilon: float = 1e-6,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load a single .npz file containing 'reflectances' and 'thicknesses',
    then split into train/test/val according to `splits` = (train_frac, test_frac, val_frac).
    Splits must each be in [0,1] and sum to 1 within `epsilon`. Zero is allowed.

    Returns:
        r_train, t_train, r_test, t_test, r_val, t_val
    """
    # Validate splits
    if any(s < 0 or s > 1 for s in splits):
        raise ValueError(f"All split fractions must be between 0 and 1. Got {splits}")
    total = sum(splits)
    if abs(total - 1.0) > epsilon:
        raise ValueError(
            f"Splits must sum to 1 (±{epsilon}). Got {splits} (sum={total:.6f})"
        )

    # Check file
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: {filepath}")

    # Load data arrays
    data = np.load(filepath)
    if "reflectances" not in data or "thicknesses" not in data:
        raise KeyError("File must contain 'reflectances' and 'thicknesses' arrays")
    all_refl = data["reflectances"]
    all_thick = data["thicknesses"]
    n = all_refl.shape[0]
    if all_thick.shape[0] != n:
        raise ValueError(
            "Mismatched first dimension between reflectances and thicknesses"
        )

    # Compute counts
    n_train = int(np.floor(splits[0] * n))
    n_test = int(np.floor(splits[1] * n))
    n_val = n - n_train - n_test  # absorb rounding remainder

    # Shuffle indices
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)

    # Slice indices
    train_idx = idx[:n_train]
    test_idx = idx[n_train : n_train + n_test]
    val_idx = idx[n_train + n_test :]

    # Helper to build empty arrays of correct shape
    def _slice(arr, indices, count):
        if count > 0:
            return arr[indices]
        # create empty of shape (0, feature_dims...)
        return np.empty((0,) + arr.shape[1:], dtype=arr.dtype)

    r_train = _slice(all_refl, train_idx, n_train)
    t_train = _slice(all_thick, train_idx, n_train)
    r_test = _slice(all_refl, test_idx, n_test)
    t_test = _slice(all_thick, test_idx, n_test)
    r_val = _slice(all_refl, val_idx, n_val)
    t_val = _slice(all_thick, val_idx, n_val)

    return r_train, t_train, r_test, t_test, r_val, t_val


def train_and_save_model(
    surrogate_name: str,
    dataset_name: str,
    epochs: int,
    batch_size: int,
    seed: int | None = None,
    device: str = "cpu",
    training_id: str | None = None,
):
    # setup and data loading
    set_random_seeds(seed, device)
    train_in, train_out, test_in, test_out, val_in, val_out = load_dataset(dataset_name)

    # model instantiation and normalization
    Surrogate = get_surrogate(surrogate_name)
    cfg = get_model_config(surrogate_name)
    model = Surrogate(device=device, n_evals=train_in.shape[1], config=cfg)
    model.fit_normalization(train_in, train_out)
    train_in_n, test_in_n, val_in_n = model.normalize(
        (train_in, test_in, val_in), model.in_min, model.in_max
    )
    train_out_n, test_out_n, val_out_n = model.normalize(
        (train_out, test_out, val_out), model.out_min, model.out_max
    )

    # prepare data loaders (validation is not used in training)
    train_loader, test_loader, val_loader = model.prepare_data(
        dataset_train=(train_in_n, train_out_n),
        dataset_test=(test_in_n, test_out_n),
        dataset_val=(val_in_n, val_out_n),
        batch_size=batch_size,
        shuffle=True,
    )

    # training
    model.fit(train_loader=train_loader, test_loader=test_loader, epochs=epochs)

    # saving
    model_name = f"{surrogate_name.lower()}_{dataset_name}".replace("__", "_")
    save_id = training_id or dataset_name
    model.save(
        model_name=model_name,
        training_id=save_id,
        base_dir=os.path.join(os.getcwd(), "trained"),
    )
