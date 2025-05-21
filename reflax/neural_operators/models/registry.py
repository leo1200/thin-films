# codes/surrogates/registry.py

from .abstract import AbstractSurrogate
from .deeponet import DeepOperatorNetwork
from .direct_fcnn import DirectFCNN
from .direct_fcnn_plain import DirectFCNNPlain
from .hyperparameters import (
    DeepOperatorConfig,
    DirectFCNNConfig,
    DirectFCNNPlainConfig,
    OperatorFCNNConfig,
)
from .operator_fcnn import OperatorFCNN

# map name → class
SURROGATE_REGISTRY: dict[str, type[AbstractSurrogate]] = {
    "direct_fcnn": DirectFCNN,
    "operator_fcnn": OperatorFCNN,
    "deeponet": DeepOperatorNetwork,
    "direct_fcnn_plain": DirectFCNNPlain,
}

# map name → config dataclass
CONFIG_REGISTRY: dict[str, type] = {
    "direct_fcnn": DirectFCNNConfig,
    "operator_fcnn": OperatorFCNNConfig,
    "deeponet": DeepOperatorConfig,
    "direct_fcnn_plain": DirectFCNNPlainConfig,
}


def get_surrogate(name: str) -> type[AbstractSurrogate]:
    try:
        return SURROGATE_REGISTRY[name]
    except KeyError:
        raise ValueError(f"Unknown surrogate: {name}")


def get_model_config(name: str):
    try:
        cfg_cls = CONFIG_REGISTRY[name]
        return cfg_cls()
    except KeyError:
        raise ValueError(f"No config defined for surrogate: {name}")
