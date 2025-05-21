# codes/surrogates/__init__.py

from .abstract import AbstractSurrogate
from .registry import get_model_config, get_surrogate

__all__ = [
    "AbstractSurrogate",
    "get_model_config",
    "get_surrogate",
]
