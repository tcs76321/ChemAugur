# models/registry.py
"""
Model registry — factory for constructing models by name.

Provides a single entry point for building any supported model architecture.
New models are added by registering them in the MODEL_REGISTRY dict.
The training pipeline and prediction module use this factory so they remain
model-agnostic.
"""
from typing import Dict, Any

import torch.nn as nn

from chem_augur.models.gnn.gcn_model import SimpleGCN


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

MODEL_REGISTRY: Dict[str, type] = {
    "gcn": SimpleGCN,
}

# MACE is registered lazily to avoid import errors when mace-torch is not
# installed. The factory function handles the import at construction time.
try:
    from chem_augur.models.gnn.mace_model import MACEWrapper
    MODEL_REGISTRY["mace"] = MACEWrapper
except ImportError:
    pass


def build_model(name: str, **kwargs: Any) -> nn.Module:
    """
    Build a model by name.

    Args:
        name: Model architecture name (e.g., "gcn", "mace").
        **kwargs: Keyword arguments passed to the model constructor.
            See each model's docstring for parameters.

    Returns:
        Instantiated nn.Module.

    Raises:
        ValueError: If the model name is not recognized.
        ImportError: If the model's dependencies are not installed.

    Example:
        >>> model = build_model("mace", cutoff=12.0, hidden_channels=128)
        >>> model = build_model("gcn", num_node_features=34, hidden_channels=64)
    """
    if name not in MODEL_REGISTRY:
        available = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(
            f"Unknown model '{name}'. Available models: {available}"
        )

    return MODEL_REGISTRY[name](**kwargs)


def list_available_models() -> list:
    """Return list of registered model names."""
    return sorted(MODEL_REGISTRY.keys())


def register_model(name: str, model_cls: type) -> None:
    """
    Register a new model class.

    Args:
        name: Unique name for the model.
        model_cls: nn.Module subclass.

    Raises:
        ValueError: If the name is already registered.
    """
    if name in MODEL_REGISTRY:
        raise ValueError(f"Model '{name}' is already registered.")
    MODEL_REGISTRY[name] = model_cls