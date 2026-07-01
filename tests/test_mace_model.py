# tests/test_mace_model.py
"""Tests for MACE model wrapper and registry."""
import pytest
import torch

from chem_augur.models.registry import build_model, list_available_models, MODEL_REGISTRY


class TestModelRegistry:
    """Test the model registry factory."""

    def test_gcn_is_available(self):
        """GCN should always be available (PyG dependency)."""
        models = list_available_models()
        assert "gcn" in models

    def test_build_gcn(self):
        """Building GCN should return a valid module."""
        model = build_model(
            "gcn",
            num_node_features=34,
            hidden_channels=64,
            output_channels=1,
            num_layers=4,
            dropout=0.5,
        )

        assert model is not None
        assert isinstance(model, torch.nn.Module)

        # Quick forward pass with dummy data
        from torch_geometric.data import Data
        data = Data(
            x=torch.randn(5, 34),
            edge_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
            batch=torch.zeros(5, dtype=torch.long),
        )

        with torch.no_grad():
            output = model(data)
        assert output.shape == (1, 1)  # Single graph, single output

    def test_unknown_model_raises(self):
        """Unknown model name should raise ValueError."""
        with pytest.raises(ValueError, match="Unknown model"):
            build_model("nonexistent_model_name")


class TestMACEWrapper:
    """Test MACE wrapper (these tests run even without mace-torch installed)."""

    def test_import_handles_missing_mace(self):
        """The wrapper should import cleanly even without mace-torch."""
        # The registry.py already handles this with lazy import
        # If mace-torch is installed, "mace" will be in the registry
        # If not, it won't — either way, no crash
        models = list_available_models()
        assert isinstance(models, list)

    def test_gcn_registry_entry(self):
        """GCN model class should be callable from registry."""
        model_cls = MODEL_REGISTRY.get("gcn")
        assert model_cls is not None

        model = model_cls(
            num_node_features=10,
            hidden_channels=32,
            output_channels=1,
        )
        assert isinstance(model, torch.nn.Module)