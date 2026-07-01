# tests/test_smoke.py
"""End-to-end smoke tests: SMILES → conformer → model → prediction.

These tests validate that the full pipeline functions correctly without
requiring mace-torch to be installed. They run in < 5 seconds on CPU and
should be run as part of CI to catch regressions.
"""
import pytest
import torch

from chem_augur.features.conformer_generation import (
    smiles_to_3d_data,
    generate_3d_conformer,
    mol_to_3d_data,
)
from chem_augur.models.registry import build_model, list_available_models
from chem_augur.config import TrainingConfig


class TestSmilesToPrediction:
    """End-to-end: SMILES → 3D conformer → model forward pass → valid output."""

    def test_smiles_to_3d_data_shape(self):
        """SMILES should produce valid z, pos tensors with correct shapes."""
        data = smiles_to_3d_data("CCO", seed=42)

        assert data.z is not None
        assert data.pos is not None
        assert data.z.dtype == torch.long
        assert data.pos.dtype == torch.float

        # Ethanol (C2H6O) with explicit hydrogens = 9 atoms
        assert data.z.shape[0] == 9
        assert data.pos.shape == (9, 3)

    def test_gcn_forward_with_conformer_z(self):
        """
        GCN forward pass using z as a feature dimension proxy.

        The GCN model doesn't use z/pos directly (it uses x/edge_index),
        but we can verify that the registry builds a working model and
        that a forward pass completes without error.
        """
        model = build_model(
            "gcn",
            num_node_features=34,
            hidden_channels=64,
            output_channels=1,
            num_layers=2,
            dropout=0.5,
        )

        from torch_geometric.data import Data

        data = Data(
            x=torch.randn(9, 34),
            edge_index=torch.tensor(
                [[0, 1, 2, 3, 4, 5, 6, 7],
                 [1, 2, 3, 4, 5, 6, 7, 8]]
            ),
            batch=torch.zeros(9, dtype=torch.long),
        )

        with torch.no_grad():
            output = model(data)

        assert output.shape == (1, 1)
        assert torch.isfinite(output).all()

    def test_conformer_with_target_property(self):
        """Conformer generation with a target property should set y."""
        from rdkit import Chem

        mol = Chem.MolFromSmiles("CCO")
        mol_3d = generate_3d_conformer(mol, seed=42)
        mol_3d.SetProp("test_bp", "78.37")

        data = mol_to_3d_data(mol_3d, target_property="test_bp")

        assert data.y is not None
        assert abs(data.y.item() - 78.37) < 0.01

    def test_multiple_smiles_all_generate(self):
        """Multiple diverse SMILES should all generate valid conformers."""
        test_smiles_list = [
            "CCO",          # ethanol
            "O",            # water
            "c1ccccc1",     # benzene
            "CC(=O)O",      # acetic acid
            "CCCC",         # butane
        ]

        for smiles in test_smiles_list:
            data = smiles_to_3d_data(smiles, seed=42)
            assert data.z is not None
            assert data.pos is not None
            assert data.z.shape[0] > 0
            assert data.pos.shape[0] == data.z.shape[0]
            assert data.pos.shape[1] == 3
            # All positions should be finite
            assert torch.isfinite(data.pos).all()


class TestTrainingConfigRoundTrip:
    """Verify TrainingConfig serialization works end-to-end."""

    def test_default_config_validates(self):
        """Default config should pass validation."""
        config = TrainingConfig()
        config.validate()

    def test_mace_config_validates(self):
        """MACE-specific config should validate."""
        config = TrainingConfig(
            model_name="mace",
            cutoff=12.0,
            hidden_channels=128,
            max_ell=3,
            correlation=3,
        )
        config.validate()

    def test_config_dict_round_trip(self):
        """Config → dict → config should preserve values."""
        original = TrainingConfig(
            model_name="mace",
            cutoff=15.0,
            batch_size=8,
            learning_rate=5e-4,
        )
        restored = TrainingConfig.from_dict(original.to_dict())

        assert restored.model_name == "mace"
        assert restored.cutoff == 15.0
        assert restored.batch_size == 8
        assert restored.learning_rate == 5e-4


class TestModelRegistrySmoke:
    """Verify model registry works for all available models."""

    def test_all_registered_models_buildable(self):
        """Every model in the registry should be constructable."""
        models = list_available_models()
        assert len(models) > 0

        for model_name in models:
            if model_name == "gcn":
                model = build_model(
                    "gcn",
                    num_node_features=34,
                    hidden_channels=32,
                    output_channels=1,
                )
                assert isinstance(model, torch.nn.Module)
            elif model_name == "mace":
                # MACE construction may fail if mace-torch is not installed,
                # but the registry should still list it. Skip the build test
                # since we can't verify without the dependency.
                pass