# tests/test_mixture.py
"""Tests for multi-molecule mixture builder."""
import pytest
import torch

from chem_augur.features.mixture_builder import build_mixture


class TestBuildMixture:
    """Test mixture/multi-molecule system construction."""

    def test_pure_substance(self):
        """Single species should produce valid z and pos."""
        z, pos = build_mixture(
            smiles_list=["CCO"],
            mole_ratios=[1.0],
            num_molecules=5,
            seed=42,
            relax=False,
        )

        assert z is not None
        assert pos is not None
        assert z.dtype == torch.long
        assert pos.dtype == torch.float

        # 5 ethanol molecules × 9 atoms each = 45 atoms
        assert z.shape[0] == 45  # 5 * 9
        assert pos.shape == (45, 3)

        # All atoms should be C (6), H (1), or O (8)
        unique_z = set(z.tolist())
        assert unique_z.issubset({1, 6, 8})

    def test_binary_mixture(self):
        """Two-species mixture should produce valid z and pos."""
        z, pos = build_mixture(
            smiles_list=["CCO", "O"],
            mole_ratios=[0.5, 0.5],
            num_molecules=6,
            seed=42,
            relax=False,
        )

        assert z is not None
        assert pos is not None

        # 3 ethanol (9 atoms) + 3 water (3 atoms) = 36 atoms
        assert z.shape[0] == 36
        assert pos.shape == (36, 3)

    def test_ratios_must_sum_to_one(self):
        """Ratios not summing to 1.0 should raise ValueError."""
        with pytest.raises(ValueError, match="sum to 1.0"):
            build_mixture(
                smiles_list=["CCO", "O"],
                mole_ratios=[0.6, 0.6],
                num_molecules=4,
                seed=42,
                relax=False,
            )

    def test_mismatched_lengths_raises(self):
        """Different length lists should raise ValueError."""
        with pytest.raises(ValueError, match="must have the same length"):
            build_mixture(
                smiles_list=["CCO", "O", "CO"],
                mole_ratios=[0.5, 0.5],
                num_molecules=4,
                seed=42,
                relax=False,
            )

    def test_invalid_smiles_raises(self):
        """Invalid SMILES should raise ValueError."""
        with pytest.raises(ValueError, match="Could not parse"):
            build_mixture(
                smiles_list=["not_a_smiles!!!"],
                mole_ratios=[1.0],
                num_molecules=4,
                seed=42,
                relax=False,
            )

    def test_reproducibility(self):
        """Same seed should produce identical placement."""
        z1, pos1 = build_mixture(
            smiles_list=["CCO"],
            mole_ratios=[1.0],
            num_molecules=5,
            seed=42,
            relax=False,
        )
        z2, pos2 = build_mixture(
            smiles_list=["CCO"],
            mole_ratios=[1.0],
            num_molecules=5,
            seed=42,
            relax=False,
        )

        assert torch.equal(z1, z2)
        assert torch.allclose(pos1, pos2, atol=1e-5)

    def test_no_steric_clashes(self):
        """Placed molecules should not have atoms too close together."""
        z, pos = build_mixture(
            smiles_list=["CCO"],
            mole_ratios=[1.0],
            num_molecules=5,
            seed=42,
            relax=False,
        )

        # Each ethanol molecule has 9 atoms
        # Within each molecule, atoms can be close (bonded)
        # Between molecules, no atom pair should be < 2.0 Å
        mol_size = 9
        for i in range(5):
            for j in range(i + 1, 5):
                start_i = i * mol_size
                end_i = start_i + mol_size
                start_j = j * mol_size
                end_j = start_j + mol_size

                pos_i = pos[start_i:end_i]
                pos_j = pos[start_j:end_j]

                # Compute all pairwise distances between molecules i and j
                diff = pos_i.unsqueeze(1) - pos_j.unsqueeze(0)
                distances = torch.sqrt((diff * diff).sum(dim=2))

                min_dist = distances.min().item()
                assert min_dist >= 1.9, (
                    f"Steric clash detected: molecules {i} and {j} "
                    f"have atoms at {min_dist:.2f} Å"
                )

    def test_relaxation(self):
        """Relaxation should not crash and should produce valid positions."""
        z, pos = build_mixture(
            smiles_list=["CCO"],
            mole_ratios=[1.0],
            num_molecules=3,
            seed=42,
            relax=True,
            relax_steps=50,
        )

        assert z.shape[0] == 27  # 3 * 9
        assert pos.shape == (27, 3)
        assert not torch.any(torch.isnan(pos))
        assert not torch.any(torch.isinf(pos))

    def test_custom_box_size(self):
        """Custom box size should be respected."""
        z, pos = build_mixture(
            smiles_list=["O"],
            mole_ratios=[1.0],
            num_molecules=5,
            seed=42,
            box_size=50.0,
            relax=False,
        )

        # All positions should be within [0, box_size]
        assert pos.min() >= -1.0  # Allow small margin
        assert pos.max() <= 51.0