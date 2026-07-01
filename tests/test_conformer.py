# tests/test_conformer.py
"""Tests for 3D conformer generation."""
import pytest
from rdkit import Chem

from chem_augur.features.conformer_generation import (
    generate_3d_conformer,
    mol_to_3d_data,
    smiles_to_3d_data,
)


class TestGenerate3DConformer:
    """Test ETKDG + MMFF conformer generation."""

    def test_ethanol_conformer(self):
        """Ethanol (CCO) should generate a valid 3D conformer."""
        mol = Chem.MolFromSmiles("CCO")
        mol_3d = generate_3d_conformer(mol, seed=42)

        assert mol_3d is not None
        assert mol_3d.GetNumConformers() > 0
        conf = mol_3d.GetConformer()
        assert conf.Is3D()

        # Ethanol: C2H6O = 9 atoms
        assert mol_3d.GetNumAtoms() == 9  # With explicit hydrogens

    def test_water_conformer(self):
        """Water should generate a valid 3D conformer."""
        mol = Chem.MolFromSmiles("O")
        mol_3d = generate_3d_conformer(mol, seed=42)

        assert mol_3d is not None
        conf = mol_3d.GetConformer()
        assert conf.Is3D()

        # H2O = 3 atoms
        assert mol_3d.GetNumAtoms() == 3

    def test_benzene_conformer(self):
        """Benzene should generate a planar conformer."""
        mol = Chem.MolFromSmiles("c1ccccc1")
        mol_3d = generate_3d_conformer(mol, seed=42)

        assert mol_3d is not None
        conf = mol_3d.GetConformer()
        assert conf.Is3D()

        # C6H6 = 12 atoms
        assert mol_3d.GetNumAtoms() == 12

    def test_reproducibility(self):
        """Same seed should produce the same conformer."""
        mol1 = Chem.MolFromSmiles("CCO")
        mol2 = Chem.MolFromSmiles("CCO")

        conf1 = generate_3d_conformer(mol1, seed=42).GetConformer()
        conf2 = generate_3d_conformer(mol2, seed=42).GetConformer()

        pos1 = conf1.GetPositions()
        pos2 = conf2.GetPositions()

        # Positions should be identical with same seed
        import numpy as np
        assert np.allclose(pos1, pos2, atol=1e-4)

    def test_empty_molecule_raises(self):
        """Empty molecule should raise ValueError."""
        mol = Chem.Mol()
        with pytest.raises(ValueError, match="zero atoms"):
            generate_3d_conformer(mol)

    def test_multiple_calls_different_seeds(self):
        """Different seeds should produce different conformers."""
        mol1 = Chem.MolFromSmiles("CCCC")  # Butane
        mol2 = Chem.MolFromSmiles("CCCC")

        conf1 = generate_3d_conformer(mol1, seed=42).GetConformer()
        conf2 = generate_3d_conformer(mol2, seed=123).GetConformer()

        pos1 = conf1.GetPositions()
        pos2 = conf2.GetPositions()

        import numpy as np
        # Different seeds should produce different geometries
        assert not np.allclose(pos1, pos2, atol=1e-4)


class TestMolTo3DData:
    """Test conversion from RDKit mol to PyG Data."""

    def test_ethanol_data(self):
        """Ethanol should produce valid z, pos, and y."""
        mol = Chem.MolFromSmiles("CCO")
        mol_3d = generate_3d_conformer(mol, seed=42)

        data = mol_to_3d_data(mol_3d, target_property=None)

        assert data.z is not None
        assert data.pos is not None
        assert data.z.dtype == torch.long
        assert data.pos.dtype == torch.float

        # z should have atomic numbers for C, H, O
        assert data.z.shape[0] == 9  # 9 atoms with hydrogens
        assert data.pos.shape == (9, 3)

        # Carbon = 6, Oxygen = 8, Hydrogen = 1
        unique_z = set(data.z.tolist())
        assert 6 in unique_z  # Carbon
        assert 8 in unique_z  # Oxygen
        assert 1 in unique_z  # Hydrogen

    def test_with_target_property(self):
        """Molecule with a property should have y set."""
        mol = Chem.MolFromSmiles("CCO")
        mol_3d = generate_3d_conformer(mol, seed=42)
        mol_3d.SetProp("test_prop", "78.37")

        data = mol_to_3d_data(mol_3d, target_property="test_prop")

        assert data.y is not None
        assert abs(data.y.item() - 78.37) < 0.01

    def test_missing_target_property_raises(self):
        """Missing property should raise KeyError."""
        mol = Chem.MolFromSmiles("CCO")
        mol_3d = generate_3d_conformer(mol, seed=42)

        with pytest.raises(KeyError, match="not found"):
            mol_to_3d_data(mol_3d, target_property="nonexistent")

    def test_no_conformer_raises(self):
        """Molecule without conformer should raise ValueError."""
        mol = Chem.MolFromSmiles("CCO")

        with pytest.raises(ValueError, match="no 3D conformer|Bad Conformer"):
            mol_to_3d_data(mol)

    def test_none_molecule_raises(self):
        """None molecule should raise ValueError."""
        with pytest.raises(ValueError, match="None or has zero atoms"):
            mol_to_3d_data(None)


class TestSmilesTo3DData:
    """Test SMILES-to-3D-data convenience function."""

    def test_valid_smiles(self):
        """Valid SMILES should produce a Data object."""
        data = smiles_to_3d_data("CCO", seed=42)

        assert data.z is not None
        assert data.pos is not None
        assert data.z.shape[0] == 9
        assert data.pos.shape == (9, 3)

    def test_invalid_smiles_raises(self):
        """Invalid SMILES should raise ValueError."""
        with pytest.raises(ValueError, match="Could not parse"):
            smiles_to_3d_data("not_a_smiles!!!")


# Need torch for tensor assertions
import torch