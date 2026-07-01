# features/conformer_generation.py
"""
3D conformer generation for molecular property prediction.

Generates physically reasonable 3D geometries from 2D molecular graphs using
RDKit's ETKDG algorithm (experimental torsion knowledge from the Cambridge
Structural Database) followed by MMFF94 force-field optimization.

The output is the minimal representation needed for geometry-based models:
atomic numbers z and 3D positions pos — no bond types, hybridization labels,
or other hand-crafted features.
"""
from typing import Optional, Tuple

from rdkit import Chem
from rdkit.Chem import AllChem
import torch
from torch_geometric.data import Data


def generate_3d_conformer(mol: Chem.Mol, seed: int = 42) -> Chem.Mol:
    """
    Generate a single low-energy 3D conformer for a molecule.

    Uses ETKDG (Experimental Torsion Knowledge Distance Geometry) to embed
    the molecule in 3D space, then refines with MMFF94 force-field
    optimization. ETKDG uses experimental torsion angle preferences from
    the Cambridge Structural Database to produce physically realistic
    conformations for drug-like organic molecules.

    Args:
        mol: RDKit molecule with explicit hydrogens. The molecule will be
             modified in-place — a copy is made first.
        seed: Random seed for reproducible conformer generation.

    Returns:
        A copy of the molecule with a 3D conformer added. Returns None if
        embedding or optimization fails.

    Raises:
        ValueError: If the molecule has no atoms.
    """
    if mol.GetNumAtoms() == 0:
        raise ValueError("Cannot generate conformer for molecule with zero atoms.")

    mol = Chem.Mol(mol)  # Copy to avoid mutating input
    mol = Chem.AddHs(mol)

    # ETKDG embedding with reproducible seed
    params = AllChem.ETKDG()
    params.randomSeed = seed

    status = AllChem.EmbedMolecule(mol, params)
    if status != 0:
        raise RuntimeError(
            "ETKDG embedding failed. The molecule may have unusual geometry "
            "or be too constrained for distance geometry embedding."
        )

    # MMFF94 force-field optimization
    opt_status = AllChem.MMFFOptimizeMolecule(mol)
    if opt_status != 0:
        # Optimization returned non-zero — this is often benign (convergence
        # criterion not fully met) but the geometry is still improved.
        # Only fail if the conformer is completely invalid.
        try:
            conf = mol.GetConformer()
            if conf is None or not conf.Is3D():
                raise RuntimeError("MMFF optimization produced invalid conformer.")
        except Exception:
            raise RuntimeError("MMFF optimization failed to produce valid 3D geometry.")

    return mol


def mol_to_3d_data(
    mol: Chem.Mol,
    target_property: Optional[str] = None,
) -> Data:
    """
    Convert an RDKit molecule with 3D conformer to a PyG Data object.

    Extracts only atomic numbers (z) and 3D positions (pos). No bond types,
    hybridization labels, aromaticity flags, or other hand-crafted features.
    The model learns all chemical semantics from geometry alone.

    Args:
        mol: RDKit molecule with a 3D conformer (from generate_3d_conformer
             or from an SDF file with 3D coordinates).
        target_property: Optional property name to extract as label y.

    Returns:
        PyG Data object with:
          - z: [num_atoms] atomic numbers (torch.long)
          - pos: [num_atoms, 3] 3D coordinates in Angstroms (torch.float)
          - y: [1] regression target (only if target_property provided and
                present on the molecule)
    """
    if mol is None or mol.GetNumAtoms() == 0:
        raise ValueError("Molecule is None or has zero atoms.")

    conf = mol.GetConformer()
    if conf is None or not conf.Is3D():
        raise ValueError(
            "Molecule has no 3D conformer. Call generate_3d_conformer() first, "
            "or use an SDF file with 3D coordinates."
        )

    # Atomic numbers — the only node feature
    z = torch.tensor(
        [atom.GetAtomicNum() for atom in mol.GetAtoms()],
        dtype=torch.long,
    )

    # 3D positions from the conformer
    pos = torch.tensor(conf.GetPositions(), dtype=torch.float)

    # Optional label
    y = None
    if target_property is not None:
        if mol.HasProp(target_property):
            y = torch.tensor([[float(mol.GetProp(target_property))]], dtype=torch.float)
        else:
            raise KeyError(
                f"Property '{target_property}' not found on molecule."
            )

    return Data(z=z, pos=pos, y=y)


def smiles_to_3d_data(
    smiles: str,
    target_property: Optional[str] = None,
    seed: int = 42,
) -> Data:
    """
    Convert a SMILES string directly to a PyG Data object with 3D geometry.

    Convenience wrapper that parses the SMILES, generates a conformer, and
    extracts z/pos in one call.

    Args:
        smiles: SMILES string (e.g., "CCO" for ethanol).
        target_property: Optional property name (for labeled data).
        seed: Random seed for conformer generation.

    Returns:
        PyG Data object with z, pos, and optionally y.

    Raises:
        ValueError: If SMILES cannot be parsed.
        RuntimeError: If conformer generation fails.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    mol_3d = generate_3d_conformer(mol, seed=seed)
    return mol_to_3d_data(mol_3d, target_property=target_property)