# features/molecular_graph_3d.py
"""
3D molecular graph utilities for geometry-based models.

Provides thin utility functions to extract atomic numbers (z) and 3D positions
(pos) from RDKit molecules with conformers, and to build radius graphs for
models that require explicit edge indices.

MACE and other geometry-based models compute edges, distances, angles, and
dihedrals internally from pos — so the edge_index returned here is primarily
for debugging, inspection, or models that don't build their own radius graph.
"""
from typing import Optional, Tuple

from rdkit import Chem
import torch
from torch_geometric.nn import radius_graph

from chem_augur.features.conformer_generation import (
    generate_3d_conformer,
    mol_to_3d_data,
)


def z_pos_from_mol(mol: Chem.Mol) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract atomic numbers and 3D positions from an RDKit molecule.

    Args:
        mol: RDKit molecule with a 3D conformer.

    Returns:
        z: [N] atomic numbers (torch.long)
        pos: [N, 3] 3D coordinates in Angstroms (torch.float)
    """
    data = mol_to_3d_data(mol)
    return data.z, data.pos


def z_pos_from_smiles(
    smiles: str,
    seed: int = 42,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Parse SMILES, generate 3D conformer, extract z and pos.

    Args:
        smiles: SMILES string.
        seed: Random seed for conformer generation.

    Returns:
        z: [N] atomic numbers (torch.long)
        pos: [N, 3] 3D coordinates in Angstroms (torch.float)
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")

    mol_3d = generate_3d_conformer(mol, seed=seed)
    return z_pos_from_mol(mol_3d)


def build_radius_graph(
    pos: torch.Tensor,
    cutoff: float = 12.0,
    batch: Optional[torch.Tensor] = None,
    max_num_neighbors: int = 64,
) -> torch.Tensor:
    """
    Build an edge index from 3D positions using a distance cutoff.

    Uses PyG's radius_graph which returns directed edges for all atom pairs
    within the cutoff distance. This is primarily for debugging and inspection
    — MACE builds its own radius graph internally.

    Args:
        pos: [N, 3] 3D atom positions.
        cutoff: Maximum distance (Å) for edges.
        batch: [N] batch indices for multi-graph batching.
        max_num_neighbors: Maximum neighbors per atom (limits memory for
                          very dense systems).

    Returns:
        edge_index: [2, num_edges] directed edge indices.
    """
    return radius_graph(
        pos,
        r=cutoff,
        batch=batch,
        max_num_neighbors=max_num_neighbors,
    )


def num_edges_in_radius(
    pos: torch.Tensor,
    cutoff: float = 12.0,
    batch: Optional[torch.Tensor] = None,
) -> int:
    """
    Count the number of edges within a cutoff radius.

    Useful for estimating computational cost before training.

    Args:
        pos: [N, 3] 3D atom positions.
        cutoff: Maximum distance for edges.
        batch: [N] batch indices.

    Returns:
        Number of directed edges in the radius graph.
    """
    edge_index = build_radius_graph(pos, cutoff=cutoff, batch=batch)
    return edge_index.shape[1]