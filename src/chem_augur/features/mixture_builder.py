# features/mixture_builder.py
"""
Multi-molecule mixture builder.

Places multiple molecules from different chemical species in a shared 3D
coordinate space, producing a single combined z (atomic numbers) and pos
(3D positions) tensor that can be fed directly into a geometry-based model.

The same pipeline handles pure substances (all molecules of the same species)
and mixtures (multiple species at specified mole ratios) identically. The
model sees only atoms in space — it has no concept of "molecule identity."

Algorithm:
  1. Generate 3D conformer for each species independently
  2. Determine number of each species based on mole ratios
  3. Place molecules via random translation + rotation with steric clash
     rejection (reject placements where any intermolecular atom pair < 2.0 Å)
  4. Quick force-field relaxation of the combined system (MMFF)
  5. Return combined z and pos for all atoms
"""
from typing import List, Tuple, Optional

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, rdMolTransforms
import torch

from chem_augur.features.conformer_generation import generate_3d_conformer


# Minimum allowed distance between any two atoms of different molecules
# during placement. This prevents physically impossible interpenetration.
MIN_INTERMOLECULAR_DISTANCE = 2.0  # Å


# Maximum placement attempts before increasing box size
MAX_PLACEMENT_ATTEMPTS = 100


def _random_rotation_matrix(rng: np.random.Generator) -> np.ndarray:
    """Generate a random 3D rotation matrix (uniform on SO(3))."""
    # Use the Arvo method: random quaternion → rotation matrix
    u = rng.uniform(0, 1, 3)
    theta = 2 * np.pi * u[0]
    phi = 2 * np.pi * u[1]
    r = np.sqrt(u[2])
    w = np.sqrt(1 - u[2])

    q = np.array([w, r * np.cos(theta), r * np.sin(theta) * np.cos(phi),
                  r * np.sin(theta) * np.sin(phi)])

    # Quaternion to rotation matrix
    q0, q1, q2, q3 = q
    return np.array([
        [1 - 2*q2*q2 - 2*q3*q3, 2*q1*q2 - 2*q0*q3,     2*q1*q3 + 2*q0*q2],
        [2*q1*q2 + 2*q0*q3,     1 - 2*q1*q1 - 2*q3*q3, 2*q2*q3 - 2*q0*q1],
        [2*q1*q3 - 2*q0*q2,     2*q2*q3 + 2*q0*q1,     1 - 2*q1*q1 - 2*q2*q2],
    ])


def _has_steric_clash(
    new_positions: np.ndarray,
    existing_positions: np.ndarray,
    min_distance: float = MIN_INTERMOLECULAR_DISTANCE,
) -> bool:
    """
    Check if any atom in new_positions is too close to any atom in
    existing_positions.

    Uses brute-force pairwise distance check. For large systems (>1000 atoms),
    this could be optimized with a spatial hash or KD-tree, but for typical
    mixture sizes (10-100 molecules, 30-3000 atoms) this is fine.
    """
    if len(existing_positions) == 0:
        return False

    # Compute all pairwise distances efficiently with broadcasting
    # new_positions: [M, 3], existing_positions: [N, 3]
    # diff: [M, N, 3]
    diff = new_positions[:, np.newaxis, :] - existing_positions[np.newaxis, :, :]
    # distances: [M, N]
    distances = np.sqrt(np.sum(diff * diff, axis=2))

    return np.any(distances < min_distance)


def _place_molecule(
    mol: Chem.Mol,
    rng: np.random.Generator,
    box_size: float,
    existing_positions: np.ndarray,
    max_attempts: int = MAX_PLACEMENT_ATTEMPTS,
) -> Optional[np.ndarray]:
    """
    Attempt to place a molecule in the shared box without steric clashes.

    Tries random translations and rotations up to max_attempts times.
    If no valid placement is found, returns None (caller should increase
    box_size and retry).

    Args:
        mol: RDKit molecule with 3D conformer.
        rng: NumPy random generator.
        box_size: Side length of the cubic placement box.
        existing_positions: [N, 3] positions of already-placed atoms.
        max_attempts: Maximum placement attempts.

    Returns:
        [num_atoms, 3] numpy array of placed positions, or None.
    """
    conf = mol.GetConformer()
    original_positions = np.array(conf.GetPositions(), dtype=np.float64)

    # Center the molecule at origin for rotation
    centroid = original_positions.mean(axis=0)
    centered = original_positions - centroid

    for _ in range(max_attempts):
        # Random translation within box
        translation = rng.uniform(0, box_size, 3)

        # Random rotation
        rot_matrix = _random_rotation_matrix(rng)
        rotated = centered @ rot_matrix.T

        # Apply translation
        placed = rotated + translation

        # Check for steric clashes with existing atoms
        if not _has_steric_clash(placed, existing_positions):
            return placed

    return None


def _estimate_box_size(
    total_atoms: int,
    target_density: float = 0.01,
) -> float:
    """
    Estimate a reasonable box size based on total atom count.

    Typical molecular liquids have ~0.01 atoms/Å³ (close to water's density).
    This gives a rough starting point — the placement algorithm adjusts
    automatically if it's too small.

    Args:
        total_atoms: Total number of atoms across all molecules.
        target_density: Target atom density in atoms/Å³.

    Returns:
        Box side length in Å.
    """
    volume = total_atoms / target_density
    return volume ** (1 / 3)


def build_mixture(
    smiles_list: List[str],
    mole_ratios: List[float],
    num_molecules: int = 10,
    seed: int = 42,
    box_size: Optional[float] = None,
    relax: bool = True,
    relax_steps: int = 100,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Build a multi-molecule system for geometry-based property prediction.

    Places molecules from multiple species in a shared 3D coordinate space
    with steric clash avoidance and optional force-field relaxation. The
    output is a single combined z and pos tensor that can be fed directly
    into MACE or any other geometry-based model.

    For pure substances, pass a single-element smiles_list with ratio [1.0].

    Args:
        smiles_list: List of SMILES strings, one per species.
        mole_ratios: Mole fraction for each species. Should sum to 1.0 and
            have the same length as smiles_list.
        num_molecules: Total number of molecules to place (default 10).
            Larger values give better statistics for mixture properties
            but increase computational cost.
        seed: Random seed for reproducible placement.
        box_size: Side length of the cubic placement box in Å. If None,
            estimated automatically from atom count.
        relax: Whether to perform MMFF force-field relaxation of the
            combined system after placement (recommended).
        relax_steps: Maximum MMFF optimization steps if relax=True.

    Returns:
        z: [total_atoms] atomic numbers (torch.long)
        pos: [total_atoms, 3] 3D positions in Angstroms (torch.float)

    Raises:
        ValueError: If smiles_list and mole_ratios have different lengths,
            or if ratios don't sum to ~1.0, or if a SMILES can't be parsed.
        RuntimeError: If conformer generation fails for any species.
    """
    if len(smiles_list) != len(mole_ratios):
        raise ValueError(
            f"smiles_list ({len(smiles_list)}) and mole_ratios "
            f"({len(mole_ratios)}) must have the same length."
        )

    if abs(sum(mole_ratios) - 1.0) > 1e-6:
        raise ValueError(
            f"Mole ratios must sum to 1.0, got {sum(mole_ratios)}"
        )

    rng = np.random.default_rng(seed)

    # --- Step 1: Generate conformers for each species ---
    species_mols = []
    species_conformer_positions = []
    species_z = []

    for i, smiles in enumerate(smiles_list):
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Could not parse SMILES: {smiles}")

        mol_3d = generate_3d_conformer(mol, seed=seed + i)
        conf = mol_3d.GetConformer()
        positions = np.array(conf.GetPositions(), dtype=np.float64)
        z = np.array([atom.GetAtomicNum() for atom in mol_3d.GetAtoms()],
                     dtype=np.int64)

        species_mols.append(mol_3d)
        species_conformer_positions.append(positions)
        species_z.append(z)

    # --- Step 2: Determine molecule counts per species ---
    counts = []
    remaining = num_molecules
    for i, ratio in enumerate(mole_ratios[:-1]):
        count = max(1, int(round(ratio * num_molecules)))
        counts.append(count)
        remaining -= count
    counts.append(max(1, remaining))  # Last species gets the remainder

    # --- Step 3: Estimate box size ---
    total_atoms = sum(
        counts[i] * len(species_z[i]) for i in range(len(smiles_list))
    )
    if box_size is None:
        box_size = _estimate_box_size(total_atoms)

    # --- Step 4: Place molecules ---
    all_z = []
    all_positions = []
    existing_positions = np.zeros((0, 3), dtype=np.float64)

    # Shuffle placement order to avoid systematic bias
    placement_order = []
    for i, count in enumerate(counts):
        for _ in range(count):
            placement_order.append(i)
    rng.shuffle(placement_order)

    for species_idx in placement_order:
        placed_pos = _place_molecule(
            species_mols[species_idx],
            rng,
            box_size,
            existing_positions,
        )

        if placed_pos is None:
            # Increase box size and retry
            box_size *= 1.5
            rng2 = np.random.default_rng(seed + 9999)
            placed_pos = _place_molecule(
                species_mols[species_idx],
                rng2,
                box_size,
                existing_positions,
                max_attempts=MAX_PLACEMENT_ATTEMPTS * 2,
            )
            if placed_pos is None:
                raise RuntimeError(
                    f"Could not place molecule of species {species_idx} "
                    f"even after increasing box size to {box_size:.1f} Å. "
                    "Try reducing num_molecules or increasing box_size."
                )

        all_z.append(torch.tensor(species_z[species_idx], dtype=torch.long))
        all_positions.append(placed_pos)
        existing_positions = np.vstack([existing_positions, placed_pos])

    z = torch.cat(all_z)
    pos = torch.tensor(np.vstack(all_positions), dtype=torch.float)

    # --- Step 5: Optional force-field relaxation ---
    if relax:
        # Build a combined RDKit molecule for MMFF optimization
        combined_mol = Chem.RWMol()
        atom_map = []  # (mol_idx, atom_idx) for each atom in combined system

        for species_idx in placement_order:
            mol = species_mols[species_idx]
            for atom in mol.GetAtoms():
                new_atom = Chem.Atom(atom.GetAtomicNum())
                combined_mol.AddAtom(new_atom)
                atom_map.append((species_idx, atom.GetIdx()))

        # Set conformer positions
        conf = Chem.Conformer(combined_mol.GetNumAtoms())
        for i, pos_i in enumerate(pos.numpy()):
            conf.SetAtomPosition(i, pos_i)
        combined_mol.AddConformer(conf)

        # MMFF optimization
        try:
            AllChem.MMFFOptimizeMolecule(combined_mol, maxIters=relax_steps)
            # Extract optimized positions
            opt_conf = combined_mol.GetConformer()
            pos = torch.tensor(
                np.array(opt_conf.GetPositions(), dtype=np.float32),
                dtype=torch.float,
            )
        except Exception:
            # MMFF may fail for unusual elements or disconnected systems.
            # Fall back to unrelaxed positions — they're still physically
            # reasonable from the steric clash avoidance.
            pass

    return z, pos