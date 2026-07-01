# features/geometry_dataset.py
"""
Geometry3DDataset — a PyTorch Geometric Dataset that loads molecules from SDF
files, generates 3D conformers, and produces z (atomic numbers) + pos (3D
positions) Data objects for geometry-based models (MACE).

Features:
  - Reads SDF files with optional 2D or 3D coordinates
  - Generates 3D conformers via ETKDG + MMFF when 3D coords are not present
  - Conformer caching to disk (keyed by SMILES hash) for fast reload
  - Graceful handling of failed molecules (skip + log, don't crash)
  - Train / val / test split utilities
  - Target normalization via StandardScaler

Design philosophy:
    The dataset produces only z and pos — no bond types, hybridization labels,
    aromaticity flags, or any hand-crafted features. The model learns all
    chemical semantics from geometry alone.
"""
from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
from typing import List, Optional, Callable, Tuple

import torch
from rdkit import Chem
from torch_geometric.data import Dataset, Data

from chem_augur.features.conformer_generation import (
    generate_3d_conformer,
    mol_to_3d_data,
)
from chem_augur.features.molecular_dataset import StandardScaler

logger = logging.getLogger(__name__)

# Default conformer cache directory
DEFAULT_CACHE_DIR = os.path.join(
    os.path.expanduser("~"), ".cache", "chem_augur", "conformers"
)


def _smiles_to_cache_key(smiles: str, seed: int) -> str:
    """Generate a deterministic cache key from SMILES and seed."""
    digest = hashlib.sha256(f"{smiles}:{seed}".encode()).hexdigest()
    return digest


def _load_cached_conformer(cache_path: str) -> Optional[Chem.Mol]:
    """Load a cached conformer from disk. Returns None if not found."""
    if not os.path.exists(cache_path):
        return None
    try:
        mol = Chem.Mol(cache_path)
        if mol is not None:
            # RDKit's Mol format doesn't preserve all properties —
            # re-sanitize to ensure the molecule is usable
            Chem.SanitizeMol(mol)
        return mol
    except Exception:
        # Cache file corrupted — delete it and return None
        os.remove(cache_path)
        return None


def _save_conformer_to_cache(mol: Chem.Mol, cache_path: str) -> None:
    """Save a conformer to disk cache."""
    os.makedirs(os.path.dirname(cache_path), exist_ok=True)
    try:
        # Use MOL format (not SDF) to avoid property dependency issues
        Chem.MolToMolFile(mol, cache_path)
    except Exception:
        # Non-fatal — caching is an optimization, not a requirement
        pass


def _mol_from_sdf_supplier(
    mol: Chem.Mol,
    target_property: str,
    seed: int,
    cache_dir: str,
) -> Optional[Data]:
    """
    Process a single molecule from an SDF supplier.

    Handles both 2D (no conformer) and 3D (has conformer) molecules.
    Generates conformers when needed, with disk caching.

    Returns:
        PyG Data with z, pos, and y (target value), or None on failure.
    """
    if mol is None or mol.GetNumAtoms() == 0:
        return None

    # Extract target property value
    if not mol.HasProp(target_property):
        return None

    try:
        target_value = float(mol.GetProp(target_property))
    except (ValueError, KeyError):
        return None

    # Get or generate 3D conformer
    conf = mol.GetConformer()
    has_3d = conf is not None and conf.Is3D()

    if not has_3d:
        # Generate 3D conformer — check cache first
        smiles = Chem.MolToSmiles(mol)
        if smiles:
            cache_key = _smiles_to_cache_key(smiles, seed)
            cache_path = os.path.join(cache_dir, f"{cache_key}.mol")

            cached_mol = _load_cached_conformer(cache_path)
            if cached_mol is not None:
                mol = cached_mol
            else:
                try:
                    mol = generate_3d_conformer(mol, seed=seed)
                    if smiles:
                        _save_conformer_to_cache(mol, cache_path)
                except Exception as e:
                    logger.debug(f"Conformer generation failed: {e}")
                    return None
        else:
            try:
                mol = generate_3d_conformer(mol, seed=seed)
            except Exception as e:
                logger.debug(f"Conformer generation failed (no SMILES): {e}")
                return None

    # Convert to Data object
    try:
        data = mol_to_3d_data(mol, target_property=None)
        data.y = torch.tensor([[target_value]], dtype=torch.float)
        # Store SMILES for reference (useful for debugging)
        if not has_3d:
            smiles = Chem.MolToSmiles(mol)
            if smiles:
                data.smiles = smiles
        return data
    except Exception as e:
        logger.debug(f"mol_to_3d_data failed: {e}")
        return None


class Geometry3DDataset(Dataset):
    """
    PyG Dataset that produces 3D geometry data (z, pos, y) from SDF files.

    Each sample is a Data object with:
      - z: [num_atoms] atomic numbers (torch.long)
      - pos: [num_atoms, 3] 3D coordinates in Angstroms (torch.float)
      - y: [1] regression target value (torch.float)

    Args:
        root_dir: Root directory for caching processed graphs.
        sdf_files: List of paths to .sdf files.
        target_property: Molecule property name to use as regression target.
        seed: Random seed for conformer generation (reproducibility).
        cache_dir: Directory for conformer caching. Defaults to
                   ~/.cache/chem_augur/conformers/.
        transform: Standard PyG transform callable.
        pre_filter: Standard PyG pre-filter callable.
        pre_transform: Standard PyG pre-transform callable.
    """

    def __init__(
        self,
        root_dir: str,
        sdf_files: List[str],
        target_property: str = "PUBCHEM_EXACT_MASS",
        seed: int = 42,
        cache_dir: Optional[str] = None,
        transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        self.target_property = target_property
        self.sdf_files = sdf_files
        self.seed = seed
        self.cache_dir = cache_dir or DEFAULT_CACHE_DIR

        sdf_hash = "_".join(Path(f).stem for f in sdf_files)
        processed_dir = os.path.join(
            root_dir, f"processed_3d_{sdf_hash}_{target_property}"
        )
        super().__init__(
            root=root_dir,
            processed_dir=processed_dir,
            transform=transform,
            pre_filter=pre_filter,
            pre_transform=pre_transform,
        )
        self._collated_data = None
        self._collated_slices = None

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    @property
    def raw_file_names(self) -> List[str]:
        return self.sdf_files

    @property
    def processed_file_names(self) -> List[str]:
        return ["graphs_3d.pt"]

    def _ensure_loaded(self):
        """Load processed data from disk once and cache it."""
        if self._collated_data is None:
            self._collated_data, self._collated_slices = torch.load(
                self.processed_paths[0], weights_only=False
            )

    def len(self) -> int:
        self._ensure_loaded()
        return len(self._collated_data)

    def get(self, idx: int) -> Data:
        self._ensure_loaded()
        return self._collated_data[idx]

    # ------------------------------------------------------------------
    # Processing
    # ------------------------------------------------------------------

    def process(self):
        """
        Process all SDF files: read molecules, generate 3D conformers,
        extract z/pos/y, and cache to disk.
        """
        data_list = []
        skipped_no_property = 0
        skipped_no_conformer = 0
        total_read = 0

        for sdf_path in self.sdf_files:
            if not os.path.exists(sdf_path):
                logger.warning(f"SDF file not found: {sdf_path}")
                continue

            supplier = Chem.SDMolSupplier(sdf_path)
            for mol in supplier:
                total_read += 1
                if mol is None:
                    continue

                data = _mol_from_sdf_supplier(
                    mol,
                    target_property=self.target_property,
                    seed=self.seed,
                    cache_dir=self.cache_dir,
                )

                if data is None:
                    if not mol.HasProp(self.target_property):
                        skipped_no_property += 1
                    else:
                        skipped_no_conformer += 1
                    continue

                if self.pre_filter is not None and not self.pre_filter(data):
                    continue
                if self.pre_transform is not None:
                    data = self.pre_transform(data)

                data_list.append(data)

        logger.info(
            f"Geometry3DDataset: {total_read} molecules read, "
            f"{len(data_list)} valid graphs generated. "
            f"Skipped: {skipped_no_property} (no property), "
            f"{skipped_no_conformer} (conformer failed)"
        )

        if len(data_list) == 0:
            raise RuntimeError(
                "No valid graphs generated. Check that SDF files contain "
                f"molecules with the '{self.target_property}' property."
            )

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        logger.info(
            f"Processed {len(data_list)} 3D graphs -> {self.processed_paths[0]}"
        )

    # ------------------------------------------------------------------
    # Split utilities
    # ------------------------------------------------------------------

    @staticmethod
    def split(
        dataset: "Geometry3DDataset",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> Tuple[List[int], List[int], List[int]]:
        """
        Random split into train / val / test index lists.

        Args:
            dataset: The dataset to split.
            train_ratio: Fraction for training.
            val_ratio: Fraction for validation.
            test_ratio: Fraction for testing.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_indices, val_indices, test_indices).
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        N = len(dataset)
        indices = torch.randperm(
            N, generator=torch.Generator().manual_seed(seed)
        )
        train_idx = indices[: int(N * train_ratio)].tolist()
        val_idx = indices[
            int(N * train_ratio) : int(N * (train_ratio + val_ratio))
        ].tolist()
        test_idx = indices[int(N * (train_ratio + val_ratio)) :].tolist()
        return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Normalization helper
# ---------------------------------------------------------------------------

def normalize_geometry_dataset(
    dataset: Geometry3DDataset,
    train_indices: List[int],
) -> StandardScaler:
    """
    Normalize target values in a Geometry3DDataset.

    Computes mean/std from the training split and applies z-score
    normalization to all samples in-place.

    Args:
        dataset: The dataset to normalize.
        train_indices: Indices of the training split.

    Returns:
        Fitted StandardScaler (for inverse-transforming predictions).
    """
    scaler = StandardScaler()
    ys = torch.stack([dataset[idx].y for idx in train_indices])
    scaler.fit(ys)

    for idx in range(len(dataset)):
        data = dataset[idx]
        data.y = scaler.transform(data.y)

    return scaler