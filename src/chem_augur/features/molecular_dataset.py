# features/molecular_dataset.py
"""
MolecularDataset – a PyTorch Geometric Dataset that loads molecules from SDF files,
converts them to graph Data objects with rich features, and supports train/val/test
splits with target normalization.
"""
import os
from pathlib import Path
from typing import List, Optional, Callable

from rdkit import Chem
import torch
from torch_geometric.data import Dataset, Data

from chem_augur.features.molecular_graph import mol_to_graph


class MolecularDataset(Dataset):
    """
    Dataset backed by one or more SDF files.

    Args:
        root_dir:          Root directory for caching processed graphs.
        sdf_files:         List of paths to .sdf files.
        target_property:   Molecule property name to use as regression target.
        transform:         Standard PyG transform callable.
        pre_filter:        Standard PyG pre-filter callable.
        pre_transform:     Standard PyG pre-transform callable (applied before caching).
    """

    def __init__(
        self,
        root_dir: str,
        sdf_files: List[str],
        target_property: str = "PUBCHEM_EXACT_MASS",
        transform: Optional[Callable] = None,
        pre_filter: Optional[Callable] = None,
        pre_transform: Optional[Callable] = None,
    ):
        sdf_hash = "_".join(Path(f).stem for f in sdf_files)
        processed_dir = os.path.join(root_dir, f"processed_{sdf_hash}_{target_property}")
        super().__init__(root=root_dir, processed_dir=processed_dir,
                         transform=transform, pre_filter=pre_filter,
                         pre_transform=pre_transform)
        self.target_property = target_property
        self.sdf_files = sdf_files

    # ---------------------------------------------------------------
    # Dataset interface
    # ---------------------------------------------------------------
    @property
    def raw_file_names(self) -> List[str]:
        return self.sdf_files

    @property
    def processed_file_names(self) -> List[str]:
        return ["graphs.pt"]

    def len(self) -> int:
        data, _ = torch.load(self.processed_paths[0])
        return len(data)

    def get(self, idx: int) -> Data:
        data, _ = torch.load(self.processed_paths[0])
        return data[idx]

    # ---------------------------------------------------------------
    # Processing
    # ---------------------------------------------------------------
    def process(self):
        data_list = []
        for sdf_path in self.sdf_files:
            supplier = Chem.SDMolSupplier(sdf_path)
            for mol in supplier:
                if mol is None:
                    continue
                graph = mol_to_graph(mol, target_property=self.target_property)
                if graph is None:
                    continue
                if self.pre_filter is not None and not self.pre_filter(graph):
                    continue
                if self.pre_transform is not None:
                    graph = self.pre_transform(graph)
                data_list.append(graph)

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        print(f"Processed {len(data_list)} graphs -> {self.processed_paths[0]}")

    # ---------------------------------------------------------------
    # Split utilities
    # ---------------------------------------------------------------
    @staticmethod
    def split(
        dataset: "MolecularDataset",
        train_ratio: float = 0.8,
        val_ratio: float = 0.1,
        test_ratio: float = 0.1,
        seed: int = 42,
    ) -> tuple:
        """Random split into train / val / test index lists."""
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6
        N = len(dataset)
        indices = torch.randperm(N, generator=torch.Generator().manual_seed(seed))
        train_idx = indices[: int(N * train_ratio)].tolist()
        val_idx = indices[int(N * train_ratio): int(N * (train_ratio + val_ratio))].tolist()
        test_idx = indices[int(N * (train_ratio + val_ratio)):].tolist()
        return train_idx, val_idx, test_idx


# ---------------------------------------------------------------------------
# Target normalization helpers
# ---------------------------------------------------------------------------

class StandardScaler:
    """Compute mean/std on training targets; transform anywhere."""

    def __init__(self):
        self.mean = 0.0
        self.std = 1.0

    def fit(self, y: torch.Tensor):
        self.mean = y.mean().item()
        self.std = y.std().item() if y.std() > 0 else 1.0

    def transform(self, y: torch.Tensor) -> torch.Tensor:
        return (y - self.mean) / self.std

    def inverse_transform(self, y: torch.Tensor) -> torch.Tensor:
        return y * self.std + self.mean

    def state_dict(self):
        return {"mean": self.mean, "std": self.std}

    def load_state_dict(self, state):
        self.mean = state["mean"]
        self.std = state["std"]


def normalize_dataset(dataset: MolecularDataset, indices: List[int]):
    """
    Normalize target values in *dataset* using statistics computed from
    the molecules at *indices* (typically the training split).

    Returns the fitted StandardScaler.
    """
    scaler = StandardScaler()
    ys = torch.stack([dataset[idx].y for idx in indices])
    scaler.fit(ys)
    # Overwrite y in-place for every sample
    for idx in range(len(dataset)):
        data = dataset[idx]
        data.y = scaler.transform(data.y)
        # PyG datasets store references, so mutating is safe
    return scaler