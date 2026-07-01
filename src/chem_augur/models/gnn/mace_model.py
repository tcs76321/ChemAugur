# models/gnn/mace_model.py
"""
MACE (Multi-body Atomic Cluster Expansion) model wrapper for ChemAugur.

Wraps the MACE equivariant message-passing model in a PyG-compatible interface
that takes atomic numbers z and 3D positions pos and predicts a scalar property.

MACE captures distances, angles, dihedrals, and higher-order many-body
correlations through a hierarchical ACE basis expansion with O(3)-equivariant
tensor features (l=0,1,2,3). This is the most complete geometry-based model
for molecular property prediction from first principles.

Architecture:
    z, pos, batch → MACE (internal radius_graph + many-body expansion)
                   → global pooling → output MLP → scalar prediction

The wrapper isolates MACE internals behind a clean forward(z, pos, batch)
signature, so the training pipeline and prediction module are model-agnostic.
"""
from typing import Optional, Dict, Any

import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Fallback implementation when mace-torch is not installed.
# This is a placeholder that raises a clear error on forward() so users
# know exactly what to install.
# ---------------------------------------------------------------------------

class _MaceNotInstalledError(nn.Module):
    """Placeholder that raises a helpful error when mace-torch is absent."""

    def __init__(self):
        super().__init__()
        self._msg = (
            "mace-torch is not installed. Install it with:\n"
            "    pip install mace-torch\n"
            "Then import this module again to use the full MACE model."
        )

    def forward(self, z, pos, batch=None):
        raise ImportError(self._msg)


# Attempt to import the real MACE model
try:
    # mace-torch provides several model classes. The `MACE` class is the
    # core equivariant model. We wrap it to match our forward() signature.
    from mace.modules.models import MACE  # type: ignore

    _MACE_AVAILABLE = True
except ImportError:
    _MACE_AVAILABLE = False


# ---------------------------------------------------------------------------
# Main wrapper class
# ---------------------------------------------------------------------------

class MACEWrapper(nn.Module):
    """
    PyG-compatible wrapper around the MACE equivariant model.

    Takes only atomic numbers (z) and 3D positions (pos) — no bond types,
    hybridization labels, aromaticity flags, or any hand-crafted features.
    MACE learns all chemical semantics from geometry via its many-body
    expansion over the radius graph.

    Args:
        num_elements: Maximum atomic number + 1 (for embedding table).
            Default 100 covers all elements up to Fermium.
        hidden_channels: Dimension of internal node embeddings (default 128).
        num_interactions: Number of message-passing interaction blocks.
            Each block performs one round of equivariant message passing.
        max_ell: Maximum spherical harmonic order (l) for tensor features.
            0=scalar only, 1=scalar+vector, 2=scalar+vector+tensor,
            3=full (default, for best accuracy).
        num_cutoff_basis: Number of radial basis functions for distance
            expansion inside the cutoff radius.
        correlation: Many-body correlation order. 2=pairs+triplets,
            3=pairs+triplets+quartets (dihedrals). Higher values capture
            more geometric detail at increased computational cost.
        cutoff: Interaction cutoff radius in Angstroms (default 12.0).
            Higher values capture more solvation shells and long-range
            interactions but increase computational cost quadratically.
        out_channels: Number of output channels. 1 for single-property
            regression, >1 for multi-task prediction.
        r_max: Maximum interatomic distance for the radial basis. Should
            typically match or slightly exceed `cutoff`.
        num_radial_basis: Number of radial basis functions (alternative to
            num_cutoff_basis, depending on MACE version).
        num_before_skip: Number of residual layers before skip connection.
        num_after_skip: Number of residual layers after skip connection.
        avg_num_neighbors: Average number of neighbors per atom (used for
            normalization, MACE computes this automatically if None).
        **kwargs: Additional keyword arguments passed to MACE constructor.
    """

    def __init__(
        self,
        num_elements: int = 100,
        hidden_channels: int = 128,
        num_interactions: int = 2,
        max_ell: int = 3,
        num_cutoff_basis: int = 5,
        correlation: int = 3,
        cutoff: float = 12.0,
        out_channels: int = 1,
        r_max: Optional[float] = None,
        num_radial_basis: int = 8,
        num_before_skip: int = 1,
        num_after_skip: int = 2,
        avg_num_neighbors: Optional[float] = None,
        **kwargs,
    ):
        super().__init__()

        if not _MACE_AVAILABLE:
            self._model = _MaceNotInstalledError()
            return

        if r_max is None:
            r_max = cutoff

        self.cutoff = cutoff
        self.out_channels = out_channels

        # MACE model construction.
        # The MACE class from mace-torch takes configuration through
        # keyword arguments. We map our clean parameter names to theirs.
        self._model = MACE(
            r_max=r_max,
            num_bessel=num_radial_basis,
            num_polynomial_cutoff=num_cutoff_basis,
            max_ell=max_ell,
            correlation=correlation,
            num_interactions=num_interactions,
            hidden_irreps=f"{hidden_channels}x0e + {hidden_channels}x1o + {hidden_channels}x2e",
            MLP_irreps=f"{hidden_channels}x0e",
            atomic_energies=None,  # No per-element bias for property prediction
            avg_num_neighbors=avg_num_neighbors,
            atomic_numbers=range(num_elements),
            num_radial_basis=num_radial_basis,
            num_cutoff_basis=num_cutoff_basis,
            **kwargs,
        )

        # Output head: pool over all atoms → predict scalar property
        self.output_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.SiLU(),
            nn.Linear(hidden_channels // 2, out_channels),
        )

        self._hidden_channels = hidden_channels
        self._num_interactions = num_interactions
        self._max_ell = max_ell
        self._correlation = correlation
        self._num_elements = num_elements

    def forward(
        self,
        z: torch.Tensor,
        pos: torch.Tensor,
        batch: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass: z + pos → property prediction.

        Args:
            z: [N] atomic numbers (torch.long). Values should be in [1, 100].
            pos: [N, 3] 3D atom positions in Angstroms (torch.float).
            batch: [N] batch indices for batched graphs (torch.long).
                   If None, assumes a single graph.

        Returns:
            Tensor of shape [batch_size, out_channels] with predicted
            property values (un-normalized).
        """
        if batch is None:
            batch = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)

        # MACE forward pass — the model internally builds the radius graph,
        # computes pairwise distances, angles, dihedrals, and applies
        # equivariant message passing.
        node_features = self._model(
            dict(
                positions=pos,
                numbers=z,
                batch=batch,
            )
        )

        # node_features shape: [N, hidden_channels] or dictionary of irreps
        # Handle dictionary-style output (MACE returns per-irrep tensors)
        if isinstance(node_features, dict):
            # Get scalar (l=0) features for pooling
            scalar_key = [k for k in node_features.keys() if "0e" in str(k)]
            if scalar_key:
                node_features = node_features[scalar_key[0]]
            else:
                # Fallback: use first available feature
                node_features = list(node_features.values())[0]

        # Global mean pooling over atoms in each graph
        num_graphs = batch.max().item() + 1 if len(batch) > 0 else 1
        pooled = torch.zeros(num_graphs, node_features.shape[1],
                            device=node_features.device)

        for i in range(num_graphs):
            mask = (batch == i)
            if mask.any():
                pooled[i] = node_features[mask].mean(dim=0)

        # Output head
        return self.output_head(pooled)

    def get_config(self) -> Dict[str, Any]:
        """Return model hyperparameters for checkpoint serialization."""
        return {
            "hidden_channels": self._hidden_channels,
            "num_interactions": self._num_interactions,
            "max_ell": self._max_ell,
            "cutoff": self.cutoff,
            "correlation": self._correlation,
            "num_elements": self._num_elements,
            "out_channels": self.out_channels,
        }


# ---------------------------------------------------------------------------
# Checkpoint utilities
# ---------------------------------------------------------------------------

def save_mace_checkpoint(
    path: str,
    model: MACEWrapper,
    scaler: Any = None,
    **metadata,
) -> None:
    """
    Save model, scaler, and metadata to a single checkpoint file.

    Args:
        path: Path to save the checkpoint (.pt file).
        model: The MACEWrapper model instance.
        scaler: Optional StandardScaler for target normalization.
        **metadata: Additional metadata (epoch, metrics, config, etc.).
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "model_config": model.get_config(),
    }
    if scaler is not None:
        checkpoint["scaler_state"] = scaler.state_dict()
    checkpoint.update(metadata)
    torch.save(checkpoint, path)


def load_mace_checkpoint(
    path: str,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """
    Load a checkpoint and reconstruct the model + scaler.

    Args:
        path: Path to the checkpoint file.
        device: Device to load the model on.

    Returns:
        Dictionary with keys:
          - "model": MACEWrapper instance (in eval mode)
          - "scaler_params": dict of scaler state (or None)
          - "metadata": dict of additional metadata
    """
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model = MACEWrapper(**checkpoint["model_config"])
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scaler_params = checkpoint.get("scaler_state", None)

    metadata = {
        k: v for k, v in checkpoint.items()
        if k not in ("model_state_dict", "model_config", "scaler_state")
    }

    return {
        "model": model,
        "scaler_params": scaler_params,
        "metadata": metadata,
    }