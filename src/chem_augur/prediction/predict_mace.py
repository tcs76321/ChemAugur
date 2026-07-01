# prediction/predict_mace.py
"""
Prediction module for geometry-based molecular property prediction.

Loads a trained MACE (or any registered model) checkpoint and predicts
properties for pure substances or mixtures from SMILES strings alone.

The model uses only atomic numbers z and 3D positions pos — no bond types,
hybridization labels, or hand-crafted features. Conformers are generated
automatically from SMILES using ETKDG + MMFF optimization.

Usage (CLI):
    # Pure substance
    python -m chem_augur.prediction.predict_mace "CCO"

    # Mixture
    python -m chem_augur.prediction.predict_mace --mixture "CCO" "O" --ratios 0.5 0.5

    # With custom checkpoint
    python -m chem_augur.prediction.predict_mace "CCO" --checkpoint path/to/best.pt

Usage (API):
    from chem_augur.prediction.predict_mace import predict_property, predict_mixture

    bp = predict_property("CCO")
    bp = predict_mixture(["CCO", "O"], [0.5, 0.5])
"""
from __future__ import annotations

import os
import logging
from typing import Optional, List, Dict, Any

import torch

from chem_augur.features.conformer_generation import smiles_to_3d_data
from chem_augur.features.mixture_builder import build_mixture
from chem_augur.features.molecular_dataset import StandardScaler
from chem_augur.models.registry import build_model
from chem_augur.models.gnn.mace_model import load_mace_checkpoint

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Default checkpoint path
# ---------------------------------------------------------------------------

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "..", "checkpoints", "best_model.pt"
)

# ---------------------------------------------------------------------------
# Module-level model cache — load once, predict many times
# ---------------------------------------------------------------------------

_cached_model: Optional[torch.nn.Module] = None
_cached_scaler: Optional[StandardScaler] = None
_cached_device: Optional[torch.device] = None
_cached_checkpoint: Optional[str] = None


def _get_device() -> torch.device:
    """Get the best available device."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def _load_model(
    checkpoint_path: str,
    device: torch.device,
) -> tuple:
    """
    Load model and scaler from a checkpoint, with caching.

    If the same checkpoint was already loaded, returns the cached model.
    This avoids reloading the model for every prediction call.
    """
    global _cached_model, _cached_scaler, _cached_device, _cached_checkpoint

    checkpoint_path = os.path.abspath(checkpoint_path)

    # Return cached if same checkpoint
    if (
        _cached_model is not None
        and _cached_checkpoint == checkpoint_path
        and _cached_device == device
    ):
        return _cached_model, _cached_scaler

    logger.info(f"Loading model from {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Reconstruct model from saved config
    config = checkpoint.get("config", checkpoint.get("model_config", {}))
    model_name = config.get("model_name", "mace")

    model = build_model(
        model_name,
        hidden_channels=config.get("hidden_channels", 128),
        num_interactions=config.get("num_interactions", 2),
        max_ell=config.get("max_ell", 3),
        num_cutoff_basis=config.get("num_cutoff_basis", 5),
        correlation=config.get("correlation", 3),
        cutoff=config.get("cutoff", 12.0),
        out_channels=config.get("out_channels", 1),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    # Reconstruct scaler
    scaler = StandardScaler()
    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    # Cache
    _cached_model = model
    _cached_scaler = scaler
    _cached_device = device
    _cached_checkpoint = checkpoint_path

    return model, scaler


# ---------------------------------------------------------------------------
# Prediction API
# ---------------------------------------------------------------------------

def predict_property(
    smiles: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    seed: int = 42,
) -> float:
    """
    Predict a physical property for a single pure substance.

    Generates a 3D conformer from the SMILES string, extracts atomic numbers
    and positions, and passes them through the trained model. Returns the
    prediction in the original (un-normalized) property scale.

    Args:
        smiles: SMILES string (e.g., "CCO" for ethanol).
        checkpoint_path: Path to a trained model checkpoint.
        seed: Random seed for conformer generation.

    Returns:
        Predicted property value (float).

    Raises:
        ValueError: If the SMILES cannot be parsed.
        FileNotFoundError: If the checkpoint doesn't exist.
        RuntimeError: If conformer generation or prediction fails.

    Example:
        >>> bp = predict_property("CCO")
        >>> print(f"Predicted boiling point: {bp:.1f} °C")
    """
    device = _get_device()
    model, scaler = _load_model(checkpoint_path, device)

    # Generate 3D geometry from SMILES
    data = smiles_to_3d_data(smiles, seed=seed)
    z = data.z.to(device)
    pos = data.pos.to(device)

    # Run prediction
    with torch.no_grad():
        batch = torch.zeros(z.shape[0], dtype=torch.long, device=device)
        pred = model(z, pos, batch)

    # Inverse-transform to original scale
    original_pred = scaler.inverse_transform(pred)
    return original_pred.item()


def predict_mixture(
    smiles_list: List[str],
    mole_ratios: List[float],
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    num_molecules: int = 10,
    seed: int = 42,
    relax: bool = True,
) -> float:
    """
    Predict a physical property for a mixture of chemical species.

    Places molecules of each species in a shared 3D coordinate space at
    the specified mole ratios, then passes the combined atomic numbers and
    positions through the trained model.

    The model has no concept of "molecule identity" — it sees only atoms
    in space and learns how distance/angle/dihedral patterns between
    different elements produce properties.

    Args:
        smiles_list: List of SMILES strings, one per species.
        mole_ratios: Mole fractions (should sum to 1.0).
        checkpoint_path: Path to a trained model checkpoint.
        num_molecules: Total number of molecules to place (default 10).
            More molecules give better statistics but cost more compute.
        seed: Random seed for placement and conformer generation.
        relax: Whether to perform force-field relaxation after placement.

    Returns:
        Predicted mixture property value (float).

    Raises:
        ValueError: If SMILES cannot be parsed or ratios don't sum to 1.0.
        FileNotFoundError: If the checkpoint doesn't exist.

    Example:
        >>> bp = predict_mixture(["CCO", "O"], [0.5, 0.5])
        >>> print(f"Predicted mixture boiling point: {bp:.1f} °C")
    """
    device = _get_device()
    model, scaler = _load_model(checkpoint_path, device)

    # Build multi-molecule system
    z, pos = build_mixture(
        smiles_list=smiles_list,
        mole_ratios=mole_ratios,
        num_molecules=num_molecules,
        seed=seed,
        relax=relax,
    )
    z = z.to(device)
    pos = pos.to(device)

    # Run prediction
    with torch.no_grad():
        batch = torch.zeros(z.shape[0], dtype=torch.long, device=device)
        pred = model(z, pos, batch)

    # Inverse-transform to original scale
    original_pred = scaler.inverse_transform(pred)
    return original_pred.item()


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(
    smiles_list: List[str],
    checkpoint_path: str = DEFAULT_CHECKPOINT,
    seed: int = 42,
) -> List[float]:
    """
    Predict properties for a batch of SMILES strings.

    Each SMILES is processed independently (single-molecule prediction).

    Args:
        smiles_list: List of SMILES strings.
        checkpoint_path: Path to a trained model checkpoint.
        seed: Random seed for conformer generation.

    Returns:
        List of predicted property values, same order as input.
    """
    results = []
    for smiles in smiles_list:
        try:
            val = predict_property(smiles, checkpoint_path, seed)
            results.append(val)
        except Exception as e:
            logger.error(f"Failed to predict {smiles}: {e}")
            results.append(float("nan"))
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    """Parse command-line arguments for prediction."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Predict molecular properties from SMILES using a "
                    "geometry-based model (MACE)."
    )
    parser.add_argument(
        "smiles",
        nargs="*",
        help="SMILES string(s) to predict. For mixtures, use --mixture instead.",
    )
    parser.add_argument(
        "--mixture",
        action="store_true",
        default=False,
        help="Treat SMILES arguments as a mixture.",
    )
    parser.add_argument(
        "--ratios",
        type=float,
        nargs="+",
        default=None,
        help="Mole ratios for mixture prediction (must sum to 1.0).",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=DEFAULT_CHECKPOINT,
        help="Path to trained model checkpoint.",
    )
    parser.add_argument(
        "--num-molecules",
        type=int,
        default=10,
        help="Number of molecules to place for mixture prediction.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for conformer generation.",
    )
    parser.add_argument(
        "--no-relax",
        action="store_true",
        default=False,
        help="Skip force-field relaxation for mixtures.",
    )

    return parser.parse_args()


def main() -> None:
    """CLI entry point for property prediction."""
    args = _parse_args()

    if not args.smiles:
        print("Error: At least one SMILES string is required.")
        print("Example: python -m chem_augur.prediction.predict_mace CCO")
        print("Mixture: python -m chem_augur.prediction.predict_mace --mixture CCO O --ratios 0.5 0.5")
        return

    try:
        if args.mixture:
            if args.ratios is None:
                # Default to equal ratios if not specified
                ratios = [1.0 / len(args.smiles)] * len(args.smiles)
            else:
                ratios = args.ratios

            value = predict_mixture(
                smiles_list=args.smiles,
                mole_ratios=ratios,
                checkpoint_path=args.checkpoint,
                num_molecules=args.num_molecules,
                seed=args.seed,
                relax=not args.no_relax,
            )
            mixture_str = " + ".join(args.smiles)
            print(f"Predicted property for mixture ({mixture_str}): {value:.4f}")
        else:
            for smiles in args.smiles:
                value = predict_property(
                    smiles,
                    checkpoint_path=args.checkpoint,
                    seed=args.seed,
                )
                print(f"Predicted property for '{smiles}': {value:.4f}")

    except FileNotFoundError as e:
        print(f"Error: Checkpoint not found: {e}")
        print(f"Make sure '{args.checkpoint}' exists (train the model first).")
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    main()