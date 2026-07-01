# training/train_mace.py
"""
Unified training pipeline for geometry-based molecular property prediction.

Trains a MACE (or any registered model) on single-molecule or mixture data
using only atomic numbers z and 3D positions pos. The same training loop
handles pure substances and mixtures identically — the model sees only
atoms in space.

Features:
  - 3D conformer generation with disk caching
  - Train / validation / test splits
  - Target normalization (StandardScaler fitted on training split)
  - Epoch-based training with early stopping
  - MAE + RMSE metrics every epoch
  - Model checkpointing (best validation loss)
  - Learning-rate scheduling (ReduceLROnPlateau)
  - TensorBoard logging
  - Python logging module (no print() statements)
  - Configuration via TrainingConfig dataclass or TOML file
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime
from typing import Optional, List, Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from torch.utils.data import Subset
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data

from chem_augur.config import TrainingConfig
from chem_augur.features.conformer_generation import (
    generate_3d_conformer,
    mol_to_3d_data,
)
from chem_augur.features.molecular_dataset import StandardScaler
from chem_augur.features.geometry_dataset import (
    Geometry3DDataset,
    normalize_geometry_dataset,
)
from chem_augur.models.registry import build_model

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Optional[str] = None, level: int = logging.INFO) -> None:
    """Configure Python logging with console and optional file output."""
    handlers: List[logging.Handler] = [logging.StreamHandler()]

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_path = os.path.join(log_dir, f"train_{timestamp}.log")
        handlers.append(logging.FileHandler(log_path))

    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Mean absolute error."""
    return (pred - target).abs().mean().item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    """Root mean squared error."""
    return torch.sqrt(((pred - target) ** 2).mean()).item()


# ---------------------------------------------------------------------------
# 3D Dataset (geometric, no 2D features)
# ---------------------------------------------------------------------------

class GeometryDataset(torch.utils.data.Dataset):
    """
    Dataset of 3D molecular geometries with optional property labels.

    Each sample is a Data object with z, pos, and optionally y.
    Supports both single-molecule and mixture data.

    For single molecules: each sample is one molecule with a 3D conformer.
    For mixtures: each sample is a multi-molecule system built by
    mixture_builder.build_mixture().

    Args:
        samples: List of PyG Data objects, each with z, pos, and optionally y.
    """

    def __init__(self, samples: List[Data]):
        self._samples = samples

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Data:
        return self._samples[idx]


# ---------------------------------------------------------------------------
# Training and evaluation
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: optim.Optimizer,
    loss_fn: nn.Module,
    scaler: StandardScaler,
    device: torch.device,
    grad_clip_norm: Optional[float] = None,
) -> Tuple[float, float, float]:
    """Run one training epoch. Returns (avg_loss, mae, rmse)."""
    model.train()
    total_loss = 0.0
    total_graphs = 0
    all_preds, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch.z, batch.pos, batch.batch)
        loss = loss_fn(pred, batch.y)

        loss.backward()

        if grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)

        optimizer.step()

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

        all_preds.append(scaler.inverse_transform(pred.detach()))
        all_targets.append(scaler.inverse_transform(batch.y.detach()))

    avg_loss = total_loss / total_graphs
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return avg_loss, mae(preds, targets), rmse(preds, targets)


@torch.no_grad()
def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    scaler: StandardScaler,
    device: torch.device,
) -> Tuple[float, float, float]:
    """Evaluate model on a dataset. Returns (avg_loss, mae, rmse)."""
    model.eval()
    total_loss = 0.0
    total_graphs = 0
    all_preds, all_targets = [], []

    for batch in loader:
        batch = batch.to(device)
        pred = model(batch.z, batch.pos, batch.batch)
        loss = loss_fn(pred, batch.y)

        total_loss += loss.item() * batch.num_graphs
        total_graphs += batch.num_graphs

        all_preds.append(scaler.inverse_transform(pred))
        all_targets.append(scaler.inverse_transform(batch.y))

    avg_loss = total_loss / total_graphs
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return avg_loss, mae(preds, targets), rmse(preds, targets)


# ---------------------------------------------------------------------------
# Checkpointing
# ---------------------------------------------------------------------------

def save_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: optim.Optimizer,
    scaler: StandardScaler,
    epoch: int,
    best_val_loss: float,
    config: TrainingConfig,
) -> None:
    """Save training checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state": scaler.state_dict(),
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "config": config.to_dict(),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    torch.save(checkpoint, path)
    logger.info(f"Checkpoint saved to {path}")


def load_checkpoint(
    path: str,
    model: nn.Module,
    optimizer: Optional[optim.Optimizer] = None,
    device: torch.device = torch.device("cpu"),
) -> dict:
    """Load training checkpoint. Returns metadata dict."""
    checkpoint = torch.load(path, map_location=device, weights_only=False)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    scaler = StandardScaler()
    if "scaler_state" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler_state"])

    return {
        "epoch": checkpoint.get("epoch", 0),
        "best_val_loss": checkpoint.get("best_val_loss", float("inf")),
        "scaler": scaler,
        "config": checkpoint.get("config", {}),
    }


# ---------------------------------------------------------------------------
# Data preparation
# ---------------------------------------------------------------------------

def prepare_geometry_data(
    config: TrainingConfig,
    use_sdf: bool = False,
) -> Tuple[GeometryDataset, StandardScaler, List[int], List[int], List[int]]:
    """
    Load or generate 3D geometry data, split into train/val/test, and
    normalize targets.

    When use_sdf=True and SDF files are available, uses Geometry3DDataset
    to load real data with conformer generation and caching. Otherwise,
    falls back to a small synthetic dataset for testing.

    Returns:
        dataset: GeometryDataset with all samples.
        scaler: Fitted StandardScaler for target normalization.
        train_idx, val_idx, test_idx: Split indices.
    """
    logger.info("Preparing geometry data...")

    # Check if real SDF data is available
    sdf_available = any(
        os.path.exists(f) for f in config.sdf_files
    )

    if use_sdf and sdf_available:
        logger.info("Loading real data from SDF files via Geometry3DDataset...")
        try:
            geo_dataset = Geometry3DDataset(
                root_dir=config.root_dir,
                sdf_files=config.sdf_files,
                target_property=config.target_property,
                seed=config.seed,
            )

            if len(geo_dataset) == 0:
                logger.warning(
                    "Geometry3DDataset produced 0 samples. "
                    "Falling back to synthetic data."
                )
            else:
                train_idx, val_idx, test_idx = Geometry3DDataset.split(
                    geo_dataset,
                    train_ratio=config.train_ratio,
                    val_ratio=config.val_ratio,
                    test_ratio=config.test_ratio,
                    seed=config.seed,
                )

                logger.info(
                    f"Split: train={len(train_idx)}, val={len(val_idx)}, "
                    f"test={len(test_idx)}"
                )

                scaler = normalize_geometry_dataset(geo_dataset, train_idx)
                logger.info(
                    f"Target stats: mean={scaler.mean:.4f}, "
                    f"std={scaler.std:.4f}"
                )

                return geo_dataset, scaler, train_idx, val_idx, test_idx
        except Exception as e:
            logger.warning(
                f"Geometry3DDataset failed: {e}. "
                "Falling back to synthetic data."
            )

    # Fallback: synthetic dataset for testing
    logger.info("Using synthetic placeholder dataset.")
    test_smiles = [
        "CCO",          # ethanol
        "CC(C)O",       # isopropanol
        "CCCC",         # butane
        "CC(=O)O",      # acetic acid
        "c1ccccc1",     # benzene
        "CC(=O)OC",     # methyl acetate
        "CCCCCC",       # hexane
        "O",            # water
        "CO",           # methanol
        "CC(C)=O",      # acetone
    ]

    synthetic_properties = [
        78.37,   # ethanol
        82.5,    # isopropanol
        -0.5,    # butane
        118.0,   # acetic acid
        80.1,    # benzene
        57.0,    # methyl acetate
        69.0,    # hexane
        100.0,   # water
        64.7,    # methanol
        56.0,    # acetone
    ]

    samples = []
    for smiles, prop in zip(test_smiles, synthetic_properties):
        try:
            data = _generate_single_sample(smiles, prop, config)
            if data is not None:
                samples.append(data)
        except Exception as e:
            logger.warning(f"Failed to generate data for {smiles}: {e}")

    logger.info(f"Generated {len(samples)} synthetic geometry samples")

    dataset = GeometryDataset(samples)

    n = len(dataset)
    indices = torch.randperm(
        n, generator=torch.Generator().manual_seed(config.seed)
    ).tolist()

    train_end = int(n * config.train_ratio)
    val_end = train_end + int(n * config.val_ratio)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    logger.info(
        f"Split: train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    scaler = StandardScaler()
    ys = torch.cat([dataset[i].y for i in train_idx])
    scaler.fit(ys)
    logger.info(f"Target stats: mean={scaler.mean:.4f}, std={scaler.std:.4f}")

    for i in range(len(dataset)):
        data = dataset[i]
        data.y = scaler.transform(data.y)

    return dataset, scaler, train_idx, val_idx, test_idx


def _generate_single_sample(
    smiles: str,
    property_value: float,
    config: TrainingConfig,
) -> Optional[Data]:
    """
    Generate a single 3D geometry sample from a SMILES string.

    For pure substances: one molecule placed in space.
    For mixtures (future): multiple molecules in shared space.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    mol_3d = generate_3d_conformer(mol, seed=config.seed)
    data = mol_to_3d_data(mol_3d, target_property=None)
    data.y = torch.tensor([[property_value]], dtype=torch.float)
    return data


# ---------------------------------------------------------------------------
# TensorBoard setup
# ---------------------------------------------------------------------------

def _setup_tensorboard(config: TrainingConfig):
    """Initialize TensorBoard writer if enabled."""
    if not config.tensorboard:
        return None

    try:
        from torch.utils.tensorboard import SummaryWriter
    except ImportError:
        logger.warning(
            "TensorBoard not available (install tensorboard). "
            "Disabling TensorBoard logging."
        )
        return None

    os.makedirs(config.log_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = config.experiment_name or f"mace_{timestamp}"
    log_path = os.path.join(config.log_dir, exp_name)
    return SummaryWriter(log_path)


# ---------------------------------------------------------------------------
# Main training loop
# ---------------------------------------------------------------------------

def train(
    config: TrainingConfig,
    resume_path: Optional[str] = None,
    use_sdf: bool = False,
    num_workers: int = 0,
) -> None:
    """
    Run the full training pipeline.

    Args:
        config: TrainingConfig with all hyperparameters.
        resume_path: Path to checkpoint to resume from (None = fresh start).
        use_sdf: Whether to use real SDF data via Geometry3DDataset.
        num_workers: Number of DataLoader worker processes.
    """
    config.validate()

    # Reproducibility
    torch.manual_seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Setup logging and TensorBoard
    setup_logging(config.log_dir)
    writer = _setup_tensorboard(config)

    # ---- Data ----
    dataset, scaler, train_idx, val_idx, test_idx = prepare_geometry_data(
        config, use_sdf=use_sdf,
    )

    train_loader = DataLoader(
        Subset(dataset, train_idx),
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        Subset(dataset, val_idx),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    test_loader = DataLoader(
        Subset(dataset, test_idx),
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    # ---- Model ----
    logger.info(f"Building model: {config.model_name}")
    model = build_model(
        config.model_name,
        hidden_channels=config.hidden_channels,
        num_interactions=config.num_interactions,
        max_ell=config.max_ell,
        num_cutoff_basis=config.num_cutoff_basis,
        correlation=config.correlation,
        cutoff=config.cutoff,
        out_channels=1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model parameters: {n_params:,}")

    # ---- Optimizer and scheduler ----
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=config.lr_factor,
        patience=config.lr_patience,
        verbose=True,
    )

    # ---- Resume from checkpoint ----
    start_epoch = 1
    best_val_loss = float("inf")
    patience_counter = 0

    if resume_path is not None:
        logger.info(f"Resuming from checkpoint: {resume_path}")
        meta = load_checkpoint(resume_path, model, optimizer, device)
        start_epoch = meta["epoch"] + 1
        best_val_loss = meta["best_val_loss"]
        ckpt_scaler = meta["scaler"]
        # Use the checkpoint's scaler for inverse transforms
        if ckpt_scaler is not None:
            scaler = ckpt_scaler
        logger.info(
            f"Resumed from epoch {meta['epoch']} "
            f"(best_val_loss={best_val_loss:.4f}), "
            f"starting at epoch {start_epoch}"
        )

    # ---- Training loop ----
    start_time = time.time()

    os.makedirs(config.checkpoint_dir, exist_ok=True)

    for epoch in range(start_epoch, config.max_epochs + 1):
        t0 = time.time()

        train_loss, train_mae_val, train_rmse_val = train_epoch(
            model, train_loader, optimizer, loss_fn, scaler, device,
            grad_clip_norm=config.grad_clip_norm,
        )
        val_loss, val_mae_val, val_rmse_val = eval_epoch(
            model, val_loader, loss_fn, scaler, device,
        )
        elapsed = time.time() - t0

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        logger.info(
            f"Epoch {epoch:3d}/{config.max_epochs}  "
            f"Train Loss={train_loss:.4f}  MAE={train_mae_val:.4f}  "
            f"RMSE={train_rmse_val:.4f}  "
            f"Val Loss={val_loss:.4f}  MAE={val_mae_val:.4f}  "
            f"RMSE={val_rmse_val:.4f}  "
            f"LR={current_lr:.2e}  {elapsed:.1f}s"
        )

        # TensorBoard
        if writer is not None:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("MAE/train", train_mae_val, epoch)
            writer.add_scalar("MAE/val", val_mae_val, epoch)
            writer.add_scalar("RMSE/train", train_rmse_val, epoch)
            writer.add_scalar("RMSE/val", val_rmse_val, epoch)
            writer.add_scalar("LR", current_lr, epoch)

        # Early stopping and checkpointing
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pt")
            save_checkpoint(
                ckpt_path, model, optimizer, scaler,
                epoch, best_val_loss, config,
            )
            logger.info(f"  -> New best model (val_loss={val_loss:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(
                    f"Early stopping at epoch {epoch} "
                    f"(patience={config.patience})"
                )
                break

    # ---- Final test evaluation ----
    logger.info("Loading best checkpoint for final evaluation...")
    ckpt_path = os.path.join(config.checkpoint_dir, "best_model.pt")
    if os.path.exists(ckpt_path):
        meta = load_checkpoint(ckpt_path, model, device=device)
        test_scaler = meta["scaler"]
        logger.info(
            f"Loaded checkpoint from epoch {meta['epoch']} "
            f"(best_val_loss={meta['best_val_loss']:.4f})"
        )
    else:
        test_scaler = scaler
        logger.warning(
            "No checkpoint found — evaluating with current model state."
        )

    test_loss, test_mae_val, test_rmse_val = eval_epoch(
        model, test_loader, loss_fn, test_scaler, device,
    )
    logger.info(
        f"Test  Loss={test_loss:.4f}  MAE={test_mae_val:.4f}  "
        f"RMSE={test_rmse_val:.4f}"
    )

    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time:.0f}s")

    if writer is not None:
        writer.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for training."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Train a geometry-based model for molecular property prediction."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to TOML configuration file.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Model architecture (overrides config).",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=None,
        help="Interaction cutoff in Å (overrides config).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Maximum training epochs (overrides config).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config).",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config).",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--sdf",
        action="store_true",
        default=False,
        help="Use real SDF data via Geometry3DDataset (with conformer generation).",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="Number of DataLoader worker processes.",
    )
    args = parser.parse_args()

    # Load config
    if args.config:
        config = TrainingConfig.from_toml(args.config)
        logger.info(f"Loaded config from {args.config}")
    else:
        config = TrainingConfig()

    # Apply environment variable overrides
    config.apply_env_overrides()

    # Apply CLI overrides
    if args.model:
        config.model_name = args.model
    if args.cutoff:
        config.cutoff = args.cutoff
    if args.epochs:
        config.max_epochs = args.epochs
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.lr:
        config.learning_rate = args.lr

    train(
        config,
        resume_path=args.resume,
        use_sdf=args.sdf,
        num_workers=args.num_workers,
    )


if __name__ == "__main__":
    main()