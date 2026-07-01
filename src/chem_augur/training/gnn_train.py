# training/gnn_train.py
"""
Full training pipeline for the SimpleGCN molecular property predictor.

Features:
  - MolecularDataset with disk caching
  - Train / validation / test splits
  - Target normalization (StandardScaler fitted on training split)
  - Epoch-based training with early stopping
  - MAE + RMSE metrics every epoch
  - Model checkpointing (best validation MAE)
  - Learning-rate scheduling (ReduceLROnPlateau)
"""
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.loader import DataLoader

from chem_augur.features.molecular_graph import ATOM_FEATURE_DIM
from chem_augur.features.molecular_dataset import (
    MolecularDataset,
    StandardScaler,
    normalize_dataset,
)
from chem_augur.models.gnn.gcn_model import SimpleGCN


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
SDF_FILES = [
    "src/chem_augur/data/Compound_000000001_000500000.sdf",
    "src/chem_augur/data/Compound_000500001_001000000.sdf",
    "src/chem_augur/data/Compound_001000001_001500000.sdf",
]
TARGET_PROPERTY = "pIC50"
ROOT_DIR = "."
MODEL_DIR = "src/chem_augur/models/gnn"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyper-parameters
HIDDEN_CHANNELS = 64
NUM_LAYERS = 4
DROPOUT = 0.5
LEARNING_RATE = 1e-3
BATCH_SIZE = 32
MAX_EPOCHS = 200
PATIENCE = 25          # early-stopping patience
WEIGHT_DECAY = 1e-5
LR_PATIENCE = 10       # ReduceLROnPlateau patience
LR_FACTOR = 0.5

# Train / val / test ratios
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def mae(pred: torch.Tensor, target: torch.Tensor) -> float:
    return (pred - target).abs().mean().item()


def rmse(pred: torch.Tensor, target: torch.Tensor) -> float:
    return torch.sqrt(((pred - target) ** 2).mean()).item()


def train_epoch(model, loader, optimizer, loss_fn, scaler: StandardScaler):
    model.train()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        optimizer.zero_grad()
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * batch.num_graphs
        # inverse-transform for metrics in original scale
        all_preds.append(scaler.inverse_transform(pred))
        all_targets.append(scaler.inverse_transform(batch.y))
    avg_loss = total_loss / sum(b.num_graphs for b in loader)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return avg_loss, mae(preds, targets), rmse(preds, targets)


@torch.no_grad()
def eval_epoch(model, loader, loss_fn, scaler: StandardScaler):
    model.eval()
    total_loss = 0.0
    all_preds, all_targets = [], []
    for batch in loader:
        batch = batch.to(DEVICE)
        pred = model(batch)
        loss = loss_fn(pred, batch.y)
        total_loss += loss.item() * batch.num_graphs
        all_preds.append(scaler.inverse_transform(pred))
        all_targets.append(scaler.inverse_transform(batch.y))
    avg_loss = total_loss / sum(b.num_graphs for b in loader)
    preds = torch.cat(all_preds)
    targets = torch.cat(all_targets)
    return avg_loss, mae(preds, targets), rmse(preds, targets)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"Device: {DEVICE}")

    # 1. Load or create dataset ------------------------------------------------
    dataset = MolecularDataset(
        root_dir=ROOT_DIR,
        sdf_files=SDF_FILES,
        target_property=TARGET_PROPERTY,
    )
    print(f"Dataset size: {len(dataset)} molecules")

    # 2. Splits ----------------------------------------------------------------
    train_idx, val_idx, test_idx = MolecularDataset.split(
        dataset, train_ratio=TRAIN_RATIO, val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO, seed=SEED,
    )
    print(f"Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}")

    # 3. Normalize targets -----------------------------------------------------
    scaler = normalize_dataset(dataset, train_idx)
    print(f"Target scaler  mean={scaler.mean:.4f}, std={scaler.std:.4f}")

    # 4. DataLoaders -----------------------------------------------------------
    train_loader = DataLoader(dataset, train_indices=train_idx, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(dataset, train_indices=val_idx, batch_size=BATCH_SIZE)
    test_loader = DataLoader(dataset, train_indices=test_idx, batch_size=BATCH_SIZE)

    # 5. Model -----------------------------------------------------------------
    model = SimpleGCN(
        num_node_features=ATOM_FEATURE_DIM,
        hidden_channels=HIDDEN_CHANNELS,
        output_channels=1,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT,
    ).to(DEVICE)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 6. Optimizer + scheduler -------------------------------------------------
    loss_fn = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=LR_FACTOR, patience=LR_PATIENCE, verbose=True
    )

    # 7. Training loop ---------------------------------------------------------
    best_val_mae = float("inf")
    best_val_loss = float("inf")
    patience_counter = 0
    start_time = time.time()

    for epoch in range(1, MAX_EPOCHS + 1):
        t0 = time.time()
        train_loss, train_mae, train_rmse = train_epoch(model, train_loader, optimizer, loss_fn, scaler)
        val_loss, val_mae, val_rmse = eval_epoch(model, val_loader, loss_fn, scaler)
        elapsed = time.time() - t0
        scheduler.step(val_loss)

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"Epoch {epoch:3d}/{MAX_EPOCHS}  "
            f"Train Loss={train_loss:.4f}  MAE={train_mae:.4f}  RMSE={train_rmse:.4f}  "
            f"Val Loss={val_loss:.4f}  MAE={val_mae:.4f}  RMSE={val_rmse:.4f}  "
            f"LR={current_lr:.2e}  {elapsed:.1f}s"
        )

        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_mae = val_mae
            patience_counter = 0
            # Save best checkpoint
            checkpoint = {
                "model_state_dict": model.state_dict(),
                "scaler_state": scaler.state_dict(),
                "epoch": epoch,
                "val_mae": best_val_mae,
                "val_rmse": val_rmse,
                "hidden_channels": HIDDEN_CHANNELS,
                "num_layers": NUM_LAYERS,
                "dropout": DROPOUT,
            }
            ckpt_path = os.path.join(MODEL_DIR, "gcn_best_checkpoint.pt")
            torch.save(checkpoint, ckpt_path)
            print(f"  -> New best checkpoint saved (val_mae={best_val_mae:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch} (patience={PATIENCE})")
                break

    # 8. Final evaluation on test set ------------------------------------------
    print("\n" + "=" * 60)
    print("Loading best checkpoint for final test evaluation...")
    ckpt_path = os.path.join(MODEL_DIR, "gcn_best_checkpoint.pt")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_loss, test_mae, test_rmse = eval_epoch(model, test_loader, loss_fn, scaler)
    print(f"Test  Loss={test_loss:.4f}  MAE={test_mae:.4f}  RMSE={test_rmse:.4f}")
    print(f"Total training time: {time.time() - start_time:.0f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()