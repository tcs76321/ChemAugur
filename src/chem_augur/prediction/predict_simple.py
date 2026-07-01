# prediction/predict_simple.py
"""
Prediction utilities for the SimpleGCN molecular property predictor.

Loads the checkpoint produced by gnn_train.py (which bundles both model weights
and the StandardScaler) and returns predictions in the original property scale.
"""
import os
import torch
from torch_geometric.data import Data

from chem_augur.features.molecular_graph import (
    smiles_to_pyg_graph_simple,
    ATOM_FEATURE_DIM,
)
from chem_augur.features.molecular_dataset import StandardScaler
from chem_augur.models.gnn.gcn_model import SimpleGCN

DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(__file__), "..", "models", "gnn", "gcn_best_checkpoint.pt"
)


def _load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model + scaler from a single checkpoint file."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = SimpleGCN(
        num_node_features=ATOM_FEATURE_DIM,
        hidden_channels=checkpoint.get("hidden_channels", 64),
        output_channels=1,
        num_layers=checkpoint.get("num_layers", 4),
        dropout=checkpoint.get("dropout", 0.5),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    scaler = StandardScaler()
    scaler.load_state_dict(checkpoint["scaler_state"])
    return model, scaler


def predict_property(
    smiles: str,
    checkpoint_path: str = DEFAULT_CHECKPOINT,
) -> float:
    """
    Predict the target property for a single SMILES string.

    Returns the prediction in the **original** (un-normalized) scale.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    graph = smiles_to_pyg_graph_simple(smiles)
    if graph is None:
        raise ValueError(f"Could not parse SMILES: {smiles}")
    graph.to(device)

    model, scaler = _load_checkpoint(checkpoint_path, device)

    with torch.no_grad():
        pred = model(Data(x=graph.x.unsqueeze(0), edge_index=graph.edge_index,
                          batch=torch.tensor([0])))
    # Inverse-transform to original scale
    original_pred = scaler.inverse_transform(pred)
    return original_pred.item()


# ------------------------------------------------------------------
# CLI entry-point
# ------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Predict molecular property with SimpleGCN")
    parser.add_argument("smiles", help="SMILES string to predict")
    parser.add_argument(
        "--checkpoint", default=DEFAULT_CHECKPOINT,
        help="Path to checkpoint file (default: gcn_best_checkpoint.pt)",
    )
    args = parser.parse_args()

    try:
        value = predict_property(args.smiles, args.checkpoint)
        print(f"Predicted property for '{args.smiles}': {value:.4f}")
    except Exception as exc:
        print(f"Error: {exc}")