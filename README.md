# ChemAugur

Graph neural networks for predicting molecular properties from structure alone.

ChemAugur converts molecular structures into graph representations and trains Graph Convolutional Networks (GCNs) to predict physical and chemical properties. It uses RDKit for chemistry, PyTorch Geometric for graph learning, and PubChem as the data source — no hand-crafted descriptors or quantum calculations required.

## Architecture

Molecules are represented as graphs: atoms are nodes, bonds are edges.

| Component | Details |
|---|---|
| **Node features** | 34-dim vector per atom: atomic number, degree, formal charge, hybridization, aromaticity, chirality, hydrogen count, valence, ring membership |
| **Edge features** | 7-dim vector per bond: bond type (single/double/triple/aromatic), conjugation, ring membership |
| **Model** | Multi-layer GCN with ReLU activation, dropout, and global mean pooling |
| **Output** | Single regression value (configurable target property) |

## Project Structure

```
src/chem_augur/
├── data/
│   ├── pubchem_sdf_download.py   # Download SDF files from PubChem FTP
│   └── sdf_inspector.py          # Inspect and analyze SDF contents
├── features/
│   ├── molecular_graph.py        # Convert RDKit molecules → PyG Data objects
│   └── molecular_dataset.py      # Dataset class with caching, splits, normalization
├── models/gnn/
│   └── gcn_model.py              # SimpleGCN model definition
├── training/
│   ├── gnn_train.py              # Full training pipeline (recommended)
│   └── train_simple.py           # Minimal training example (8 alkanes)
└── prediction/
    └── predict_simple.py         # Predict property from a SMILES string
```

## Prerequisites

- Python 3.11+
- GPU recommended for large datasets; CPU works for small-scale experiments

## Installation

```bash
pip install -r reqs/development.txt
pip install -e .
```

## Quick Start

### 1. Download training data

Downloads compressed SDF files from PubChem with MD5 verification and automatic decompression:

```bash
python src/chem_augur/data/pubchem_sdf_download.py
```

### 2. Inspect your data

Review molecule counts and available properties in the downloaded files:

```bash
python src/chem_augur/data/sdf_inspector.py
```

### 3. Train

Train the GCN on PubChem data with early stopping, learning-rate scheduling, and checkpointing:

```bash
PYTHONPATH=src python src/chem_augur/training/gnn_train.py
```

The best model (by validation MAE) is saved to `src/chem_augur/models/gnn/gcn_best_checkpoint.pt`.

### 4. Predict

Predict a property for any molecule by SMILES:

```bash
PYTHONPATH=src python src/chem_augur/prediction/predict_simple.py "CCO" --checkpoint src/chem_augur/models/gnn/gcn_best_checkpoint.pt
```

Or use the API programmatically:

```python
from chem_augur.prediction.predict_simple import predict_property

value = predict_property("CCO")
print(f"Predicted: {value:.4f}")
```

## Configuration

Training hyperparameters are defined as module-level constants in `src/chem_augur/training/gnn_train.py`:

| Parameter | Default | Description |
|---|---|---|
| `HIDDEN_CHANNELS` | 64 | Hidden layer dimension |
| `NUM_LAYERS` | 4 | Number of GCN convolution layers |
| `DROPOUT` | 0.5 | Dropout rate |
| `LEARNING_RATE` | 1e-3 | Initial learning rate |
| `BATCH_SIZE` | 32 | Training batch size |
| `MAX_EPOCHS` | 200 | Maximum training epochs |
| `PATIENCE` | 25 | Early-stopping patience |
| `TARGET_PROPERTY` | "pIC50" | Molecule property to predict |
| `SDF_FILES` | 3 files | Paths to input SDF files |

Edit these values directly in `gnn_train.py` before running training.

## Development

Regenerate locked requirements after editing `pyproject.toml`:

```bash
pip-compile --extra dev --output-file=reqs/development.txt pyproject.toml
```

## Roadmap

- [ ] Benchmark against MoleculeNet tasks (ESOL, FreeSolv, Lipophilicity)
- [ ] Add message-passing variants: GAT, MPNN, GIN
- [ ] Multi-task heads for joint property prediction
- [ ] Uncertainty estimates via MC dropout or deep ensembles
- [ ] Hugging Face Hub model card with reproducible evaluation
- [ ] JOSS submission once benchmarks are published

## License

GPL-3.0