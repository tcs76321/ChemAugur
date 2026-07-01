# ChemAugur

Graph neural networks for predicting molecular properties from geometry alone.

ChemAugur predicts physical properties of chemicals — pure substances and mixtures — using only atomic numbers and 3D positions. No bond types, hybridization labels, aromaticity flags, or hand-crafted features. The model learns all chemical semantics from geometry.

## Architecture

Molecules are represented by their fundamental physical description: which atoms are present (`z`) and where they are in 3D space (`pos`). Everything else — bond types, hybridization, hydrogen bonding, π-stacking, solvent effects — is discovered by the model from the geometry.

| Component | Details |
|---|---|
| **Input** | Atomic numbers (`z`) and 3D positions (`pos`) — nothing else |
| **Model** | MACE — multi-body atomic cluster expansion with equivariant message passing |
| **Geometry** | Distances, angles, dihedrals, and many-body correlations via ACE basis |
| **Tensor features** | O(3)-equivariant tensors (l=0,1,2,3) for scalar, vector, and tensor properties |
| **Cutoff** | Configurable interaction radius (default 12 Å) — captures multiple solvation shells |
| **Output** | Single regression value (configurable target property) |

### What the Model Captures

| Chemical concept | How the model learns it |
|---|---|
| Bond type (single/double/triple) | Distance between atoms (1.54 vs 1.34 vs 1.20 Å) |
| Aromaticity | ~1.40 Å distances + 120° angles in rings |
| Hybridization | Bond angle patterns (109.5° vs 120° vs 180°) |
| Ring strain | Deviations from ideal angles (60° in cyclopropane) |
| Hydrogen bonding | O-H···O distance (~2.5-3.0 Å) + angle pattern |
| π-stacking | Parallel ring planes at ~3.5 Å |
| Steric effects | Close non-bonded contacts |
| Solvent effects | All atoms in the system, including solvent molecules |
| Mixture non-ideality | Intermolecular distance-angle-dihedral patterns |
| Conformational flexibility | Dihedral angle patterns (gauche vs anti) |

### Unified Architecture

The same model handles pure substances and mixtures identically:

```
Pure ethanol:  9 atoms (1 molecule) → MACE → boiling point
Pure water:    3 atoms (1 molecule) → MACE → boiling point
50:50 mixture: 5 ethanol + 5 water molecules placed in shared space
               → MACE → mixture boiling point
```

The model has no concept of "molecule identity" — it sees only atoms in space and learns how distance/angle/dihedral patterns between different elements produce properties.

## Project Structure

```
src/chem_augur/
├── config.py                   # TrainingConfig dataclass with TOML support
├── data/
│   ├── pubchem_sdf_download.py # Download SDF files from PubChem FTP
│   └── sdf_inspector.py        # Inspect and analyze SDF contents
├── features/
│   ├── molecular_graph.py      # Legacy 2D graph features (GCN)
│   ├── molecular_graph_3d.py   # 3D geometry utilities
│   ├── molecular_dataset.py    # Dataset class with caching, splits, normalization
│   ├── conformer_generation.py # ETKDG + MMFF conformer generation
│   └── mixture_builder.py      # Multi-molecule placement for mixtures
├── models/
│   ├── registry.py             # Model factory
│   └── gnn/
│       ├── gcn_model.py        # Legacy 2D GCN model
│       └── mace_model.py       # MACE equivariant model wrapper
├── training/
│   ├── gnn_train.py            # Legacy 2D GCN training pipeline
│   └── train_mace.py           # Unified 3D training pipeline
└── prediction/
    ├── predict_simple.py       # Legacy 2D GCN prediction
    └── predict_mace.py         # 3D prediction CLI + API
tests/                          # Test suite
```

## Prerequisites

- Python 3.11+
- GPU recommended for large datasets; CPU works for small-scale experiments
- RDKit, PyTorch, PyTorch Geometric

## Installation

```bash
pip install -r reqs/development.txt
pip install -e .
```

For MACE support (the default model):
```bash
pip install mace-torch
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

Train the MACE model on 3D geometry data with early stopping, learning-rate scheduling, and checkpointing:

```bash
chem-augur-train --cutoff 12.0 --epochs 200
```

Or with a configuration file:
```bash
chem-augur-train --config config.toml
```

The best model is saved to `checkpoints/best_model.pt`.

### 4. Predict

Predict a property for any molecule by SMILES:

```bash
# Pure substance
chem-augur-predict "CCO"

# Mixture
chem-augur-predict --mixture "CCO" "O" --ratios 0.5 0.5
```

Or use the API programmatically:

```python
from chem_augur.prediction.predict_mace import predict_property, predict_mixture

# Pure substance
bp = predict_property("CCO")
print(f"Predicted: {bp:.4f}")

# Mixture
bp = predict_mixture(["CCO", "O"], [0.5, 0.5])
print(f"Predicted mixture: {bp:.4f}")
```

## Configuration

Training is configured via a `TrainingConfig` dataclass, TOML file, environment variables, or CLI arguments.

### Key Parameters

| Parameter | Default | Description |
|---|---|---|
| `model_name` | `"mace"` | Model architecture (`"mace"` or `"gcn"`) |
| `cutoff` | 12.0 | Interaction cutoff radius in Å |
| `hidden_channels` | 128 | Hidden dimension |
| `num_interactions` | 2 | Message-passing blocks |
| `max_ell` | 3 | Max spherical harmonic order (0-3) |
| `correlation` | 3 | Many-body order (3 = up to dihedrals) |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 1e-3 | Initial learning rate |
| `max_epochs` | 200 | Maximum training epochs |
| `patience` | 25 | Early-stopping patience |
| `seed` | 42 | Random seed |

### Configuration Sources (in order of precedence)

1. **CLI arguments** — `--cutoff 15.0`
2. **Environment variables** — `CHEMAUGUR_CUTOFF=15.0`
3. **TOML file** — `config.toml`
4. **Defaults** — as listed above

## Development Setup

See [CONTRIBUTING.md](CONTRIBUTING.md) for full development setup instructions.

Quick start:
```bash
git clone git@github.com:tcs76321/ChemAugur.git
cd ChemAugur
python -m venv .venv && source .venv/bin/activate
pip install -r reqs/development.txt
pip install -e ".[test,dev]"
pytest tests/ -v
```

## Known Limitations

### Conformer Generation
- Uses ETKDG + MMFF for conformer generation — approximate, not DFT-level accuracy
- Single conformer per molecule (no conformational sampling yet)
- May fail for unusual elements or highly constrained ring systems

### Mixture Handling
- Random placement with steric clash avoidance — not a full molecular dynamics simulation
- Box size estimation is approximate
- Periodic boundary conditions not yet implemented

### Model Architecture
- No explicit long-range electrostatics (Ewald summation) — relies on cutoff
- MACE integration requires `mace-torch` as a separate dependency
- Training on large mixture systems (>1000 atoms) is computationally intensive

## Troubleshooting

### PyTorch Geometric Installation
```bash
# If PyG fails to install with pip, try:
pip install torch-geometric --index-url https://data.pyg.org/whl/torch-2.6.0+cpu.html
```

### MACE Installation
```bash
# mace-torch may require specific dependency versions
pip install mace-torch
```

### RDKit Issues
```bash
# RDKit is best installed via conda:
conda install -c conda-forge rdkit
# Or via pip:
pip install rdkit
```

### CUDA Memory
For large cutoff values or many molecules, reduce batch size:
```bash
CHEMAUGUR_BATCH_SIZE=4 chem-augur-train
```

### SDF Parsing
If PubChem SDF files fail to parse, verify the files are not corrupted:
```bash
python src/chem_augur/data/sdf_inspector.py
```

## Roadmap

- [x] 2D GCN baseline (legacy)
- [x] Critical bug fixes
- [x] 3D conformer generation (ETKDG + MMFF)
- [x] MACE model integration with configurable cutoff
- [x] Unified pure substance + mixture architecture
- [x] Mixture builder with multi-molecule placement
- [x] TOML configuration and environment variable support
- [x] Test suite
- [ ] Conformational sampling (multiple conformers per molecule)
- [ ] Benchmark against MoleculeNet (ESOL, FreeSolv, Lipophilicity)
- [ ] NIST ThermoData Engine / Dortmund Data Bank mixture benchmarks
- [ ] Periodic boundary conditions for bulk-phase simulations
- [ ] Long-range electrostatics (Ewald summation)
- [ ] Multi-task property prediction
- [ ] Uncertainty estimates via MC dropout or deep ensembles
- [ ] MACE internal fork for thermodynamic-specific modifications
- [ ] JOSS submission once benchmarks are published

## License

GPL-3.0