# ChemAugur

Predict physical properties of chemicals — pure substances and mixtures — from geometry alone. No bond types, no hybridization labels, no hand-crafted features. Just atomic numbers and 3D positions.

## How It Works

ChemAugur uses **MACE** (Multi-body Atomic Cluster Expansion), an equivariant graph neural network that captures distances, angles, dihedrals, and many-body correlations directly from 3D geometry. The model learns that 1.54 Å means a single bond, 120° means sp² hybridization, and 2.5-3.0 Å with the right angle means a hydrogen bond — without being told.

The same model handles pure substances and mixtures identically:
```
Pure ethanol:  9 atoms → MACE → boiling point
50:50 mixture: 5 ethanol + 5 water molecules in shared space → MACE → mixture boiling point
```

## Installation

```bash
pip install -r reqs/development.txt
pip install -e .
pip install mace-torch  # Required for the default model
```

## Quick Start

```bash
# Download data
python src/chem_augur/data/pubchem_sdf_download.py

# Inspect data
python src/chem_augur/data/sdf_inspector.py

# Train
chem-augur-train --cutoff 12.0 --epochs 200

# Predict
chem-augur-predict "CCO"
chem-augur-predict --mixture "CCO" "O" --ratios 0.5 0.5
```

## API

```python
from chem_augur.prediction.predict_mace import predict_property, predict_mixture

bp = predict_property("CCO")                    # Pure substance
bp = predict_mixture(["CCO", "O"], [0.5, 0.5])  # Mixture
```

## Configuration

Configuration via TOML file, environment variables (`CHEMAUGUR_*`), or CLI arguments:

| Parameter | Default | Description |
|---|---|---|
| `cutoff` | 12.0 | Interaction radius (Å) — captures multiple solvation shells |
| `hidden_channels` | 128 | Hidden dimension |
| `num_interactions` | 2 | Message-passing interaction blocks |
| `max_ell` | 3 | Spherical harmonic order (0-3) |
| `correlation` | 3 | Many-body order (3 = up to dihedrals) |
| `batch_size` | 16 | Training batch size |
| `learning_rate` | 1e-3 | Initial learning rate |
| `max_epochs` | 200 | Maximum training epochs |
| `patience` | 25 | Early stopping patience |

```toml
# config.toml
cutoff = 15.0
hidden_channels = 256
batch_size = 8
```

```bash
chem-augur-train --config config.toml
CHEMAUGUR_CUTOFF=15.0 chem-augur-train
```

## Project Structure

```
src/chem_augur/
├── config.py                   # TrainingConfig dataclass
├── data/                       # PubChem SDF download + inspection
├── features/
│   ├── conformer_generation.py # ETKDG + MMFF 3D conformers
│   ├── mixture_builder.py      # Multi-molecule placement
│   ├── molecular_graph_3d.py   # z/pos utilities
│   ├── molecular_graph.py      # Legacy 2D features
│   └── molecular_dataset.py    # Dataset with caching + normalization
├── models/
│   ├── registry.py             # Model factory
│   └── gnn/
│       ├── mace_model.py       # MACE wrapper
│       └── gcn_model.py        # Legacy 2D GCN
├── training/
│   ├── train_mace.py           # Unified 3D training pipeline
│   └── gnn_train.py            # Legacy 2D training
└── prediction/
    ├── predict_mace.py         # 3D prediction CLI + API
    └── predict_simple.py       # Legacy 2D prediction
tests/
```

## Architecture Philosophy

A molecular system is fully described by two things: **which atoms are present** (`z`) and **where they are** (`pos`). Bond types, hybridization, aromaticity, hydrogen bonding, solvent effects — all are derivable from geometry. The model learns these patterns rather than being told them with labels that fail for unusual chemistry (transition metals, 3-center bonds, frustrated Lewis pairs).

Mixtures are not fundamentally different from pure substances — just more atoms in the same space. Intermolecular interactions (hydrogen bonds, van der Waals, π-stacking) use the same distance-angle-dihedral mechanism as intramolecular ones.

## Known Limitations

- Conformer generation uses ETKDG + MMFF (approximate, not DFT accuracy)
- Single conformer per molecule (no conformational sampling yet)
- No explicit long-range electrostatics (relies on cutoff radius)
- `mace-torch` is a separate dependency
- Mixture placement uses random placement + steric clash avoidance, not full MD

## Development

```bash
pip install -e ".[test,dev]"
pytest tests/ -v
black src/ tests/
```

See [CONTRIBUTING.md](CONTRIBUTING.md) for full guidelines.

## Roadmap

- [ ] Conformational sampling (multiple conformers per molecule)
- [ ] MoleculeNet benchmarks (ESOL, FreeSolv, Lipophilicity)
- [ ] NIST/Dortmund mixture benchmarks
- [ ] Periodic boundary conditions for bulk-phase simulations
- [ ] Long-range electrostatics (Ewald summation)
- [ ] Multi-task property prediction
- [ ] MACE internal fork for thermodynamic-specific modifications

## License

GPL-3.0