# ChemAugur

**Graph neural networks for predicting physical properties of small molecules from structure alone.**

ChemAugur is an open-source PyTorch Geometric project that trains graph convolutional networks (GCNs) on molecular graphs — atoms as nodes, bonds as edges — to predict physical and chemical properties without requiring hand-engineered descriptors or quantum mechanical calculations.



## Why GCNs for molecular property prediction?

Molecules are graphs. Traditional QSAR pipelines flatten them into fingerprints (Morgan, MACCS) or descriptor vectors, losing structural context in the process. Message-passing GNNs operate directly on the graph, learning representations that respect the symmetries of chemistry: permutation invariance over atoms, locality of bonding, and the relevance of neighborhood topology to properties like LogP, solubility, and polar surface area.


## Status

**Early stage. Single-property regression baseline working end-to-end on PubChem SDF data.**

Current capabilities:
- Bulk download and inspection of PubChem SDF compound files
- Conversion of SDF records to PyTorch Geometric `Data` objects
- A simple GCN baseline with training and single-molecule inference scripts
- Reproducible environment via pinned `pyproject.toml` and `pip-compile`

Not yet implemented (see roadmap): multi-task prediction, uncertainty quantification, message-passing variants beyond GCN, published benchmark results, model cards.

## Installation

Requires Python 3.10+ and a working PyTorch install. CPU-only is fine for the baseline; GPU recommended for larger datasets.

```bash
pip install -r reqs/development.txt
pip install -e .
```

## Quick start

Download a PubChem SDF slice:

```bash
python src/chem_augur/data/pubchem_sdf_download.py
```

Inspect what you got:

```bash
python src/chem_augur/data/sdf_inspector.py
```

Train the baseline GCN:

```bash
PYTHONPATH=src python src/chem_augur/training/train_simple.py
```

Predict a property for a molecule the model has not seen:

```bash
PYTHONPATH=src python src/chem_augur/prediction/predict_simple.py
```

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
