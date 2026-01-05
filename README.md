# ChemAugur

### An open source project aiming to use graph neural networks to predict the physical properties of chemicals based only on their molecular structure and atomic makeup.


## Set Up
```
pip install -r reqs/development.txt
```
```
pip install -e .
```

## Common Commands
Update requirements
```
pip-compile --extra dev --output-file=reqs/development.txt pyproject.toml
```
Download SDF data files
```
python src/chem_augur/data/pubchem_sdf_download.py
```
Inspect SDF files
```
python src/chem_augur/data/sdf_inspector.py
```
Train model
```
PYTHONPATH=src python src/chem_augur/training/train_simple.py
```
Predict property of untrained molecule
```
PYTHONPATH=src python src/chem_augur/prediction/predict_simple.py
```
