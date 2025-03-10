# ChemAugur
#### An open source project aiming to use graph neural networks to predict the physical properties of chemicals based only on their molecular structure and atomic makeup.

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
```
PYTHONPATH=src python src/chem_augur/training/train_simple.py
```
```
PYTHONPATH=src python src/chem_augur/prediction/predict_simple.py
```