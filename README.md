# ChemAugur

#### An open source project aiming to use graph neural networks to predict the physical properties of chemicals based only on their molecular structure and atomic makeup.

### Current Status: Base functionality verified with hard coded data using Pytorch, PyG (PyTorch Geometric), and RDkit

### Next Goal: Identify large open datasets for training, formulate ideal ETL process

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
Train model
```
PYTHONPATH=src python src/chem_augur/training/train_simple.py
```
Predict property of untrained molecule
```
PYTHONPATH=src python src/chem_augur/prediction/predict_simple.py
```
