# ChemAugur

### An open source project aiming to use graph neural networks to predict the physical properties of chemicals based only on their molecular structure and atomic makeup.

## Research and Development Plan:
- ✅ Fundamental Pytorch, Torch Geometric, RDKit architecture setup and verified
- ✅ First Dataset identified, PubChem Compound SDFs via FTP
- ✅ Finished data script to download, hashcheck, and manage SDF files from PubChem
- ✅ Recognized issues and limitations with PubChem Compound SDF Datasource
  - Small percent of molecules in files were invalid and stereochemistry warnings
  - Lack of chemicals properties anticipated to be present, boiling point not included
- ☑️ Identify Possible Superior Data Sources
  - https://www.aatbio.com/data-sets/boiling-point-bp-and-melting-point-mp-reference-table
  - 
- ☑️ Architect multifaceted ETL pipeline to begin building up a purpose built SQL data warehouse
  - Obtain valid and warning-free SMILES from PubChem SDFs to act as IDs for each entry, and the basis for building molecular graphs
  - Use aatbio bp and mp data-set to obtain boiling points, and possibly more compounds, revalidate with RDKit
  - 
- ☑️ Finish building up an adequate dataset in the warehouse
- ☑️ Refine and expand training procedure, focusing on picking best properties
- ☑️ Train on the data warehouse, ideally in a maintainable way
- ☑️ Experiment with properties other than normal boiling point
  - Toxicity
  - Psychotropic Effects
  - Receptor affinities

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
