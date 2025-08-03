# ChemAugur

### An open source project aiming to use graph neural networks to predict the physical properties of chemicals based only on their molecular structure and atomic makeup.

## Research and Development Plan:
- ✅ Fundamental Pytorch, Torch Geometric, RDKit architecture setup and verified
- ✅ First dataset identified, PubChem Compound SDFs via FTP
- ✅ Finished data script to download, hashcheck, and manage SDF files from PubChem
- ✅ Recognized issues and limitations with using only a PubChem Compound SDF Datasource
- ❌ Identify possible superior data sources; some options, but no complete solutions
- 👨‍💻 Architect and develop ETL pipeline and a data warehouse to begin building up
  - Multi-faceted approach
    - Obtain valid and stereochemistry warning-free SMILES from PubChem SDFs, maybe to act as IDs for each entry, and then the basis for building molecular graphs with RDkit
    - Use aatbio bp and mp data-set to obtain boiling points, and possibly more compounds, revalidate with RDKit at each step
    - Focus on relevant chemicals with an inclusive variety of bonds and stereochemical features
- ☑️ Build up an adequate dataset in the warehouse to train on, at least thousands, likely need tens of thousands of molecules
- ☑️ Refine and expand training procedure, focusing on picking best properties
- ☑️ Train on the data warehouse
- ☑️ Experiment with properties other than normal boiling point
  - Toxicity
  - Psychotropic Effects
  - Receptor affinities
- ☑️ Repeat from: Build up a larger database, or refactor datawarehouse, or refactor pipeline

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
