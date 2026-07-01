# ChemAugur Roadmap

This document outlines the development trajectory for ChemAugur. Near-term milestones are detailed; longer-term goals are sketched at a higher level since they depend on what we learn along the way.

---

## Completed

- [x] 2D GCN baseline with PubChem SDF data pipeline
- [x] Critical bug fixes (batch handling, layer count, DataLoader, reproducibility)
- [x] 3D conformer generation via ETKDG + MMFF
- [x] MACE model integration with configurable cutoff (12-15 Å)
- [x] Unified pure substance + mixture architecture (same model, same pipeline)
- [x] Multi-molecule mixture builder (random placement + steric clash avoidance + MMFF relaxation)
- [x] TrainingConfig dataclass with TOML and env-var support
- [x] Training pipeline with logging, TensorBoard, early stopping, checkpointing
- [x] Prediction CLI and programmatic API for pure substances and mixtures
- [x] Model registry for architecture-swappable training
- [x] Test suite (conformer, mixture, config, model registry)
- [x] Documentation (README, INTRO.md, CONTRIBUTING.md)

---

## Phase 1: Production-Ready Training (Next 2-4 Weeks)

### 1.1 SDF → 3D Data Pipeline
**Goal**: Train on real PubChem data with 3D conformers.

- [ ] `GeometryDataset` subclass that reads SDF files, generates conformers, and caches them to disk
- [ ] Conformer caching directory with hash-based invalidation (SMILES → conformer mapping)
- [ ] Handle SDF molecules that lack 3D coordinates (generate conformers on-the-fly)
- [ ] Handle molecules where ETKDG fails (skip + log, don't crash the pipeline)
- [ ] Multi-process conformer generation for large datasets
- [ ] Progress bar integration (tqdm) for long-running conformer generation

### 1.2 Small Test Dataset
**Goal**: A ~100-molecule dataset for CI smoke tests and rapid iteration.

- [ ] Select ~100 molecules from PubChem with diverse sizes and functional groups
- [ ] Include known experimental properties (boiling point, logP) from public sources
- [ ] Commit the dataset (or a download script) to the repository
- [ ] Add `tests/test_smoke.py` — end-to-end: SMILES → conformer → MACE → prediction → validate output shape

### 1.3 Training Infrastructure
**Goal**: Run a real training job end-to-end without manual intervention.

- [ ] Add `--resume` flag to resume from checkpoint (restore optimizer state, epoch, RNG)
- [ ] Add `--validate-only` flag to run evaluation on a test set without training
- [ ] Log hyperparameters + metrics to TensorBoard for experiment comparison
- [ ] Add CSV logging alongside TensorBoard for easy analysis in pandas/Excel
- [ ] Add `--dry-run` flag that generates conformers + builds loaders without training
- [ ] Verify training works on CPU (for CI) and GPU (for production)

### 1.4 MACE Integration Hardening
**Goal**: Ensure the MACE wrapper works reliably with `mace-torch` installed.

- [ ] Test MACE forward pass with real conformer data (not just synthetics)
- [ ] Test MACE backward pass (gradients flow correctly through all parameters)
- [ ] Test checkpoint save/load round-trip (model + scaler + optimizer + config)
- [ ] Benchmark MACE on a small dataset (10 molecules, 100 epochs) to verify loss decreases
- [ ] Document MACE installation quirks (version pinning, CUDA compatibility)
- [ ] Add `mace-torch` to optional dependencies in `pyproject.toml` with version constraint

---

## Phase 2: Benchmarking and Validation (2-6 Weeks)

### 2.1 MoleculeNet Benchmarks
**Goal**: Validate against established benchmarks to measure progress.

- [ ] ESOL (water solubility) — 1,128 compounds, small molecules
- [ ] FreeSolv (hydration free energy) — 642 compounds
- [ ] Lipophilicity (logP) — 4,200 compounds
- [ ] Compare against published results (GCN, SchNet, DimeNet++, etc.)
- [ ] Document results in `benchmarks/` directory with reproducible scripts

### 2.2 Mixture Benchmark Targets
**Goal**: Identify and acquire mixture property data for training.

- [ ] Research NIST ThermoData Engine (TDE) access — boiling points, vapor pressures, activity coefficients for binary mixtures
- [ ] Research Dortmund Data Bank (DDB) access — vapor-liquid equilibrium (VLE) data, azeotrope data, excess enthalpies
- [ ] Research Minnesota Solvation Database (MNSOL) — solvation free energies in diverse solvents
- [ ] Identify 2-3 specific mixture systems with high-quality data for initial experiments (e.g., ethanol-water, acetone-chloroform, hexane-benzene)
- [ ] Download or acquire data, format as CSV/JSON, commit scripts to `data/`

### 2.3 Single-Molecule Model Training
**Goal**: Train a MACE model on single-molecule properties and establish baseline performance.

- [ ] Train on ESOL (1,128 compounds) — measure MAE, RMSE, R²
- [ ] Train on FreeSolv (642 compounds) — measure MAE, RMSE, R²
- [ ] Train on Lipophilicity (4,200 compounds) — measure MAE, RMSE, R²
- [ ] Hyperparameter sweep: cutoff (8, 10, 12, 15 Å), hidden_channels (64, 128, 256), num_interactions (1, 2, 3)
- [ ] Document best hyperparameters and results in `results/` directory
- [ ] Compare against published GCN/DimeNet++/GemNet results on the same benchmarks

---

## Phase 3: Mixture Training (4-8 Weeks)

### 3.1 Mixture Dataset Construction
**Goal**: Build a dataset of mixture properties from acquired data.

- [ ] Implement `MixtureDataset` class that generates multi-molecule systems on-the-fly
- [ ] Support variable number of molecules per sample (configurable)
- [ ] Support multiple conformer samples per species (for conformational averaging)
- [ ] Cache generated mixture configurations to disk for reproducibility
- [ ] Add data augmentation: multiple random placements per mixture composition

### 3.2 Mixture Model Training
**Goal**: Train MACE on mixture properties.

- [ ] Train on binary mixture boiling points (ethanol-water, acetone-chloroform, etc.)
- [ ] Train on mixture vapor-liquid equilibrium data (activity coefficients)
- [ ] Compare pure substance predictions vs mixture predictions for the same species
- [ ] Analyze whether the model learns non-ideal mixing behavior from geometry alone
- [ ] Document which mixture properties are learnable from geometry and which require additional physics

### 3.3 Mixture Analysis Tools
**Goal**: Understand what the model is learning about mixtures.

- [ ] Implement attention/importance visualization: which inter-molecular distances contribute most to the prediction?
- [ ] Implement species contribution analysis: which species dominates the prediction?
- [ ] Implement distance-resolved analysis: how does prediction change as we vary cutoff?
- [ ] Generate predicted vs experimental parity plots for mixture properties

---

## Phase 4: Architecture Improvements (6-12 Weeks)

### 4.1 Conformational Sampling
**Goal**: Move from single conformer to ensemble-based predictions.

- [ ] Generate N conformers per molecule (N=10-100) using ETKDG with different seeds
- [ ] Implement energy-weighted averaging: weight conformers by MMFF energy (lower energy = higher weight)
- [ ] Implement Boltzmann-weighted predictions: pred = Σ(pred_i * exp(-E_i/kT)) / Σ(exp(-E_i/kT))
- [ ] Compare single-conformer vs multi-conformer prediction accuracy
- [ ] Analyze which properties benefit most from conformational sampling (entropy-driven properties like boiling point should benefit significantly)

### 4.2 Periodic Boundary Conditions
**Goal**: Simulate bulk-phase behavior more accurately.

- [ ] Implement cubic periodic boundary conditions for mixture placement
- [ ] Minimum image convention for distance calculations (atom interacts with nearest periodic image)
- [ ] Handle cutoff radius < half box size constraint
- [ ] Compare PBC vs non-PBC predictions on mixture properties
- [ ] Benchmark computational cost of PBC vs non-PBC

### 4.3 Long-Range Electrostatics
**Goal**: Handle charged species and ionic interactions.

- [ ] Research Ewald summation implementation for point charges in PyTorch
- [ ] Implement particle-mesh Ewald (PME) or smooth particle-mesh Ewald (SPME) as a layer
- [ ] Partial charges from MMFF or Gasteiger (fast, approximate) as auxiliary input
- [ ] Compare predictions with and without long-range electrostatics on charged systems
- [ ] Evaluate whether the benefit justifies the computational cost

### 4.4 Model Architecture Improvements
**Goal**: Explore model variants beyond the base MACE wrapper.

- [ ] Implement MACE-MP (pre-trained on Materials Project) as a starting point
- [ ] Experiment with attention-based pooling instead of mean pooling
- [ ] Experiment with multi-task heads (predict multiple properties simultaneously)
- [ ] Experiment with physics-informed loss terms (e.g., penalize predictions that violate thermodynamic constraints)
- [ ] Evaluate newer MACE versions as they're released

---

## Phase 5: Production Features (8-16 Weeks)

### 5.1 Multi-Task Property Prediction
**Goal**: One model predicts multiple properties.

- [ ] Implement multi-task output head: shared MACE backbone → separate MLPs per property
- [ ] Task weighting: balance gradients from different properties (uncertainty weighting, GradNorm)
- [ ] Train on dataset with multiple property labels per molecule
- [ ] Evaluate whether multi-task learning improves single-task performance (transfer learning)

### 5.2 Uncertainty Quantification
**Goal**: Know when the model is guessing.

- [ ] Implement MC dropout: multiple forward passes with dropout enabled → variance
- [ ] Implement deep ensemble: train N models with different seeds → variance
- [ ] Calibrate uncertainty estimates against prediction error
- [ ] Add uncertainty output to prediction API: `predict_property(smiles) → (value, uncertainty)`
- [ ] Flag predictions with high uncertainty for human review

### 5.3 Web Interface
**Goal**: Make predictions accessible without a command line.

- [ ] Simple Flask/FastAPI server for prediction
- [ ] Web form: paste SMILES → get prediction + uncertainty
- [ ] Mixture builder UI: select species, set ratios, predict
- [ ] Visualization: 3D molecule viewer showing the conformer used for prediction
- [ ] Deploy to Hugging Face Spaces or similar for public access

### 5.4 Pre-Trained Models
**Goal**: Distribute trained models for immediate use.

- [ ] Train a general-purpose model on combined MoleculeNet benchmarks
- [ ] Train a mixture model on combined binary mixture data
- [ ] Upload models to Hugging Face Hub with model cards
- [ ] Add `chem-augur-download` CLI to fetch pre-trained models
- [ ] Document model limitations, training data, and expected accuracy

---

## Phase 6: Advanced Research Directions (6-18 Months)

### 6.1 MACE Fork for Thermodynamics
**Goal**: Modify MACE internals for thermodynamic property prediction.

- [ ] Fork `mace-torch` repository
- [ ] Add thermodynamic-specific loss terms (e.g., Gibbs free energy consistency)
- [ ] Add temperature-dependence: predict property as function of T
- [ ] Add pressure-dependence for vapor-liquid equilibrium
- [ ] Explore graph-level auxiliary tasks (predicting molecular volume, polarizability, dipole moment as intermediate representations)

### 6.2 Generative Inverse Design
**Goal**: Given a desired property, suggest molecular structures or mixture compositions.

- [ ] Train a generative model (VAE, diffusion) conditioned on MACE embeddings
- [ ] Latent space interpolation between known molecules
- [ ] Property-conditional generation: "generate molecules with boiling point near 100°C"
- [ ] Mixture composition optimization: "find the ethanol:water ratio that maximizes X"

### 6.3 Active Learning Loop
**Goal**: Automatically select the most informative molecules to measure next.

- [ ] Uncertainty-based acquisition: select molecules with highest prediction uncertainty
- [ ] Diversity-based acquisition: select molecules that are geometrically different from training data
- [ ] Interface with laboratory data (CSV import, API for experimental results)
- [ ] Automated retraining when new data arrives

### 6.4 Reaction Prediction
**Goal**: Extend from static properties to chemical reactions.

- [ ] Represent reactions as before/after geometry pairs
- [ ] Predict reaction energy barriers from geometry changes
- [ ] Predict reaction products given reactants
- [ ] This is a major pivot — may require a separate project

---

## Phase 7: Community and Dissemination (Ongoing)

### 7.1 Academic Publication
- [ ] Write paper on geometry-only mixture property prediction
- [ ] Target: JOSS (Journal of Open Source Software) for software paper
- [ ] Target: J. Chem. Inf. Model. or Digital Discovery for methods paper
- [ ] Include benchmarks against existing methods
- [ ] Release all data, code, and trained models with paper

### 7.2 Community Building
- [ ] Add issue templates (bug report, feature request, benchmark submission)
- [ ] Add discussion forum (GitHub Discussions)
- [ ] Create tutorial notebooks (Jupyter) for common use cases
- [ ] Present at cheminformatics/ML conferences (ACS, ICCS, NeurIPS workshops)
- [ ] Write blog posts about the "geometry-only" philosophy and results

### 7.3 Integration Ecosystem
- [ ] DeepChem integration (use ChemAugur as a DeepChem model)
- [ ] ASE calculator for molecular dynamics (use MACE model as force field)
- [ ] RDKit contribution: conformer generation improvements based on our findings
- [ ] PubChem integration: link to ChemAugur predictions from PubChem compound pages

---

## Timeline Summary

| Phase | Timeframe | Key Deliverable |
|---|---|---|
| 1: Production Training | 2-4 weeks | SDF → 3D pipeline, small test dataset, training runs end-to-end |
| 2: Benchmarks | 2-6 weeks | MoleculeNet results, mixture data acquisition, single-molecule baselines |
| 3: Mixture Training | 4-8 weeks | Mixture dataset, mixture model training, analysis tools |
| 4: Architecture | 6-12 weeks | Conformational sampling, PBC, long-range electrostatics, model improvements |
| 5: Production | 8-16 weeks | Multi-task, uncertainty, web UI, pre-trained models |
| 6: Research | 6-18 months | MACE fork, inverse design, active learning, reactions |
| 7: Community | Ongoing | Publications, tutorials, integrations |

## Prioritization Principles

When deciding what to work on next, prioritize based on:

1. **Does it unlock new data?** (e.g., SDF pipeline → can train on real data)
2. **Does it validate the core hypothesis?** (e.g., mixture benchmarks → proves geometry-only works)
3. **Does it remove a known limitation?** (e.g., conformational sampling → fixes single-conformer problem)
4. **Does it make the project usable by others?** (e.g., pre-trained models, web UI)
5. **Is it novel research?** (e.g., MACE fork, inverse design — exciting but lower priority until fundamentals are solid)