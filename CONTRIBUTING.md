# Contributing to ChemAugur

Thanks for your interest in contributing! ChemAugur is an open-source project using geometry-based graph neural networks (MACE) to predict physical properties of chemicals — pure substances and mixtures — from only atomic numbers and 3D positions.

## Development Setup

1. **Fork and clone** the repository
   ```bash
   git clone git@github.com:YOUR_USERNAME/ChemAugur.git
   cd ChemAugur
   ```

2. **Create a virtual environment** (Python 3.11+)
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # Linux/macOS
   ```

3. **Install in editable mode** with dev and test dependencies
   ```bash
   pip install -r reqs/development.txt
   pip install -e ".[test,dev]"
   ```

4. **Verify installation**
   ```bash
   python -c "import chem_augur; print(chem_augur.__version__)"
   ```

## Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest --cov=src/chem_augur tests/ -v

# Run specific test file
pytest tests/test_conformer.py -v
```

## Code Style

ChemAugur uses [Black](https://github.com/psf/black) for code formatting:

```bash
# Check formatting
black --check src/ tests/

# Auto-format
black src/ tests/
```

## Making Changes

1. Create a feature branch from `main`
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and add tests

3. Run the test suite
   ```bash
   pytest tests/ -v
   ```

4. Format your code
   ```bash
   black src/ tests/
   ```

5. Commit with a descriptive message
   ```bash
   git commit -m "Add feature: description of change"
   ```

6. Push and open a pull request
   ```bash
   git push origin feature/your-feature-name
   ```

## Pull Request Guidelines

- Include tests for new functionality
- Update documentation if needed
- Keep PRs focused on a single change
- Ensure all existing tests pass
- Use type hints for all new functions
- Add docstrings following the existing style

## Project Structure

```
src/chem_augur/
├── config.py              # TrainingConfig dataclass
├── data/                  # Data download and inspection
├── features/              # 3D conformer generation, mixture builder, graph utils
├── models/                # MACE wrapper, GCN (legacy), model registry
├── training/              # Training pipelines
└── prediction/            # Prediction CLI and API
tests/                     # Test suite
```

## Architecture Philosophy

ChemAugur uses only `z` (atomic numbers) and `pos` (3D positions) as input. No bond types, hybridization labels, aromaticity flags, or hand-crafted features. The model learns all chemical semantics from geometry alone. This is a deliberate design choice — read the README for the full rationale.

## Reporting Issues

Open an issue on GitHub with:
- A clear description of the problem
- Steps to reproduce
- Expected vs actual behavior
- Python/PyTorch/PyG/RDKit versions

## License

By contributing, you agree that your contributions will be licensed under the GPL-3.0 license.