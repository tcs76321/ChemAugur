# config.py
"""
Training configuration system for ChemAugur.

Provides a TrainingConfig dataclass with TOML serialization and environment
variable override support. Replaces the module-level constants previously
scattered across training scripts.

Usage:
    # From defaults
    config = TrainingConfig()

    # From TOML file
    config = TrainingConfig.from_toml("config.toml")

    # Override with environment variables
    config.apply_env_overrides()

    # Save for reproducibility
    config.to_toml("experiment_config.toml")
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field, asdict
from typing import Optional, Dict, Any, List


@dataclass
class TrainingConfig:
    """Configuration for a ChemAugur training run.

    All fields can be set via TOML file, constructor keyword arguments,
    or environment variables (prefixed with CHEMAUGUR_).

    Environment variable override: any field can be overridden by setting
    an environment variable named CHEMAUGUR_{FIELD_NAME_UPPER}. For example,
    ``CHEMAUGUR_CUTOFF=15.0`` overrides the ``cutoff`` field.
    """

    # ---- Model ----
    model_name: str = "mace"
    """Model architecture: 'mace' or 'gcn'."""

    hidden_channels: int = 128
    """Hidden dimension for node embeddings."""

    num_interactions: int = 2
    """Number of message-passing interaction blocks (MACE only)."""

    max_ell: int = 3
    """Maximum spherical harmonic order (l=0,1,2,3 for MACE only)."""

    num_cutoff_basis: int = 5
    """Number of radial basis functions for distance expansion."""

    correlation: int = 3
    """Many-body correlation order (3 = up to quartets/dihedrals, MACE only)."""

    cutoff: float = 12.0
    """Interaction cutoff radius in Angstroms."""

    # ---- GCN-specific (ignored by MACE) ----
    num_node_features: int = 34
    """Number of input node features (GCN only)."""

    num_layers: int = 4
    """Number of GCN convolutional layers (GCN only)."""

    dropout: float = 0.5
    """Dropout probability (GCN only)."""

    # ---- Training ----
    batch_size: int = 16
    """Training batch size."""

    learning_rate: float = 1e-3
    """Initial learning rate."""

    max_epochs: int = 200
    """Maximum number of training epochs."""

    patience: int = 25
    """Early stopping patience (epochs without improvement)."""

    weight_decay: float = 1e-5
    """Weight decay (L2 regularization) for Adam optimizer."""

    lr_patience: int = 10
    """ReduceLROnPlateau patience (epochs)."""

    lr_factor: float = 0.5
    """Factor by which to reduce learning rate on plateau."""

    seed: int = 42
    """Random seed for reproducibility."""

    grad_clip_norm: Optional[float] = 1.0
    """Maximum gradient norm for clipping (None to disable)."""

    # ---- Data ----
    target_property: str = "pIC50"
    """Molecule property name to use as regression target."""

    train_ratio: float = 0.8
    """Fraction of data for training."""

    val_ratio: float = 0.1
    """Fraction of data for validation."""

    test_ratio: float = 0.1
    """Fraction of data for testing."""

    num_molecules_per_sample: int = 10
    """Number of molecules to place per mixture/pure sample."""

    sdf_files: List[str] = field(default_factory=lambda: [
        "src/chem_augur/data/Compound_000000001_000500000.sdf",
        "src/chem_augur/data/Compound_000500001_001000000.sdf",
        "src/chem_augur/data/Compound_001000001_001500000.sdf",
    ])
    """Paths to SDF data files."""

    # ---- Paths ----
    checkpoint_dir: str = "checkpoints"
    """Directory for saving model checkpoints."""

    log_dir: str = "logs"
    """Directory for TensorBoard logs."""

    root_dir: str = "."
    """Root directory for dataset caching."""

    model_dir: str = "src/chem_augur/models/gnn"
    """Directory for legacy GCN model checkpoints."""

    # ---- Experiment tracking ----
    experiment_name: Optional[str] = None
    """Optional experiment name for logging. Auto-generated if None."""

    tensorboard: bool = True
    """Whether to log metrics to TensorBoard."""

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to a plain dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "TrainingConfig":
        """Create config from a dictionary, ignoring unknown keys."""
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered = {k: v for k, v in d.items() if k in valid_keys}
        return cls(**filtered)

    def to_toml(self, path: str) -> None:
        """Save configuration to a TOML file."""
        try:
            import tomllib  # Python 3.11+
            import tomli_w  # type: ignore
        except ImportError:
            raise ImportError(
                "tomli_w is required for TOML serialization. "
                "Install with: pip install tomli-w"
            )

        with open(path, "w") as f:
            tomli_w.dump(self.to_dict(), f)

    @classmethod
    def from_toml(cls, path: str) -> "TrainingConfig":
        """Load configuration from a TOML file."""
        try:
            import tomllib
        except ImportError:
            try:
                import tomli as tomllib  # type: ignore
            except ImportError:
                raise ImportError(
                    "tomli or Python 3.11+ is required for TOML parsing. "
                    "Install with: pip install tomli"
                )

        with open(path, "rb") as f:
            data = tomllib.load(f)

        return cls.from_dict(data)

    # ------------------------------------------------------------------
    # Environment variable overrides
    # ------------------------------------------------------------------

    def apply_env_overrides(self) -> None:
        """
        Override config fields from environment variables.

        Environment variables should be named ``CHEMAUGUR_{FIELD_NAME}``
        in uppercase. For example::

            export CHEMAUGUR_CUTOFF=15.0
            export CHEMAUGUR_BATCH_SIZE=32
            export CHEMAUGUR_MODEL_NAME=mace

        Fields with list values (e.g., sdf_files) are not supported for
        environment variable override.
        """
        for field_def in self.__dataclass_fields__.values():
            env_key = f"CHEMAUGUR_{field_def.name.upper()}"
            env_value = os.environ.get(env_key)
            if env_value is not None:
                field_type = field_def.type
                # Parse string to appropriate type
                if field_type == bool or field_type == "bool":
                    setattr(self, field_def.name,
                            env_value.lower() in ("true", "1", "yes"))
                elif field_type == int or field_type == "int":
                    setattr(self, field_def.name, int(env_value))
                elif field_type == float or field_type == "float":
                    setattr(self, field_def.name, float(env_value))
                elif field_type == str or field_type == "str":
                    setattr(self, field_def.name, env_value)
                elif field_type == Optional[str]:
                    setattr(self, field_def.name, env_value if env_value else None)
                elif field_type == Optional[float]:
                    setattr(self, field_def.name,
                            float(env_value) if env_value else None)
                # List fields are skipped for env override

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validate(self) -> None:
        """
        Validate configuration consistency.

        Raises:
            ValueError: If any configuration values are invalid.
        """
        if self.train_ratio + self.val_ratio + self.test_ratio > 1.0 + 1e-6:
            raise ValueError(
                f"Data split ratios sum to "
                f"{self.train_ratio + self.val_ratio + self.test_ratio}, "
                f"must sum to 1.0"
            )

        if self.cutoff <= 0:
            raise ValueError(f"Cutoff must be positive, got {self.cutoff}")

        if self.batch_size < 1:
            raise ValueError(f"Batch size must be >= 1, got {self.batch_size}")

        if self.learning_rate <= 0:
            raise ValueError(
                f"Learning rate must be positive, got {self.learning_rate}"
            )

        if self.patience < 1:
            raise ValueError(f"Patience must be >= 1, got {self.patience}")

        if self.seed < 0:
            raise ValueError(f"Seed must be non-negative, got {self.seed}")

        valid_models = {"gcn", "mace"}
        if self.model_name not in valid_models:
            raise ValueError(
                f"Unknown model '{self.model_name}'. "
                f"Valid options: {valid_models}"
            )

        if self.max_ell not in (0, 1, 2, 3) and self.model_name == "mace":
            raise ValueError(
                f"max_ell must be 0, 1, 2, or 3 for MACE, got {self.max_ell}"
            )

        if self.correlation < 1 and self.model_name == "mace":
            raise ValueError(
                f"correlation must be >= 1 for MACE, got {self.correlation}"
            )