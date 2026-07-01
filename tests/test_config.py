# tests/test_config.py
"""Tests for TrainingConfig dataclass and serialization."""
import os
import tempfile

import pytest

from chem_augur.config import TrainingConfig


class TestTrainingConfigDefaults:
    """Test default configuration values."""

    def test_default_config(self):
        """Default config should have sensible values."""
        config = TrainingConfig()

        assert config.model_name == "mace"
        assert config.hidden_channels == 128
        assert config.cutoff == 12.0
        assert config.correlation == 3
        assert config.max_ell == 3
        assert config.batch_size == 16
        assert config.learning_rate == 1e-3
        assert config.max_epochs == 200
        assert config.patience == 25
        assert config.seed == 42

    def test_config_validation_passes(self):
        """Default config should validate without errors."""
        config = TrainingConfig()
        config.validate()

    def test_invalid_cutoff_raises(self):
        """Negative cutoff should raise ValueError."""
        config = TrainingConfig(cutoff=-5.0)
        with pytest.raises(ValueError, match="Cutoff must be positive"):
            config.validate()

    def test_invalid_ratios_raises(self):
        """Split ratios > 1.0 should raise ValueError."""
        config = TrainingConfig(train_ratio=0.8, val_ratio=0.5, test_ratio=0.5)
        with pytest.raises(ValueError, match="sum to 1.0"):
            config.validate()

    def test_invalid_model_raises(self):
        """Unknown model should raise ValueError."""
        config = TrainingConfig(model_name="nonexistent")
        with pytest.raises(ValueError, match="Unknown model"):
            config.validate()

    def test_invalid_max_ell_raises(self):
        """Invalid max_ell for MACE should raise ValueError."""
        config = TrainingConfig(model_name="mace", max_ell=5)
        with pytest.raises(ValueError, match="max_ell must be"):
            config.validate()


class TestTrainingConfigSerialization:
    """Test TOML serialization and deserialization."""

    def test_to_dict_and_back(self):
        """Dict round-trip should preserve values."""
        original = TrainingConfig(
            model_name="mace",
            hidden_channels=256,
            cutoff=15.0,
            batch_size=8,
        )

        d = original.to_dict()
        restored = TrainingConfig.from_dict(d)

        assert restored.model_name == original.model_name
        assert restored.hidden_channels == original.hidden_channels
        assert restored.cutoff == original.cutoff
        assert restored.batch_size == original.batch_size

    def test_from_dict_ignores_unknown_keys(self):
        """Unknown keys in dict should be silently ignored."""
        d = {"model_name": "gcn", "unknown_field": 999}
        config = TrainingConfig.from_dict(d)

        assert config.model_name == "gcn"
        # unknown_field should not cause an error

    def test_toml_round_trip(self):
        """TOML save/load should preserve values."""
        original = TrainingConfig(
            model_name="mace",
            cutoff=15.0,
            correlation=3,
            batch_size=8,
            learning_rate=5e-4,
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".toml", delete=False
        ) as f:
            original.to_toml(f.name)
            toml_path = f.name

        try:
            loaded = TrainingConfig.from_toml(toml_path)
            assert loaded.model_name == "mace"
            assert loaded.cutoff == 15.0
            assert loaded.correlation == 3
            assert loaded.batch_size == 8
            assert loaded.learning_rate == 5e-4
        finally:
            os.unlink(toml_path)


class TestEnvironmentOverrides:
    """Test environment variable overrides."""

    def test_env_override_string(self, monkeypatch):
        """String field should be overridable via env var."""
        monkeypatch.setenv("CHEMAUGUR_MODEL_NAME", "gcn")
        config = TrainingConfig()
        config.apply_env_overrides()

        assert config.model_name == "gcn"

    def test_env_override_float(self, monkeypatch):
        """Float field should be overridable via env var."""
        monkeypatch.setenv("CHEMAUGUR_CUTOFF", "15.0")
        config = TrainingConfig()
        config.apply_env_overrides()

        assert config.cutoff == 15.0

    def test_env_override_int(self, monkeypatch):
        """Int field should be overridable via env var."""
        monkeypatch.setenv("CHEMAUGUR_BATCH_SIZE", "64")
        config = TrainingConfig()
        config.apply_env_overrides()

        assert config.batch_size == 64

    def test_env_override_bool(self, monkeypatch):
        """Bool field should be overridable via env var."""
        monkeypatch.setenv("CHEMAUGUR_TENSORBOARD", "false")
        config = TrainingConfig()
        config.apply_env_overrides()

        assert config.tensorboard is False

    def test_env_not_set_leaves_default(self, monkeypatch):
        """Missing env var should leave default unchanged."""
        # Ensure the env var is not set
        monkeypatch.delenv("CHEMAUGUR_CUTOFF", raising=False)
        config = TrainingConfig()
        config.apply_env_overrides()

        assert config.cutoff == 12.0  # Default value