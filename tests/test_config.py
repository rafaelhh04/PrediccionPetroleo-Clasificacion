"""Tests for configuration loading and environment overrides."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from brent_forecast.config import load_settings
from conftest import CONFIG_PATH


def test_default_config_loads() -> None:
    settings = load_settings(CONFIG_PATH)

    assert settings.seed == 42
    assert settings.split.test_start == date(2024, 1, 1)
    assert settings.validation.tuning_splits == 5
    assert settings.validation.purge == 1
    assert settings.validation.max_train_size is None  # expanding by default
    assert settings.preprocessing.vif_threshold == 10.0
    assert settings.paths.plots_dir == Path("results/plots")


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRENT_SEED", "7")
    monkeypatch.setenv("BRENT_SPLIT__TEST_START", "2021-01-01")
    monkeypatch.setenv("BRENT_MODELS__RANDOM_FOREST__GRID", '{"max_depth": [5]}')

    settings = load_settings(CONFIG_PATH)

    assert settings.seed == 7
    assert settings.split.test_start == date(2021, 1, 1)
    assert settings.validation.embargo == 5  # untouched section kept
    assert settings.models.random_forest.grid["max_depth"] == [5]


def test_kwargs_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRENT_SEED", "7")

    assert load_settings(CONFIG_PATH, seed=3).seed == 3


@pytest.mark.parametrize(
    ("variable", "value", "field"),
    [
        ("BRENT_VALIDATION__MODE", "shuffled", "mode"),
        ("BRENT_VALIDATION__PURGE", "-1", "purge"),
        ("BRENT_VALIDATION__TUNING_SPLITS", "1", "tuning_splits"),
    ],
)
def test_invalid_validation_settings_are_rejected(
    monkeypatch: pytest.MonkeyPatch, variable: str, value: str, field: str
) -> None:
    monkeypatch.setenv(variable, value)

    with pytest.raises(ValidationError, match=field):
        load_settings(CONFIG_PATH)


def test_rolling_mode_sets_a_training_window(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRENT_VALIDATION__MODE", "rolling")

    validation = load_settings(CONFIG_PATH).validation

    assert validation.max_train_size == validation.rolling_train_size


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.yaml")


def test_grid_overrides_are_deep_merged(monkeypatch: pytest.MonkeyPatch) -> None:
    """Documented semantics: overriding one grid key keeps the other keys from the YAML."""
    default = load_settings(CONFIG_PATH).models.random_forest.grid
    monkeypatch.setenv("BRENT_MODELS__RANDOM_FOREST__GRID", '{"max_depth": [3]}')

    grid = load_settings(CONFIG_PATH).models.random_forest.grid

    assert grid["max_depth"] == [3]
    assert grid["n_estimators"] == default["n_estimators"]
    assert grid["min_samples_leaf"] == default["min_samples_leaf"]


def test_data_date_bounds_are_validated(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRENT_DATA__EXPECTED_START_MIN", "2012-01-01")

    with pytest.raises(ValidationError, match="expected_start_min"):
        load_settings(CONFIG_PATH)


def test_models_dir_lives_under_results() -> None:
    settings = load_settings(CONFIG_PATH)

    assert settings.paths.models_dir == settings.paths.results_dir / "models"
