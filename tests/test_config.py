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
    assert settings.split.train_end == date(2022, 1, 1)
    assert settings.split.val_end == date(2024, 1, 1)
    assert settings.cv.n_splits == 5
    assert settings.preprocessing.vif_threshold == 10.0
    assert settings.paths.plots_dir == Path("results/plots")


def test_env_overrides_yaml(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRENT_SEED", "7")
    monkeypatch.setenv("BRENT_SPLIT__TRAIN_END", "2021-01-01")
    monkeypatch.setenv("BRENT_MODELS__RANDOM_FOREST__GRID", '{"max_depth": [5]}')

    settings = load_settings(CONFIG_PATH)

    assert settings.seed == 7
    assert settings.split.train_end == date(2021, 1, 1)
    assert settings.split.val_end == date(2024, 1, 1)  # untouched sibling kept
    assert settings.models.random_forest.grid["max_depth"] == [5]


def test_kwargs_override_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRENT_SEED", "7")

    assert load_settings(CONFIG_PATH, seed=3).seed == 3


def test_invalid_split_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRENT_SPLIT__TRAIN_END", "2030-01-01")

    with pytest.raises(ValidationError, match="train_end"):
        load_settings(CONFIG_PATH)


def test_missing_config_file(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(tmp_path / "missing.yaml")
