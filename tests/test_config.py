"""Tests for configuration loading and environment overrides."""

from datetime import date
from pathlib import Path

import pytest
from pydantic import ValidationError

from brent_forecast.config import ModelSettings, SearchSpace, load_settings
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


@pytest.mark.parametrize(
    ("space", "message"),
    [
        ({"type": "int", "low": 5, "high": 5}, "low` < `high"),
        ({"type": "float", "low": 1.0}, "low` < `high"),
        ({"type": "float", "low": 0.0, "high": 1.0, "log": True}, "low` > 0"),
        ({"type": "categorical"}, "choices"),
    ],
)
def test_invalid_search_spaces_are_rejected(space: dict, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        SearchSpace(**space)


def test_a_model_needs_exactly_one_search_definition() -> None:
    space = {"C": SearchSpace(type="float", low=0.1, high=1.0)}

    with pytest.raises(ValidationError, match="exactly one"):
        ModelSettings(params={}, grid={"C": [1.0]}, space=space)
    with pytest.raises(ValidationError, match="exactly one"):
        ModelSettings(params={})
    assert ModelSettings(params={}, space=space).grid == {}


def test_lightgbm_is_tuned_with_optuna() -> None:
    settings = load_settings(CONFIG_PATH)

    assert settings.models.lightgbm.space
    assert not settings.models.lightgbm.grid
    assert settings.tuning.timeout is None  # reproducible by default


def test_serving_alias_can_be_switched_by_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    assert load_settings(CONFIG_PATH).serving.alias == "champion"

    monkeypatch.setenv("BRENT_SERVING__ALIAS", "challenger")

    serving = load_settings(CONFIG_PATH).serving
    assert serving.alias == "challenger"
    assert serving.live is True


def test_layered_files_are_deep_merged(tmp_path: Path) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("seed: 7\nmodels:\n  svm:\n    grid:\n      C: [1.0]\n")
    default = load_settings(CONFIG_PATH)

    settings = load_settings([CONFIG_PATH, overlay])

    assert settings.seed == 7
    assert settings.models.svm.grid["C"] == [1.0]  # lists are replaced...
    assert settings.models.svm.grid["gamma"] == default.models.svm.grid["gamma"]  # ...maps merged
    assert settings.models.svm.params == default.models.svm.params


def test_environment_still_overrides_layered_files(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    overlay = tmp_path / "overlay.yaml"
    overlay.write_text("tuning:\n  n_trials: 5\n")
    monkeypatch.setenv("BRENT_TUNING__N_TRIALS", "2")

    assert load_settings([CONFIG_PATH, overlay]).tuning.n_trials == 2


def test_demo_overlay_is_smaller_but_keeps_the_protocol() -> None:
    default = load_settings(CONFIG_PATH)
    demo = load_settings([CONFIG_PATH, CONFIG_PATH.parent / "demo.yaml"])

    assert demo.tuning.n_trials < default.tuning.n_trials
    assert demo.evaluation.bootstrap_resamples < default.evaluation.bootstrap_resamples
    # Same data, split, leakage controls, candidates and statistics: only the budget shrinks.
    for section in ("paths", "data", "split", "preprocessing", "backtest", "tracking"):
        assert getattr(demo, section) == getattr(default, section), section
    assert (demo.validation.purge, demo.validation.embargo) == (1, 5)
    assert demo.evaluation.alpha == default.evaluation.alpha
    assert demo.models.model_fields_set == default.models.model_fields_set


@pytest.mark.parametrize("paths", [[], ["missing.yaml"]])
def test_layered_files_must_exist(paths: list[str], tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings([CONFIG_PATH, *(tmp_path / p for p in paths)] if paths else [])
