"""Tests for the Optuna search in brent_forecast.models.tuning."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brent_forecast.config import SearchSpace, Settings, TuningSettings
from brent_forecast.models.registry import build_pipeline
from brent_forecast.models.tuning import optuna_search
from brent_forecast.validation.walk_forward import PurgedWalkForwardSplit

SPACE = {
    "C": SearchSpace(type="float", low=1e-3, high=10.0, log=True),
    "fit_intercept": SearchSpace(type="categorical", choices=[True, False]),
    "max_iter": SearchSpace(type="int", low=100, high=200),
}


def _data(seed: int = 0, n: int = 400) -> tuple[pd.DataFrame, pd.Series]:
    rng = np.random.default_rng(seed)
    X = pd.DataFrame(rng.normal(size=(n, 3)), columns=["a", "b", "lag_ret_1"])
    y = pd.Series((X["a"] + rng.normal(size=n) > 0).astype(float))
    return X, y


def _search(settings: Settings, tuning: TuningSettings, trials_file: Path | None = None) -> dict:
    X, y = _data()
    return dict(
        optuna_search(
            build_pipeline("logistic_regression", {}, settings.preprocessing),
            SPACE,
            X,
            y,
            cv=PurgedWalkForwardSplit(3, purge=1, embargo=2),
            scoring="roc_auc",
            tuning=tuning,
            seed=0,
            model_name="LR",
            trials_file=trials_file,
        )
    )


def test_optuna_search_finds_a_good_configuration_inside_the_space(
    settings: Settings, tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    tuning = TuningSettings(n_trials=8, timeout=None, pruning=False, startup_trials=3)

    best = _search(settings, tuning, trials_file=tmp_path / "trials.csv")

    params = best["params"]
    assert set(params) == {"model__C", "model__fit_intercept", "model__max_iter"}
    assert 1e-3 <= params["model__C"] <= 10.0
    assert 100 <= params["model__max_iter"] <= 200
    assert best["mean_cv_auc"] > 0.7  # the signal in column "a" is found
    assert best["std_cv_auc"] >= 0
    trials = pd.read_csv(tmp_path / "trials.csv")
    assert len(trials) == 8
    assert (trials["state"] == "COMPLETE").all()
    assert "Optuna: 8 trials (8 complete, 0 pruned)" in caplog.text


def test_optuna_search_is_reproducible_with_a_seed(settings: Settings) -> None:
    tuning = TuningSettings(n_trials=5, timeout=None, pruning=True, startup_trials=2)

    assert _search(settings, tuning) == _search(settings, tuning)


def test_pruning_stops_bad_trials(settings: Settings, tmp_path: Path) -> None:
    tuning = TuningSettings(n_trials=25, timeout=None, pruning=True, startup_trials=2)

    _search(settings, tuning, trials_file=tmp_path / "trials.csv")

    states = pd.read_csv(tmp_path / "trials.csv")["state"]
    assert (states == "PRUNED").any()
    assert (states == "COMPLETE").any()
