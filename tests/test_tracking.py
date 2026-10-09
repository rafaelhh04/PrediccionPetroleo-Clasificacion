"""Tests for brent_forecast.tracking (MLflow tracking and model registry)."""

import json
import subprocess
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
import pytest
from mlflow import MlflowClient

from brent_forecast.config import Settings
from brent_forecast.evaluation.report import generate_report
from brent_forecast.pipeline import run
from brent_forecast.tracking import (
    CHALLENGER,
    CHAMPION,
    TRUSTED_TYPES,
    Decision,
    git_info,
    load_model,
    log_evaluation,
    log_training,
    registry_aliases,
    select_champion,
)

MODELS = {
    "Logistic Regression": {"key": "logistic_regression", "kind": "model"},
    "LightGBM": {"key": "lightgbm", "kind": "model"},
    "Persistence": {"key": "persistence", "kind": "baseline"},
    "Majority class": {"key": "majority", "kind": "baseline"},
}


def _report(lr_auc: float, lgbm_auc: float, significant: bool) -> dict[str, Any]:
    auc = {"Logistic Regression": lr_auc, "LightGBM": lgbm_auc, "Persistence": 0.51}
    auc["Majority class"] = 0.5
    return {
        "candidates": {n: {"auc": {"estimate": a}} for n, a in auc.items()},
        "reference_baseline": "Persistence",
        "comparisons": {
            "Logistic Regression": {"significant": significant, "p_holm": 0.01},
            "LightGBM": {"significant": False, "p_holm": 0.9},
        },
    }


# ── promotion rule ────────────────────────────────────────────


def test_a_significant_model_becomes_champion() -> None:
    decision = select_champion(_report(0.60, 0.52, significant=True), {"models": MODELS})

    assert decision == Decision(
        champion="logistic_regression",
        challenger="persistence",
        reason=decision.reason,
    )
    assert "significantly" in decision.reason


def test_without_significance_the_best_baseline_is_champion() -> None:
    decision = select_champion(_report(0.53, 0.55, significant=False), {"models": MODELS})

    assert decision.champion == "persistence"
    assert decision.challenger == "lightgbm"  # best model by AUC, kept under watch
    assert "No model beats" in decision.reason


def test_the_rule_only_promotes_the_best_model_by_auc() -> None:
    """A significant but not top-ranked model is not promoted on its own."""
    report = _report(0.53, 0.55, significant=True)  # LR significant, LightGBM better

    decision = select_champion(report, {"models": MODELS})

    assert decision.champion == "persistence"
    assert decision.challenger == "lightgbm"


# ── git metadata ──────────────────────────────────────────────


def test_git_info_inside_and_outside_a_repository(tmp_path: Path) -> None:
    inside = git_info(Path(__file__).parent)
    outside = git_info(tmp_path)

    assert len(inside["git_commit"]) == 40
    assert inside["git_dirty"] in {"true", "false"}
    assert outside == {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": "unknown"}


def test_git_info_when_git_is_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    def missing(*args: Any, **kwargs: Any) -> Any:
        raise FileNotFoundError("git")

    monkeypatch.setattr(subprocess, "run", missing)

    assert git_info()["git_commit"] == "unknown"


# ── end to end: train -> report -> registry -> load by alias ──


@pytest.fixture
def tracked(fast_settings: Settings, tmp_path: Path) -> Settings:
    tracking = fast_settings.tracking.model_copy(update={"enabled": True})
    return fast_settings.model_copy(update={"tracking": tracking})


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_training_and_evaluation_are_tracked_and_the_champion_loads_by_alias(
    tracked: Settings, small_raw_data_dir: Path
) -> None:
    result = run(tracked)
    run_id = log_training(tracked, result)
    generate_report(tracked)
    decision = log_evaluation(tracked)

    client = MlflowClient(tracked.tracking.uri)
    parent = client.get_run(run_id)
    assert parent.data.tags["stage"] == "evaluate"
    assert (
        len(parent.data.tags["git_commit"]) in {7, 40}
        or parent.data.tags["git_commit"] == "unknown"
    )
    assert parent.data.tags["data_verified"] == "None"
    assert parent.data.params["validation.purge"] == "1"
    assert "logistic_regression.p_holm" in parent.data.metrics
    artifacts = {a.path for a in client.list_artifacts(run_id)}
    assert {"results", "plots", "config", "report"} <= artifacts

    info = json.loads(tracked.paths.run_info_file.read_text())
    assert set(info["children"]) == {m["key"] for m in result.summary.values()}
    assert set(info["models"]) == set(info["children"])
    assert all(uri.startswith("models:/") for uri in info["models"].values())
    child = client.get_run(info["children"]["lightgbm"])
    assert child.data.tags["kind"] == "model"
    assert "oos_auc_roc" in child.data.metrics
    assert len(client.get_metric_history(child.info.run_id, "window_auc")) >= 2
    assert "num_leaves" in child.data.params

    # Synthetic data: no model can beat the baselines, so a baseline is champion.
    assert decision.champion in {"majority", "persistence", "stratified", "buy_and_hold"}
    aliases = registry_aliases(tracked)
    assert set(aliases) == {CHAMPION, CHALLENGER}
    assert aliases[CHAMPION]["candidate"] == decision.champion
    assert aliases[CHAMPION]["promotion_reason"] == decision.reason

    champion = load_model(tracked)
    features = result.input_example
    proba = champion.predict_proba(features)[:, 1]
    expected = result.final_models[decision.champion].predict_proba(features)[:, 1]
    pd.testing.assert_series_equal(pd.Series(proba), pd.Series(expected))
    assert load_model(tracked, CHALLENGER).predict_proba(features).shape == (5, 2)

    promotion = json.loads((tracked.paths.results_dir / "promotion.json").read_text())
    assert promotion["champion"] == decision.champion

    # A second evaluation registers new versions and moves the aliases.
    log_evaluation(tracked)
    again = registry_aliases(tracked)
    assert int(again[CHAMPION]["version"]) > int(aliases[CHAMPION]["version"])


def test_trusted_types_cover_every_candidate(fast_settings: Settings) -> None:
    """skops refuses unknown types on load: every candidate must be covered by the allowlist."""
    import numpy as np
    import skops.io as sio

    from brent_forecast.models.baselines import BASELINE_NAMES, build_baseline
    from brent_forecast.models.registry import MODEL_NAMES, build_pipeline

    rng = np.random.default_rng(0)
    X = pd.DataFrame(rng.normal(size=(200, 3)), columns=["a", "lag_ret_1", "vix"])
    y = pd.Series((X["a"] > 0).astype(float))
    for key in MODEL_NAMES:
        config = getattr(fast_settings.models, key)
        model = build_pipeline(key, config.params, fast_settings.preprocessing).fit(X, y)
        assert set(sio.get_untrusted_types(data=sio.dumps(model))) <= set(TRUSTED_TYPES), key
    for key in BASELINE_NAMES:
        model = build_baseline(key, seed=0).fit(X, y)
        assert set(sio.get_untrusted_types(data=sio.dumps(model))) <= set(TRUSTED_TYPES), key


def test_evaluation_requires_a_tracked_training_run(tracked: Settings) -> None:
    with pytest.raises(FileNotFoundError, match="tracking enabled"):
        log_evaluation(tracked)


def test_registry_is_empty_before_the_first_promotion(tracked: Settings) -> None:
    assert registry_aliases(tracked) == {}


def test_tracking_never_writes_outside_the_configured_locations(
    tracked: Settings, tmp_path: Path
) -> None:
    registry_aliases(tracked)

    assert Path(tracked.tracking.uri.removeprefix("sqlite:///")).is_file()
    assert mlflow.get_tracking_uri() == tracked.tracking.uri
