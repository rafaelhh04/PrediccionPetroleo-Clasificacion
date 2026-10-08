"""End-to-end test of brent_forecast.pipeline on a small synthetic dataset."""

import json
from pathlib import Path

import joblib
import pandas as pd
import pytest

from brent_forecast.config import Settings
from brent_forecast.pipeline import run

MODELS = ["Logistic Regression", "SVM (RBF)", "Random Forest", "MLP NumPy"]
BASELINES = ["Majority class", "Persistence", "Stratified random", "Buy & hold"]
KEYS = ["logistic_regression", "svm", "random_forest", "mlp"]
BASELINE_KEYS = ["majority", "persistence", "stratified", "buy_and_hold"]
METRIC_KEYS = {"accuracy", "precision", "recall", "f1", "auc_roc"}


def test_run_writes_predictions_metrics_models_and_plots(
    fast_settings: Settings, small_raw_data_dir: Path
) -> None:
    run(fast_settings)
    paths = fast_settings.paths

    metrics = json.loads(paths.metrics_file.read_text())
    protocol = metrics["protocol"]
    assert protocol["test_start"] == "2024-01-01"
    assert (protocol["purge"], protocol["embargo"]) == (1, 5)
    assert protocol["n_windows"] >= 2
    assert list(metrics["models"]) == MODELS + BASELINES
    kinds = [m["kind"] for m in metrics["models"].values()]
    assert kinds == ["model"] * 4 + ["baseline"] * 4
    for per_model in metrics["models"].values():
        assert set(per_model) == {"kind", "cv", "oos", "windows"}
        assert set(per_model["oos"]) == METRIC_KEYS
        assert all(0.0 <= v <= 1.0 for v in per_model["oos"].values())
        assert 0.0 <= per_model["cv"]["auc_mean"] <= 1.0
        assert len(per_model["windows"]) == protocol["n_windows"]

    predictions = pd.read_csv(paths.predictions_file, parse_dates=["date"])
    probas = [f"proba_{k}" for k in KEYS + BASELINE_KEYS]
    assert list(predictions.columns) == ["date", "label", "window", *probas]
    assert predictions["date"].is_monotonic_increasing
    assert predictions["window"].nunique() == protocol["n_windows"]
    assert predictions[probas].notna().all().all()
    assert (predictions["proba_buy_and_hold"] == 1.0).all()
    buy_and_hold = metrics["models"]["Buy & hold"]["oos"]
    assert buy_and_hold["accuracy"] == pytest.approx(predictions["label"].mean())
    assert buy_and_hold["auc_roc"] == 0.5

    plots = {p.name for p in paths.plots_dir.glob("*.png")}
    assert {
        "roc_oos_comparison.png",
        "oos_auc_per_window.png",
        "training_curves_mlp_numpy.png",
        "learning_curve_mlp_numpy.png",
        "confusion_mlp_numpy_oos.png",
    } <= plots
    # Per model: out-of-sample confusion matrix and learning curve; plus 3 summary figures.
    assert len(plots) == 4 * 2 + 3

    for key in KEYS:
        model = joblib.load(paths.models_dir / f"{key}.joblib")
        assert hasattr(model, "predict_proba")


def test_run_is_deterministic(fast_settings: Settings, small_raw_data_dir: Path) -> None:
    run(fast_settings)
    first = (
        fast_settings.paths.metrics_file.read_bytes(),
        fast_settings.paths.predictions_file.read_bytes(),
    )

    run(fast_settings)

    assert fast_settings.paths.metrics_file.read_bytes() == first[0]
    assert fast_settings.paths.predictions_file.read_bytes() == first[1]
