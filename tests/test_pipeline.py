"""End-to-end test of brent_forecast.pipeline on a small synthetic dataset."""

import json
from pathlib import Path

import pytest

from brent_forecast.config import Settings
from brent_forecast.pipeline import run

MODELS = ["Logistic Regression", "SVM (RBF)", "Random Forest", "MLP NumPy"]


@pytest.mark.filterwarnings("ignore::FutureWarning")  # SVC(probability=True), see roadmap
def test_run_writes_metrics_and_plots(fast_settings: Settings, small_raw_data_dir: Path) -> None:
    run(fast_settings)

    metrics = json.loads(fast_settings.paths.metrics_file.read_text())
    assert list(metrics) == MODELS
    for per_model in metrics.values():
        assert set(per_model) == {"train", "val", "test"}
        for split in per_model.values():
            assert set(split) == {"accuracy", "precision", "recall", "f1", "auc_roc"}
            assert all(0.0 <= v <= 1.0 for v in split.values())

    plots = {p.name for p in fast_settings.paths.plots_dir.glob("*.png")}
    assert {
        "roc_comparison.png",
        "roc_test_comparison.png",
        "train_val_test_summary.png",
        "training_curves_mlp_numpy.png",
        "learning_curve_mlp_numpy.png",
    } <= plots
    # Per model: confusion (val + test), ROC (val) and learning curve; plus 4 summary figures.
    assert len(plots) == 4 * 4 + 4

    models_dir = fast_settings.paths.models_dir
    assert sorted(p.name for p in models_dir.glob("*.joblib")) == [
        "logistic_regression.joblib",
        "mlp.joblib",
        "random_forest.joblib",
        "svm.joblib",
    ]


@pytest.mark.filterwarnings("ignore::FutureWarning")
def test_run_is_deterministic(fast_settings: Settings, small_raw_data_dir: Path) -> None:
    run(fast_settings)
    first = fast_settings.paths.metrics_file.read_bytes()

    run(fast_settings)

    assert fast_settings.paths.metrics_file.read_bytes() == first
