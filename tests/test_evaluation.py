"""Tests for brent_forecast.evaluation: metrics, plots, out-of-sample summaries and curves."""

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

from brent_forecast.evaluation.learning_curves import plot_learning_curve, plot_training_curves
from brent_forecast.evaluation.metrics import (
    compute_metrics,
    log_full_metrics,
    plot_confusion_matrix,
    safe_name,
)
from brent_forecast.evaluation.oos import (
    log_oos_summary,
    oos_metrics,
    plot_roc_oos,
    plot_window_auc,
    window_metrics,
)
from brent_forecast.models.mlp import NumpyMLPClassifier
from brent_forecast.validation.walk_forward import PurgedWalkForwardSplit


def _is_png(path: Path) -> bool:
    return path.is_file() and path.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"


# ── metrics ───────────────────────────────────────────────────


def test_compute_metrics_perfect_predictions() -> None:
    y = np.array([0, 1, 0, 1])

    metrics = compute_metrics(y, y, np.array([0.1, 0.9, 0.2, 0.8]))

    assert metrics == {"accuracy": 1.0, "precision": 1.0, "recall": 1.0, "f1": 1.0, "auc_roc": 1.0}


def test_compute_metrics_macro_averages_and_zero_division() -> None:
    y_true = np.array([0, 0, 0, 1])
    y_pred = np.array([0, 0, 0, 0])  # never predicts class 1

    metrics = compute_metrics(y_true, y_pred, np.array([0.1, 0.2, 0.3, 0.4]))

    assert metrics["accuracy"] == 0.75
    assert metrics["precision"] == pytest.approx((0.75 + 0.0) / 2)
    assert metrics["recall"] == pytest.approx((1.0 + 0.0) / 2)
    assert metrics["auc_roc"] == 1.0
    assert all(isinstance(v, float) for v in metrics.values())


def test_log_full_metrics(caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    metrics = {"accuracy": 0.5, "precision": 0.25, "recall": 0.75, "f1": 0.4, "auc_roc": 0.6}

    log_full_metrics(metrics, "Val", "Model X")

    assert "[Model X — Val]" in caplog.text
    assert "AUC-ROC    : 0.6000" in caplog.text


@pytest.mark.parametrize(
    ("name", "slug"),
    [
        ("Logistic Regression", "logistic_regression"),
        ("SVM (RBF)", "svm_rbf"),
        ("MLP NumPy", "mlp_numpy"),
    ],
)
def test_safe_name(name: str, slug: str) -> None:
    assert safe_name(name) == slug


def test_confusion_matrix_plot(plots_dir: Path) -> None:
    y = np.array([0, 1, 0, 1, 1])
    pred = np.array([0, 1, 1, 1, 0])

    path = plot_confusion_matrix(y, pred, "SVM (RBF)", plots_dir)

    assert path == plots_dir / "confusion_svm_rbf.png"
    assert _is_png(path)


# ── out-of-sample summaries ───────────────────────────────────


def _oos(n_windows: int = 3, size: int = 40) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    rng = np.random.default_rng(0)
    n = n_windows * size
    index = pd.RangeIndex(500, 500 + n)
    dates = pd.Series(pd.bdate_range("2024-01-01", periods=n), index=index)
    y = pd.Series(rng.integers(0, 2, n).astype(float), index=index)
    proba = pd.Series(np.clip(0.5 + 0.3 * (y - 0.5) + rng.normal(0, 0.2, n), 0, 1), index=index)
    window = pd.Series(np.repeat(np.arange(n_windows), size), index=index)
    return dates, y, proba, window


def test_oos_metrics_threshold() -> None:
    _, y, proba, _ = _oos()

    metrics = oos_metrics(y, proba)

    assert metrics == compute_metrics(y, (proba >= 0.5).astype(int), proba)
    assert metrics["auc_roc"] > 0.7


def test_window_metrics() -> None:
    dates, y, proba, window = _oos()
    y.iloc[80:] = 1.0  # last window has a single class: AUC undefined

    rows = window_metrics(dates, y, proba, window)

    assert [r["window"] for r in rows] == [0, 1, 2]
    assert [r["n"] for r in rows] == [40, 40, 40]
    assert rows[0]["start"] == "2024-01-01"
    assert rows[0]["end"] == str(dates.iloc[39].date())
    assert rows[0]["auc_roc"] is not None
    assert rows[2]["auc_roc"] is None
    assert 0 <= rows[2]["accuracy"] <= 1


def test_oos_plots_and_summary(plots_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    dates, y, proba, window = _oos()
    rows = window_metrics(dates, y, proba, window)
    summary = {
        "Model A": {
            "cv": {"auc_mean": 0.51, "auc_std": 0.02},
            "oos": oos_metrics(y, proba),
            "windows": rows,
        },
        "Model B": {
            "kind": "baseline",
            "cv": {"auc_mean": 0.49, "auc_std": 0.01},
            "oos": oos_metrics(y, 1 - proba),
            "windows": rows,
        },
    }

    assert _is_png(plot_roc_oos(y, {"Model A": proba, "Model B": 1 - proba}, plots_dir))
    assert _is_png(plot_window_auc({n: m["windows"] for n, m in summary.items()}, plots_dir))
    log_oos_summary(summary)

    assert "Walk-forward summary" in caplog.text
    assert "0.5100 ± 0.020" in caplog.text
    assert caplog.text.count("Model B") == 1
    assert "Model B *" in caplog.text
    assert "* naive baseline" in caplog.text


# ── learning / training curves ────────────────────────────────


@pytest.mark.parametrize(
    ("estimator", "slug"),
    [
        (LogisticRegression(), "logistic_regression"),
        (NumpyMLPClassifier(hidden_1=4, hidden_2=2, max_epochs=2, random_state=0), "mlp_numpy"),
    ],
)
def test_learning_curve(
    estimator: Any,
    slug: str,
    toy_xy: tuple[np.ndarray, np.ndarray],
    plots_dir: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    caplog.set_level("INFO")
    X, y = toy_xy
    name = "Logistic Regression" if slug == "logistic_regression" else "MLP NumPy"

    path = plot_learning_curve(
        estimator,
        X,
        y,
        name,
        cv=PurgedWalkForwardSplit(2, purge=1),
        scoring="roc_auc",
        train_sizes=[0.5, 1.0],
        plots_dir=plots_dir,
    )

    assert path == plots_dir / f"learning_curve_{slug}.png"
    assert _is_png(path)
    assert caplog.text.count("CV AUC") == 2


def test_training_curves(toy_xy: tuple[np.ndarray, np.ndarray], plots_dir: Path) -> None:
    X, y = toy_xy
    mlp = NumpyMLPClassifier(max_epochs=4, random_state=0).fit(X, y)

    path = plot_training_curves(mlp.history_, plots_dir)

    assert path == plots_dir / "training_curves_mlp_numpy.png"
    assert _is_png(path)
