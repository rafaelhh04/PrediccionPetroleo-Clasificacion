"""Tests for brent_forecast.evaluation: metrics, plots, final evaluation and learning curves."""

from pathlib import Path
from typing import Any

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

from brent_forecast.evaluation.final import (
    evaluate_on_test,
    log_final_summary_table,
    plot_roc_test_comparison,
    plot_train_val_test_summary,
)
from brent_forecast.evaluation.learning_curves import (
    plot_learning_curve_mlp,
    plot_learning_curve_sklearn,
)
from brent_forecast.evaluation.metrics import (
    compute_metrics,
    log_full_metrics,
    log_summary_table,
    plot_confusion_matrix,
    plot_roc_comparison,
    plot_roc_curve,
    safe_name,
)
from brent_forecast.models.neural_network import train_mlp
from brent_forecast.models.tuning import make_time_series_cv

MLP_KWARGS: dict[str, Any] = {
    "hidden_1": 8,
    "hidden_2": 4,
    "dropout_p": 0.1,
    "learning_rate": 0.05,
    "batch_size": 32,
    "max_epochs": 3,
    "patience": 2,
    "random_state": 0,
}


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
    [("Logistic Regression", "logistic_regression"), ("SVM (RBF)", "svm_rbf"), ("MLP NumPy", "mlp_numpy")],
)
def test_safe_name(name: str, slug: str) -> None:
    assert safe_name(name) == slug


def test_single_model_plots(plots_dir: Path) -> None:
    y = np.array([0, 1, 0, 1, 1])
    proba = np.array([0.2, 0.7, 0.4, 0.9, 0.3])

    cm = plot_confusion_matrix(y, (proba > 0.5).astype(int), "SVM (RBF)", plots_dir)
    roc = plot_roc_curve(y, proba, "SVM (RBF)", plots_dir)

    assert cm == plots_dir / "confusion_svm_rbf.png"
    assert roc == plots_dir / "roc_svm_rbf.png"
    assert _is_png(cm)
    assert _is_png(roc)


def _fake_result(name: str, rng: np.random.Generator, y: np.ndarray) -> dict[str, Any]:
    metrics = {"accuracy": 0.5, "precision": 0.5, "recall": 0.5, "f1": 0.5, "auc_roc": 0.55}
    return {
        "model_name": name,
        "metrics_train": metrics,
        "metrics_val": metrics,
        "metrics_test": {**metrics, "auc_roc": 0.45},
        "y_proba_val": rng.random(len(y)),
        "y_proba_test": rng.random(len(y)),
    }


def test_comparison_plots_and_tables(plots_dir: Path, caplog: pytest.LogCaptureFixture) -> None:
    caplog.set_level("INFO")
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 50)
    results = [_fake_result("Model A", rng, y), _fake_result("Model B", rng, y)]

    assert _is_png(plot_roc_comparison(results, y, plots_dir))
    assert _is_png(plot_roc_test_comparison(results, y, plots_dir))
    assert _is_png(plot_train_val_test_summary(results, plots_dir))
    log_summary_table(results)
    log_final_summary_table(results)

    assert "Validation summary" in caplog.text
    assert "Final summary" in caplog.text
    assert "-0.1000" in caplog.text  # test - val gap
    assert caplog.text.count("Model B") >= 2


# ── final evaluation ──────────────────────────────────────────


def test_evaluate_on_test_handles_sklearn_and_mlp(
    toy_xy: tuple[np.ndarray, np.ndarray], plots_dir: Path
) -> None:
    X, y = toy_xy
    X_tr, y_tr, X_te, y_te = X[:200], y[:200], X[200:], y[200:]
    logreg = LogisticRegression().fit(X_tr, y_tr)
    weights, history = train_mlp(X_tr, y_tr, X_te, y_te, verbose=False, **MLP_KWARGS)
    fitted = [
        {"model_name": "LogReg", "model": logreg},
        {"model_name": "MLP", "model": {"params": weights, "history": history, "config": {}}},
    ]

    results = evaluate_on_test(fitted, X_te, y_te, plots_dir)

    assert [r["model_name"] for r in results] == ["LogReg", "MLP"]
    for r in results:
        assert set(r["metrics_test"]) == {"accuracy", "precision", "recall", "f1", "auc_roc"}
        assert r["y_proba_test"].shape == (100,)
        assert set(np.unique(r["y_pred_test"])) <= {0, 1}
    np.testing.assert_allclose(results[0]["y_proba_test"], logreg.predict_proba(X_te)[:, 1])
    assert results[0]["metrics_test"]["auc_roc"] > 0.8  # the toy problem is learnable
    assert _is_png(plots_dir / "confusion_logreg_test.png")
    assert "metrics_test" not in fitted[0]  # inputs are not mutated


# ── learning curves ───────────────────────────────────────────


def test_learning_curve_sklearn(toy_xy: tuple[np.ndarray, np.ndarray], plots_dir: Path) -> None:
    X, y = toy_xy

    path = plot_learning_curve_sklearn(
        LogisticRegression(),
        X,
        y,
        "Logistic Regression",
        cv=make_time_series_cv(2),
        scoring="roc_auc",
        train_sizes=[0.5, 1.0],
        plots_dir=plots_dir,
    )

    assert path == plots_dir / "learning_curve_logistic_regression.png"
    assert _is_png(path)


def test_learning_curve_mlp(
    toy_xy: tuple[np.ndarray, np.ndarray], plots_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level("INFO")
    X, y = toy_xy

    path = plot_learning_curve_mlp(
        X,
        y,
        {**MLP_KWARGS, "unused_key": 1},
        cv=make_time_series_cv(2),
        train_sizes=[0.5, 1.0],
        plots_dir=plots_dir,
    )

    assert path == plots_dir / "learning_curve_mlp_numpy.png"
    assert _is_png(path)
    assert caplog.text.count("frac=") == 2
