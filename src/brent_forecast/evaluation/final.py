"""Final evaluation on the test set.

PROJECT RULE: this module is called **exactly once** per run. Iterating on a
model based on test metrics overfits the test set and breaks the
methodology; if test metrics are bad, they are reported as they are.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve

from brent_forecast._types import ModelResult
from brent_forecast.evaluation.metrics import (
    compute_metrics,
    log_full_metrics,
    plot_confusion_matrix,
    save_figure,
)

logger = logging.getLogger(__name__)


def evaluate_on_test(
    fitted_results: Sequence[ModelResult],
    X_test: Any,
    y_test: Any,
    plots_dir: Path,
) -> list[ModelResult]:
    """Evaluate every fitted model (an estimator with ``predict_proba``) on the test set.

    Returns
    -------
    list of dict
        Copies of the input results with ``metrics_test``, ``y_pred_test`` and
        ``y_proba_test`` added.
    """
    logger.info(
        "Final evaluation on X_test (single pass; the test set is not used for anything else)"
    )

    test_results = []
    for r in fitted_results:
        name = r["model_name"]
        model = r["model"]

        y_proba = model.predict_proba(X_test)[:, 1]

        y_pred = (y_proba >= 0.5).astype(int)
        metrics_test = compute_metrics(y_test, y_pred, y_proba)

        log_full_metrics(metrics_test, "Test", name)
        plot_confusion_matrix(y_test, y_pred, f"{name} (test)", plots_dir)

        test_results.append(
            {
                **r,
                "metrics_test": metrics_test,
                "y_pred_test": y_pred,
                "y_proba_test": y_proba,
            }
        )

    return test_results


def plot_roc_test_comparison(
    test_results: Sequence[ModelResult], y_test: Any, plots_dir: Path
) -> Path:
    """Save the test ROC curves of all models overlaid in one figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in test_results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba_test"])
        auc = r["metrics_test"]["auc_roc"]
        ax.plot(fpr, tpr, linewidth=2, label=f"{r['model_name']} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Final ROC on TEST — 4 models")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, plots_dir, "roc_test_comparison.png")


def plot_train_val_test_summary(test_results: Sequence[ModelResult], plots_dir: Path) -> Path:
    """Save a bar chart of train/val/test AUC per model (visualises the shift)."""
    names = [r["model_name"] for r in test_results]
    auc_tr = [r["metrics_train"]["auc_roc"] for r in test_results]
    auc_va = [r["metrics_val"]["auc_roc"] for r in test_results]
    auc_te = [r["metrics_test"]["auc_roc"] for r in test_results]

    x = np.arange(len(names))
    width = 0.27

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.bar(x - width, auc_tr, width, label="AUC train", color="C0")
    ax.bar(x, auc_va, width, label="AUC val", color="C1")
    ax.bar(x + width, auc_te, width, label="AUC test", color="C2")
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Random")
    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=20, ha="right")
    ax.set_ylabel("ROC AUC")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("AUC on train / val / test — 4 models")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, plots_dir, "train_val_test_summary.png")


def log_final_summary_table(test_results: Sequence[ModelResult]) -> None:
    """Log the final comparison table: train / val / test AUC per model."""
    lines = [
        "=" * 78,
        f"{'Model':<25}{'AUC(tr)':>12}{'AUC(val)':>12}{'AUC(test)':>13}{'Gap te-val':>14}",
        "-" * 78,
    ]
    for r in test_results:
        auc_tr = r["metrics_train"]["auc_roc"]
        auc_va = r["metrics_val"]["auc_roc"]
        auc_te = r["metrics_test"]["auc_roc"]
        gap = auc_te - auc_va
        lines.append(
            f"{r['model_name']:<25}{auc_tr:>12.4f}{auc_va:>12.4f}{auc_te:>13.4f}{gap:>+14.4f}"
        )
    lines.append("=" * 78)
    logger.info("Final summary:\n%s", "\n".join(lines))
