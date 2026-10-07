"""Shared metrics and plots, so that the 4 models are directly comparable.

Metrics (the project's standard five): accuracy, precision (macro),
recall (macro), F1 (macro) and ROC AUC.
"""

import logging
from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless backend: never block the pipeline on a display
import matplotlib.pyplot as plt
import numpy.typing as npt
from matplotlib.figure import Figure
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from brent_forecast._types import Metrics, ModelResult

logger = logging.getLogger(__name__)


def compute_metrics(
    y_true: npt.ArrayLike, y_pred: npt.ArrayLike, y_proba: npt.ArrayLike
) -> Metrics:
    """Compute the five project metrics on one set.

    Parameters
    ----------
    y_true
        True labels (0/1).
    y_pred
        Hard predictions (0/1).
    y_proba
        Probabilities ``P(y=1)``, needed for the ROC AUC.

    Returns
    -------
    dict
        Keys ``accuracy``, ``precision``, ``recall``, ``f1``, ``auc_roc``.
    """
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "auc_roc": float(roc_auc_score(y_true, y_proba)),
    }


def log_full_metrics(metrics: Metrics, set_name: str, model_name: str) -> None:
    """Log the five metrics of one set (train/val/test) in a readable block."""
    logger.info(
        "[%s — %s]\n"
        "  Accuracy   : %.4f\n"
        "  Precision  : %.4f   (macro)\n"
        "  Recall     : %.4f   (macro)\n"
        "  F1         : %.4f   (macro)\n"
        "  AUC-ROC    : %.4f",
        model_name,
        set_name,
        metrics["accuracy"],
        metrics["precision"],
        metrics["recall"],
        metrics["f1"],
        metrics["auc_roc"],
    )


def plot_confusion_matrix(
    y_true: npt.ArrayLike,
    y_pred: npt.ArrayLike,
    model_name: str,
    save_dir: Path,
) -> Path:
    """Save the confusion matrix of a model as a PNG and return its path."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Down (0)", "Up (1)"])
    ax.set_yticklabels(["Down (0)", "Up (1)"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"Confusion matrix — {model_name}")

    for i in range(2):
        for j in range(2):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center", color=color)

    fig.colorbar(im, ax=ax, fraction=0.045)
    fig.tight_layout()
    return save_figure(fig, save_dir, f"confusion_{safe_name(model_name)}.png")


def plot_roc_curve(
    y_true: npt.ArrayLike,
    y_proba: npt.ArrayLike,
    model_name: str,
    save_dir: Path,
) -> Path:
    """Save the ROC curve of a model as a PNG and return its path."""
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    auc = roc_auc_score(y_true, y_proba)

    fig, ax = plt.subplots(figsize=(5, 4.5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.3f}", linewidth=2)
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False positive rate (FPR)")
    ax.set_ylabel("True positive rate (TPR)")
    ax.set_title(f"ROC curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, save_dir, f"roc_{safe_name(model_name)}.png")


def plot_roc_comparison(
    results: Sequence[ModelResult], y_val: npt.ArrayLike, save_dir: Path
) -> Path:
    """Save the validation ROC curves of all models overlaid in one figure."""
    fig, ax = plt.subplots(figsize=(6, 5))
    for r in results:
        fpr, tpr, _ = roc_curve(y_val, r["y_proba_val"])
        auc = r["metrics_val"]["auc_roc"]
        ax.plot(fpr, tpr, linewidth=2, label=f"{r['model_name']} (AUC={auc:.3f})")

    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random")
    ax.set_xlabel("False positive rate (FPR)")
    ax.set_ylabel("True positive rate (TPR)")
    ax.set_title("ROC comparison — all models (validation)")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, save_dir, "roc_comparison.png")


def log_summary_table(results: Sequence[ModelResult]) -> None:
    """Log a train/validation comparison table of the models."""
    lines = [
        "=" * 82,
        f"{'Model':<25}"
        f"{'Acc(tr)':>10}{'Acc(val)':>10}{'Prec(val)':>11}"
        f"{'Rec(val)':>10}{'F1(val)':>9}{'AUC(val)':>10}",
        "-" * 82,
    ]
    for r in results:
        m_tr = r["metrics_train"]
        m_va = r["metrics_val"]
        lines.append(
            f"{r['model_name']:<25}"
            f"{m_tr['accuracy']:>10.4f}"
            f"{m_va['accuracy']:>10.4f}"
            f"{m_va['precision']:>11.4f}"
            f"{m_va['recall']:>10.4f}"
            f"{m_va['f1']:>9.4f}"
            f"{m_va['auc_roc']:>10.4f}"
        )
    lines.append("=" * 82)
    logger.info("Validation summary:\n%s", "\n".join(lines))


def safe_name(name: str) -> str:
    """Turn a model name into a file-name slug: 'SVM (RBF)' -> 'svm_rbf'."""
    return name.lower().replace(" ", "_").replace("(", "").replace(")", "")


def save_figure(fig: Figure, save_dir: Path, filename: str) -> Path:
    """Save ``fig`` as ``save_dir/filename`` (dpi=120), close it and return the path."""
    save_dir.mkdir(parents=True, exist_ok=True)
    path = save_dir / filename
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
