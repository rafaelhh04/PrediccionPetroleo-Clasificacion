"""Shared metrics and plots, so that the 4 models are directly comparable.

Metrics (the project's standard five): accuracy, precision (macro),
recall (macro), F1 (macro) and ROC AUC.
"""

import logging
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
)

from brent_forecast._types import Metrics

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
