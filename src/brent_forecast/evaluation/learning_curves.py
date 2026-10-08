"""Learning curves (any estimator) and the per-epoch training curves of the NumPy MLP.

Figures are saved as ``<plots_dir>/learning_curve_<model>.png`` and
``<plots_dir>/training_curves_mlp_numpy.png``.
"""

import logging
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from sklearn.model_selection import learning_curve

from brent_forecast._types import FloatArray
from brent_forecast.evaluation.metrics import safe_name, save_figure
from brent_forecast.models.neural_network import EpochRecord

logger = logging.getLogger(__name__)


def plot_learning_curve(
    estimator: Any,
    X_train: Any,
    y_train: Any,
    model_name: str,
    *,
    cv: Any,
    scoring: str,
    train_sizes: Sequence[float],
    plots_dir: Path,
) -> Path:
    """Compute and save the learning curve of an estimator or pipeline.

    ``learning_curve`` clones the estimator and refits it on growing prefixes
    of every training fold, so the fitted preprocessing steps are refitted too.

    Returns
    -------
    pathlib.Path
        Path of the saved PNG.
    """
    logger.info("Computing learning curve for %s", model_name)
    sizes, train_scores, val_scores = learning_curve(
        estimator,
        X_train,
        y_train,
        train_sizes=np.asarray(train_sizes),
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )
    for size, tr, va in zip(sizes, train_scores, val_scores, strict=True):
        logger.info(
            "  n=%d -> train AUC %.4f | CV AUC %.4f", size, float(np.mean(tr)), float(np.mean(va))
        )
    return _save_learning_curve_plot(sizes, train_scores, val_scores, model_name, plots_dir)


def plot_training_curves(history: Sequence[EpochRecord], plots_dir: Path) -> Path:
    """Save the NumPy MLP's train/validation loss and validation AUC per epoch."""
    epochs = [h["epoch"] for h in history]
    train_l = [h["train_loss"] for h in history]
    val_l = [h["val_loss"] for h in history]
    val_aucs = [h["val_auc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    axes[0].plot(epochs, train_l, label="train_loss")
    axes[0].plot(epochs, val_l, label="val_loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("BCE loss")
    axes[0].set_title("Loss curve — MLP NumPy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(epochs, val_aucs, color="C2", label="val_auc")
    axes[1].axhline(0.5, linestyle="--", color="gray", label="Random")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Val AUC")
    axes[1].set_title("Validation AUC per epoch (early-stopping hold-out)")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    return save_figure(fig, plots_dir, "training_curves_mlp_numpy.png")


def _save_learning_curve_plot(
    sizes: npt.NDArray[np.int_],
    train_scores: FloatArray,
    val_scores: FloatArray,
    model_name: str,
    plots_dir: Path,
) -> Path:
    """Plot mean ± std of train and CV AUC per training-set size."""
    train_mean = train_scores.mean(axis=1)
    train_std = train_scores.std(axis=1)
    val_mean = val_scores.mean(axis=1)
    val_std = val_scores.std(axis=1)

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(sizes, train_mean, "o-", color="C0", label="AUC train")
    ax.fill_between(sizes, train_mean - train_std, train_mean + train_std, alpha=0.15, color="C0")
    ax.plot(sizes, val_mean, "s-", color="C1", label="AUC CV (val fold)")
    ax.fill_between(sizes, val_mean - val_std, val_mean + val_std, alpha=0.15, color="C1")
    ax.axhline(0.5, linestyle="--", color="gray", alpha=0.5, label="Random")
    ax.set_xlabel("Training subset size")
    ax.set_ylabel("ROC AUC")
    ax.set_title(f"Learning curve — {model_name}")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return save_figure(fig, plots_dir, f"learning_curve_{safe_name(model_name)}.png")
