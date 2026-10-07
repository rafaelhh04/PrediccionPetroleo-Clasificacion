"""Learning curves with TimeSeriesSplit.

sklearn models (LogReg/SVM/RF) use ``sklearn.model_selection.learning_curve``.
The NumPy MLP is not sklearn-compatible, so its curve is computed manually by
retraining ``train_mlp`` on growing subsets of every fold.

Figures are saved as ``<plots_dir>/learning_curve_<model>.png``.
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
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import TimeSeriesSplit, learning_curve

from brent_forecast._types import FloatArray
from brent_forecast.evaluation.metrics import safe_name, save_figure
from brent_forecast.models.neural_network import TRAIN_KWARGS, predict_proba, train_mlp

logger = logging.getLogger(__name__)


def plot_learning_curve_sklearn(
    estimator: Any,
    X_train: FloatArray,
    y_train: FloatArray,
    model_name: str,
    *,
    cv: TimeSeriesSplit,
    scoring: str,
    train_sizes: Sequence[float],
    plots_dir: Path,
) -> Path:
    """Compute and save the learning curve of an sklearn estimator.

    ``learning_curve`` clones the estimator and refits it on every fold.

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
    return _save_learning_curve_plot(sizes, train_scores, val_scores, model_name, plots_dir)


def plot_learning_curve_mlp(
    X_train: FloatArray,
    y_train: FloatArray,
    mlp_params: dict[str, Any],
    *,
    cv: TimeSeriesSplit,
    train_sizes: Sequence[float],
    plots_dir: Path,
    model_name: str = "MLP NumPy",
) -> Path:
    """Compute and save the learning curve of the NumPy MLP.

    For every relative size, the MLP is trained on the first N samples of each
    TimeSeriesSplit training fold (keeping temporal order); mean and std of the
    train and validation AUC are reported.

    Returns
    -------
    pathlib.Path
        Path of the saved PNG.
    """
    logger.info("Computing manual learning curve for %s", model_name)
    sizes_rel = np.asarray(train_sizes)
    train_scores = []
    val_scores = []

    train_kwargs = {k: v for k, v in mlp_params.items() if k in TRAIN_KWARGS}

    for frac in sizes_rel:
        size_train = []
        size_val = []
        for tr_idx, va_idx in cv.split(X_train):
            n_use = max(50, int(len(tr_idx) * frac))
            tr_sub = tr_idx[:n_use]

            X_tr, y_tr = X_train[tr_sub], y_train[tr_sub]
            X_va, y_va = X_train[va_idx], y_train[va_idx]
            best_params, _ = train_mlp(X_tr, y_tr, X_va, y_va, verbose=False, **train_kwargs)

            size_train.append(roc_auc_score(y_tr, predict_proba(X_tr, best_params)))
            size_val.append(roc_auc_score(y_va, predict_proba(X_va, best_params)))

        train_scores.append(size_train)
        val_scores.append(size_val)
        logger.info(
            "  frac=%.2f -> train AUC %.4f | CV AUC %.4f",
            frac,
            np.mean(size_train),
            np.mean(size_val),
        )

    sizes_abs = (sizes_rel * len(X_train)).astype(int)
    return _save_learning_curve_plot(
        sizes_abs, np.array(train_scores), np.array(val_scores), model_name, plots_dir
    )


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
