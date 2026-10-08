"""Training and validation evaluation shared by every model pipeline."""

import logging
from pathlib import Path
from typing import Any

import numpy as np

from brent_forecast._types import ModelResult
from brent_forecast.evaluation.metrics import (
    compute_metrics,
    log_full_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)

logger = logging.getLogger(__name__)


def fit_and_evaluate(
    model: Any,
    model_name: str,
    X_train: Any,
    y_train: Any,
    X_val: Any,
    y_val: Any,
    plots_dir: Path,
) -> ModelResult:
    """Fit a classifier (or pipeline) on train and evaluate it on train and validation.

    Logs the metrics of both sets and saves the validation confusion matrix
    and ROC curve.

    Returns
    -------
    dict
        ``model_name``, fitted ``model``, ``metrics_train``, ``metrics_val``,
        ``y_pred_val`` and ``y_proba_val``.
    """
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val = model.predict_proba(X_val)[:, 1]

    metrics_train = compute_metrics(y_train, y_pred_train, y_proba_train)
    metrics_val = compute_metrics(y_val, y_pred_val, y_proba_val)

    log_full_metrics(metrics_train, "Train", model_name)
    log_full_metrics(metrics_val, "Val", model_name)

    plot_confusion_matrix(y_val, y_pred_val, model_name, plots_dir)
    plot_roc_curve(y_val, y_proba_val, model_name, plots_dir)

    return {
        "model_name": model_name,
        "model": model,
        "metrics_train": metrics_train,
        "metrics_val": metrics_val,
        "y_pred_val": y_pred_val,
        "y_proba_val": y_proba_val,
    }


def log_top_features(pipeline: Any, model_name: str, top_k: int = 5) -> None:
    """Log the most influential features of a fitted pipeline, when the model exposes them.

    Uses ``|coef_|`` for linear models and ``feature_importances_`` for trees;
    other models (calibrated SVM, MLP) are skipped.
    """
    model = pipeline.named_steps["model"]
    names = pipeline[:-1].get_feature_names_out()
    if hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        order = np.argsort(np.abs(coefs))[::-1][:top_k]
        lines = [
            f"    {'+' if coefs[i] > 0 else '-'} {names[i]:<25} coef = {coefs[i]:+.4f}"
            for i in order
        ]
        logger.info("[%s] Top %d features by |coef|:\n%s", model_name, top_k, "\n".join(lines))
    elif hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1][:top_k]
        lines = [f"    {names[i]:<25} importance = {importances[i]:.4f}" for i in order]
        logger.info("[%s] Top %d features by importance:\n%s", model_name, top_k, "\n".join(lines))
