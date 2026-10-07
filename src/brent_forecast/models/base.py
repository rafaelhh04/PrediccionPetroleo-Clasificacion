"""Training and validation evaluation shared by the sklearn models."""

import logging
from pathlib import Path
from typing import Any

from brent_forecast._types import FloatArray, ModelResult
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
    X_train: FloatArray,
    y_train: FloatArray,
    X_val: FloatArray,
    y_val: FloatArray,
    plots_dir: Path,
) -> ModelResult:
    """Fit an sklearn classifier on train and evaluate it on train and validation.

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
