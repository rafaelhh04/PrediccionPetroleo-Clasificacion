"""Baseline model: L2-regularised logistic regression.

It is the minimum reference: any more complex model (SVM, Random Forest,
NumPy MLP) must beat its validation metrics to justify the extra complexity.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit

from brent_forecast._types import FloatArray, ModelResult
from brent_forecast.config import ModelSettings
from brent_forecast.models.base import fit_and_evaluate
from brent_forecast.models.tuning import grid_search

logger = logging.getLogger(__name__)

MODEL_NAME = "Logistic Regression"


def tune_hyperparameters(
    X_train: FloatArray,
    y_train: FloatArray,
    cv: TimeSeriesSplit,
    config: ModelSettings,
    scoring: str,
    seed: int,
) -> dict[str, Any]:
    """Grid-search ``config.grid`` with time-series CV.

    Returns
    -------
    dict
        Full hyperparameters: ``config.params`` + ``random_state`` + best grid values.
    """
    default_params = {**config.params, "random_state": seed}
    best = grid_search(
        LogisticRegression(**default_params),
        config.grid,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        model_name=MODEL_NAME,
    )
    return {**default_params, **best}


def train_and_evaluate(
    X_train: FloatArray,
    y_train: FloatArray,
    X_val: FloatArray,
    y_val: FloatArray,
    feature_cols: list[str],
    params: dict[str, Any],
    plots_dir: Path,
) -> ModelResult:
    """Train on train, evaluate on validation and log the top coefficients.

    ``penalty`` is not passed: its default is L2, and passing ``penalty='l2'``
    explicitly raises a FutureWarning in scikit-learn >= 1.8.
    """
    logger.info("[%s] Training with params: %s", MODEL_NAME, params)
    model = LogisticRegression(**params)
    result = fit_and_evaluate(model, MODEL_NAME, X_train, y_train, X_val, y_val, plots_dir)
    _log_top_coefficients(model, feature_cols, top_k=5)
    return result


def _log_top_coefficients(model: LogisticRegression, feature_cols: list[str], top_k: int) -> None:
    """Log the features with the largest absolute coefficient."""
    coefs = model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    lines = [
        f"    {'+' if coefs[i] > 0 else '-'} {feature_cols[i]:<25} coef = {coefs[i]:+.4f}"
        for i in order[:top_k]
    ]
    logger.info("[%s] Top %d features by |coef|:\n%s", MODEL_NAME, top_k, "\n".join(lines))
