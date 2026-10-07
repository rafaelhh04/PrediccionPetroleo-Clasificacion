"""Random Forest: bagged ensemble of decision trees.

Trees do not need scaled features, but the same scaled ``X_train`` as the
other models is used to keep the comparison strict.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit

from brent_forecast._types import FloatArray, ModelResult
from brent_forecast.config import ModelSettings
from brent_forecast.models.base import fit_and_evaluate
from brent_forecast.models.tuning import grid_search

logger = logging.getLogger(__name__)

MODEL_NAME = "Random Forest"


def tune_hyperparameters(
    X_train: FloatArray,
    y_train: FloatArray,
    cv: TimeSeriesSplit,
    config: ModelSettings,
    scoring: str,
    seed: int,
) -> dict[str, Any]:
    """Grid-search ``config.grid`` with time-series CV.

    The outer search runs with ``n_jobs=1`` to avoid nesting parallelism with
    the forest's own ``n_jobs=-1`` (oversubscription).

    Returns
    -------
    dict
        Full hyperparameters: ``config.params`` + ``random_state`` + best grid values.
    """
    default_params = {**config.params, "random_state": seed}
    best = grid_search(
        RandomForestClassifier(**default_params),
        config.grid,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=1,
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
    """Train on train, evaluate on validation and log the top feature importances."""
    logger.info("[%s] Training with params: %s", MODEL_NAME, params)
    model = RandomForestClassifier(**params)
    result = fit_and_evaluate(model, MODEL_NAME, X_train, y_train, X_val, y_val, plots_dir)
    _log_top_features(model, feature_cols, top_k=5)
    return result


def _log_top_features(model: RandomForestClassifier, feature_cols: list[str], top_k: int) -> None:
    """Log the features with the highest importance (mean decrease in impurity)."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    lines = [f"    {feature_cols[i]:<25} importance = {importances[i]:.4f}" for i in order[:top_k]]
    logger.info("[%s] Top %d features by importance:\n%s", MODEL_NAME, top_k, "\n".join(lines))
