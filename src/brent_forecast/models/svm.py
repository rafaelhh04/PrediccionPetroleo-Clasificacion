"""Support Vector Machine with an RBF kernel.

The RBF kernel assumes the relation between the features and the price
direction is non-linear and smooth — a reasonable hypothesis in finance.
"""

import logging
from pathlib import Path
from typing import Any

from sklearn.model_selection import TimeSeriesSplit
from sklearn.svm import SVC

from brent_forecast._types import FloatArray, ModelResult
from brent_forecast.config import ModelSettings
from brent_forecast.models.base import fit_and_evaluate
from brent_forecast.models.tuning import grid_search

logger = logging.getLogger(__name__)

MODEL_NAME = "SVM (RBF)"


def tune_hyperparameters(
    X_train: FloatArray,
    y_train: FloatArray,
    cv: TimeSeriesSplit,
    config: ModelSettings,
    scoring: str,
    seed: int,
) -> dict[str, Any]:
    """Grid-search ``config.grid`` with time-series CV.

    The search uses ``probability=False`` (~2x faster: ROC AUC is computed from
    ``decision_function``); the returned parameters keep ``probability=True``
    for the final model.

    Returns
    -------
    dict
        Full hyperparameters: ``config.params`` + ``random_state`` + best grid values.
    """
    default_params = {**config.params, "random_state": seed}
    cv_params = {**default_params, "probability": False}
    best = grid_search(
        SVC(**cv_params),
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
    """Train on train and evaluate on validation (RBF SVM scales as O(n^2)).

    ``feature_cols`` is unused; it keeps the signature shared by all models.
    """
    logger.info("[%s] Training with params: %s", MODEL_NAME, params)
    model = SVC(**params)
    result = fit_and_evaluate(model, MODEL_NAME, X_train, y_train, X_val, y_val, plots_dir)
    logger.info(
        "[%s] Support vectors: %d of %d", MODEL_NAME, model.support_.shape[0], X_train.shape[0]
    )
    return result
