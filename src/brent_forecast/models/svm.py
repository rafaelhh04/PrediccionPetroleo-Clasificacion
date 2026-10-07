"""
Support Vector Machine con kernel RBF.

Modelo no lineal por excelencia. La elección del kernel RBF asume que
las relaciones entre features y dirección del precio son no lineales y
suaves — hipótesis razonable en finanzas.

Fase 5: añadida `tune_hyperparameters` con TimeSeriesSplit + GridSearchCV.
Durante la búsqueda se usa probability=False para acelerar (~2× speedup);
el modelo final se entrena con probability=True para reportar AUC val.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from brent_forecast.evaluation.metrics import (
    compute_metrics,
    log_full_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from brent_forecast.config import ModelSettings
from brent_forecast.models.tuning import log_grid_results

logger = logging.getLogger(__name__)

MODEL_NAME = "SVM (RBF)"


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv,
    config: ModelSettings,
    scoring: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Grid search con TimeSeriesSplit.
    Truco: durante CV usamos probability=False (sklearn calcula AUC con
    decision_function, no necesita predict_proba) → ~2× más rápido.
    El modelo final se entrenará con probability=True.
    """
    default_params = {**config.params, "random_state": seed}
    cv_params = {**default_params, "probability": False}
    base = SVC(**cv_params)
    grid = GridSearchCV(base, config.grid, cv=cv, scoring=scoring, n_jobs=-1, refit=False)
    grid.fit(X_train, y_train)

    results = [
        {
            "params":      grid.cv_results_["params"][i],
            "mean_cv_auc": float(grid.cv_results_["mean_test_score"][i]),
            "std_cv_auc":  float(grid.cv_results_["std_test_score"][i]),
        }
        for i in range(len(grid.cv_results_["params"]))
    ]
    log_grid_results(MODEL_NAME, results, top_k=5)

    return {**default_params, **grid.best_params_}


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    feature_cols: list,
    params: Dict[str, Any],
    plots_dir: Path,
) -> dict:
    """
    Entrena SVM RBF.

    Coste: SVM RBF escala O(n²). Con ~3.000 filas tarda segundos.
    """
    logger.info("[%s] Training with params: %s", MODEL_NAME, params)

    model = SVC(**params)
    model.fit(X_train, y_train)

    y_pred_train  = model.predict(X_train)
    y_pred_val    = model.predict(X_val)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val   = model.predict_proba(X_val)[:, 1]

    metrics_train = compute_metrics(y_train, y_pred_train, y_proba_train)
    metrics_val   = compute_metrics(y_val,   y_pred_val,   y_proba_val)

    log_full_metrics(metrics_train, "Train", MODEL_NAME)
    log_full_metrics(metrics_val,   "Val",   MODEL_NAME)
    logger.info(
        "[%s] Support vectors: %d of %d", MODEL_NAME, model.support_.shape[0], X_train.shape[0]
    )

    plot_confusion_matrix(y_val, y_pred_val, MODEL_NAME, plots_dir)
    plot_roc_curve(y_val, y_proba_val, MODEL_NAME, plots_dir)

    return {
        "model_name":    MODEL_NAME,
        "model":         model,
        "metrics_train": metrics_train,
        "metrics_val":   metrics_val,
        "y_pred_val":    y_pred_val,
        "y_proba_val":   y_proba_val,
    }
