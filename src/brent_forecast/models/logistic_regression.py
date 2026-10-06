"""
Modelo baseline: Regresión Logística con regularización L2.

Sirve como referencia mínima: cualquier modelo más complejo del proyecto
(SVM, Random Forest, MLP-NumPy) debe superar las métricas de val obtenidas
aquí para justificar su mayor complejidad.

Fase 5: añadida `tune_hyperparameters` con TimeSeriesSplit + GridSearchCV.
`train_and_evaluate` acepta `params=None` (defaults Fase 3) o un dict de
hyperparámetros encontrados por el tuner.
"""

import logging
from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.linear_model import LogisticRegression
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

MODEL_NAME = "Logistic Regression"


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv,
    config: ModelSettings,
    scoring: str,
    seed: int,
) -> Dict[str, Any]:
    """
    Grid search con TimeSeriesSplit sobre `config.grid`.
    Devuelve un dict de hyperparámetros completos (config.params + seed + best).
    """
    default_params = {**config.params, "random_state": seed}
    base = LogisticRegression(**default_params)
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
    Entrena LogReg sobre train, predice sobre val, calcula métricas y plots.

    Nota: el parámetro `penalty` se omite — su default es L2, y pasar
    `penalty='l2'` explícitamente lanza FutureWarning en sklearn ≥1.8.
    """
    logger.info("[%s] Training with params: %s", MODEL_NAME, params)

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    y_pred_train  = model.predict(X_train)
    y_pred_val    = model.predict(X_val)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val   = model.predict_proba(X_val)[:, 1]

    metrics_train = compute_metrics(y_train, y_pred_train, y_proba_train)
    metrics_val   = compute_metrics(y_val,   y_pred_val,   y_proba_val)

    log_full_metrics(metrics_train, "Train", MODEL_NAME)
    log_full_metrics(metrics_val,   "Val",   MODEL_NAME)

    plot_confusion_matrix(y_val, y_pred_val, MODEL_NAME, plots_dir)
    plot_roc_curve(y_val, y_proba_val, MODEL_NAME, plots_dir)

    _print_top_coefficients(model, feature_cols, top_k=5)

    return {
        "model_name":    MODEL_NAME,
        "model":         model,
        "metrics_train": metrics_train,
        "metrics_val":   metrics_val,
        "y_pred_val":    y_pred_val,
        "y_proba_val":   y_proba_val,
    }


def _print_top_coefficients(model: LogisticRegression, feature_cols: list, top_k: int = 5) -> None:
    """Registra las features con mayor valor absoluto del coeficiente."""
    coefs = model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    lines = [
        f"    {'+' if coefs[i] > 0 else '-'} {feature_cols[i]:<25} coef = {coefs[i]:+.4f}"
        for i in order[:top_k]
    ]
    logger.info("[%s] Top %d features by |coef|:\n%s", MODEL_NAME, top_k, "\n".join(lines))
