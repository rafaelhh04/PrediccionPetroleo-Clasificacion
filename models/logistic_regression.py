"""
Modelo baseline: Regresión Logística con regularización L2.

Sirve como referencia mínima: cualquier modelo más complejo del proyecto
(SVM, Random Forest, MLP-NumPy) debe superar las métricas de val obtenidas
aquí para justificar su mayor complejidad.

Fase 5: añadida `tune_hyperparameters` con TimeSeriesSplit + GridSearchCV.
`train_and_evaluate` acepta `params=None` (defaults Fase 3) o un dict de
hyperparámetros encontrados por el tuner.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

from utils.evaluation import (
    compute_metrics,
    print_full_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from utils.tuning import make_time_series_cv, print_grid_results, SCORING

MODEL_NAME = "Logistic Regression"


DEFAULT_PARAMS: Dict[str, Any] = {
    "C": 1.0,
    "solver": "lbfgs",
    "max_iter": 1000,
    "random_state": 42,
}

GRID: Dict[str, list] = {
    "C": [0.01, 0.1, 1.0, 10.0],
}


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv=None,
) -> Dict[str, Any]:
    """
    Grid search con TimeSeriesSplit sobre el espacio definido en GRID.
    Devuelve un dict de hyperparámetros completos (DEFAULT_PARAMS + best).
    """
    cv = cv or make_time_series_cv()
    base = LogisticRegression(**DEFAULT_PARAMS)
    grid = GridSearchCV(base, GRID, cv=cv, scoring=SCORING, n_jobs=-1, refit=False)
    grid.fit(X_train, y_train)

    results = [
        {
            "params":      grid.cv_results_["params"][i],
            "mean_cv_auc": float(grid.cv_results_["mean_test_score"][i]),
            "std_cv_auc":  float(grid.cv_results_["std_test_score"][i]),
        }
        for i in range(len(grid.cv_results_["params"]))
    ]
    print_grid_results(MODEL_NAME, results, top_k=5)

    return {**DEFAULT_PARAMS, **grid.best_params_}


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val:   np.ndarray,
    y_val:   np.ndarray,
    feature_cols: list,
    params: Optional[Dict[str, Any]] = None,
) -> dict:
    """
    Entrena LogReg sobre train, predice sobre val, calcula métricas y plots.

    Si `params` is None usa DEFAULT_PARAMS (comportamiento Fase 3).
    En Fase 5 se pasa el dict devuelto por `tune_hyperparameters`.

    Nota: el parámetro `penalty` se omite — su default es L2, y pasar
    `penalty='l2'` explícitamente lanza FutureWarning en sklearn ≥1.8.
    """
    print(f"\n[{MODEL_NAME}] Entrenando con params: {params or DEFAULT_PARAMS}...")
    params = params or DEFAULT_PARAMS

    model = LogisticRegression(**params)
    model.fit(X_train, y_train)

    y_pred_train  = model.predict(X_train)
    y_pred_val    = model.predict(X_val)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val   = model.predict_proba(X_val)[:, 1]

    metrics_train = compute_metrics(y_train, y_pred_train, y_proba_train)
    metrics_val   = compute_metrics(y_val,   y_pred_val,   y_proba_val)

    print_full_metrics(metrics_train, "Train", MODEL_NAME)
    print_full_metrics(metrics_val,   "Val",   MODEL_NAME)

    plot_confusion_matrix(y_val, y_pred_val, MODEL_NAME)
    plot_roc_curve(y_val, y_proba_val, MODEL_NAME)

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
    """Imprime las features con mayor valor absoluto del coeficiente."""
    coefs = model.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]
    print(f"[{MODEL_NAME}] Top {top_k} features por |coef|:")
    for i in order[:top_k]:
        sign = "+" if coefs[i] > 0 else "-"
        print(f"    {sign} {feature_cols[i]:<25} coef = {coefs[i]:+.4f}")
