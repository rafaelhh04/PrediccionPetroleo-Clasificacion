"""
Support Vector Machine con kernel RBF.

Modelo no lineal por excelencia. La elección del kernel RBF asume que
las relaciones entre features y dirección del precio son no lineales y
suaves — hipótesis razonable en finanzas.

Fase 5: añadida `tune_hyperparameters` con TimeSeriesSplit + GridSearchCV.
Durante la búsqueda se usa probability=False para acelerar (~2× speedup);
el modelo final se entrena con probability=True para reportar AUC val.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV

from brent_forecast.evaluation.metrics import (
    compute_metrics,
    print_full_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from brent_forecast.models.tuning import make_time_series_cv, print_grid_results, SCORING

MODEL_NAME = "SVM (RBF)"


DEFAULT_PARAMS: Dict[str, Any] = {
    "kernel":       "rbf",
    "C":            1.0,
    "gamma":        "scale",
    "probability":  True,
    "random_state": 42,
}

GRID: Dict[str, list] = {
    "C":     [0.1, 1.0, 10.0],
    "gamma": ["scale", 0.01, 0.1],
}


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv=None,
) -> Dict[str, Any]:
    """
    Grid search con TimeSeriesSplit.
    Truco: durante CV usamos probability=False (sklearn calcula AUC con
    decision_function, no necesita predict_proba) → ~2× más rápido.
    El modelo final se entrenará con probability=True.
    """
    cv = cv or make_time_series_cv()
    cv_params = {**DEFAULT_PARAMS, "probability": False}
    base = SVC(**cv_params)
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
    Entrena SVM RBF.

    Si `params` is None usa DEFAULT_PARAMS (comportamiento Fase 3).
    Coste: SVM RBF escala O(n²). Con ~3.000 filas tarda segundos.
    """
    params = params or DEFAULT_PARAMS
    print(f"\n[{MODEL_NAME}] Entrenando con params: {params}...")

    model = SVC(**params)
    model.fit(X_train, y_train)

    y_pred_train  = model.predict(X_train)
    y_pred_val    = model.predict(X_val)
    y_proba_train = model.predict_proba(X_train)[:, 1]
    y_proba_val   = model.predict_proba(X_val)[:, 1]

    metrics_train = compute_metrics(y_train, y_pred_train, y_proba_train)
    metrics_val   = compute_metrics(y_val,   y_pred_val,   y_proba_val)

    print_full_metrics(metrics_train, "Train", MODEL_NAME)
    print_full_metrics(metrics_val,   "Val",   MODEL_NAME)
    print(f"[{MODEL_NAME}] Support vectors: {model.support_.shape[0]} de {X_train.shape[0]}")

    plot_confusion_matrix(y_val, y_pred_val, MODEL_NAME)
    plot_roc_curve(y_val, y_proba_val, MODEL_NAME)

    return {
        "model_name":    MODEL_NAME,
        "model":         model,
        "metrics_train": metrics_train,
        "metrics_val":   metrics_val,
        "y_pred_val":    y_pred_val,
        "y_proba_val":   y_proba_val,
    }
