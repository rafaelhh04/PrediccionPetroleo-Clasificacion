"""
Random Forest: ensemble de árboles de decisión con bagging.

A diferencia de SVM/LogReg, no requiere features escaladas (los árboles
trabajan con thresholds univariantes). Aun así se usa el mismo X_train_sc
escalado que los demás modelos para mantener la comparabilidad estricta.

Fase 5: añadida `tune_hyperparameters` con TimeSeriesSplit + GridSearchCV
sobre n_estimators, max_depth y min_samples_leaf.
"""

from typing import Any, Dict, Optional

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from brent_forecast.evaluation.metrics import (
    compute_metrics,
    print_full_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from brent_forecast.models.tuning import make_time_series_cv, print_grid_results, SCORING

MODEL_NAME = "Random Forest"


DEFAULT_PARAMS: Dict[str, Any] = {
    "n_estimators":     200,
    "max_depth":        10,
    "min_samples_leaf": 20,
    "n_jobs":           -1,
    "random_state":     42,
}

GRID: Dict[str, list] = {
    "n_estimators":     [100, 200],
    "max_depth":        [5, 10, 20],
    "min_samples_leaf": [10, 20, 50],
}


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv=None,
) -> Dict[str, Any]:
    """Grid search con TimeSeriesSplit. 18 combos × 5 folds = 90 fits."""
    cv = cv or make_time_series_cv()
    # n_jobs=1 en el GridSearchCV externo para no anidar paralelismo con n_jobs=-1
    # del RF interno (joblib gestiona, pero explicitar evita oversubscription).
    base = RandomForestClassifier(**DEFAULT_PARAMS)
    grid = GridSearchCV(base, GRID, cv=cv, scoring=SCORING, n_jobs=1, refit=False)
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
    Entrena Random Forest.

    Si `params` is None usa DEFAULT_PARAMS (comportamiento Fase 3).
    """
    params = params or DEFAULT_PARAMS
    print(f"\n[{MODEL_NAME}] Entrenando con params: {params}...")

    model = RandomForestClassifier(**params)
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

    _print_top_features(model, feature_cols, top_k=5)

    return {
        "model_name":    MODEL_NAME,
        "model":         model,
        "metrics_train": metrics_train,
        "metrics_val":   metrics_val,
        "y_pred_val":    y_pred_val,
        "y_proba_val":   y_proba_val,
    }


def _print_top_features(model: RandomForestClassifier, feature_cols: list, top_k: int = 5) -> None:
    """Imprime las features con mayor importancia (mean decrease in impurity)."""
    importances = model.feature_importances_
    order = np.argsort(importances)[::-1]
    print(f"[{MODEL_NAME}] Top {top_k} features por importancia:")
    for i in order[:top_k]:
        print(f"    {feature_cols[i]:<25} importance = {importances[i]:.4f}")
