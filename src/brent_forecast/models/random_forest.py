"""
Random Forest: ensemble de árboles de decisión con bagging.

A diferencia de SVM/LogReg, no requiere features escaladas (los árboles
trabajan con thresholds univariantes). Aun así se usa el mismo X_train_sc
escalado que los demás modelos para mantener la comparabilidad estricta.

Fase 5: añadida `tune_hyperparameters` con TimeSeriesSplit + GridSearchCV
sobre n_estimators, max_depth y min_samples_leaf.
"""

from pathlib import Path
from typing import Any, Dict

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from brent_forecast.evaluation.metrics import (
    compute_metrics,
    print_full_metrics,
    plot_confusion_matrix,
    plot_roc_curve,
)
from brent_forecast.config import ModelSettings
from brent_forecast.models.tuning import print_grid_results

MODEL_NAME = "Random Forest"


def tune_hyperparameters(
    X_train: np.ndarray,
    y_train: np.ndarray,
    cv,
    config: ModelSettings,
    scoring: str,
    seed: int,
) -> Dict[str, Any]:
    """Grid search con TimeSeriesSplit. 18 combos × 5 folds = 90 fits."""
    default_params = {**config.params, "random_state": seed}
    # n_jobs=1 en el GridSearchCV externo para no anidar paralelismo con n_jobs=-1
    # del RF interno (joblib gestiona, pero explicitar evita oversubscription).
    base = RandomForestClassifier(**default_params)
    grid = GridSearchCV(base, config.grid, cv=cv, scoring=scoring, n_jobs=1, refit=False)
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
    Entrena Random Forest.
    """
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

    plot_confusion_matrix(y_val, y_pred_val, MODEL_NAME, plots_dir)
    plot_roc_curve(y_val, y_proba_val, MODEL_NAME, plots_dir)

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
