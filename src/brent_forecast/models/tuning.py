"""
Helpers para hyperparameter tuning con TimeSeriesSplit.

Centraliza la creación del CV y el reporte de resultados de la búsqueda
para garantizar consistencia entre los 4 modelos del proyecto.

Convenciones del proyecto:
- CV siempre TimeSeriesSplit (nunca KFold/StratifiedKFold) → respeta orden temporal.
- Métrica de scoring: roc_auc.
- Búsqueda solo sobre X_train (X_val queda como holdout independiente).
"""

from typing import List, Dict, Any
from sklearn.model_selection import TimeSeriesSplit


CV_N_SPLITS = 5
SCORING     = "roc_auc"


def make_time_series_cv(n_splits: int = CV_N_SPLITS) -> TimeSeriesSplit:
    """
    Devuelve un TimeSeriesSplit configurado del proyecto.

    Esquema "expanding window": cada fold acumula todo el histórico
    disponible hasta el inicio del val fold. Apropiado para series
    financieras donde el modelo de producción tendría acceso a todo
    el pasado.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def print_grid_results(
    model_name: str,
    results: List[Dict[str, Any]],
    top_k: int = 5,
) -> None:
    """
    Imprime las top-K combinaciones de hyperparámetros ordenadas por
    mean_cv_auc descendente.

    `results` es una lista de dicts con keys: 'params', 'mean_cv_auc',
    'std_cv_auc'.
    """
    sorted_res = sorted(results, key=lambda r: r["mean_cv_auc"], reverse=True)
    print(f"\n[{model_name}] Top {min(top_k, len(sorted_res))} configuraciones por AUC CV:")
    for i, r in enumerate(sorted_res[:top_k], start=1):
        print(f"  {i}. AUC = {r['mean_cv_auc']:.4f} ± {r['std_cv_auc']:.4f}  | params = {r['params']}")
    print(f"[{model_name}] Best params: {sorted_res[0]['params']}")
