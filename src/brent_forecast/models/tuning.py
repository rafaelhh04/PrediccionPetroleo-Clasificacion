"""Hyperparameter search helpers.

Project conventions:

- CV is always a purged walk-forward splitter (never KFold/StratifiedKFold), so
  that the temporal order is respected and labels never straddle a fold edge.
- The search only sees the development period; the out-of-sample period is
  evaluated afterwards by walk-forward.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from sklearn.model_selection import GridSearchCV

logger = logging.getLogger(__name__)


class GridResult(TypedDict):
    """Cross-validation result of one hyperparameter combination."""

    params: dict[str, Any]
    mean_cv_auc: float
    std_cv_auc: float


def grid_search(
    estimator: Any,
    grid: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Sequence[Any]]],
    X_train: Any,
    y_train: Any,
    *,
    cv: Any,
    scoring: str,
    n_jobs: int,
    model_name: str,
) -> GridResult:
    """Run ``GridSearchCV`` (no refit), log the top configurations and return the best one.

    Parameters
    ----------
    estimator
        Unfitted sklearn estimator with the fixed hyperparameters.
    grid
        Search space (a dict, or a list of dicts as accepted by ``GridSearchCV``).
    X_train, y_train
        Training data.
    cv
        Time-series cross-validator.
    scoring
        sklearn scoring name.
    n_jobs
        Parallel jobs of the outer search.
    model_name
        Name used in the logs.

    Returns
    -------
    GridResult
        Best configuration (only the searched keys) with its mean and std CV score.
    """
    search = GridSearchCV(estimator, grid, cv=cv, scoring=scoring, n_jobs=n_jobs, refit=False)
    search.fit(X_train, y_train)

    cv_results = search.cv_results_
    results: list[GridResult] = [
        {
            "params": cv_results["params"][i],
            "mean_cv_auc": float(cv_results["mean_test_score"][i]),
            "std_cv_auc": float(cv_results["std_test_score"][i]),
        }
        for i in range(len(cv_results["params"]))
    ]
    log_grid_results(model_name, results, top_k=5)

    return max(results, key=lambda r: r["mean_cv_auc"])


def log_grid_results(
    model_name: str,
    results: Sequence[GridResult],
    top_k: int = 5,
) -> None:
    """Log the ``top_k`` configurations sorted by decreasing mean CV AUC."""
    sorted_res = sorted(results, key=lambda r: r["mean_cv_auc"], reverse=True)
    lines = [
        f"  {i}. AUC = {r['mean_cv_auc']:.4f} ± {r['std_cv_auc']:.4f}  | params = {r['params']}"
        for i, r in enumerate(sorted_res[:top_k], start=1)
    ]
    logger.info(
        "[%s] Top %d configurations by CV AUC:\n%s",
        model_name,
        min(top_k, len(sorted_res)),
        "\n".join(lines),
    )
    logger.info("[%s] Best params: %s", model_name, sorted_res[0]["params"])
