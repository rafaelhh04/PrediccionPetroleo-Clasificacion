"""Hyperparameter tuning helpers built on TimeSeriesSplit.

Project conventions:

- CV is always ``TimeSeriesSplit`` (never KFold/StratifiedKFold), so that the
  temporal order is respected.
- The search only sees ``X_train``; ``X_val`` remains an independent holdout.
"""

import logging
from collections.abc import Mapping, Sequence
from typing import Any, TypedDict

from sklearn.model_selection import GridSearchCV, TimeSeriesSplit

from brent_forecast._types import FloatArray

logger = logging.getLogger(__name__)


class GridResult(TypedDict):
    """Cross-validation result of one hyperparameter combination."""

    params: dict[str, Any]
    mean_cv_auc: float
    std_cv_auc: float


def make_time_series_cv(n_splits: int) -> TimeSeriesSplit:
    """Return the project's expanding-window ``TimeSeriesSplit``.

    Each fold uses the whole history available before its validation block,
    as a production model would.
    """
    return TimeSeriesSplit(n_splits=n_splits)


def grid_search(
    estimator: Any,
    grid: Mapping[str, Sequence[Any]],
    X_train: FloatArray,
    y_train: FloatArray,
    *,
    cv: TimeSeriesSplit,
    scoring: str,
    n_jobs: int,
    model_name: str,
) -> dict[str, Any]:
    """Run ``GridSearchCV`` (no refit), log the top configurations and return the best one.

    Parameters
    ----------
    estimator
        Unfitted sklearn estimator with the fixed hyperparameters.
    grid
        Search space.
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
    dict
        Best hyperparameters found (only the searched keys).
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

    best: dict[str, Any] = search.best_params_
    return best


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
