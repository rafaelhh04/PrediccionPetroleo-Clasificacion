"""Hyperparameter search helpers.

Project conventions:

- CV is always a purged walk-forward splitter (never KFold/StratifiedKFold), so
  that the temporal order is respected and labels never straddle a fold edge.
- The search only sees the development period; the out-of-sample period is
  evaluated afterwards by walk-forward.
- Small, discrete spaces use an exhaustive grid search; larger continuous ones
  (gradient boosting) use Optuna's TPE sampler with median pruning.
"""

import logging
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
import optuna
from sklearn.base import clone
from sklearn.metrics import check_scoring
from sklearn.model_selection import GridSearchCV

from brent_forecast.config import SearchSpace, TuningSettings

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


def _suggest(trial: optuna.Trial, name: str, spec: SearchSpace) -> Any:
    """Sample one hyperparameter from its configured distribution."""
    if spec.type == "categorical":
        assert spec.choices is not None  # guaranteed by SearchSpace validation
        return trial.suggest_categorical(name, spec.choices)
    assert spec.low is not None and spec.high is not None
    if spec.type == "int":
        return trial.suggest_int(name, int(spec.low), int(spec.high), log=spec.log)
    return trial.suggest_float(name, spec.low, spec.high, log=spec.log)


def optuna_search(
    estimator: Any,
    space: Mapping[str, SearchSpace],
    X_train: Any,
    y_train: Any,
    *,
    cv: Any,
    scoring: str,
    tuning: TuningSettings,
    seed: int,
    model_name: str,
    trials_file: Path | None = None,
) -> GridResult:
    """Bayesian search (TPE) over ``space`` on the ``model`` step of ``estimator``.

    Every trial is scored on the folds of ``cv`` in temporal order. With
    ``tuning.pruning`` the running mean score is reported after each fold and
    a :class:`~optuna.pruners.MedianPruner` stops trials that fall below the
    median of earlier trials at the same fold.

    Parameters
    ----------
    estimator
        Unfitted pipeline with the fixed hyperparameters.
    space
        Hyperparameter name (without the ``model__`` prefix) -> distribution.
    X_train, y_train
        Development data (pandas).
    cv
        Time-series cross-validator.
    scoring
        sklearn scoring name.
    tuning
        Number of trials, timeout and pruning options.
    seed
        Seed of the TPE sampler (the study is reproducible without a timeout).
    model_name
        Name used in the logs.
    trials_file
        If given, every trial (parameters, score, state) is written there as CSV.

    Returns
    -------
    GridResult
        Best trial, with ``model__``-prefixed parameters like :func:`grid_search`.
    """
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    scorer = check_scoring(estimator, scoring=scoring)
    folds = list(cv.split(X_train, y_train))

    def objective(trial: optuna.Trial) -> float:
        params = {f"model__{n}": _suggest(trial, n, spec) for n, spec in space.items()}
        scores: list[float] = []
        for step, (train, test) in enumerate(folds):
            model = clone(estimator).set_params(**params)
            model.fit(X_train.iloc[train], y_train.iloc[train])
            scores.append(float(scorer(model, X_train.iloc[test], y_train.iloc[test])))
            if tuning.pruning:
                trial.report(float(np.mean(scores)), step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
        trial.set_user_attr("std_cv_auc", float(np.std(scores)))
        return float(np.mean(scores))

    pruner: optuna.pruners.BasePruner = (
        optuna.pruners.MedianPruner(n_startup_trials=tuning.startup_trials, n_warmup_steps=1)
        if tuning.pruning
        else optuna.pruners.NopPruner()
    )
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=seed, n_startup_trials=tuning.startup_trials),
        pruner=pruner,
        study_name=model_name,
    )
    study.optimize(objective, n_trials=tuning.n_trials, timeout=tuning.timeout)

    complete = study.get_trials(states=[optuna.trial.TrialState.COMPLETE])
    pruned = len(study.get_trials(states=[optuna.trial.TrialState.PRUNED]))
    logger.info(
        "[%s] Optuna: %d trials (%d complete, %d pruned)",
        model_name,
        len(study.trials),
        len(complete),
        pruned,
    )
    if trials_file is not None:
        trials_file.parent.mkdir(parents=True, exist_ok=True)
        study.trials_dataframe().to_csv(trials_file, index=False)

    results: list[GridResult] = [
        {
            "params": {f"model__{k}": v for k, v in t.params.items()},
            "mean_cv_auc": float(t.value) if t.value is not None else float("nan"),
            "std_cv_auc": float(t.user_attrs["std_cv_auc"]),
        }
        for t in complete
    ]
    log_grid_results(model_name, results, top_k=5)
    best = study.best_trial
    return {
        "params": {f"model__{k}": v for k, v in best.params.items()},
        "mean_cv_auc": float(best.value) if best.value is not None else float("nan"),
        "std_cv_auc": float(best.user_attrs["std_cv_auc"]),
    }
