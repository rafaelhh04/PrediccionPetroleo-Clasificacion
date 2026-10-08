"""
End-to-end training pipeline, invoked from the CLI with ``brent train``.

Stages: load -> build the dataset (stateless steps) -> chronological split ->
tune every model pipeline with TimeSeriesSplit -> fit on train -> validation
-> learning curves -> single final evaluation on the test set.
Logging is configured by the caller (see ``brent_forecast.logging_config``).
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np

from brent_forecast._types import ModelResult
from brent_forecast.config import Settings
from brent_forecast.data.load import DateBounds, load_oil_data
from brent_forecast.evaluation.final import (
    evaluate_on_test,
    log_final_summary_table,
    plot_roc_test_comparison,
    plot_train_val_test_summary,
)
from brent_forecast.evaluation.learning_curves import plot_learning_curve, plot_training_curves
from brent_forecast.evaluation.metrics import log_summary_table, plot_roc_comparison
from brent_forecast.features.preprocessing import build_dataset, split_temporal
from brent_forecast.models.base import fit_and_evaluate, log_top_features
from brent_forecast.models.registry import (
    MODEL_NAMES,
    build_pipeline,
    model_params,
    param_grid,
)
from brent_forecast.models.tuning import grid_search, make_time_series_cv

logger = logging.getLogger(__name__)

# The forest parallelises internally; nesting a parallel search would oversubscribe.
_SEARCH_N_JOBS = {"random_forest": 1}


def run(settings: Settings) -> None:
    """Run the full pipeline with the given settings."""
    np.random.seed(settings.seed)
    paths = settings.paths
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Brent direction classifier — training run (seed=%d)", settings.seed)

    # ─── 1. Load, build the dataset and split ────────────────────
    logger.info("[1/6] Loading data and building the dataset")
    data = settings.data
    df = load_oil_data(
        paths.data_dir / data.oil_filename,
        paths.data_dir / data.events_filename,
        DateBounds(data.expected_start_min, data.expected_start_max, data.expected_end_min),
    )
    dataset = build_dataset(df)
    parts = split_temporal(dataset.frame, settings.split)
    cols = dataset.feature_cols
    X_train, y_train = parts.train[cols], parts.train["label"]
    X_val, y_val = parts.val[cols], parts.val["label"]
    X_test, y_test = parts.test[cols], parts.test["label"]

    # ─── 2. Hyperparameter tuning with TimeSeriesSplit ───────────
    logger.info(
        "[2/6] Hyperparameter tuning with TimeSeriesSplit (n_splits=%d)", settings.cv.n_splits
    )
    cv = make_time_series_cv(settings.cv.n_splits)
    best_params = {}
    for key, name in MODEL_NAMES.items():
        config = getattr(settings.models, key)
        fixed = {**config.params, "random_state": settings.seed}
        best = grid_search(
            build_pipeline(key, fixed, settings.preprocessing, for_search=True),
            param_grid(key, config.grid),
            X_train,
            y_train,
            cv=cv,
            scoring=settings.cv.scoring,
            n_jobs=_SEARCH_N_JOBS.get(key, -1),
            model_name=name,
        )
        best_params[key] = {**fixed, **model_params(best)}
        logger.info("Best hyperparameters | %s: %s", name, best_params[key])

    # ─── 3. Fit on train, evaluate on validation ─────────────────
    logger.info("[3/6] Training the %d model pipelines on the training set", len(MODEL_NAMES))
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    results: list[ModelResult] = []
    for key, name in MODEL_NAMES.items():
        pipeline = build_pipeline(key, best_params[key], settings.preprocessing)
        logger.info("[%s] Training with params: %s", name, best_params[key])
        result = fit_and_evaluate(pipeline, name, X_train, y_train, X_val, y_val, paths.plots_dir)
        log_top_features(pipeline, name)
        artefact = paths.models_dir / f"{key}.joblib"
        joblib.dump(pipeline, artefact)
        logger.info("[%s] Pipeline saved to %s", name, artefact)
        results.append(result)

    log_summary_table(results)
    roc_path = plot_roc_comparison(results, y_val, paths.plots_dir)
    logger.info("Validation ROC comparison saved to %s", roc_path)
    mlp = results[list(MODEL_NAMES).index("mlp")]["model"].named_steps["model"]
    plot_training_curves(mlp.history_, paths.plots_dir)

    # ─── 4. Learning curves ──────────────────────────────────────
    logger.info("[4/6] Learning curves (TimeSeriesSplit)")
    for r in results:
        plot_learning_curve(
            r["model"],
            X_train,
            y_train,
            r["model_name"],
            cv=cv,
            scoring=settings.cv.scoring,
            train_sizes=settings.evaluation.learning_curve_train_sizes,
            plots_dir=paths.plots_dir,
        )

    # ─── 5. Final evaluation on TEST (single pass) ───────────────
    logger.info("[5/6] Final evaluation on the test set (single pass)")
    final_results = evaluate_on_test(results, X_test, y_test, paths.plots_dir)

    plot_roc_test_comparison(final_results, y_test, paths.plots_dir)
    plot_train_val_test_summary(final_results, paths.plots_dir)
    log_final_summary_table(final_results)

    # ─── 6. Persist metrics ──────────────────────────────────────
    logger.info("[6/6] Saving metrics")
    _save_metrics(final_results, paths.metrics_file)
    logger.info("Final metrics saved to %s", paths.metrics_file)

    best_test = max(final_results, key=lambda r: r["metrics_test"]["auc_roc"])
    logger.info(
        "Done. Best model by test AUC: %s (AUC = %.4f)",
        best_test["model_name"],
        best_test["metrics_test"]["auc_roc"],
    )


def _save_metrics(final_results: list[ModelResult], path: Path) -> None:
    """Write the train/val/test metrics of every model as JSON."""
    payload = {
        r["model_name"]: {
            "train": r["metrics_train"],
            "val": r["metrics_val"],
            "test": r["metrics_test"],
        }
        for r in final_results
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
