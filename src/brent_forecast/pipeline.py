"""
End-to-end training pipeline, invoked from the CLI with ``brent train``.

Protocol:

1. Load the data and build the dataset (stateless, past-only steps).
2. Development period (``date < split.test_start``): tune every model pipeline
   with purged walk-forward cross-validation.
3. Out-of-sample period (``date >= split.test_start``): walk-forward
   evaluation. Before each window of ``validation.test_window`` days the model
   is refitted on all older data minus the purge/embargo gap (expanding or
   rolling) and then predicts that window.
4. Persist the out-of-sample predictions and metrics, figures, and every
   pipeline refitted on all available data (the model one would deploy).

Logging is configured by the caller (see ``brent_forecast.logging_config``).
"""

import json
import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from brent_forecast.config import Settings
from brent_forecast.data.load import DateBounds, load_oil_data
from brent_forecast.evaluation.learning_curves import plot_learning_curve, plot_training_curves
from brent_forecast.evaluation.metrics import log_full_metrics, plot_confusion_matrix
from brent_forecast.evaluation.oos import (
    log_oos_summary,
    oos_metrics,
    plot_roc_oos,
    plot_window_auc,
    window_metrics,
)
from brent_forecast.features.preprocessing import build_dataset, split_by_date
from brent_forecast.models.base import log_top_features
from brent_forecast.models.registry import (
    MODEL_NAMES,
    build_pipeline,
    model_params,
    param_grid,
)
from brent_forecast.models.tuning import grid_search
from brent_forecast.validation.walk_forward import PurgedWalkForwardSplit, walk_forward_predict

logger = logging.getLogger(__name__)

# The forest parallelises internally; nesting a parallel search would oversubscribe.
_SEARCH_N_JOBS = {"random_forest": 1}


def run(settings: Settings) -> None:
    """Run the full pipeline with the given settings."""
    np.random.seed(settings.seed)
    paths, val = settings.paths, settings.validation
    paths.plots_dir.mkdir(parents=True, exist_ok=True)
    paths.models_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Brent direction classifier — training run (seed=%d)", settings.seed)

    # ─── 1. Load and build the dataset ───────────────────────────
    logger.info("[1/5] Loading data and building the dataset")
    data = settings.data
    df = load_oil_data(
        paths.data_dir / data.oil_filename,
        paths.data_dir / data.events_filename,
        DateBounds(data.expected_start_min, data.expected_start_max, data.expected_end_min),
    )
    dataset = build_dataset(df)
    parts = split_by_date(dataset.frame, settings.split)
    cols = dataset.feature_cols
    X_dev, y_dev = parts.dev[cols], parts.dev["label"]
    X_all, y_all = dataset.features, dataset.target

    # ─── 2. Tuning on the development period ─────────────────────
    tuning_cv = PurgedWalkForwardSplit(
        val.tuning_splits, purge=val.purge, embargo=val.embargo, max_train_size=val.max_train_size
    )
    logger.info("[2/5] Hyperparameter tuning on the development period with %r", tuning_cv)
    best_params: dict[str, dict[str, Any]] = {}
    cv_scores: dict[str, dict[str, float]] = {}
    for key, name in MODEL_NAMES.items():
        config = getattr(settings.models, key)
        fixed = {**config.params, "random_state": settings.seed}
        best = grid_search(
            build_pipeline(key, fixed, settings.preprocessing, for_search=True),
            param_grid(key, config.grid),
            X_dev,
            y_dev,
            cv=tuning_cv,
            scoring=val.scoring,
            n_jobs=_SEARCH_N_JOBS.get(key, -1),
            model_name=name,
        )
        best_params[key] = {**fixed, **model_params(best["params"])}
        cv_scores[name] = {"auc_mean": best["mean_cv_auc"], "auc_std": best["std_cv_auc"]}
        logger.info("Best hyperparameters | %s: %s", name, best_params[key])

    # ─── 3. Walk-forward evaluation on the out-of-sample period ──
    eval_cv = PurgedWalkForwardSplit(
        None,
        test_size=val.test_window,
        first_test_index=len(parts.dev),
        purge=val.purge,
        embargo=val.embargo,
        max_train_size=val.max_train_size,
    )
    n_windows = eval_cv.get_n_splits(X_all)
    logger.info("[3/5] Walk-forward evaluation: %d windows with %r", n_windows, eval_cv)
    oos = parts.test[["date", "label"]].copy()
    summary: dict[str, dict[str, Any]] = {}
    for key, name in MODEL_NAMES.items():
        preds = walk_forward_predict(
            build_pipeline(key, best_params[key], settings.preprocessing), X_all, y_all, eval_cv
        )
        oos["window"] = preds.window
        oos[f"proba_{key}"] = preds.proba
        metrics = oos_metrics(oos["label"], preds.proba)
        log_full_metrics(metrics, "Out-of-sample", name)
        plot_confusion_matrix(
            oos["label"], (preds.proba >= 0.5).astype(int), f"{name} (oos)", paths.plots_dir
        )
        summary[name] = {
            "cv": cv_scores[name],
            "oos": metrics,
            "windows": window_metrics(oos["date"], oos["label"], preds.proba, preds.window),
        }

    log_oos_summary(summary)
    probas = {name: oos[f"proba_{key}"] for key, name in MODEL_NAMES.items()}
    plot_roc_oos(oos["label"], probas, paths.plots_dir)
    plot_window_auc({name: m["windows"] for name, m in summary.items()}, paths.plots_dir)

    # ─── 4. Learning curves on the development period ────────────
    logger.info("[4/5] Learning curves on the development period")
    for key, name in MODEL_NAMES.items():
        plot_learning_curve(
            build_pipeline(key, best_params[key], settings.preprocessing),
            X_dev,
            y_dev,
            name,
            cv=tuning_cv,
            scoring=val.scoring,
            train_sizes=settings.evaluation.learning_curve_train_sizes,
            plots_dir=paths.plots_dir,
        )

    # ─── 5. Final models on all data and artefacts ───────────────
    logger.info("[5/5] Refitting every model on all data and saving artefacts")
    for key, name in MODEL_NAMES.items():
        pipeline = build_pipeline(key, best_params[key], settings.preprocessing).fit(X_all, y_all)
        log_top_features(pipeline, name)
        joblib.dump(pipeline, paths.models_dir / f"{key}.joblib")
        if key == "mlp":
            plot_training_curves(pipeline.named_steps["model"].history_, paths.plots_dir)

    oos.to_csv(paths.predictions_file, index=False, date_format="%Y-%m-%d")
    _save_metrics(summary, settings, n_windows, paths.metrics_file)
    logger.info(
        "Saved predictions to %s, metrics to %s and models to %s",
        paths.predictions_file,
        paths.metrics_file,
        paths.models_dir,
    )
    top = max(summary, key=lambda n: summary[n]["oos"]["auc_roc"])
    logger.info(
        "Done. Best model by out-of-sample AUC: %s (%.4f)", top, summary[top]["oos"]["auc_roc"]
    )


def _save_metrics(
    summary: dict[str, dict[str, Any]], settings: Settings, n_windows: int, path: Path
) -> None:
    """Write the evaluation protocol and every model's CV / out-of-sample metrics as JSON."""
    val = settings.validation
    payload = {
        "protocol": {
            "test_start": str(settings.split.test_start),
            "mode": val.mode,
            "test_window": val.test_window,
            "rolling_train_size": val.max_train_size,
            "purge": val.purge,
            "embargo": val.embargo,
            "tuning_splits": val.tuning_splits,
            "n_windows": n_windows,
        },
        "models": summary,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
