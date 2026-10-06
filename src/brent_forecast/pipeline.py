"""
End-to-end training pipeline, invoked from the CLI with ``brent train``.

Stages: load -> preprocess -> tune (TimeSeriesSplit) -> train the 4 models
-> learning curves -> single final evaluation on the test set.
Logging is configured by the caller (see ``brent_forecast.logging_config``).
"""

import json
import logging
import os
from pathlib import Path

import numpy as np

from brent_forecast._types import ModelResult
from brent_forecast.config import Settings
from brent_forecast.data.load import load_oil_data
from brent_forecast.evaluation.final import (
    evaluate_on_test,
    log_final_summary_table,
    plot_roc_test_comparison,
    plot_train_val_test_summary,
)
from brent_forecast.evaluation.learning_curves import (
    plot_learning_curve_mlp,
    plot_learning_curve_sklearn,
)
from brent_forecast.evaluation.metrics import log_summary_table, plot_roc_comparison
from brent_forecast.features.preprocessing import preprocess
from brent_forecast.models.logistic_regression import (
    train_and_evaluate as run_logreg,
)
from brent_forecast.models.logistic_regression import (
    tune_hyperparameters as tune_logreg,
)
from brent_forecast.models.neural_network import (
    train_and_evaluate as run_mlp,
)
from brent_forecast.models.neural_network import (
    tune_hyperparameters as tune_mlp,
)
from brent_forecast.models.random_forest import (
    train_and_evaluate as run_rf,
)
from brent_forecast.models.random_forest import (
    tune_hyperparameters as tune_rf,
)
from brent_forecast.models.svm import (
    train_and_evaluate as run_svm,
)
from brent_forecast.models.svm import (
    tune_hyperparameters as tune_svm,
)
from brent_forecast.models.tuning import make_time_series_cv

logger = logging.getLogger(__name__)


def run(settings: Settings) -> None:
    """Run the full pipeline with the given settings."""
    np.random.seed(settings.seed)
    paths = settings.paths
    os.makedirs(paths.plots_dir, exist_ok=True)
    logger.info("Brent direction classifier — training run (seed=%d)", settings.seed)

    # ─── 1. Load and merge datasets ──────────────────────────────
    logger.info("[1/6] Loading and merging datasets")
    df = load_oil_data(
        paths.data_dir / settings.data.oil_filename,
        paths.data_dir / settings.data.events_filename,
    )

    # ─── 2. Preprocessing ────────────────────────────────────────
    logger.info("[2/6] Preprocessing")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols = preprocess(
        df, settings.split, settings.preprocessing
    )
    logger.info(
        "Scaler fitted on train: %s (n_features=%d); X_test held out until the final step",
        type(scaler).__name__,
        scaler.n_features_in_,
    )

    # ─── 3. Hyperparameter tuning with TimeSeriesSplit ───────────
    logger.info(
        "[3/6] Hyperparameter tuning with TimeSeriesSplit (n_splits=%d)", settings.cv.n_splits
    )
    cv = make_time_series_cv(settings.cv.n_splits)
    models = settings.models
    scoring, seed = settings.cv.scoring, settings.seed
    best_params = {
        "Logistic Regression": tune_logreg(
            X_train, y_train, cv, models.logistic_regression, scoring=scoring, seed=seed
        ),
        "SVM (RBF)": tune_svm(X_train, y_train, cv, models.svm, scoring=scoring, seed=seed),
        "Random Forest": tune_rf(
            X_train, y_train, cv, models.random_forest, scoring=scoring, seed=seed
        ),
        "MLP NumPy": tune_mlp(X_train, y_train, cv, models.mlp, scoring=scoring, seed=seed),
    }
    for name, params in best_params.items():
        logger.info("Best hyperparameters | %s: %s", name, params)

    # ─── 4. Train the 4 models with the best hyperparameters ─────
    logger.info("[4/6] Training the 4 models with the best hyperparameters")
    runners = [
        ("Logistic Regression", run_logreg),
        ("SVM (RBF)", run_svm),
        ("Random Forest", run_rf),
        ("MLP NumPy", run_mlp),
    ]
    results = []
    for name, runner in runners:
        results.append(
            runner(
                X_train,
                y_train,
                X_val,
                y_val,
                feature_cols,
                params=best_params[name],
                plots_dir=paths.plots_dir,
            )
        )

    log_summary_table(results)
    roc_path = plot_roc_comparison(results, y_val, paths.plots_dir)
    logger.info("Validation ROC comparison saved to %s", roc_path)

    # ─── 5. Learning curves ──────────────────────────────────────
    logger.info("[5/6] Learning curves (TimeSeriesSplit)")
    sklearn_names = ["Logistic Regression", "SVM (RBF)", "Random Forest"]
    train_sizes = settings.evaluation.learning_curve_train_sizes
    for r, name in zip(results[:3], sklearn_names, strict=True):
        # sklearn.learning_curve clones the fitted estimator and refits it per fold.
        plot_learning_curve_sklearn(
            r["model"],
            X_train,
            y_train,
            name,
            cv=cv,
            scoring=settings.cv.scoring,
            train_sizes=train_sizes,
            plots_dir=paths.plots_dir,
        )
    plot_learning_curve_mlp(
        X_train,
        y_train,
        best_params["MLP NumPy"],
        cv=cv,
        train_sizes=train_sizes,
        plots_dir=paths.plots_dir,
    )

    # ─── 6. Final evaluation on TEST (single pass) ───────────────
    logger.info("[6/6] Final evaluation on X_test (single pass)")
    final_results = evaluate_on_test(results, X_test, y_test, paths.plots_dir)

    plot_roc_test_comparison(final_results, y_test, paths.plots_dir)
    plot_train_val_test_summary(final_results, paths.plots_dir)
    log_final_summary_table(final_results)
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
