"""
Proyecto de Aprendizaje Automático — Predicción del Precio del Brent
Curso 2025-26

Orquestación de la pipeline completa. Se invoca desde la CLI:

    brent train

Toda la salida por consola se duplica en results/run_log.txt (Tee).
"""

import json
import os
import sys
from pathlib import Path
import numpy as np
from datetime import datetime

from brent_forecast.config import Settings
from brent_forecast.data.load import load_oil_data
from brent_forecast.features.preprocessing import preprocess
from brent_forecast.evaluation.metrics import print_summary_table, plot_roc_comparison
from brent_forecast.models.tuning import make_time_series_cv
from brent_forecast.evaluation.learning_curves import plot_learning_curve_sklearn, plot_learning_curve_mlp
from brent_forecast.evaluation.final import (
    evaluate_on_test,
    plot_roc_test_comparison,
    plot_train_val_test_summary,
    print_final_summary_table,
)

from brent_forecast.models.logistic_regression import (
    train_and_evaluate   as run_logreg,
    tune_hyperparameters as tune_logreg,
)
from brent_forecast.models.svm import (
    train_and_evaluate   as run_svm,
    tune_hyperparameters as tune_svm,
)
from brent_forecast.models.random_forest import (
    train_and_evaluate   as run_rf,
    tune_hyperparameters as tune_rf,
)
from brent_forecast.models.neural_network import (
    train_and_evaluate   as run_mlp,
    tune_hyperparameters as tune_mlp,
)


class Tee:
    """
    Duplica las escrituras a varios streams a la vez (típicamente stdout + fichero).

    Se usa para que la salida por consola quede registrada también en un .txt
    sin perder la visualización en vivo. Llama a flush() en cada write para que
    el archivo se vea actualizado en tiempo real si el usuario lo abre durante
    la ejecución.
    """

    def __init__(self, *streams):
        self.streams = streams

    def write(self, data):
        for s in self.streams:
            s.write(data)
            s.flush()

    def flush(self):
        for s in self.streams:
            s.flush()


def _run_pipeline(settings: Settings) -> None:
    """Pipeline real (sin la lógica de logging)."""
    print("=" * 60)
    print("PROYECTO ML — PREDICCIÓN DIRECCIÓN PRECIO BRENT")
    print("Curso 2025-26")
    print(f"Ejecución: {datetime.now().isoformat(timespec='seconds')}")
    print("=" * 60)

    paths = settings.paths
    os.makedirs(paths.plots_dir, exist_ok=True)

    # ─── FASE 1 — Carga y fusión de datasets ─────────────────────
    print("\n[1/3] Cargando y fusionando datasets...")
    df = load_oil_data(
        paths.data_dir / settings.data.oil_filename,
        paths.data_dir / settings.data.events_filename,
    )

    # ─── FASE 2 — Preprocesado completo ──────────────────────────
    print("\n[2/3] Preprocesando datos...")
    X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_cols = preprocess(
        df, settings.split, settings.preprocessing
    )
    print(f"[main] Scaler ajustado en train: {type(scaler).__name__} "
          f"(n_features={scaler.n_features_in_}). X_test reservado para Fase 6.")

    # ─── FASE 5 — Hyperparameter tuning con TimeSeriesSplit ──────
    print(f"\n[3/4] Hyperparameter tuning con TimeSeriesSplit (n_splits={settings.cv.n_splits})...")
    cv = make_time_series_cv(settings.cv.n_splits)
    models = settings.models
    tune_kwargs = {"scoring": settings.cv.scoring, "seed": settings.seed}
    best_params = {
        "Logistic Regression": tune_logreg(X_train, y_train, cv, models.logistic_regression, **tune_kwargs),
        "SVM (RBF)":           tune_svm(X_train, y_train, cv, models.svm, **tune_kwargs),
        "Random Forest":       tune_rf(X_train, y_train, cv, models.random_forest, **tune_kwargs),
        "MLP NumPy":           tune_mlp(X_train, y_train, cv, models.mlp, **tune_kwargs),
    }
    print("\n[main] Mejores hyperparámetros por modelo:")
    for name, params in best_params.items():
        print(f"  {name}: {params}")

    # ─── FASES 3 + 4 — Modelos entrenados con los mejores params ─
    print("\n[4/6] Entrenando los 4 modelos con los mejores hyperparámetros...")
    runners = [
        ("Logistic Regression", run_logreg),
        ("SVM (RBF)",           run_svm),
        ("Random Forest",       run_rf),
        ("MLP NumPy",           run_mlp),
    ]
    results = []
    for name, runner in runners:
        results.append(runner(X_train, y_train, X_val, y_val, feature_cols,
                              params=best_params[name], plots_dir=paths.plots_dir))

    print_summary_table(results)
    roc_path = plot_roc_comparison(results, y_val, paths.plots_dir)
    print(f"[main] ROC comparativa (val) guardada en: {roc_path}")

    # ─── FASE 6.1 — Curvas de aprendizaje ────────────────────────
    print("\n[5/6] Generando curvas de aprendizaje (TimeSeriesSplit)...")
    sklearn_names = ["Logistic Regression", "SVM (RBF)", "Random Forest"]
    train_sizes = settings.evaluation.learning_curve_train_sizes
    for r, name in zip(results[:3], sklearn_names):
        # Re-instanciar el estimator con los best_params (refit limpio para learning_curve)
        # Usamos el modelo ya entrenado: sklearn.learning_curve hace su propio refit por fold.
        plot_learning_curve_sklearn(
            r["model"], X_train, y_train, name,
            cv=cv, scoring=settings.cv.scoring, train_sizes=train_sizes,
            plots_dir=paths.plots_dir,
        )
    plot_learning_curve_mlp(
        X_train, y_train, best_params["MLP NumPy"],
        cv=cv, train_sizes=train_sizes, plots_dir=paths.plots_dir,
    )

    # ─── FASE 6.2 — Evaluación final sobre TEST (una sola pasada) ─
    print("\n[6/6] Evaluación final sobre X_test (UNA SOLA PASADA)...")
    final_results = evaluate_on_test(results, X_test, y_test, paths.plots_dir)

    plot_roc_test_comparison(final_results, y_test, paths.plots_dir)
    plot_train_val_test_summary(final_results, paths.plots_dir)
    print_final_summary_table(final_results)
    _save_metrics(final_results, paths.metrics_file)
    print(f"[main] Métricas finales guardadas en: {paths.metrics_file}")

    best_test = max(final_results, key=lambda r: r["metrics_test"]["auc_roc"])
    print(f"\n Fase 6 completada. Mejor modelo por AUC(test): "
          f"{best_test['model_name']} (AUC = {best_test['metrics_test']['auc_roc']:.4f})")


def _save_metrics(final_results: list, path: Path) -> None:
    """Guarda las métricas train/val/test de cada modelo en JSON (legible por máquina)."""
    payload = {
        r["model_name"]: {
            "train": r["metrics_train"],
            "val":   r["metrics_val"],
            "test":  r["metrics_test"],
        }
        for r in final_results
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def run(settings: Settings) -> None:
    """
    Envoltorio: prepara el Tee (stdout + results/run_log.txt) y ejecuta la pipeline.
    Garantiza que stdout vuelva a su estado original incluso si la pipeline falla.
    """
    np.random.seed(settings.seed)
    log_path = settings.paths.log_file
    os.makedirs(log_path.parent, exist_ok=True)

    # Asegurar que stdout puede emitir UTF-8 (la consola Windows por defecto
    # usa cp1252 y rompería en caracteres como →, ✅, ─). 'replace' garantiza
    # que cualquier glifo no representable se sustituya en vez de lanzar excepción.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    log_file = open(log_path, "w", encoding="utf-8")
    original_stdout = sys.stdout
    sys.stdout = Tee(original_stdout, log_file)

    try:
        _run_pipeline(settings)
        print(f"\n[main] Log completo guardado en: {log_path}")
    finally:
        sys.stdout = original_stdout
        log_file.close()
        # Mensaje final solo por consola (ya no se duplica en el log)
        print(f"[main] (stdout restaurado; log cerrado: {log_path})")

