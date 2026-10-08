"""Model explanations (``brent explain``): permutation importance and SHAP.

Every explained model is **refitted on the development period** with the
hyperparameters of its saved final pipeline and explained on the
**out-of-sample period**, so the importances describe what generalises, not
what the model memorised.

- **Permutation importance** (Breiman, 2001) is model-agnostic and works on
  the raw feature frame: each column is shuffled ``permutation_repeats``
  times and the drop in ROC AUC is recorded. A feature removed by the VIF
  filter has exactly zero importance.
- **SHAP** (Lundberg & Lee, 2017) explains the inputs of the final estimator
  (after VIF selection, winsorisation and scaling): exact TreeSHAP for tree
  ensembles, the linear closed form for logistic regression (both in
  log-odds) and the model-agnostic permutation explainer on ``P(up)``
  otherwise. Global view: mean ``|SHAP|`` and a beeswarm plot. Local view: a
  waterfall plot of the most recent out-of-sample day.
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
from lightgbm import LGBMClassifier
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from brent_forecast.config import Settings
from brent_forecast.evaluation.metrics import safe_name, save_figure
from brent_forecast.models.registry import MODEL_NAMES

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ShapResult:
    """SHAP values of one model on the explained rows."""

    explanation: Any
    """``shap.Explanation`` with values, base values and the model inputs."""
    output: str
    """Unit of the values: ``"log-odds"`` or ``"probability"``."""


def permutation_importances(
    pipeline: Pipeline, X: pd.DataFrame, y: pd.Series, *, n_repeats: int, seed: int
) -> pd.DataFrame:
    """Drop in ROC AUC when each raw feature is shuffled, sorted by decreasing mean."""
    result = permutation_importance(
        pipeline, X, y, scoring="roc_auc", n_repeats=n_repeats, random_state=seed, n_jobs=1
    )
    table = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return table.sort_values("importance_mean", ascending=False, ignore_index=True)


def shap_values(
    pipeline: Pipeline, X_background: pd.DataFrame, X_explain: pd.DataFrame, *, seed: int
) -> ShapResult:
    """SHAP values of the final estimator on its (preprocessed) inputs."""
    preprocess, model = pipeline[:-1], pipeline[-1]
    background = preprocess.transform(X_background)
    inputs = preprocess.transform(X_explain)
    if isinstance(model, LGBMClassifier | RandomForestClassifier):
        explanation = shap.TreeExplainer(model, model_output="raw")(inputs, check_additivity=True)
        if explanation.values.ndim == 3:  # scikit-learn forests: one output per class
            explanation = explanation[:, :, 1]
        output = "log-odds" if isinstance(model, LGBMClassifier) else "probability"
        return ShapResult(explanation, output)
    if isinstance(model, LogisticRegression):
        explainer = shap.LinearExplainer(model, shap.maskers.Independent(background))
        return ShapResult(explainer(inputs), "log-odds")

    def predict_up(values: np.ndarray) -> np.ndarray:
        frame = pd.DataFrame(values, columns=inputs.columns)
        return np.asarray(model.predict_proba(frame)[:, 1])

    explainer = shap.PermutationExplainer(
        predict_up, shap.maskers.Independent(background), seed=seed
    )
    return ShapResult(explainer(inputs), "probability")


def shap_importances(result: ShapResult) -> pd.DataFrame:
    """Mean absolute SHAP value per model input, sorted by decreasing importance."""
    values = np.abs(np.asarray(result.explanation.values))
    table = pd.DataFrame(
        {"feature": result.explanation.feature_names, "mean_abs_shap": values.mean(axis=0)}
    )
    return table.sort_values("mean_abs_shap", ascending=False, ignore_index=True)


def plot_permutation(table: pd.DataFrame, name: str, plots_dir: Path, top: int = 15) -> Path:
    """Horizontal bar chart of the ``top`` permutation importances with ±1 std."""
    rows = table.head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7, 0.35 * len(rows) + 1.5))
    ax.barh(rows["feature"], rows["importance_mean"], xerr=rows["importance_std"], color="tab:blue")
    ax.axvline(0, color="gray", linewidth=1)
    ax.set_xlabel("Drop in out-of-sample ROC AUC when shuffled")
    ax.set_title(f"Permutation importance — {name}")
    fig.tight_layout()
    return save_figure(fig, plots_dir, f"permutation_importance_{safe_name(name)}.png")


def plot_shap(result: ShapResult, name: str, plots_dir: Path) -> tuple[Path, Path]:
    """Global beeswarm plot and local waterfall plot of the last explained day."""
    shap.plots.beeswarm(result.explanation, max_display=15, show=False)
    fig = plt.gcf()
    fig.suptitle(f"SHAP values ({result.output}) — {name}")
    fig.tight_layout()
    beeswarm = save_figure(fig, plots_dir, f"shap_beeswarm_{safe_name(name)}.png")

    shap.plots.waterfall(result.explanation[-1], max_display=12, show=False)
    fig = plt.gcf()
    fig.suptitle(f"Most recent day ({result.output}) — {name}")
    fig.tight_layout()
    waterfall = save_figure(fig, plots_dir, f"shap_waterfall_{safe_name(name)}.png")
    return beeswarm, waterfall


def generate_explanations(settings: Settings) -> Path:
    """Explain every model of ``settings.explain.models``; return the summary path.

    Raises
    ------
    FileNotFoundError
        If a model has not been trained (``brent train``) yet.
    """
    from brent_forecast.pipeline import prepare_data

    paths, cfg = settings.paths, settings.explain
    unknown = sorted(set(cfg.models) - set(MODEL_NAMES))
    if unknown:
        raise KeyError(f"Unknown models in explain.models: {unknown}")
    for key in cfg.models:
        if not (paths.models_dir / f"{key}.joblib").is_file():
            raise FileNotFoundError(
                f"{paths.models_dir / key}.joblib not found; run `brent train`."
            )

    dataset, parts = prepare_data(settings)
    cols = dataset.feature_cols
    X_dev, y_dev = parts.dev[cols], parts.dev["label"]
    X_oos, y_oos = parts.test[cols], parts.test["label"]
    background = X_dev.sample(min(cfg.background_rows, len(X_dev)), random_state=settings.seed)
    explained = X_oos.iloc[-cfg.max_explained_rows :]
    paths.explain_dir.mkdir(parents=True, exist_ok=True)

    sections: list[str] = []
    for key in cfg.models:
        name = MODEL_NAMES[key]
        logger.info("Explaining %s (refitted on the development period)", name)
        pipeline = clone(joblib.load(paths.models_dir / f"{key}.joblib")).fit(X_dev, y_dev)

        perm = permutation_importances(
            pipeline, X_oos, y_oos, n_repeats=cfg.permutation_repeats, seed=settings.seed
        )
        perm.to_csv(paths.explain_dir / f"permutation_{key}.csv", index=False)
        plot_permutation(perm, name, paths.plots_dir)

        result = shap_values(pipeline, background, explained, seed=settings.seed)
        importance = shap_importances(result)
        importance.to_csv(paths.explain_dir / f"shap_{key}.csv", index=False)
        plot_shap(result, name, paths.plots_dir)

        sections.append(_section(name, perm, importance, result, parts.test["date"].iloc[-1]))

    summary = paths.explain_dir / "explanations.md"
    summary.write_text(_render(sections), encoding="utf-8")
    return summary


def _section(
    name: str, perm: pd.DataFrame, importance: pd.DataFrame, result: ShapResult, last_day: Any
) -> str:
    significant = perm[perm["importance_mean"] > 2 * perm["importance_std"]]
    top = significant["feature"].head(5).tolist()
    lines = [
        f"## {name}",
        "",
        "| Rank | Permutation (ΔAUC ± std) | SHAP (mean abs, " + result.output + ") |",
        "|---|---|---|",
    ]
    for i in range(min(5, len(perm), len(importance))):
        p, s = perm.iloc[i], importance.iloc[i]
        lines.append(
            f"| {i + 1} | {p['feature']} ({p['importance_mean']:+.4f} ± "
            f"{p['importance_std']:.4f}) | {s['feature']} ({s['mean_abs_shap']:.4f}) |"
        )
    lines += [
        "",
        f"Features whose AUC drop exceeds twice its std: {', '.join(top) if top else 'none'}.",
        f"Local explanation of the most recent day ({pd.Timestamp(last_day).date()}): "
        f"`plots/shap_waterfall_{safe_name(name)}.png`.",
        "",
    ]
    return "\n".join(lines)


def _render(sections: list[str]) -> str:
    header = [
        "# Model explanations",
        "",
        "Each model is refitted on the development period and explained on the out-of-sample "
        "period. Permutation importance is the drop in ROC AUC when a raw feature is shuffled "
        "(model-agnostic); SHAP explains the inputs of the final estimator after preprocessing.",
        "Importances close to zero, or within their noise, mean the model has no stable signal "
        "to attribute. The 'twice its std' screen is applied to every feature, so about one "
        "feature per model passes it by chance alone (multiple comparisons).",
        "",
    ]
    return "\n".join(header + sections)
