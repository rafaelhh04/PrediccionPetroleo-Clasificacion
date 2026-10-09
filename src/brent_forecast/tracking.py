"""Experiment tracking and model registry with MLflow.

``brent train`` opens one parent run per training run, with one nested run per
candidate (models and baselines):

- parent: evaluation protocol and seed (params), git commit/branch and data
  SHA-256 (tags), ``metrics.json``, ``predictions.csv``, the resolved
  configuration and every plot (artefacts);
- child: hyperparameters, development-CV and out-of-sample metrics, AUC per
  walk-forward window (as steps) and the pipeline refitted on all data,
  logged as an MLflow scikit-learn model.

``brent report`` reopens the parent run, logs the statistics (intervals,
DeLong, backtest) and applies the **promotion rule** (:func:`select_champion`):

- the best model by out-of-sample AUC becomes ``champion`` only if it beats the
  best naive baseline significantly (Holm-adjusted DeLong *and* a block-
  bootstrap interval of the AUC difference above zero, see ``brent report``);
  the best baseline is then ``challenger``;
- otherwise the best **baseline** is the ``champion`` (deploying a model that
  cannot beat it would add risk without value) and the best model is the
  ``challenger``, kept under watch.

Both are registered as versions of ``tracking.registered_model``; the alias is
the stable handle (``models:/<name>@champion``) used for inference.
"""

import json
import logging
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import mlflow
from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from brent_forecast.config import Settings
from brent_forecast.pipeline import TrainingResult

logger = logging.getLogger(__name__)

CHAMPION = "champion"
CHALLENGER = "challenger"

# Models are stored with skops (no arbitrary pickle code on load). These are the
# non-standard types our candidates contain; a test fails if a new model needs more.
TRUSTED_TYPES: tuple[str, ...] = (
    "brent_forecast.features.transformers.VIFSelector",
    "brent_forecast.features.transformers.Winsorizer",
    "brent_forecast.models.mlp.NumpyMLPClassifier",
    "brent_forecast.models.baselines.PersistenceClassifier",
    "collections.OrderedDict",
    "lightgbm.basic.Booster",
    "lightgbm.sklearn.LGBMClassifier",
    "sklearn.calibration._CalibratedClassifier",
    "sklearn.calibration._SigmoidCalibration",
    "sklearn.tree._tree.Tree",
)

# Pinned runtime requirements of the logged models. Passing them explicitly
# skips MLflow's slow requirement inference (it imports and profiles the model).
_RUNTIME_PACKAGES = ("scikit-learn", "lightgbm", "numpy", "pandas", "joblib")


@dataclass(frozen=True)
class Decision:
    """Outcome of the promotion rule."""

    champion: str
    """Candidate key promoted to ``champion``."""
    challenger: str
    """Candidate key kept as ``challenger``."""
    reason: str


def select_champion(report: Mapping[str, Any], metrics: Mapping[str, Any]) -> Decision:
    """Apply the promotion rule to a ``brent report`` result (see the module docstring)."""
    candidates: Mapping[str, Any] = metrics["models"]
    comparisons: Mapping[str, Any] = report["comparisons"]
    best_model = max(comparisons, key=lambda n: report["candidates"][n]["auc"]["estimate"])
    reference = report["reference_baseline"]
    model_key, baseline_key = candidates[best_model]["key"], candidates[reference]["key"]
    if comparisons[best_model]["significant"]:
        return Decision(
            champion=model_key,
            challenger=baseline_key,
            reason=(
                f"{best_model} beats the best baseline ({reference}) significantly "
                f"(Holm-adjusted DeLong p = {comparisons[best_model]['p_holm']:.3f})."
            ),
        )
    return Decision(
        champion=baseline_key,
        challenger=model_key,
        reason=(
            f"No model beats the best baseline ({reference}) significantly; the baseline is "
            f"deployed and the best model ({best_model}) stays as challenger."
        ),
    )


def git_info(cwd: Path | None = None) -> dict[str, str]:
    """Commit, branch and dirty flag of the working tree (``unknown`` outside git)."""

    def git(*args: str) -> str:
        out = subprocess.run(
            ["git", *args], cwd=cwd, capture_output=True, text=True, check=True, timeout=10
        )
        return out.stdout.strip()

    try:
        return {
            "git_commit": git("rev-parse", "HEAD"),
            "git_branch": git("rev-parse", "--abbrev-ref", "HEAD"),
            "git_dirty": str(bool(git("status", "--porcelain"))).lower(),
        }
    except (OSError, subprocess.SubprocessError):
        return {"git_commit": "unknown", "git_branch": "unknown", "git_dirty": "unknown"}


def _pip_requirements() -> list[str]:
    requirements = []
    for package in _RUNTIME_PACKAGES:
        try:
            requirements.append(f"{package}=={version(package)}")
        except PackageNotFoundError:  # pragma: no cover - all are dependencies
            requirements.append(package)
    return requirements


def _flatten(prefix: str, values: Mapping[str, Any]) -> dict[str, Any]:
    """Flatten nested mappings: ``{"a": {"b": 1}}`` -> ``{"prefix.a.b": 1}`` (drop ``None``)."""
    flat: dict[str, Any] = {}
    for key, value in values.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flat.update(_flatten(name, value))
        elif value is not None:
            flat[name] = value
    return flat


@contextmanager
def _session(settings: Settings) -> Iterator[MlflowClient]:
    """Point MLflow at the configured store and experiment."""
    cfg = settings.tracking
    if cfg.uri.startswith("sqlite:///"):
        Path(cfg.uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.uri)
    client = MlflowClient(tracking_uri=cfg.uri)
    if client.get_experiment_by_name(cfg.experiment) is None:
        artifacts = cfg.artifact_dir.resolve()
        artifacts.mkdir(parents=True, exist_ok=True)
        client.create_experiment(cfg.experiment, artifact_location=artifacts.as_uri())
    mlflow.set_experiment(cfg.experiment)
    yield client


def log_training(settings: Settings, result: TrainingResult) -> str:
    """Log a ``brent train`` run (parent + one nested run per candidate); return its id."""
    paths, val = settings.paths, settings.validation
    metrics_payload = json.loads(paths.metrics_file.read_text(encoding="utf-8"))
    data = metrics_payload.get("data", {})
    with _session(settings), mlflow.start_run(run_name="train") as parent:
        mlflow.set_tags(
            {
                **git_info(),
                "stage": "train",
                "data_verified": str(data.get("matches_recorded_checksums")),
                **{f"data_sha256.{k}": v for k, v in data.get("sha256", {}).items()},
            }
        )
        mlflow.log_params(
            {
                "seed": settings.seed,
                "test_start": str(settings.split.test_start),
                **_flatten("validation", val.model_dump()),
                **_flatten("preprocessing", settings.preprocessing.model_dump()),
                "n_windows": result.n_windows,
            }
        )
        mlflow.log_dict(settings.model_dump(mode="json"), "config/resolved_config.json")
        for path in (paths.metrics_file, paths.predictions_file):
            mlflow.log_artifact(str(path), "results")
        for trials in paths.results_dir.glob("optuna_*.csv"):
            mlflow.log_artifact(str(trials), "results")
        if paths.plots_dir.is_dir():
            mlflow.log_artifacts(str(paths.plots_dir), "plots")

        children: dict[str, str] = {}
        for name, entry in result.summary.items():
            key = entry["key"]
            with mlflow.start_run(run_name=key, nested=True) as child:
                mlflow.set_tags({"candidate": name, "kind": entry["kind"]})
                if key in result.best_params:
                    mlflow.log_params(_flatten("", result.best_params[key]))
                mlflow.log_metrics(
                    {
                        "cv_auc_mean": entry["cv"]["auc_mean"],
                        "cv_auc_std": entry["cv"]["auc_std"],
                        **{f"oos_{k}": v for k, v in entry["oos"].items()},
                    }
                )
                for step, window in enumerate(entry["windows"]):
                    if window["auc_roc"] is not None:
                        mlflow.log_metric("window_auc", window["auc_roc"], step=step)
                mlflow.sklearn.log_model(
                    result.final_models[key],
                    name="model",
                    input_example=result.input_example,
                    pip_requirements=_pip_requirements(),
                    serialization_format="skops",
                    skops_trusted_types=list(TRUSTED_TYPES),
                )
                children[key] = child.info.run_id

    run_id = str(parent.info.run_id)
    info = {"run_id": run_id, "children": children, "uri": settings.tracking.uri}
    paths.run_info_file.write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    logger.info("MLflow run %s logged to %s", run_id, settings.tracking.uri)
    return run_id


def log_evaluation(settings: Settings) -> Decision:
    """Log ``brent report`` into the training run, register candidates and move the aliases.

    Raises
    ------
    FileNotFoundError
        If the training run was not tracked (``mlflow_run.json`` missing).
    """
    paths, cfg = settings.paths, settings.tracking
    if not paths.run_info_file.is_file():
        raise FileNotFoundError(
            f"{paths.run_info_file} not found; run `brent train` with tracking enabled first."
        )
    info = json.loads(paths.run_info_file.read_text(encoding="utf-8"))
    report = json.loads(paths.report_data_file.read_text(encoding="utf-8"))
    metrics = json.loads(paths.metrics_file.read_text(encoding="utf-8"))
    decision = select_champion(report, metrics)
    keys = {name: m["key"] for name, m in metrics["models"].items()}

    with _session(settings) as client:
        with mlflow.start_run(run_id=info["run_id"]):
            stats: dict[str, float] = {}
            for name, c in report["candidates"].items():
                key = keys[name]
                stats |= {
                    f"{key}.auc_ci_low": c["auc"]["low"],
                    f"{key}.auc_ci_high": c["auc"]["high"],
                }
                stats[f"{key}.brier"] = c["brier"]
            for name, c in report["comparisons"].items():
                stats[f"{keys[name]}.delta_auc"] = c["delta_auc"]["estimate"]
                stats[f"{keys[name]}.p_holm"] = c["p_holm"]
            for name, st in report["backtest"]["strategies"].items():
                if name in keys:
                    stats[f"{keys[name]}.sharpe"] = st["sharpe"]
            mlflow.log_metrics(stats)
            mlflow.set_tags({"stage": "evaluate", "champion": decision.champion})
            for path in (paths.report_file, paths.report_data_file):
                mlflow.log_artifact(str(path), "report")

        versions: dict[str, str] = {}
        for alias, key in ((CHAMPION, decision.champion), (CHALLENGER, decision.challenger)):
            run_id = info["children"][key]
            mv = mlflow.register_model(f"runs:/{run_id}/model", cfg.registered_model)
            tags = {
                "candidate": key,
                "kind": metrics["models"][_name(keys, key)]["kind"],
                "training_run": info["run_id"],
                "promotion_reason": decision.reason,
            }
            for tag, value in tags.items():
                client.set_model_version_tag(cfg.registered_model, mv.version, tag, value)
            client.set_registered_model_alias(cfg.registered_model, alias, mv.version)
            versions[alias] = mv.version

    logger.info(
        "Registry %s: champion = %s (v%s), challenger = %s (v%s). %s",
        cfg.registered_model,
        decision.champion,
        versions[CHAMPION],
        decision.challenger,
        versions[CHALLENGER],
        decision.reason,
    )
    decision_file = paths.results_dir / "promotion.json"
    decision_file.write_text(
        json.dumps({**asdict(decision), "versions": versions}, indent=2) + "\n", encoding="utf-8"
    )
    return decision


def _name(keys: Mapping[str, str], key: str) -> str:
    return next(name for name, k in keys.items() if k == key)


def load_model(settings: Settings, alias: str = CHAMPION) -> Any:
    """Load the registered model behind ``alias`` (a fitted scikit-learn estimator)."""
    with _session(settings):
        uri = f"models:/{settings.tracking.registered_model}@{alias}"
        return mlflow.sklearn.load_model(uri)


def registry_aliases(settings: Settings) -> dict[str, dict[str, str]]:
    """Alias -> version and tags of the registered model (empty if not registered yet)."""
    with _session(settings) as client:
        name = settings.tracking.registered_model
        try:
            model = client.get_registered_model(name)
        except MlflowException:
            return {}
        out: dict[str, dict[str, str]] = {}
        for alias, version_number in model.aliases.items():
            mv = client.get_model_version(name, version_number)
            out[alias] = {"version": str(version_number), **mv.tags}
        return out
