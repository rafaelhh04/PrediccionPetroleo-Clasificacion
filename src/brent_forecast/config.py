"""Typed project configuration.

Values are read from a YAML file (``configs/default.yaml`` by default) and can
be overridden by environment variables prefixed with ``BRENT_``, using ``__``
as the nesting delimiter (e.g. ``BRENT_SPLIT__TRAIN_END=2021-01-01``).

Precedence, highest first: explicit keyword arguments, environment variables,
YAML file. The YAML file is the single source of default values: the models
below only declare types and constraints.
"""

from datetime import date
from pathlib import Path
from typing import Any, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, PositiveFloat, model_validator
from pydantic_settings import (
    BaseSettings,
    PydanticBaseSettingsSource,
    SettingsConfigDict,
    YamlConfigSettingsSource,
)

DEFAULT_CONFIG_PATH = Path("configs/default.yaml")


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PathsSettings(_Section):
    """Filesystem locations (relative paths resolve against the working directory)."""

    data_dir: Path
    results_dir: Path
    checksums_file: Path

    @property
    def live_dir(self) -> Path:
        """Raw files extended with recent data (``brent data ingest``), next to ``data_dir``."""
        return self.data_dir.parent / "live"

    @property
    def prediction_file(self) -> Path:
        """Latest prediction written by ``brent predict``."""
        return self.results_dir / "prediction.json"

    @property
    def dataset_file(self) -> Path:
        """Featurized dataset (Parquet) written by ``brent featurize``, next to ``data_dir``."""
        return self.data_dir.parent / "processed" / "dataset.parquet"

    @property
    def validation_file(self) -> Path:
        """Summary of the raw-data validation (``brent data validate``)."""
        return self.results_dir / "validation.json"

    @property
    def plots_dir(self) -> Path:
        """Directory for generated figures."""
        return self.results_dir / "plots"

    @property
    def log_file(self) -> Path:
        """Log file of the last training run."""
        return self.results_dir / "run.log"

    @property
    def models_dir(self) -> Path:
        """Serialised fitted pipelines (``<model>.joblib``)."""
        return self.results_dir / "models"

    @property
    def predictions_file(self) -> Path:
        """Out-of-sample predictions of every model (one row per test day)."""
        return self.results_dir / "predictions.csv"

    @property
    def metrics_file(self) -> Path:
        """Machine-readable metrics of the last training run."""
        return self.results_dir / "metrics.json"

    @property
    def run_info_file(self) -> Path:
        """MLflow run ids of the last training run (links ``train`` and ``report``)."""
        return self.results_dir / "mlflow_run.json"

    @property
    def explain_dir(self) -> Path:
        """Feature importances and explanation summary written by ``brent explain``."""
        return self.results_dir / "explain"

    @property
    def report_file(self) -> Path:
        """Markdown statistical report written by ``brent report``."""
        return self.results_dir / "report.md"

    @property
    def report_data_file(self) -> Path:
        """Machine-readable counterpart of the report."""
        return self.results_dir / "report.json"


class DataSettings(_Section):
    """Dataset source and canonical file names."""

    kaggle_dataset: str
    oil_filename: str
    events_filename: str
    expected_start_min: date
    expected_start_max: date
    expected_end_min: date

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.expected_start_min > self.expected_start_max:
            raise ValueError("data.expected_start_min must not be after expected_start_max")
        return self

    @property
    def filenames(self) -> tuple[str, str]:
        """Canonical file names, oil dataset first."""
        return (self.oil_filename, self.events_filename)


class SplitSettings(_Section):
    """Boundary between the development period and the out-of-sample (test) period."""

    test_start: date


class PreprocessingSettings(_Section):
    """Thresholds of the preprocessing pipeline."""

    vif_threshold: PositiveFloat
    winsor_lower: float = Field(ge=0.0, lt=1.0)
    winsor_upper: float = Field(gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _check_bounds(self) -> Self:
        if self.winsor_lower >= self.winsor_upper:
            raise ValueError("preprocessing.winsor_lower must be below winsor_upper")
        return self


class ValidationSettings(_Section):
    """Purged walk-forward validation (tuning folds and out-of-sample windows)."""

    purge: int = Field(ge=0)
    embargo: int = Field(ge=0)
    tuning_splits: int = Field(ge=2)
    scoring: str
    mode: Literal["expanding", "rolling"]
    test_window: int = Field(ge=1)
    rolling_train_size: int = Field(ge=1)

    @property
    def max_train_size(self) -> int | None:
        """Training window length for rolling mode; ``None`` (expanding) otherwise."""
        return self.rolling_train_size if self.mode == "rolling" else None


class EvaluationSettings(_Section):
    """Evaluation and reporting options (``brent report``)."""

    learning_curve_train_sizes: list[float] = Field(min_length=1)
    bootstrap_resamples: int = Field(ge=100)
    block_length: int = Field(ge=1)
    confidence_level: float = Field(gt=0.5, lt=1.0)
    alpha: float = Field(gt=0.0, lt=0.5)
    reliability_bins: int = Field(ge=2)


class BacktestSettings(_Section):
    """Long/flat (or long/short) trading simulation of the predictions."""

    cost_bps: float = Field(ge=0.0)
    threshold: float = Field(gt=0.0, lt=1.0)
    allow_short: bool
    periods_per_year: int = Field(ge=1)


class SearchSpace(_Section):
    """Optuna distribution of one hyperparameter."""

    type: Literal["int", "float", "categorical"]
    low: float | None = None
    high: float | None = None
    log: bool = False
    choices: list[Any] | None = None

    @model_validator(mode="after")
    def _check_distribution(self) -> Self:
        if self.type == "categorical":
            if not self.choices:
                raise ValueError("a categorical search space needs non-empty `choices`")
        elif self.low is None or self.high is None or self.low >= self.high:
            raise ValueError(f"a {self.type} search space needs `low` < `high`")
        elif self.log and self.low <= 0:
            raise ValueError("a log-scale search space needs `low` > 0")
        return self


class LiveSource(_Section):
    """Where one market series is fetched from."""

    provider: Literal["yahoo", "fred"]
    symbol: str = Field(min_length=1)


class LiveSettings(_Section):
    """Incremental ingestion of recent market data (``brent data ingest``)."""

    lookback_days: int = Field(ge=31)
    timeout_seconds: PositiveFloat
    sources: dict[Literal["brent_price", "wti_price", "dxy_index", "vix"], LiveSource]

    @model_validator(mode="after")
    def _require_brent(self) -> Self:
        if "brent_price" not in self.sources:
            raise ValueError("live.sources must include brent_price (it defines the trading days)")
        return self


class TrackingSettings(_Section):
    """MLflow experiment tracking and model registry."""

    enabled: bool
    uri: str
    artifact_dir: Path
    experiment: str = Field(min_length=1)
    registered_model: str = Field(min_length=1)


class ExplainSettings(_Section):
    """Model explanations written by ``brent explain``."""

    models: list[str] = Field(min_length=1)
    permutation_repeats: int = Field(ge=1)
    background_rows: int = Field(ge=10)
    max_explained_rows: int = Field(ge=1)


class ModelSettings(_Section):
    """Fixed hyperparameters and search space of one model.

    ``grid`` (exhaustive ``GridSearchCV``) or ``space`` (Optuna TPE), never both.
    """

    params: dict[str, Any]
    grid: dict[str, list[Any]] = Field(default_factory=dict)
    space: dict[str, SearchSpace] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _check_search(self) -> Self:
        if bool(self.grid) == bool(self.space):
            raise ValueError("define exactly one of `grid` (grid search) or `space` (Optuna)")
        return self


class TuningSettings(_Section):
    """Optuna study used for models that declare a ``space``."""

    n_trials: int = Field(ge=1)
    timeout: PositiveFloat | None
    pruning: bool
    startup_trials: int = Field(ge=1)


class ModelsSettings(_Section):
    """Per-model configuration."""

    logistic_regression: ModelSettings
    svm: ModelSettings
    random_forest: ModelSettings
    mlp: ModelSettings
    lightgbm: ModelSettings


class Settings(BaseSettings):
    """Root configuration object, injected into every pipeline stage."""

    model_config = SettingsConfigDict(
        env_prefix="BRENT_",
        env_nested_delimiter="__",
        extra="forbid",
        frozen=True,
    )

    seed: int
    paths: PathsSettings
    data: DataSettings
    split: SplitSettings
    preprocessing: PreprocessingSettings
    validation: ValidationSettings
    evaluation: EvaluationSettings
    backtest: BacktestSettings
    tuning: TuningSettings
    explain: ExplainSettings
    tracking: TrackingSettings
    live: LiveSettings
    models: ModelsSettings

    @classmethod
    def settings_customise_sources(
        cls,
        settings_cls: type[BaseSettings],
        init_settings: PydanticBaseSettingsSource,
        env_settings: PydanticBaseSettingsSource,
        dotenv_settings: PydanticBaseSettingsSource,
        file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        """Use kwargs > environment > YAML (no .env or secrets files)."""
        return (init_settings, env_settings, YamlConfigSettingsSource(settings_cls))


def load_settings(config_path: Path = DEFAULT_CONFIG_PATH, **overrides: Any) -> Settings:
    """Load settings from ``config_path``, environment variables and ``overrides``.

    Raises
    ------
    FileNotFoundError
        If the YAML file does not exist.
    pydantic.ValidationError
        If the merged configuration is invalid.
    """
    if not config_path.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    class _FileSettings(Settings):
        model_config = SettingsConfigDict(yaml_file=config_path)

    return _FileSettings(**overrides)
