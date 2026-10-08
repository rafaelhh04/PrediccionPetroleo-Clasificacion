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
from typing import Any, Self

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
    def metrics_file(self) -> Path:
        """Machine-readable metrics of the last training run."""
        return self.results_dir / "metrics.json"


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
    """Chronological train/validation/test boundaries."""

    train_end: date
    val_end: date

    @model_validator(mode="after")
    def _check_order(self) -> Self:
        if self.train_end >= self.val_end:
            raise ValueError("split.train_end must be earlier than split.val_end")
        return self


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


class CVSettings(_Section):
    """Time-series cross-validation used for tuning and learning curves."""

    n_splits: int = Field(ge=2)
    scoring: str


class EvaluationSettings(_Section):
    """Evaluation and reporting options."""

    learning_curve_train_sizes: list[float] = Field(min_length=1)


class ModelSettings(_Section):
    """Fixed hyperparameters and search grid of one model."""

    params: dict[str, Any]
    grid: dict[str, list[Any]]


class ModelsSettings(_Section):
    """Per-model configuration."""

    logistic_regression: ModelSettings
    svm: ModelSettings
    random_forest: ModelSettings
    mlp: ModelSettings


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
    cv: CVSettings
    evaluation: EvaluationSettings
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
