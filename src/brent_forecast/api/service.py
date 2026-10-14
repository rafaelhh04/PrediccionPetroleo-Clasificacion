"""The model served by the API: loading from the registry, retries and cached features.

The registry is the single source of truth: the model behind
``serving.alias`` (``champion`` by default) is loaded once at start-up, with its
version, promotion reason and input contract (the column names of the MLflow
model signature, recorded by ``brent train``). If no model can be loaded yet,
the service keeps running and retries at most every
``serving.reload_interval_seconds`` (``/health`` answers 503 meanwhile), so the
API can start before the first ``brent report``.

The model is copied to a temporary directory before loading: MLflow writes
metadata next to the artefacts it loads from ``models:/`` URIs, and in the
cloud the registry is a read-only mount of the bucket (see ``infra/``).
"""

import logging
import tempfile
import threading
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import mlflow
import pandas as pd
from mlflow.exceptions import MlflowException

from brent_forecast.config import Settings
from brent_forecast.predict import Source, latest_features
from brent_forecast.tracking import registry_aliases

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ServedModel:
    """A fitted model plus the registry metadata the API exposes."""

    alias: str
    model: Any
    columns: tuple[str, ...] | None
    """Expected feature names, in order (``None`` if the model records none)."""
    info: Mapping[str, str] = field(default_factory=dict)
    """Version and tags of the model version (``candidate``, ``promotion_reason``...)."""
    loaded_at: datetime = field(default_factory=lambda: datetime.now(UTC))


Loader = Callable[[Settings, str], ServedModel]


def load_from_registry(settings: Settings, alias: str) -> ServedModel:
    """Load ``models:/<registered_model>@<alias>`` with its signature and version tags.

    The alias is resolved to a version first, so the model, its signature and its
    tags all come from the same version even if the alias moves meanwhile. Nothing
    is written to the registry (it can be mounted read-only).

    Raises
    ------
    MlflowException
        If no version has ``alias`` (or the registry cannot be read).
    """
    name = settings.tracking.registered_model
    tags = registry_aliases(settings).get(alias)  # also points MLflow at the configured store
    if tags is None:
        raise MlflowException(f"No version of {name!r} has the alias {alias!r}.")
    uri = f"models:/{name}/{tags['version']}"
    with tempfile.TemporaryDirectory(prefix="brent-model-") as tmp:
        local = mlflow.artifacts.download_artifacts(artifact_uri=uri, dst_path=tmp)
        model = mlflow.sklearn.load_model(local)
    info = mlflow.models.get_model_info(uri)
    if info.signature is not None:
        columns: tuple[str, ...] | None = tuple(map(str, info.signature.inputs.input_names()))
    else:
        names = getattr(model, "feature_names_in_", None)
        columns = None if names is None else tuple(map(str, names))
    return ServedModel(
        alias=alias,
        model=model,
        columns=columns,
        info=tags,
    )


class ModelService:
    """Thread-safe holder of the served model (FastAPI runs sync endpoints in a thread pool)."""

    def __init__(
        self,
        settings: Settings,
        loader: Loader = load_from_registry,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.settings = settings
        self._loader = loader
        self._clock = clock
        self._lock = threading.Lock()
        self._served: ServedModel | None = None
        self._error: str | None = "not loaded yet"
        self._last_attempt: float | None = None
        self.load_failures = 0
        """Failed loading attempts since start-up (exported as a metric)."""

    @property
    def served(self) -> ServedModel | None:
        """The loaded model, without triggering a new loading attempt."""
        return self._served

    @property
    def alias(self) -> str:
        """Registry alias this service serves."""
        return self.settings.serving.alias

    @property
    def error(self) -> str | None:
        """Why the last loading attempt failed (``None`` once a model is loaded)."""
        return self._error

    def load(self) -> ServedModel | None:
        """Try to load the model now; return it, or ``None`` (the error is kept)."""
        with self._lock:
            return self._load()

    def get(self) -> ServedModel | None:
        """Return the served model, retrying the load if none is loaded and the interval elapsed."""
        if self._served is not None:
            return self._served
        with self._lock:
            interval = self.settings.serving.reload_interval_seconds
            due = self._last_attempt is None or self._clock() - self._last_attempt >= interval
            if self._served is None and due:
                self._load()
            return self._served

    def _load(self) -> ServedModel | None:
        self._last_attempt = self._clock()
        try:
            served = self._loader(self.settings, self.alias)
        except Exception as exc:  # any failure leaves the API up and not ready (503)
            self._error = f"{type(exc).__name__}: {exc}"
            self.load_failures += 1
            logger.warning("Could not load model alias %r: %s", self.alias, self._error)
            return None
        self._served, self._error = served, None
        logger.info(
            "Serving %s@%s (version %s, candidate %s)",
            self.settings.tracking.registered_model,
            self.alias,
            served.info.get("version", "unknown"),
            served.info.get("candidate", "unknown"),
        )
        return served


class LatestFeatures:
    """Feature row of the most recent day, recomputed only when a data file changes."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._lock = threading.Lock()
        self._key: tuple[tuple[int, int] | None, ...] | None = None
        self._value: tuple[pd.Timestamp, pd.DataFrame, Source] | None = None

    def _files(self) -> tuple[Path, ...]:
        paths, data = self.settings.paths, self.settings.data
        return (
            paths.live_dir / data.oil_filename,
            paths.data_dir / data.oil_filename,
            paths.data_dir / data.events_filename,
        )

    def _fingerprint(self) -> tuple[tuple[int, int] | None, ...]:
        stats: list[tuple[int, int] | None] = []
        for path in self._files():
            try:
                stat = path.stat()
            except FileNotFoundError:
                stats.append(None)
            else:
                stats.append((stat.st_mtime_ns, stat.st_size))
        return tuple(stats)

    def get(self) -> tuple[pd.Timestamp, pd.DataFrame, Source]:
        """Date, one-row feature frame and source (``live``/``snapshot``) of the latest day.

        Raises
        ------
        FileNotFoundError, ValueError
            If the data files are missing or invalid (see :func:`latest_features`).
        """
        with self._lock:
            key = self._fingerprint()
            if self._value is None or key != self._key:
                self._value = latest_features(self.settings, live=self.settings.serving.live)
                self._key = key
            return self._value
