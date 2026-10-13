"""Next-day prediction with a registered model (``brent predict``).

The features of the most recent day are built with exactly the same code as in
training (``engineer_features`` + ``handle_nulls``), but without creating a
label: the last day is precisely the one whose next-day direction is unknown.
The feature row is validated with :data:`~brent_forecast.data.schemas.INFERENCE_SCHEMA`
before it reaches the model.

:func:`predict_features` is the single prediction path: ``brent predict`` and the
HTTP API (:mod:`brent_forecast.api`) both go through it, whether the features
are built from the latest day or sent by a client.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd

from brent_forecast.config import Settings
from brent_forecast.data.load import DateBounds, load_oil_data
from brent_forecast.data.schemas import INFERENCE_SCHEMA, validate
from brent_forecast.features.preprocessing import engineer_features, handle_nulls

logger = logging.getLogger(__name__)

Source = Literal["live", "snapshot", "request"]
Direction = Literal["up", "down/flat"]


@dataclass(frozen=True)
class Prediction:
    """Probability that Brent closes higher on the trading day after ``as_of``."""

    as_of: date | None
    """Last observed day (``None`` when a client sends the features without a date)."""
    proba_up: float
    direction: Direction
    source: Source
    """``live`` (ingested data), ``snapshot`` (raw dataset) or ``request`` (API client)."""


def latest_features(settings: Settings, *, live: bool) -> tuple[pd.Timestamp, pd.DataFrame, Source]:
    """Feature row of the most recent day, its date and the data source used.

    With ``live`` the ingested file (``brent data ingest``) is used when it exists.
    """
    paths, data = settings.paths, settings.data
    live_file = paths.live_dir / data.oil_filename
    use_live = live and live_file.is_file()
    if live and not use_live:
        logger.warning("No live data in %s; predicting from the raw snapshot.", live_file)
    oil_file = live_file if use_live else paths.data_dir / data.oil_filename
    df = load_oil_data(
        oil_file,
        paths.data_dir / data.events_filename,
        DateBounds(data.expected_start_min, data.expected_start_max, data.expected_end_min),
    )
    df, feature_cols = engineer_features(df)
    df = handle_nulls(df, feature_cols)
    last = df.tail(1)
    return last["date"].iloc[0], last[feature_cols], "live" if use_live else "snapshot"


def predict_features(
    settings: Settings,
    model: Any,
    features: pd.DataFrame,
    *,
    as_of: date | None,
    source: Source,
) -> Prediction:
    """Validate a one-row feature frame and predict the next-day direction with ``model``.

    Columns are reordered as the model was fitted when it records them
    (``feature_names_in_``).

    Raises
    ------
    DataValidationError
        If a feature is missing, non-numeric or not finite.
    """
    columns = getattr(model, "feature_names_in_", None)
    if columns is not None:
        features = features[list(columns)]
    features = validate(features, INFERENCE_SCHEMA)
    proba = float(model.predict_proba(features)[0, 1])
    return Prediction(
        as_of=as_of,
        proba_up=proba,
        direction="up" if proba >= settings.backtest.threshold else "down/flat",
        source=source,
    )


def predict_next(settings: Settings, model: Any, *, live: bool = True) -> Prediction:
    """Predict the direction of the next trading day with a fitted ``model``."""
    as_of, features, source = latest_features(settings, live=live)
    prediction = predict_features(settings, model, features, as_of=as_of.date(), source=source)
    logger.info(
        "Prediction for the day after %s: P(up) = %.3f", prediction.as_of, prediction.proba_up
    )
    return prediction


def save_prediction(prediction: Prediction, path: Path, **context: str) -> Path:
    """Write the prediction (plus registry context such as alias and version) as JSON."""
    as_of = prediction.as_of.isoformat() if prediction.as_of else None
    payload = {**asdict(prediction), "as_of": as_of, **context}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
