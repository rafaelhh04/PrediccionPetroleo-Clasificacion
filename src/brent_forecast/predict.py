"""Next-day prediction with a registered model (``brent predict``).

The features of the most recent day are built with exactly the same code as in
training (``engineer_features`` + ``handle_nulls``), but without creating a
label: the last day is precisely the one whose next-day direction is unknown.
The feature row is validated with :data:`~brent_forecast.data.schemas.INFERENCE_SCHEMA`
before it reaches the model.
"""

import json
import logging
from dataclasses import asdict, dataclass
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd

from brent_forecast.config import Settings
from brent_forecast.data.load import DateBounds, load_oil_data
from brent_forecast.data.schemas import INFERENCE_SCHEMA, validate
from brent_forecast.features.preprocessing import engineer_features, handle_nulls

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Prediction:
    """Probability that Brent closes higher on the trading day after ``as_of``."""

    as_of: date
    proba_up: float
    direction: str
    source: str
    """``live`` (ingested data) or ``snapshot`` (raw dataset)."""


def latest_features(settings: Settings, *, live: bool) -> tuple[pd.Timestamp, pd.DataFrame, str]:
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


def predict_next(settings: Settings, model: Any, *, live: bool = True) -> Prediction:
    """Predict the direction of the next trading day with a fitted ``model``."""
    as_of, features, source = latest_features(settings, live=live)
    columns = getattr(model, "feature_names_in_", None)
    if columns is not None:
        features = features[list(columns)]
    features = validate(features, INFERENCE_SCHEMA)
    proba = float(model.predict_proba(features)[0, 1])
    prediction = Prediction(
        as_of=as_of.date(),
        proba_up=proba,
        direction="up" if proba >= settings.backtest.threshold else "down/flat",
        source=source,
    )
    logger.info("Prediction for the day after %s: P(up) = %.3f", prediction.as_of, proba)
    return prediction


def save_prediction(prediction: Prediction, path: Path, **context: str) -> Path:
    """Write the prediction (plus registry context such as alias and version) as JSON."""
    payload = {**asdict(prediction), "as_of": prediction.as_of.isoformat(), **context}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path
