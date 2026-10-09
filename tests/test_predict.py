"""Tests for brent_forecast.predict (`brent predict`)."""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from brent_forecast.config import Settings
from brent_forecast.data.schemas import DataValidationError
from brent_forecast.models.baselines import PersistenceClassifier
from brent_forecast.models.registry import build_pipeline
from brent_forecast.pipeline import prepare_data
from brent_forecast.predict import latest_features, predict_next, save_prediction
from conftest import EXPECTED_FEATURES, OIL_FILENAME, make_oil_frame, write_csv


@pytest.fixture
def fitted(settings: Settings, raw_data_dir: Path) -> object:
    dataset, _ = prepare_data(settings)
    model = build_pipeline("logistic_regression", {"random_state": 0}, settings.preprocessing)
    return model.fit(dataset.features, dataset.target)


def test_latest_features_use_the_last_day_even_without_a_label(
    settings: Settings, raw_data_dir: Path
) -> None:
    as_of, features, source = latest_features(settings, live=False)
    dataset, _ = prepare_data(settings)

    raw = pd.read_csv(raw_data_dir / OIL_FILENAME, parse_dates=["date"])
    assert as_of == raw["date"].max()  # the training set ends one day earlier (no label)
    assert as_of > dataset.dates.max()
    assert list(features.columns) == EXPECTED_FEATURES
    assert features.notna().all().all()
    assert source == "snapshot"


def test_live_data_is_used_when_available(settings: Settings, raw_data_dir: Path) -> None:
    longer = make_oil_frame(end="2026-03-31")
    settings.paths.live_dir.mkdir(parents=True)
    write_csv(longer, settings.paths.live_dir / OIL_FILENAME)

    as_of, _, source = latest_features(settings, live=True)

    assert as_of == longer["date"].max()
    assert source == "live"


def test_missing_live_data_falls_back_to_the_snapshot(
    settings: Settings, raw_data_dir: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _, _, source = latest_features(settings, live=True)

    assert source == "snapshot"
    assert "No live data" in caplog.text


def test_predict_next_with_a_pipeline(settings: Settings, fitted: object) -> None:
    prediction = predict_next(settings, fitted, live=False)

    assert 0.0 <= prediction.proba_up <= 1.0
    assert prediction.direction == ("up" if prediction.proba_up >= 0.5 else "down/flat")
    assert prediction.as_of == pd.Timestamp("2026-03-12").date()


def test_predict_next_with_a_baseline(settings: Settings, raw_data_dir: Path) -> None:
    dataset, _ = prepare_data(settings)
    persistence = PersistenceClassifier().fit(dataset.features, dataset.target)

    prediction = predict_next(settings, persistence, live=False)
    _, features, _ = latest_features(settings, live=False)

    assert prediction.proba_up == float(features["lag_ret_1"].iloc[0] > 0)


def test_invalid_feature_rows_never_reach_the_model(
    settings: Settings, fitted: object, monkeypatch: pytest.MonkeyPatch
) -> None:
    as_of, features, _ = latest_features(settings, live=False)
    features = features.copy()
    features.iloc[0, 0] = np.inf
    monkeypatch.setattr(
        "brent_forecast.predict.latest_features", lambda s, live: (as_of, features, "snapshot")
    )

    with pytest.raises(DataValidationError, match="inference features"):
        predict_next(settings, fitted, live=False)


def test_save_prediction(settings: Settings, fitted: object) -> None:
    prediction = predict_next(settings, fitted, live=False)

    path = save_prediction(
        prediction, settings.paths.prediction_file, alias="champion", version="3"
    )

    payload = json.loads(path.read_text())
    assert payload["as_of"] == "2026-03-12"
    assert payload["alias"] == "champion"
    assert payload["proba_up"] == prediction.proba_up
