"""The API against a real MLflow registry (SQLite in ``tmp_path``, no network).

The registry is filled directly with MLflow (one logged model per alias, with the
same options as ``brent train``), which takes seconds instead of a full training.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import mlflow
import pytest
from fastapi.testclient import TestClient
from mlflow import MlflowClient

from brent_forecast.api import create_app
from brent_forecast.api.service import load_from_registry
from brent_forecast.config import Settings
from brent_forecast.models.baselines import PersistenceClassifier
from brent_forecast.models.registry import build_pipeline
from brent_forecast.pipeline import prepare_data
from brent_forecast.predict import predict_next
from brent_forecast.tracking import TRUSTED_TYPES, load_model
from conftest import EXPECTED_FEATURES


def _register(
    settings: Settings, model: Any, example: Any, alias: str, tags: dict[str, str]
) -> str:
    cfg = settings.tracking
    with mlflow.start_run():
        logged = mlflow.sklearn.log_model(
            model,
            name="model",
            input_example=example,
            serialization_format="skops",
            skops_trusted_types=list(TRUSTED_TYPES),
            pip_requirements=["scikit-learn"],
        )
    version = mlflow.register_model(logged.model_uri, cfg.registered_model).version
    client = MlflowClient(cfg.uri)
    for key, value in tags.items():
        client.set_model_version_tag(cfg.registered_model, version, key, value)
    client.set_registered_model_alias(cfg.registered_model, alias, version)
    return str(version)


@pytest.fixture
def registry(settings: Settings, small_raw_data_dir: Path) -> Iterator[Settings]:
    """Registry: persistence baseline as champion, logistic regression as challenger."""
    cfg = settings.tracking
    Path(cfg.uri.removeprefix("sqlite:///")).parent.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(cfg.uri)
    cfg.artifact_dir.mkdir(parents=True)
    mlflow.create_experiment(cfg.experiment, artifact_location=cfg.artifact_dir.as_uri())
    mlflow.set_experiment(cfg.experiment)

    dataset, _ = prepare_data(settings)
    X, y = dataset.features, dataset.target
    example = X.head(5)
    persistence = PersistenceClassifier().fit(X, y)
    logreg = build_pipeline("logistic_regression", {"random_state": 0}, settings.preprocessing)
    reason = "No model beats the best baseline (Persistence) significantly."
    _register(
        settings,
        persistence,
        example,
        "champion",
        {"candidate": "persistence", "kind": "baseline", "promotion_reason": reason},
    )
    _register(
        settings,
        logreg.fit(X, y),
        example,
        "challenger",
        {"candidate": "logistic_regression", "kind": "model", "promotion_reason": reason},
    )
    yield settings
    mlflow.set_tracking_uri(None)


def _with_alias(settings: Settings, alias: str) -> Settings:
    serving = settings.serving.model_copy(update={"alias": alias})
    return settings.model_copy(update={"serving": serving})


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_champion_is_served_from_the_registry(registry: Settings) -> None:
    with TestClient(create_app(registry)) as client:
        health = client.get("/health").json()
        info = client.get("/model-info").json()
        prediction = client.post("/predict").json()

    assert health == {
        "status": "ok",
        "model_loaded": True,
        "alias": "champion",
        "version": "1",
        "detail": None,
    }
    assert info["candidate"] == "persistence"
    assert info["kind"] == "baseline"
    assert info["promotion_reason"].startswith("No model beats")
    # The persistence baseline records no feature names: the signature is the contract.
    assert info["features"] == EXPECTED_FEATURES
    expected = predict_next(registry, load_model(registry), live=True)
    assert prediction["proba_up"] == expected.proba_up
    assert prediction["as_of"] == str(expected.as_of)
    assert prediction["model"] == {"alias": "champion", "version": "1", "candidate": "persistence"}


@pytest.mark.filterwarnings("ignore::UserWarning")
def test_the_challenger_can_be_served_instead(registry: Settings) -> None:
    challenger = _with_alias(registry, "challenger")

    with TestClient(create_app(challenger)) as client:
        info = client.get("/model-info").json()
        latest = client.post("/predict").json()

    assert info["candidate"] == "logistic_regression"
    assert info["version"] == "2"
    assert 0.0 < latest["proba_up"] < 1.0
    served = load_from_registry(challenger, "challenger")
    assert list(served.model.feature_names_in_) == list(served.columns or ()) == info["features"]


def test_an_unknown_alias_keeps_the_api_up_but_not_ready(registry: Settings) -> None:
    with TestClient(create_app(_with_alias(registry, "missing"))) as client:
        health = client.get("/health")
        live = client.get("/health/live")

    assert health.status_code == 503
    assert health.json()["model_loaded"] is False
    assert "MlflowException" in health.json()["detail"]
    assert live.status_code == 200


def test_an_empty_registry_is_not_ready(settings: Settings) -> None:
    with TestClient(create_app(settings)) as client:
        response = client.get("/health")

    assert response.status_code == 503
