"""Tests for brent_forecast.api (FastAPI inference service), with an in-memory model."""

from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from brent_forecast.api import create_app
from brent_forecast.api.service import ModelService, ServedModel
from brent_forecast.config import Settings, load_settings
from brent_forecast.models.baselines import PersistenceClassifier
from brent_forecast.models.registry import build_pipeline
from brent_forecast.pipeline import prepare_data
from brent_forecast.predict import latest_features
from conftest import CONFIG_PATH, EXPECTED_FEATURES, _paths, _tracking

INFO = {
    "version": "7",
    "candidate": "logistic_regression",
    "kind": "model",
    "promotion_reason": "test",
    "training_run": "abc",
}


@pytest.fixture(scope="module")
def fitted(tmp_path_factory: pytest.TempPathFactory) -> Any:
    """Logistic regression fitted once per module on the synthetic data."""
    from conftest import EVENTS_FILENAME, OIL_FILENAME, make_events_frame, make_oil_frame, write_csv

    root = tmp_path_factory.mktemp("fit")
    settings = load_settings(CONFIG_PATH, paths=_paths(root), tracking=_tracking(root))
    settings.paths.data_dir.mkdir(parents=True)
    oil = make_oil_frame()
    write_csv(oil, settings.paths.data_dir / OIL_FILENAME)
    write_csv(make_events_frame(oil), settings.paths.data_dir / EVENTS_FILENAME)
    dataset, _ = prepare_data(settings)
    model = build_pipeline("logistic_regression", {"random_state": 0}, settings.preprocessing)
    return model.fit(dataset.features, dataset.target)


@pytest.fixture
def served(fitted: Any) -> ServedModel:
    return ServedModel("champion", fitted, tuple(EXPECTED_FEATURES), INFO)


@pytest.fixture
def client(settings: Settings, served: ServedModel, raw_data_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app(settings, loader=lambda s, alias: served)) as test_client:
        yield test_client


def _features(settings: Settings) -> dict[str, float]:
    _, row, _ = latest_features(settings, live=False)
    return {k: float(v) for k, v in row.iloc[0].items()}


# ── health and model info ─────────────────────────────────────


def test_health_is_ok_once_the_model_is_loaded(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_loaded": True,
        "alias": "champion",
        "version": "7",
        "detail": None,
    }
    assert client.get("/health/live").json() == {"status": "ok"}


def test_model_info_exposes_the_registry_entry_and_the_contract(client: TestClient) -> None:
    body = client.get("/model-info").json()

    assert body["candidate"] == "logistic_regression"
    assert body["promotion_reason"] == "test"
    assert body["registered_model"] == "test-model"
    assert body["features"] == EXPECTED_FEATURES
    assert body["threshold"] == 0.5


def test_without_a_model_every_model_endpoint_answers_503(settings: Settings) -> None:
    def missing(s: Settings, alias: str) -> ServedModel:
        raise RuntimeError("registry empty")

    with TestClient(create_app(settings, loader=missing)) as client:
        health = client.get("/health")
        assert health.status_code == 503
        assert health.json()["model_loaded"] is False
        assert "registry empty" in health.json()["detail"]
        assert client.get("/health/live").status_code == 200
        assert client.get("/model-info").status_code == 503
        response = client.post("/predict", json={})
        assert response.status_code == 503
        assert "No model loaded" in response.json()["detail"]


# ── predictions ───────────────────────────────────────────────


def test_predict_the_latest_day_without_a_body(client: TestClient) -> None:
    for response in (client.post("/predict"), client.post("/predict", json={})):
        assert response.status_code == 200, response.text
        body = response.json()
        assert body["as_of"] == "2026-03-12"
        assert body["source"] == "snapshot"
        assert 0.0 <= body["proba_up"] <= 1.0
        assert body["direction"] == ("up" if body["proba_up"] >= 0.5 else "down/flat")
        assert body["model"] == {
            "alias": "champion",
            "version": "7",
            "candidate": "logistic_regression",
        }


def test_predict_from_features_matches_the_latest_day(
    client: TestClient, settings: Settings
) -> None:
    latest = client.post("/predict").json()

    body = client.post(
        "/predict", json={"features": _features(settings), "as_of": "2026-03-12"}
    ).json()

    assert body["source"] == "request"
    assert body["as_of"] == "2026-03-12"
    assert body["proba_up"] == pytest.approx(latest["proba_up"])


@pytest.mark.parametrize(
    ("mutate", "kind"),
    [
        (lambda f: f.pop("vix"), "missing"),
        (lambda f: f.update(unknown=1.0), "extra_forbidden"),
        (lambda f: f.update(vix="17.0"), "float_type"),
        (lambda f: f.update(vix=True), "float_type"),
    ],
)
def test_invalid_features_are_rejected_with_422(
    client: TestClient, settings: Settings, mutate: Any, kind: str
) -> None:
    features = _features(settings)
    mutate(features)

    response = client.post("/predict", json={"features": features})

    assert response.status_code == 422
    assert kind in {error["type"] for error in response.json()["detail"]}


def test_non_finite_numbers_are_rejected(client: TestClient, settings: Settings) -> None:
    body = '{"features": {"vix": NaN}}'

    response = client.post("/predict", content=body, headers={"content-type": "application/json"})

    assert response.status_code == 422


def test_unknown_body_fields_and_lonely_as_of_are_rejected(client: TestClient) -> None:
    assert client.post("/predict", json={"feature": {}}).status_code == 422
    assert client.post("/predict", json={"as_of": "2026-03-12"}).status_code == 422


def test_a_model_without_a_contract_validates_with_the_schema(
    settings: Settings, raw_data_dir: Path
) -> None:
    import pandas as pd

    X = pd.DataFrame({"lag_ret_1": [0.1, -0.1], "vix": [1.0, 2.0]})
    model = PersistenceClassifier().fit(X, [0, 1])
    served = ServedModel("champion", model, None)
    features = _features(settings)

    with TestClient(create_app(settings, loader=lambda s, a: served)) as client:
        ok = client.post("/predict", json={"features": features})
        assert ok.status_code == 200
        assert ok.json()["proba_up"] == float(features["lag_ret_1"] > 0)
        assert ok.json()["model"]["version"] is None
        assert client.get("/model-info").json()["features"] is None

        missing = client.post("/predict", json={"features": {"vix": 1.0}})
        assert missing.status_code == 422
        assert missing.json()["detail"][0]["type"] == "missing"


def test_schema_violations_are_422(
    client: TestClient, settings: Settings, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pydantic already rejects non-finite numbers; the Pandera contract is the last guard."""
    from brent_forecast.data.schemas import DataValidationError

    def invalid(*args: Any, **kwargs: Any) -> Any:
        raise DataValidationError("inference features failed validation: vix")

    monkeypatch.setattr("brent_forecast.api.app.predict_features", invalid)

    response = client.post("/predict", json={"features": _features(settings)})

    assert response.status_code == 422
    assert response.json()["detail"][0]["loc"] == ["body", "features"]
    assert "inference features" in response.json()["detail"][0]["msg"]


def test_latest_day_without_data_is_503(settings: Settings, served: ServedModel) -> None:
    assert not settings.paths.data_dir.exists()
    with TestClient(create_app(settings, loader=lambda s, a: served)) as client:
        response = client.post("/predict")

    assert response.status_code == 503
    assert "Latest features unavailable" in response.json()["detail"]


def test_root_redirects_to_the_docs(client: TestClient) -> None:
    response = client.get("/", follow_redirects=False)

    assert response.status_code == 307
    assert response.headers["location"] == "/docs"


# ── model service ─────────────────────────────────────────────


def test_loading_is_retried_after_the_interval(settings: Settings, served: ServedModel) -> None:
    now = [0.0]
    calls: list[str] = []

    def flaky(s: Settings, alias: str) -> ServedModel:
        calls.append(alias)
        if len(calls) == 1:
            raise RuntimeError("not yet")
        return served

    service = ModelService(settings, flaky, clock=lambda: now[0])
    assert service.load() is None
    assert service.error == "RuntimeError: not yet"
    assert service.get() is None  # interval (30 s) not elapsed: no new attempt
    assert len(calls) == 1

    now[0] = 30.0
    assert service.get() is served
    assert service.error is None
    assert service.get() is served
    assert len(calls) == 2


def _refreshing(settings: Settings, interval: float) -> Settings:
    serving = settings.serving.model_copy(update={"refresh_interval_seconds": interval})
    return settings.model_copy(update={"serving": serving})


class _Registry:
    """The alias moves when ``version`` changes; loads return a model of that version."""

    def __init__(self, served: ServedModel) -> None:
        self.served, self.version, self.loads = served, "7", 0
        self.probe_error: Exception | None = None

    def load(self, s: Settings, alias: str) -> ServedModel:
        self.loads += 1
        return ServedModel(alias, self.served.model, self.served.columns, {"version": self.version})

    def probe(self, s: Settings, alias: str) -> str | None:
        if self.probe_error:
            raise self.probe_error
        return self.version


def test_a_moved_alias_is_reloaded_in_the_background(
    settings: Settings, served: ServedModel
) -> None:
    now, tasks = [0.0], []
    registry = _Registry(served)
    service = ModelService(
        _refreshing(settings, 300),
        registry.load,
        clock=lambda: now[0],
        probe=registry.probe,
        background=tasks.append,
    )
    first = service.load()
    assert first is not None and first.info["version"] == "7"

    now[0] = 299.0
    assert service.get() is first and not tasks  # not due yet

    registry.version = "8"  # e.g. `brent retrain` registered a new champion
    now[0] = 300.0
    assert service.get() is first  # the current model answers while the new one loads
    assert service.get() is first and len(tasks) == 1  # one refresh at a time
    tasks.pop()()

    refreshed = service.get()
    assert refreshed is not None and refreshed.info["version"] == "8"
    assert (registry.loads, service.refreshes) == (2, 1)


def test_an_unchanged_alias_is_not_reloaded(settings: Settings, served: ServedModel) -> None:
    now = [0.0]
    registry = _Registry(served)
    service = ModelService(
        _refreshing(settings, 60),
        registry.load,
        clock=lambda: now[0],
        probe=registry.probe,
        background=lambda task: task(),
    )
    service.load()
    for t in (60.0, 120.0, 180.0):
        now[0] = t
        service.get()

    assert (registry.loads, service.refreshes) == (1, 0)


def test_a_failed_refresh_keeps_serving_the_current_model(
    settings: Settings, served: ServedModel, caplog: pytest.LogCaptureFixture
) -> None:
    now = [0.0]
    registry = _Registry(served)
    registry.probe_error = RuntimeError("registry unreachable")
    service = ModelService(
        _refreshing(settings, 60),
        registry.load,
        clock=lambda: now[0],
        probe=registry.probe,
        background=lambda task: task(),
    )
    first = service.load()
    now[0] = 60.0

    with caplog.at_level("WARNING", logger="brent_forecast.api.service"):
        assert service.get() is first

    assert "registry unreachable" in caplog.text
    registry.probe_error, registry.version = None, "9"
    now[0] = 120.0  # the next check succeeds
    service.get()
    assert service.get().info["version"] == "9"  # type: ignore[union-attr]


def test_refresh_can_be_disabled(settings: Settings, served: ServedModel) -> None:
    tasks: list[Any] = []
    service = ModelService(
        _refreshing(settings, 0),
        lambda s, alias: served,
        clock=lambda: 1e9,
        probe=lambda s, alias: "other",
        background=tasks.append,
    )
    service.load()
    service.get()
    assert not tasks


def test_the_default_probe_reads_the_registry(settings: Settings) -> None:
    from brent_forecast.api.service import registry_version

    assert registry_version(settings, "champion") is None  # empty registry


def test_latest_features_are_cached_until_a_file_changes(
    settings: Settings, served: ServedModel, raw_data_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from brent_forecast.api import service as service_module

    calls = []
    real = service_module.latest_features

    def counting(s: Settings, *, live: bool) -> Any:
        calls.append(live)
        return real(s, live=live)

    monkeypatch.setattr(service_module, "latest_features", counting)
    cache = service_module.LatestFeatures(settings)

    first = cache.get()
    assert cache.get() is first
    assert calls == [True]

    oil = next(raw_data_dir.glob("oil_*.csv"))
    oil.write_text(oil.read_text() + "\n")
    cache.get()
    assert len(calls) == 2
