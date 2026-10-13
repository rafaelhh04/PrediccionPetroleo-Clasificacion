"""Tests for the API observability: request ids, JSON logs and Prometheus metrics."""

import json
import logging
from collections.abc import Iterator
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from brent_forecast.api import create_app
from brent_forecast.api.observability import REQUEST_ID_HEADER, new_request_id
from brent_forecast.api.service import ServedModel
from brent_forecast.config import Settings
from brent_forecast.logging_config import JsonFormatter, RequestIdFilter, request_id
from brent_forecast.models.baselines import PersistenceClassifier
from conftest import EXPECTED_FEATURES

INFO = {"version": "4", "candidate": "persistence", "kind": "baseline"}


@pytest.fixture
def served() -> ServedModel:
    model = PersistenceClassifier().fit(pd.DataFrame({"lag_ret_1": [0.1, -0.1]}), [0, 1])
    return ServedModel("champion", model, tuple(EXPECTED_FEATURES), INFO)


@pytest.fixture
def client(settings: Settings, served: ServedModel, raw_data_dir: Path) -> Iterator[TestClient]:
    with TestClient(create_app(settings, loader=lambda s, a: served)) as test_client:
        yield test_client


class _Records(logging.Handler):
    """Collect records after the request-id filter, as the console handler sees them."""

    def __init__(self) -> None:
        super().__init__()
        self.records: list[logging.LogRecord] = []
        self.addFilter(RequestIdFilter())

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@pytest.fixture
def records() -> Iterator[list[logging.LogRecord]]:
    handler = _Records()
    root = logging.getLogger()
    root.addHandler(handler)
    level, root.level = root.level, logging.DEBUG
    yield handler.records
    root.removeHandler(handler)
    root.setLevel(level)


def _metric(text: str, name: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith(name)]


# ── request ids ───────────────────────────────────────────────


def test_a_safe_client_request_id_is_kept_and_returned(client: TestClient) -> None:
    response = client.get("/health", headers={REQUEST_ID_HEADER: "trace-42"})

    assert response.headers[REQUEST_ID_HEADER] == "trace-42"


@pytest.mark.parametrize("candidate", [None, "", "bad id", "x" * 200, "a\r\nb"])
def test_missing_or_unsafe_ids_are_replaced(candidate: str | None) -> None:
    rid = new_request_id(candidate)

    assert rid != candidate
    assert len(rid) == 32 and rid.isalnum()


def test_every_response_gets_a_fresh_id(client: TestClient) -> None:
    ids = {client.get("/health/live").headers[REQUEST_ID_HEADER] for _ in range(3)}
    invalid = client.post("/predict", json={"features": {"vix": "x"}})

    assert len(ids) == 3
    assert REQUEST_ID_HEADER in invalid.headers  # errors are traced too


def test_logs_emitted_during_a_request_carry_its_id(
    client: TestClient, records: list[logging.LogRecord]
) -> None:
    client.post("/predict", headers={REQUEST_ID_HEADER: "req-7"})

    access = [r for r in records if r.name == "brent_forecast.api.access"]
    inner = [r for r in records if r.name.startswith("brent_forecast.data")]
    assert access and inner
    assert {getattr(r, "request_id", None) for r in access + inner} == {"req-7"}
    assert getattr(access[-1], "status", None) == 200
    assert access[-1].levelno == logging.INFO
    assert request_id.get() is None  # reset after the request


def test_probes_are_logged_at_debug(client: TestClient, records: list[logging.LogRecord]) -> None:
    client.get("/health")
    client.get("/metrics")

    access = [r for r in records if r.name == "brent_forecast.api.access"]
    assert [r.levelno for r in access] == [logging.DEBUG, logging.DEBUG]


# ── metrics ───────────────────────────────────────────────────


def test_metrics_count_predictions_and_time_the_model(client: TestClient) -> None:
    client.post("/predict")
    client.post("/predict")
    features = {name: 0.0 for name in EXPECTED_FEATURES} | {"lag_ret_1": -0.01}
    client.post("/predict", json={"features": features})
    client.post("/predict", json={"features": {"vix": 1.0}})  # 422: not a prediction

    text = client.get("/metrics").text

    predictions = _metric(text, "brent_predictions_total")
    assert any('source="snapshot"' in line and line.endswith(" 2.0") for line in predictions)
    assert any(
        'direction="down/flat"' in line and 'source="request"' in line for line in predictions
    )
    assert _metric(text, 'brent_model_inference_seconds_count{alias="champion"}') == [
        'brent_model_inference_seconds_count{alias="champion"} 3.0'
    ]
    assert _metric(text, "brent_model_loaded") == ['brent_model_loaded{alias="champion"} 1.0']
    info = _metric(text, "brent_model_info")
    assert info == [
        'brent_model_info{alias="champion",candidate="persistence",kind="baseline",version="4"} 1.0'
    ]
    http = _metric(text, "http_requests_total")
    assert any('status="422"' in line for line in http)
    assert not any('handler="/metrics"' in line for line in http)


def test_metrics_report_an_unavailable_model(settings: Settings) -> None:
    def broken(s: Settings, alias: str) -> ServedModel:
        raise RuntimeError("registry down")

    with TestClient(create_app(settings, loader=broken)) as client:
        text = client.get("/metrics").text

    assert _metric(text, "brent_model_loaded") == ['brent_model_loaded{alias="champion"} 0.0']
    assert _metric(text, "brent_model_info") == []
    assert _metric(text, "brent_model_load_failures_total") == [
        'brent_model_load_failures_total{alias="champion"} 1.0'
    ]


def test_metrics_are_not_part_of_the_api_contract(client: TestClient) -> None:
    assert "/metrics" not in client.get("/openapi.json").json()["paths"]


def test_two_apps_do_not_share_metrics(settings: Settings, served: ServedModel) -> None:
    first = create_app(settings, loader=lambda s, a: served)
    second = create_app(settings, loader=lambda s, a: served)

    assert first.state.metrics.registry is not second.state.metrics.registry


# ── JSON log format ───────────────────────────────────────────


def _record(**extra: object) -> logging.LogRecord:
    record = logging.LogRecord("brent.test", logging.WARNING, __file__, 1, "hello %s", ("x",), None)
    for key, value in extra.items():
        setattr(record, key, value)
    return record


def test_json_records_have_the_standard_fields_and_extras() -> None:
    line = JsonFormatter().format(_record(status=503, path=Path("/a"), request_id="r1"))

    payload = json.loads(line)
    assert payload["level"] == "WARNING"
    assert payload["logger"] == "brent.test"
    assert payload["message"] == "hello x"
    assert payload["status"] == 503
    assert payload["path"] == "/a"  # non-JSON values are stringified
    assert payload["request_id"] == "r1"
    assert payload["time"].endswith("+00:00")


def test_json_records_without_request_id_or_with_exceptions() -> None:
    try:
        raise ValueError("boom")
    except ValueError:
        import sys

        record = _record(request_id=None)
        record.exc_info = sys.exc_info()

    payload = json.loads(JsonFormatter().format(record))

    assert "request_id" not in payload
    assert "ValueError: boom" in payload["exception"]
