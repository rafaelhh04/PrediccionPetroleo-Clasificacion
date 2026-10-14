"""Tests for scripts/smoke_test.py, the post-deployment check of the CD workflow (no network)."""

import importlib.util
import json
import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
from fastapi.testclient import TestClient

from brent_forecast.api import create_app
from brent_forecast.api.service import ServedModel
from brent_forecast.config import Settings
from brent_forecast.models.baselines import PersistenceClassifier
from brent_forecast.pipeline import prepare_data

SCRIPT = Path(__file__).parents[1] / "scripts" / "smoke_test.py"


@pytest.fixture(scope="module")
def smoke() -> ModuleType:
    spec = importlib.util.spec_from_file_location("smoke_test", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _fetch_from(client: TestClient, smoke: ModuleType) -> Any:
    """A ``fetch`` that answers from the in-process app instead of the network."""

    def fetch(url: str, *, method: str = "GET", payload: Any = None, token: Any = None) -> Any:
        path = url.removeprefix("http://api")
        response = client.request(method, path, json=payload)
        return smoke.Response(response.status_code, response.json())

    return fetch


@pytest.fixture
def api(settings: Settings, raw_data_dir: Path) -> Iterator[TestClient]:
    dataset, _ = prepare_data(settings)
    model = PersistenceClassifier().fit(dataset.features, dataset.target)
    served = ServedModel("champion", model, None, {"version": "3", "candidate": "persistence"})
    with TestClient(create_app(settings, loader=lambda s, alias: served)) as client:
        yield client


def test_a_healthy_deployment_passes(smoke: ModuleType, api: TestClient) -> None:
    summary = smoke.run("http://api", fetch=_fetch_from(api, smoke), attempts=1)

    assert summary["version"] == "3"
    assert summary["candidate"] == "persistence"
    assert 0.0 <= summary["proba_up"] <= 1.0


def test_a_deployment_without_a_model_fails_after_retrying(
    smoke: ModuleType, settings: Settings
) -> None:
    def no_model(s: Settings, alias: str) -> ServedModel:
        raise RuntimeError("registry empty")

    sleeps: list[float] = []
    with (
        TestClient(create_app(settings, loader=no_model)) as client,
        pytest.raises(smoke.SmokeTestError, match="never reported a loaded model"),
    ):
        smoke.run(
            "http://api", fetch=_fetch_from(client, smoke), attempts=3, delay=5, sleep=sleeps.append
        )
    assert sleeps == [5, 5]  # no sleep after the last attempt


def test_readiness_is_polled_until_the_model_is_loaded(smoke: ModuleType) -> None:
    answers = iter(
        [
            OSError("connection reset"),  # cold start
            smoke.Response(503, {"model_loaded": False}),
            smoke.Response(200, {"model_loaded": True, "version": "1"}),
        ]
    )

    def fetch(url: str, **_: Any) -> Any:
        answer = next(answers)
        if isinstance(answer, Exception):
            raise answer
        return answer

    ready = smoke.wait_until_ready("http://api", fetch=fetch, attempts=5, sleep=lambda _: None)

    assert ready.body["version"] == "1"


@pytest.mark.parametrize(
    ("path", "response", "message"),
    [
        ("/model-info", (200, {"candidate": None}), "no candidate"),
        ("/predict", (503, {"detail": "no data"}), "/predict returned 503"),
        ("/predict", (200, {"proba_up": 1.5}), "invalid proba_up"),
        ("/predict", (200, {"proba_up": True}), "invalid proba_up"),
    ],
)
def test_a_broken_endpoint_fails_the_smoke_test(
    smoke: ModuleType, path: str, response: tuple[int, Any], message: str
) -> None:
    healthy = {
        "/health/live": (200, {"status": "ok"}),
        "/health": (200, {"model_loaded": True, "version": "1"}),
        "/model-info": (200, {"candidate": "persistence"}),
        "/predict": (200, {"proba_up": 0.5, "as_of": "2026-03-11"}),
    }
    answers = {**healthy, path: response}

    def fetch(url: str, *, method: str = "GET", payload: Any = None, **_: Any) -> Any:
        route = url.removeprefix("http://api")
        if route == "/predict" and payload is not None:  # the invalid request
            return smoke.Response(422, {"detail": []})
        return smoke.Response(*answers[route])

    with pytest.raises(smoke.SmokeTestError, match=message):
        smoke.run("http://api", fetch=fetch, attempts=1)


def test_invalid_input_must_be_rejected(smoke: ModuleType) -> None:
    def fetch(url: str, *, method: str = "GET", payload: Any = None, **_: Any) -> Any:
        if url.endswith("/health"):
            return smoke.Response(200, {"model_loaded": True})
        if url.endswith("/model-info"):
            return smoke.Response(200, {"candidate": "x"})
        return smoke.Response(200, {"proba_up": 0.4})  # accepts the invalid body too

    with pytest.raises(smoke.SmokeTestError, match="expected 422"):
        smoke.run("http://api", fetch=fetch, attempts=1)


# ── the HTTP client itself, against a local server ────────────


class _Handler(BaseHTTPRequestHandler):
    seen: list[dict[str, Any]] = []

    def _answer(self) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length).decode() if length else None
        type(self).seen.append(
            {"method": self.command, "auth": self.headers.get("Authorization"), "body": body}
        )
        if self.path == "/text":
            payload, status = b"not json", 200
        elif self.path == "/missing":
            payload, status = json.dumps({"detail": "nope"}).encode(), 404
        else:
            payload, status = json.dumps({"ok": True}).encode(), 200
        self.send_response(status)
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self) -> None:  # noqa: N802 - http.server's naming
        self._answer()

    def do_POST(self) -> None:  # noqa: N802
        self._answer()

    def log_message(self, *args: Any) -> None:  # keep pytest output clean
        pass


@pytest.fixture
def server() -> Iterator[str]:
    _Handler.seen = []
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), _Handler)
    thread = threading.Thread(target=httpd.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def test_request_decodes_json_and_http_errors(smoke: ModuleType, server: str) -> None:
    ok = smoke.request(f"{server}/ok", method="POST", payload={"a": 1}, token="t0k")
    missing = smoke.request(f"{server}/missing")
    text = smoke.request(f"{server}/text")

    assert (ok.status, ok.body) == (200, {"ok": True})
    assert (missing.status, missing.body) == (404, {"detail": "nope"})
    assert (text.status, text.body) == (200, None)
    assert _Handler.seen[0] == {"method": "POST", "auth": "Bearer t0k", "body": '{"a": 1}'}


def test_main_reports_success_and_failure(
    smoke: ModuleType, server: str, capsys: pytest.CaptureFixture[str]
) -> None:
    # The stub answers {"ok": true} everywhere: live is 200 but readiness never is.
    assert smoke.main([server, "--attempts", "1", "--delay", "0"]) == 1
    assert "Smoke test FAILED" in capsys.readouterr().err
    assert smoke.main(["http://127.0.0.1:9", "--attempts", "1"]) == 1  # connection refused
