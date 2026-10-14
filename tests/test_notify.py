"""Tests for brent_forecast.notify (chat webhook, no network)."""

import json
import logging
import urllib.error

import pytest

from brent_forecast.config import Settings, load_settings
from brent_forecast.notify import format_message, http_post, notify
from conftest import CONFIG_PATH


class FakePost:
    def __init__(self, status: int = 200, error: Exception | None = None) -> None:
        self.status, self.error = status, error
        self.calls: list[tuple[str, dict[str, str], float]] = []

    def __call__(self, url: str, body: bytes, timeout: float) -> int:
        self.calls.append((url, json.loads(body), timeout))
        if self.error:
            raise self.error
        return self.status


def test_the_message_goes_to_slack_and_discord_fields() -> None:
    post = FakePost()

    delivered = notify("https://hooks.example/x", "Retraining", ["a", "b"], timeout=3, post=post)

    assert delivered
    url, body, timeout = post.calls[0]
    assert (url, timeout) == ("https://hooks.example/x", 3)
    assert body["text"] == body["content"] == "*Retraining*\n• a\n• b"


def test_without_a_url_the_message_is_only_logged(caplog: pytest.LogCaptureFixture) -> None:
    post = FakePost()
    with caplog.at_level(logging.INFO, logger="brent_forecast.notify"):
        assert not notify(None, "Drift", ["x"], post=post)
    assert not post.calls
    assert "*Drift*" in caplog.text


@pytest.mark.parametrize(
    "post",
    [
        FakePost(status=404),
        FakePost(error=urllib.error.URLError("unreachable")),
        FakePost(error=TimeoutError()),
    ],
)
def test_a_failed_notification_never_raises(
    post: FakePost, caplog: pytest.LogCaptureFixture
) -> None:
    with caplog.at_level(logging.WARNING, logger="brent_forecast.notify"):
        assert not notify("https://hooks.example/x", "t", post=post)
    assert "Notification" in caplog.text


def test_format_message_without_lines() -> None:
    assert format_message("Only a title", []) == "*Only a title*"


def test_http_post_reports_the_status(monkeypatch: pytest.MonkeyPatch) -> None:
    class Response:
        status = 204

        def __enter__(self) -> "Response":
            return self

        def __exit__(self, *args: object) -> None:
            pass

    seen = {}

    def fake_urlopen(request: object, timeout: float) -> Response:
        seen["request"], seen["timeout"] = request, timeout
        return Response()

    monkeypatch.setattr("urllib.request.urlopen", fake_urlopen)

    assert http_post("https://hooks.example/x", b"{}", 2.0) == 204
    assert seen["timeout"] == 2.0
    assert seen["request"].get_header("Content-type") == "application/json"  # type: ignore[attr-defined]


def test_the_webhook_url_is_masked_in_dumps(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("BRENT_NOTIFY__WEBHOOK_URL", "https://hooks.example/secret-token")
    settings: Settings = load_settings(CONFIG_PATH)

    dumped = json.dumps(settings.model_dump(mode="json"))

    assert "secret-token" not in dumped
    assert settings.notify.webhook_url is not None
    assert settings.notify.webhook_url.get_secret_value().endswith("secret-token")
