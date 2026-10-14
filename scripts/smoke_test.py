#!/usr/bin/env python3
"""Post-deployment smoke test of the inference API (standard library only).

    python scripts/smoke_test.py https://candidate---brent-api-prod-xyz.a.run.app

Checks, in order, and exits non-zero on the first failure:

1. ``GET /health/live`` answers 200 (the process is up);
2. ``GET /health`` answers 200 with ``model_loaded: true`` (a model is served),
   retried while the instance is cold or still loading the model;
3. ``GET /model-info`` names the candidate being served;
4. ``POST /predict`` with an empty body returns a probability in [0, 1];
5. ``POST /predict`` with an invalid body is rejected with 422.

Cloud Run may be private (``--id-token``): the token is sent as a bearer token.
"""

import argparse
import json
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Response:
    """Status code and decoded JSON body (``None`` if the body is not JSON)."""

    status: int
    body: Any


class SmokeTestError(RuntimeError):
    """A check failed."""


def request(
    url: str,
    *,
    method: str = "GET",
    payload: Any = None,
    token: str | None = None,
    timeout: float = 30.0,
) -> Response:
    """Send one request and return the status and JSON body (HTTP errors included)."""
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json", "X-Request-ID": "smoke-test"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if token:
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read()
    try:
        body = json.loads(raw) if raw else None
    except json.JSONDecodeError:
        body = None
    return Response(status, body)


Fetch = Callable[..., Response]


def wait_until_ready(
    base: str,
    *,
    fetch: Fetch = request,
    token: str | None = None,
    attempts: int = 12,
    delay: float = 10.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Response:
    """Poll ``/health`` until it is 200 with a loaded model (cold start, first model load)."""
    last = Response(0, None)
    for attempt in range(1, attempts + 1):
        try:
            last = fetch(f"{base}/health", token=token)
        except OSError as exc:  # connection reset, timeout during a cold start
            print(f"  /health attempt {attempt}: {exc}")
        else:
            if last.status == 200 and isinstance(last.body, dict) and last.body.get("model_loaded"):
                return last
            print(f"  /health attempt {attempt}: {last.status} {last.body}")
        if attempt < attempts:
            sleep(delay)
    raise SmokeTestError(f"/health never reported a loaded model (last: {last.status} {last.body})")


def check(condition: bool, message: str) -> None:
    """Raise :class:`SmokeTestError` with ``message`` unless ``condition`` holds."""
    if not condition:
        raise SmokeTestError(message)


def run(
    base: str, *, fetch: Fetch = request, token: str | None = None, **wait: Any
) -> dict[str, Any]:
    """Run every check against ``base`` (no trailing slash); return a summary."""
    live = fetch(f"{base}/health/live", token=token)
    check(live.status == 200, f"/health/live returned {live.status}")

    health = wait_until_ready(base, fetch=fetch, token=token, **wait)

    info = fetch(f"{base}/model-info", token=token)
    check(info.status == 200, f"/model-info returned {info.status}")
    check(isinstance(info.body, dict) and bool(info.body.get("candidate")), "no candidate served")

    prediction = fetch(f"{base}/predict", method="POST", token=token)
    check(prediction.status == 200, f"/predict returned {prediction.status}: {prediction.body}")
    check(isinstance(prediction.body, dict), "/predict did not return a JSON object")
    proba = prediction.body.get("proba_up")
    valid = isinstance(proba, int | float) and not isinstance(proba, bool) and 0.0 <= proba <= 1.0
    check(valid, f"invalid proba_up: {proba!r}")

    invalid = fetch(
        f"{base}/predict", method="POST", payload={"features": {"vix": "high"}}, token=token
    )
    check(invalid.status == 422, f"an invalid body returned {invalid.status}, expected 422")

    return {
        "version": health.body.get("version"),
        "candidate": info.body.get("candidate"),
        "as_of": prediction.body.get("as_of"),
        "proba_up": proba,
    }


def main(argv: list[str] | None = None) -> int:
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("url", help="Base URL of the API (e.g. the tagged revision URL).")
    parser.add_argument("--id-token", help="Bearer token for a private service.")
    parser.add_argument("--attempts", type=int, default=12, help="Readiness polls (default 12).")
    parser.add_argument("--delay", type=float, default=10.0, help="Seconds between polls.")
    args = parser.parse_args(argv)
    try:
        summary = run(
            args.url.rstrip("/"), token=args.id_token, attempts=args.attempts, delay=args.delay
        )
    except (SmokeTestError, OSError) as exc:
        print(f"Smoke test FAILED: {exc}", file=sys.stderr)
        return 1
    print(f"Smoke test passed: {json.dumps(summary)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
