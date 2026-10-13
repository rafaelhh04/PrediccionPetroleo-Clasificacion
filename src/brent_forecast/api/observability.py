"""Observability of the API: request ids, structured access logs and Prometheus metrics.

- **Request id.** Every request gets an id: the client's ``X-Request-ID`` when it
  is a safe token, a new UUID otherwise. It is returned in the same header and
  attached to every log record emitted while the request is handled
  (:data:`brent_forecast.logging_config.request_id`).
- **Access log.** One record per request (``brent_forecast.api.access``) with the
  method, path, status and duration as structured fields; probes and scrapes are
  logged at DEBUG.
- **Metrics** (``GET /metrics``, Prometheus text format). HTTP metrics come from
  ``prometheus-fastapi-instrumentator``. The service adds:
  ``brent_predictions_total{alias,direction,source}``,
  ``brent_model_inference_seconds{alias}`` (validation + ``predict_proba``),
  ``brent_model_loaded{alias}``, ``brent_model_info{alias,version,candidate,kind}``
  and ``brent_model_load_failures_total{alias}``. Each app has its own registry,
  so several apps (tests) never collide.
"""

import logging
import re
import time
import uuid
from collections.abc import Iterator
from typing import Any

from prometheus_client import CollectorRegistry, Counter, Histogram
from prometheus_client.core import CounterMetricFamily, GaugeMetricFamily, Metric
from prometheus_client.registry import Collector
from starlette.types import ASGIApp, Message, Receive, Scope, Send

from brent_forecast.api.service import ModelService
from brent_forecast.logging_config import request_id

REQUEST_ID_HEADER = "X-Request-ID"
_SAFE_ID = re.compile(r"^[A-Za-z0-9._:-]{1,128}$")
_QUIET_PATHS = frozenset({"/health", "/health/live", "/metrics"})

access_logger = logging.getLogger("brent_forecast.api.access")

INFERENCE_BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0)
"""Seconds: a prediction takes ~20-40 ms, dominated by the Pandera validation."""


def new_request_id(candidate: str | None) -> str:
    """Return ``candidate`` if it is a safe token (no header injection), else a new UUID."""
    if candidate and _SAFE_ID.fullmatch(candidate):
        return candidate
    return uuid.uuid4().hex


class RequestContextMiddleware:
    """Pure ASGI middleware: request id in context and response, plus the access log."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        """Handle one request (other ASGI scopes pass through untouched)."""
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = dict(scope["headers"])
        rid = new_request_id(headers.get(REQUEST_ID_HEADER.lower().encode(), b"").decode("latin-1"))
        token = request_id.set(rid)
        status = 500
        start = time.perf_counter()

        async def send_with_id(message: Message) -> None:
            nonlocal status
            if message["type"] == "http.response.start":
                status = message["status"]
                message.setdefault("headers", [])
                message["headers"].append((REQUEST_ID_HEADER.lower().encode(), rid.encode()))
            await send(message)

        try:
            await self.app(scope, receive, send_with_id)
        finally:
            path = scope["path"]
            access_logger.log(
                logging.DEBUG if path in _QUIET_PATHS else logging.INFO,
                "%s %s %d",
                scope["method"],
                path,
                status,
                extra={
                    "method": scope["method"],
                    "path": path,
                    "status": status,
                    "duration_ms": round((time.perf_counter() - start) * 1000, 2),
                },
            )
            request_id.reset(token)


class _ServiceCollector(Collector):
    """Model state, read from the service at scrape time (the service knows no Prometheus)."""

    def __init__(self, service: ModelService) -> None:
        self.service = service

    def collect(self) -> Iterator[Metric]:
        """Yield the loaded flag, the model identity and the load failures."""
        service, alias = self.service, self.service.alias
        served = service.served
        loaded = GaugeMetricFamily(
            "brent_model_loaded", "1 when a model is loaded for the alias.", labels=["alias"]
        )
        loaded.add_metric([alias], 1.0 if served else 0.0)
        yield loaded
        info = GaugeMetricFamily(
            "brent_model_info",
            "Registry version and candidate of the served model (value 1).",
            labels=["alias", "version", "candidate", "kind"],
        )
        if served is not None:
            meta = served.info
            labels = [meta.get(k, "unknown") for k in ("version", "candidate", "kind")]
            info.add_metric([alias, *labels], 1.0)
        yield info
        failures = CounterMetricFamily(
            "brent_model_load_failures",
            "Failed attempts to load the model from the registry.",
            labels=["alias"],
        )
        failures.add_metric([alias], float(service.load_failures))
        yield failures


class ServingMetrics:
    """Custom metrics of the inference service, in a registry owned by one app."""

    def __init__(self, service: ModelService, registry: CollectorRegistry | None = None) -> None:
        self.registry = registry or CollectorRegistry()
        self.predictions = Counter(
            "brent_predictions",
            "Predictions served, by direction and feature source.",
            ["alias", "direction", "source"],
            registry=self.registry,
        )
        self.inference_seconds = Histogram(
            "brent_model_inference_seconds",
            "Model latency per prediction: input validation and predict_proba.",
            ["alias"],
            buckets=INFERENCE_BUCKETS,
            registry=self.registry,
        )
        self.registry.register(_ServiceCollector(service))

    def observe(self, alias: str, direction: str, source: str, seconds: float) -> None:
        """Record one prediction."""
        self.predictions.labels(alias=alias, direction=direction, source=source).inc()
        self.inference_seconds.labels(alias=alias).observe(seconds)


def instrument(app: Any, metrics: ServingMetrics) -> None:
    """Add the HTTP metrics and ``GET /metrics`` (excluded from the OpenAPI contract)."""
    from prometheus_fastapi_instrumentator import Instrumentator

    Instrumentator(
        registry=metrics.registry,
        excluded_handlers=["/metrics"],
        should_group_status_codes=False,
    ).instrument(app).expose(app, endpoint="/metrics", include_in_schema=False)
