"""FastAPI application: ``POST /predict``, ``GET /health``, ``GET /model-info``.

Status codes:

- **200**: success.
- **422**: invalid input. Pydantic rejects malformed bodies (strings, NaN,
  unknown fields); then the feature names are checked against the model's
  input contract and the row against
  :data:`~brent_forecast.data.schemas.INFERENCE_SCHEMA`. Every error uses
  FastAPI's ``{"detail": [{"loc", "msg", "type"}]}`` format.
- **503**: no model is loaded (``/health`` tells why) or, for a prediction of
  the latest day, the data files are missing or invalid.

Predictions go through :func:`brent_forecast.predict.predict_features`, exactly
like ``brent predict``. Observability (request ids, access log, ``/metrics``) is
described in :mod:`brent_forecast.api.observability`.
"""

import time
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from importlib.metadata import version
from typing import Annotated, Any

import pandas as pd
from fastapi import Body, FastAPI, HTTPException, Request, status
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.concurrency import run_in_threadpool

from brent_forecast.api.observability import (
    RequestContextMiddleware,
    ServingMetrics,
    instrument,
)
from brent_forecast.api.schemas import (
    ErrorResponse,
    HealthResponse,
    LivenessResponse,
    ModelInfoResponse,
    ModelRef,
    PredictRequest,
    PredictResponse,
)
from brent_forecast.api.service import LatestFeatures, Loader, ModelService, ServedModel
from brent_forecast.config import Settings, load_settings
from brent_forecast.data.schemas import DataValidationError
from brent_forecast.predict import Prediction, predict_features

DESCRIPTION = """
Next-day direction of Brent crude oil (up vs. down/flat), served from the MLflow
model registry. The model is the registry alias configured in `serving.alias`
(`champion` by default), which can be a naive **baseline**: a model is only
promoted if it beats the best baseline with statistical significance.

**Not investment advice.** The evaluation found no predictive signal (see the model card).
"""

_NO_MODEL: dict[int | str, dict[str, Any]] = {
    status.HTTP_503_SERVICE_UNAVAILABLE: {"model": ErrorResponse}
}


def _validation_error(loc: list[str | int], msg: str, kind: str) -> HTTPException:
    """Build a 422 in the same format as FastAPI's request validation errors."""
    detail = [{"loc": ["body", *loc], "msg": msg, "type": kind}]
    return HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=detail)


def _check_columns(features: dict[str, float], expected: tuple[str, ...]) -> None:
    missing = [name for name in expected if name not in features]
    unknown = [name for name in features if name not in expected]
    errors: list[dict[str, Any]] = [
        {"loc": ["body", "features", name], "msg": "Field required", "type": "missing"}
        for name in missing
    ]
    errors += [
        {"loc": ["body", "features", name], "msg": "Unknown feature", "type": "extra_forbidden"}
        for name in unknown
    ]
    if errors:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail=errors)


async def _request_validation_handler(_: Request, exc: Exception) -> JSONResponse:
    """Return 422 with ``loc``/``msg``/``type`` only.

    FastAPI echoes the offending ``input`` by default, which cannot be serialised
    when it is NaN or infinity (a 500 instead of a 422) and can be large.
    """
    assert isinstance(exc, RequestValidationError)
    detail = [{k: e[k] for k in ("loc", "msg", "type")} for e in exc.errors()]
    return JSONResponse({"detail": detail}, status_code=status.HTTP_422_UNPROCESSABLE_CONTENT)


def create_app(settings: Settings | None = None, *, loader: Loader | None = None) -> FastAPI:
    """Build the API for ``settings`` (``configs/default.yaml`` + ``BRENT_*`` by default).

    ``loader`` replaces the registry loader (tests use an in-memory model).
    """
    settings = settings or load_settings()
    service = ModelService(settings, loader) if loader else ModelService(settings)
    latest = LatestFeatures(settings)
    threshold = settings.backtest.threshold

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        await run_in_threadpool(service.load)
        yield

    app = FastAPI(
        title="Brent direction API",
        version=version("brent-forecast"),
        summary="Next-day Brent crude oil direction classifier.",
        description=DESCRIPTION,
        lifespan=lifespan,
        license_info={"name": "MIT", "identifier": "MIT"},
    )
    metrics = ServingMetrics(service)
    app.state.service = service
    app.state.metrics = metrics
    app.add_exception_handler(RequestValidationError, _request_validation_handler)
    instrument(app, metrics)
    app.add_middleware(RequestContextMiddleware)  # outermost: ids cover every log and response

    def timed(served: ServedModel, call: Callable[[], Prediction]) -> Prediction:
        start = time.perf_counter()
        prediction = call()
        metrics.observe(
            served.alias, prediction.direction, prediction.source, time.perf_counter() - start
        )
        return prediction

    def served_or_503() -> ServedModel:
        served = service.get()
        if served is None:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"No model loaded for alias {service.alias!r}: {service.error}",
            )
        return served

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse(url="/docs")

    @app.get(
        "/health",
        tags=["health"],
        summary="Readiness: is a model loaded?",
        response_model=HealthResponse,
        responses={status.HTTP_503_SERVICE_UNAVAILABLE: {"model": HealthResponse}},
    )
    def health() -> JSONResponse:
        """200 when the model is loaded and predictions can be served, 503 otherwise."""
        served = service.get()
        body = HealthResponse(
            status="ok" if served else "unavailable",
            model_loaded=served is not None,
            alias=service.alias,
            version=served.info.get("version") if served else None,
            detail=None if served else service.error,
        )
        code = status.HTTP_200_OK if served else status.HTTP_503_SERVICE_UNAVAILABLE
        return JSONResponse(body.model_dump(mode="json"), status_code=code)

    @app.get(
        "/health/live",
        tags=["health"],
        summary="Liveness: is the process up?",
        response_model=LivenessResponse,
    )
    def liveness() -> LivenessResponse:
        """Answer 200 while the server is up (use `/health` for readiness)."""
        return LivenessResponse()

    @app.get(
        "/model-info",
        tags=["model"],
        summary="The served model and its input contract",
        response_model=ModelInfoResponse,
        responses=_NO_MODEL,
    )
    def model_info() -> ModelInfoResponse:
        """Registry alias, version, candidate, promotion reason and expected feature names."""
        served = served_or_503()
        return ModelInfoResponse(
            registered_model=settings.tracking.registered_model,
            alias=served.alias,
            version=served.info.get("version"),
            candidate=served.info.get("candidate"),
            kind=served.info.get("kind"),
            promotion_reason=served.info.get("promotion_reason"),
            training_run=served.info.get("training_run"),
            loaded_at=served.loaded_at,
            threshold=threshold,
            features=list(served.columns) if served.columns is not None else None,
        )

    @app.post(
        "/predict",
        tags=["model"],
        summary="Predict the next-day direction",
        response_model=PredictResponse,
        responses=_NO_MODEL,
    )
    def predict(
        request: Annotated[PredictRequest | None, Body()] = None,
    ) -> PredictResponse:
        """Predict from the features in the body, or from the latest day if none are sent.

        With `features`, every expected feature (see `GET /model-info`) must be present and
        no other: names are checked against the model's input contract.
        """
        served = served_or_503()
        if request is None or request.features is None:
            prediction = _predict_latest(served)
        else:
            prediction = _predict_request(served, request)
        return PredictResponse(
            as_of=prediction.as_of,
            proba_up=prediction.proba_up,
            direction=prediction.direction,
            threshold=threshold,
            source=prediction.source,
            model=ModelRef(
                alias=served.alias,
                version=served.info.get("version"),
                candidate=served.info.get("candidate"),
            ),
        )

    def _predict_latest(served: ServedModel) -> Prediction:
        try:
            as_of, features, source = latest.get()
            return timed(
                served,
                lambda: predict_features(
                    settings, served.model, features, as_of=as_of.date(), source=source
                ),
            )
        except (FileNotFoundError, ValueError) as exc:  # DataValidationError is a ValueError
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Latest features unavailable: {exc}",
            ) from exc

    def _predict_request(served: ServedModel, request: PredictRequest) -> Prediction:
        features = request.features or {}
        if served.columns is not None:
            _check_columns(features, served.columns)
        frame = pd.DataFrame([features], columns=list(served.columns or features))
        try:
            return timed(
                served,
                lambda: predict_features(
                    settings, served.model, frame, as_of=request.as_of, source="request"
                ),
            )
        except DataValidationError as exc:
            raise _validation_error(["features"], str(exc), "value_error") from exc
        except KeyError as exc:  # a model without a recorded contract needs this column
            raise _validation_error(["features"], f"Missing feature {exc}", "missing") from exc

    return app
