"""Request and response models of the inference API (Pydantic v2).

They are the public contract of the service: every field is documented and
shows up in the OpenAPI schema (``/openapi.json``, ``/docs``).
"""

from datetime import date, datetime
from typing import Annotated, Literal, Self

from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from brent_forecast.predict import Direction, Source

FiniteFloat = Annotated[float, Field(strict=True, allow_inf_nan=False)]
"""A JSON number (never a string, a boolean, NaN or infinity)."""

_EXAMPLE_FEATURES: dict[str, JsonValue] = {
    "dxy_index": 103.2,
    "vix": 17.4,
    "brent_volatility_7d": 1.6,
    "brent_volatility_30d": 1.9,
    "brent_wti_spread": 4.1,
    "lag_ret_1": 0.0042,
    "lag_ret_3": -0.011,
    "lag_ret_7": 0.019,
    "gpr_change": -2.5,
    "event_flag_binary": 0.0,
    "high_severity_flag": 0.0,
    "vol_ratio": 0.84,
    "rsi_14": 55.2,
    "macd": 0.31,
    "macd_hist": 0.05,
    "bb_pct_b": 0.62,
    "bb_width": 0.071,
    "mom_21": 0.024,
    "mom_63": -0.035,
    "vix_chg_1": -0.012,
    "vix_chg_5": 0.031,
    "dxy_ret_1": 0.0008,
    "dxy_ret_5": -0.0021,
    "dow_sin": 0.43,
    "dow_cos": -0.90,
    "month_sin": 0.5,
    "month_cos": 0.87,
}


class _Model(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class PredictRequest(_Model):
    """Features of one day, or nothing to predict from the latest available day."""

    model_config = ConfigDict(
        json_schema_extra={
            "examples": [
                {},
                {"features": _EXAMPLE_FEATURES, "as_of": "2026-03-12"},
            ]
        }
    )

    features: dict[str, FiniteFloat] | None = Field(
        default=None,
        description=(
            "Every feature the model expects (see `GET /model-info`), by name. "
            "Omit it to use the most recent day of the data available to the service."
        ),
    )
    as_of: date | None = Field(
        default=None,
        description="Day the features describe (informative; only with `features`).",
    )

    @model_validator(mode="after")
    def _as_of_needs_features(self) -> Self:
        if self.as_of is not None and self.features is None:
            raise ValueError("`as_of` is only accepted together with `features`")
        return self


class ModelRef(_Model):
    """Registry entry that produced a prediction."""

    alias: str = Field(description="Registry alias (`champion` or `challenger`).")
    version: str | None = Field(description="Model version behind the alias.")
    candidate: str | None = Field(description="Candidate key, e.g. `persistence`.")


class PredictResponse(_Model):
    """Probability that Brent closes higher on the trading day after `as_of`."""

    as_of: date | None = Field(description="Last observed day (null if the client sent none).")
    proba_up: float = Field(ge=0.0, le=1.0, description="P(next-day close > today's close).")
    direction: Direction = Field(description="`up` when `proba_up >= threshold`.")
    threshold: float = Field(description="Decision threshold (`backtest.threshold`).")
    source: Source = Field(
        description="Where the features came from: ingested data, the raw snapshot or the client."
    )
    model: ModelRef


class HealthResponse(_Model):
    """Readiness of the service: a model must be loaded to serve predictions."""

    status: Literal["ok", "unavailable"]
    model_loaded: bool
    alias: str
    version: str | None = None
    detail: str | None = Field(default=None, description="Why no model is loaded.")


class LivenessResponse(_Model):
    """The process is up (says nothing about the model)."""

    status: Literal["ok"] = "ok"


class ModelInfoResponse(_Model):
    """The served model, why it was promoted and its input contract."""

    registered_model: str
    alias: str
    version: str | None
    candidate: str | None
    kind: str | None = Field(description="`model` or `baseline`.")
    promotion_reason: str | None
    training_run: str | None = Field(description="MLflow run id of the training run.")
    loaded_at: datetime
    threshold: float
    features: list[str] | None = Field(
        description="Expected feature names, in order (null if the model does not record them)."
    )


class ErrorResponse(_Model):
    """Error returned with a 503 (no model, no data) or a 500."""

    detail: str
