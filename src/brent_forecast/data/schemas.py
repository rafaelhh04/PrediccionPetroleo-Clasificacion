"""Pandera schemas: the data contracts of every pipeline boundary.

- :data:`OIL_SCHEMA`: the daily prices CSV (after parsing ``date``).
- :data:`EVENTS_SCHEMA`: the geopolitical events timeline.
- :func:`merged_schema`: the merged frame returned by ``load_oil_data``.
- :data:`FEATURES_SCHEMA`: the model-ready dataset (``build_dataset`` /
  ``brent featurize``) and any frame sent to a model for inference.

Ranges are deliberately loose *plausibility* bounds (they catch unit errors,
wrong files and corrupted values, not market moves): Brent has never left
(0, 1000) USD, while WTI settled at −37 USD on 2020-04-20, so WTI prices and
returns are not range-checked. Columns that the pipeline forward-fills or that
start with a warm-up gap (VIX, DXY, lags, returns) may be null.

Every validation is *lazy*: all failures are collected and raised together as a
:class:`DataValidationError` with one readable line per failed check.
"""

from typing import Any

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.errors import SchemaErrors

PRICE_RANGE = (0.0, 1000.0)
"""Brent price and lags, USD (exclusive bounds)."""
RETURN_RANGE = (-100.0, 100.0)
"""Daily Brent return in percent: a fall of 100 % or more is impossible."""
SEVERITY_RANGE = (0, 10)


class DataValidationError(ValueError):
    """A frame broke its schema; the message lists every failed check."""


def _optional(*checks: pa.Check) -> pa.Column:
    return pa.Column(float, checks=list(checks), nullable=True, coerce=True)


def _unique_increasing_dates() -> pa.Check:
    return pa.Check(
        lambda df: df["date"].is_monotonic_increasing,
        error="dates must be sorted in increasing order",
    )


OIL_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, unique=True, nullable=False),
        "brent_price": pa.Column(
            float, pa.Check.in_range(*PRICE_RANGE, include_min=False), coerce=True
        ),
        "wti_price": _optional(),
        "dxy_index": _optional(pa.Check.in_range(50.0, 200.0)),
        "vix": _optional(pa.Check.in_range(0.0, 200.0, include_min=False)),
        "gpr_index": _optional(pa.Check.ge(0.0)),
        "brent_return": _optional(pa.Check.in_range(*RETURN_RANGE, include_min=False)),
        "wti_return": _optional(),
        **{
            f"brent_lag_{k}": _optional(pa.Check.in_range(*PRICE_RANGE, include_min=False))
            for k in (1, 3, 7)
        },
        **{f"wti_lag_{k}": _optional() for k in (1, 3, 7)},
        "brent_volatility_7d": _optional(pa.Check.ge(0.0)),
        "brent_volatility_30d": _optional(pa.Check.ge(0.0)),
        "brent_wti_spread": _optional(),
    },
    strict=False,  # extra columns (e.g. WTI volatilities, oil event text) are allowed
    name="oil prices",
)

EVENTS_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, unique=True, nullable=False),
        "event_type": pa.Column(str, nullable=False, required=False),
        "event_severity": pa.Column(
            float, pa.Check.in_range(*SEVERITY_RANGE), coerce=True, required=False
        ),
    },
    strict=False,
    name="geopolitical events",
)


def merged_schema(
    expected_rows: int, start_min: Any, start_max: Any, end_min: Any
) -> pa.DataFrameSchema:
    """Build the merged-frame schema: one sorted row per oil day, events filled, sane range."""
    return pa.DataFrameSchema(
        {
            "date": pa.Column(pa.DateTime, unique=True, nullable=False),
            "event_type": pa.Column(str, nullable=False),
            "event_severity": pa.Column(
                float, pa.Check.in_range(*SEVERITY_RANGE), nullable=False, coerce=True
            ),
        },
        checks=[
            _unique_increasing_dates(),
            pa.Check(
                lambda df: len(df) == expected_rows,
                error=f"the left join must keep {expected_rows} rows (one per oil day)",
            ),
            pa.Check(
                lambda df: start_min <= df["date"].min().date() <= start_max,
                error=f"first date must be between {start_min} and {start_max}",
            ),
            pa.Check(
                lambda df: df["date"].max().date() >= end_min,
                error=f"last date must be on or after {end_min}",
            ),
        ],
        strict=False,
        name="merged dataset",
    )


_FEATURE_COLUMNS = r"^(?!date$|label$|next_return$).+$"


def _feature_column() -> pa.Column:
    """Every column that is not metadata is a feature: numeric, present and finite."""
    return pa.Column(
        float,
        pa.Check(lambda s: np.isfinite(s), error="must be finite"),
        nullable=False,
        coerce=True,
        regex=True,
    )


FEATURES_SCHEMA = pa.DataFrameSchema(
    {
        "date": pa.Column(pa.DateTime, unique=True, nullable=False),
        "label": pa.Column(float, pa.Check.isin([0.0, 1.0]), nullable=False, coerce=True),
        "next_return": pa.Column(float, nullable=False, coerce=True),
        _FEATURE_COLUMNS: _feature_column(),
    },
    checks=[_unique_increasing_dates()],
    strict=False,
    name="feature dataset",
)

INFERENCE_SCHEMA = pa.DataFrameSchema(
    {_FEATURE_COLUMNS: _feature_column()}, strict=False, name="inference features"
)
"""Feature rows sent to a model to predict: no date, label or outcome required."""


def validate(frame: pd.DataFrame, schema: pa.DataFrameSchema) -> pd.DataFrame:
    """Validate ``frame`` lazily; return the (coerced) frame.

    Raises
    ------
    DataValidationError
        Listing every failed check, e.g. ``oil prices: column 'brent_price' failed
        in_range(0.0, 1000.0) for 2 row(s), e.g. -1.0``.
    """
    try:
        return schema.validate(frame, lazy=True)
    except SchemaErrors as exc:
        raise DataValidationError(_describe(schema.name, exc.failure_cases)) from None


def _describe(name: str | None, cases: pd.DataFrame) -> str:
    lines = [f"{name or 'frame'} failed validation:"]
    grouped = cases.groupby(["schema_context", "column", "check"], dropna=False, sort=False)
    for (context, column, check), group in grouped:
        if check == "column_in_dataframe":
            missing = ", ".join(map(repr, group["failure_case"].tolist()))
            lines.append(f"  - missing required column(s): {missing}")
            continue
        where = f"column {column!r}" if context == "Column" else "frame"
        examples = ", ".join(map(str, group["failure_case"].head(3).tolist()))
        lines.append(f"  - {where} failed {check} for {len(group)} row(s), e.g. {examples}")
    return "\n".join(lines)
