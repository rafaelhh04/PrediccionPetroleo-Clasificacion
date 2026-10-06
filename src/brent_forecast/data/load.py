"""Load the oil market dataset and merge it with the geopolitical events timeline."""

import logging
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

_EVENT_COLUMNS = ("event_type", "event_description", "event_severity")


def load_oil_data(oil_path: Path, events_path: Path) -> pd.DataFrame:
    """Load both datasets and left-join the geopolitical events on ``date``.

    After the merge, ``event_type``, ``event_description`` and
    ``event_severity`` hold the geopolitical events (days without an event get
    ``"none"`` / ``0``); the homonymous columns of the oil dataset are kept as
    ``oil_event_*``.

    Parameters
    ----------
    oil_path
        CSV with daily prices and market indicators.
    events_path
        CSV with the geopolitical events timeline.

    Returns
    -------
    pandas.DataFrame
        One row per trading day, sorted by date.

    Raises
    ------
    ValueError
        If either file lacks a ``date`` column or the merged frame fails validation.
    """
    df_oil = load_oil_prices(oil_path)
    df_geo = load_geopolitical_events(events_path)

    _inspect_overlap(df_oil)

    df = pd.merge(df_oil, df_geo, on="date", how="left")
    df = _fill_no_event(df)
    df = _resolve_overlap(df)
    df = df.sort_values("date").reset_index(drop=True)

    _validate(df, expected_rows=len(df_oil))

    return df


def load_oil_prices(path: Path) -> pd.DataFrame:
    """Load the daily oil prices dataset, dropping rows with an unparsable date.

    Parameters
    ----------
    path
        CSV file to read.

    Returns
    -------
    pandas.DataFrame
        Dataset with ``date`` parsed as datetime.

    Raises
    ------
    ValueError
        If the file has no ``date`` column.
    """
    df = pd.read_csv(path, sep=",")
    df.columns = df.columns.str.strip()

    if "date" not in df.columns:
        raise ValueError(f"Column 'date' not found in {path.name}.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        n_bad = df["date"].isna().sum()
        df = df.dropna(subset=["date"])
        logger.warning("Dropping %d rows with an unparsable 'date' in %s", n_bad, path.name)

    logger.info(
        "Loaded %s: %d rows | %s -> %s",
        path.name,
        len(df),
        df["date"].min().date(),
        df["date"].max().date(),
    )

    return df


def load_geopolitical_events(path: Path) -> pd.DataFrame:
    """Load the geopolitical events timeline with its columns prefixed by ``geo_``.

    The prefix avoids clashing with the homonymous event columns of the oil
    dataset during the merge.

    Parameters
    ----------
    path
        CSV file to read.

    Returns
    -------
    pandas.DataFrame
        Events with ``date`` parsed as datetime.

    Raises
    ------
    ValueError
        If the file has no ``date`` column.
    """
    df = pd.read_csv(path, sep=",")
    df.columns = df.columns.str.strip()

    if "date" not in df.columns:
        raise ValueError(f"Column 'date' not found in {path.name}.")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"])

    rename_map = {col: f"geo_{col}" for col in _EVENT_COLUMNS if col in df.columns}
    if rename_map:
        df = df.rename(columns=rename_map)

    logger.info("Loaded %s: %d events | columns: %s", path.name, len(df), df.columns.tolist())
    return df


def _inspect_overlap(df_oil: pd.DataFrame) -> None:
    """Log how many days the oil dataset already flags as event days."""
    if "event_flag" in df_oil.columns:
        n_flagged = df_oil["event_flag"].astype(bool).sum()
        logger.info("Days with event_flag=1 in the oil dataset: %d", n_flagged)

    if "event_type" in df_oil.columns:
        vc = df_oil["event_type"].value_counts()
        logger.info("event_type distribution in the oil dataset (top 5):\n%s", vc.head())


def _fill_no_event(df: pd.DataFrame) -> pd.DataFrame:
    """Fill days without a geopolitical event with neutral values (0 / ``"none"``)."""
    geo_cols = [c for c in df.columns if c.startswith("geo_")]
    for col in geo_cols:
        if "severity" in col:
            df[col] = df[col].fillna(0)
        else:
            df[col] = df[col].fillna("none")
    return df


def _resolve_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """Promote the ``geo_*`` event columns to the canonical event column names.

    Columns of the oil dataset with the same name are renamed to ``oil_*`` so
    that downstream code always reads the geopolitical events.
    """
    for col in _EVENT_COLUMNS:
        if f"geo_{col}" in df.columns:
            if col in df.columns:
                df = df.rename(columns={col: f"oil_{col}"})
            df = df.rename(columns={f"geo_{col}": col})

    logger.info("Columns after resolving the overlap: %s", df.columns.tolist())
    return df


def _validate(df: pd.DataFrame, expected_rows: int) -> None:
    """Check the integrity of the merged frame.

    Checks: no duplicated dates, the left join neither added nor lost rows, no
    nulls in ``event_severity`` / ``event_type`` and a plausible date range.

    Raises
    ------
    ValueError
        If any check fails (every failure is logged first).
    """
    errors = []

    n_dup = df["date"].duplicated().sum()
    if n_dup > 0:
        errors.append(f"DUPLICATES in 'date': {n_dup} duplicated rows.")

    if len(df) != expected_rows:
        errors.append(
            f"ROWS: expected {expected_rows}, got {len(df)}. "
            "The left join added or lost rows (possible duplicate in the events dataset)."
        )

    if "event_severity" in df.columns:
        n_null_sev = df["event_severity"].isna().sum()
        if n_null_sev > 0:
            errors.append(f"NULLS in 'event_severity': {n_null_sev} unfilled values.")

    if "event_type" in df.columns:
        n_null_type = df["event_type"].isna().sum()
        if n_null_type > 0:
            errors.append(f"NULLS in 'event_type': {n_null_type} unfilled values.")

    min_date = df["date"].min()
    max_date = df["date"].max()
    if min_date.year < 2009 or min_date.year > 2011:
        errors.append(f"DATE RANGE: unexpected minimum date ({min_date.date()}).")
    if max_date.year < 2025:
        errors.append(f"DATE RANGE: unexpected maximum date ({max_date.date()}).")

    if errors:
        for e in errors:
            logger.error("Validation failed: %s", e)
        raise ValueError(f"Validation failed with {len(errors)} error(s); see the log above.")
    logger.info("Validation passed: no duplicated dates")
    logger.info("Validation passed: row count %d == %d", len(df), expected_rows)
    logger.info("Validation passed: no nulls in event_severity / event_type")
    logger.info("Validation passed: date range %s -> %s", min_date.date(), max_date.date())
