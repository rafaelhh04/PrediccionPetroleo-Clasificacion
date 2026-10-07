
import logging
from pathlib import Path

import pandas as pd
import numpy as np

logger = logging.getLogger(__name__)


def load_oil_data(oil_path: Path, events_path: Path) -> pd.DataFrame:
    """Load the oil geopolitics dataset and merge the geopolitical events."""
    
    # Carga de precios del petróleo y eventos geopolíticos
    df_oil = load_oil_prices(oil_path)
    df_geo = load_geopolitical_events(events_path)
    
    _inspect_overlap(df_oil, df_geo)
    
    df = pd.merge(df_oil, df_geo, on='date', how='left')
    df = _fill_no_event(df)
    df = _resolve_overlap(df)
    df = df.sort_values('date').reset_index(drop=True)

    _validate(df, expected_rows=len(df_oil))

    return df



def load_oil_prices(path: Path) -> pd.DataFrame:
    """Carga del dataset de precios del petróleo."""
    
    df=pd.read_csv(path,sep=',')
    df.columns=df.columns.str.strip() 
    
    #Validacion por si no se encuentra la columna 'date' en el dataset
    if 'date' not in df.columns:
        raise ValueError(f"Column 'date' not found in {path.name}.")
    
    #Convertir la columna 'date' a formato datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Eliminar filas con fechas no parseables
    if df['date'].isna().any():
        # Contar cuántas filas tienen fechas no parseables antes de eliminarlas
        n_bad = df['date'].isna().sum()
        df = df.dropna(subset=['date'])
        logger.warning("Dropping %d rows with an unparseable 'date' in %s", n_bad, path.name)
        
    logger.info(
        "Loaded %s: %d rows | %s -> %s",
        path.name, len(df), df['date'].min().date(), df['date'].max().date(),
    )
        
    return df

def load_geopolitical_events(path: Path) -> pd.DataFrame:
    """Carga del dataset de eventos geopolíticos."""
    
    df=pd.read_csv(path,sep=',')
    df.columns=df.columns.str.strip() 
    
    #Validacion por si no se encuentra la columna 'date' en el dataset
    if 'date' not in df.columns:
        raise ValueError(f"Column 'date' not found in {path.name}.")
    
    #Convertir la columna 'date' a formato datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.dropna(subset=['date'])
    
    # Renombrar columnas que podrían solapar con el dataset de petróleo 
    rename_map = {}
    for col in ['event_type', 'event_description', 'event_severity']:
        if col in df.columns:
            rename_map[col] = f'geo_{col}'
            
    if rename_map:
        df = df.rename(columns=rename_map)
        
    logger.info("Loaded %s: %d events | columns: %s", path.name, len(df), df.columns.tolist())
    return df

def _inspect_overlap(df_oil: pd.DataFrame, df_geo: pd.DataFrame) -> None:
    """
    Inspecciona solapamiento entre event_flag (dataset petróleo)
    y los eventos del dataset geopolítico.
    """
    
    if 'event_flag' in df_oil.columns:
        n_flagged = df_oil['event_flag'].astype(bool).sum()
        logger.info("Days with event_flag=1 in the oil dataset: %d", n_flagged)
        
    if 'event_type' in df_oil.columns:
        vc = df_oil['event_type'].value_counts()
        logger.info("event_type distribution in the oil dataset (top 5):\n%s", vc.head())
        
def _fill_no_event(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rellena con valores neutros los días sin evento geopolítico
    """
    geo_cols = [c for c in df.columns if c.startswith('geo_')]
    for col in geo_cols:
        if 'severity' in col:
            df[col] = df[col].fillna(0)
        elif 'type' in col or 'description' in col:
            df[col] = df[col].fillna('none')
        else:
            df[col] = df[col].fillna('none')
    return df

def _resolve_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resuelve solapamiento entre event_flag y eventos geopolíticos.
    Si event_flag=1 pero no hay evento geopolítico, se asigna un evento genérico.
    """
     
    if 'geo_event_type' in df.columns:
        # Si en el dataset petróleo ya había una columna 'event_type', la renombramos
        if 'event_type' in df.columns:
            df = df.rename(columns={'event_type': 'oil_event_type'})
        df = df.rename(columns={'geo_event_type': 'event_type'})

    if 'geo_event_severity' in df.columns:
        if 'event_severity' in df.columns:
            df = df.rename(columns={'event_severity': 'oil_event_severity'})
        df = df.rename(columns={'geo_event_severity': 'event_severity'})

    
    if 'geo_event_description' in df.columns:
        if 'event_description' in df.columns:
            df = df.rename(columns={'event_description': 'oil_event_description'})  # ← añadir
        df = df.rename(columns={'geo_event_description': 'event_description'})

    logger.info("Columns after resolving the overlap: %s", df.columns.tolist())
    return df
def _validate(df: pd.DataFrame, expected_rows: int) -> None:
    """
    Valida la integridad del DataFrame resultante del merge.

    Comprueba:
    1. Sin duplicados en 'date'
    2. Número de filas = dataset principal (left join no añade filas)
    3. Nulos en event_severity rellenados con 0
    4. Rango de fechas 2010–2026
    """
    errors = []

    # 1. Duplicados en date
    n_dup = df['date'].duplicated().sum()
    if n_dup > 0:
        errors.append(f"DUPLICATES in 'date': {n_dup} duplicated rows.")

    # 2. Número de filas
    if len(df) != expected_rows:
        errors.append(
            f"ROWS: expected {expected_rows}, got {len(df)}. "
            "The left join added or lost rows (possible duplicate in the events dataset)."
        )

    # 3. Nulos en event_severity
    if 'event_severity' in df.columns:
        n_null_sev = df['event_severity'].isna().sum()
        if n_null_sev > 0:
            errors.append(f"NULLS in 'event_severity': {n_null_sev} unfilled values.")

    # 4. Nulos en event_type
    if 'event_type' in df.columns:
        n_null_type = df['event_type'].isna().sum()
        if n_null_type > 0:
            errors.append(f"NULLS in 'event_type': {n_null_type} unfilled values.")

    # 5. Rango de fechas
    min_date = df['date'].min()
    max_date = df['date'].max()
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
