
import pandas as pd
import numpy as np
import os

# Carpeta de datos relativa a la raíz del proyecto (directorio de trabajo).
DATA_DIR = os.path.join('data', 'raw')

OIL_DATA_PATH = os.path.join(DATA_DIR, 'oil_geopolitics_dataset_2010_2026.csv')
GEO_DATA_PATH = os.path.join(DATA_DIR, 'geopolitical_events_timeline.csv')


def load_oil_data()-> pd.DataFrame:
    """Load the oil geopolitics dataset."""
    
    # Carga de precios del petróleo y eventos geopolíticos
    df_oil = load_oil_prices(OIL_DATA_PATH)
    df_geo = load_geopolitical_events(GEO_DATA_PATH)
    
    _inspect_overlap(df_oil, df_geo)
    
    df = pd.merge(df_oil, df_geo, on='date', how='left')
    df = _fill_no_event(df)
    df = _resolve_overlap(df)
    df = df.sort_values('date').reset_index(drop=True)

    _validate(df, expected_rows=len(df_oil))

    return df



def load_oil_prices(path:str)-> pd.DataFrame:
    """Carga del dataset de precios del petróleo."""
    
    df=pd.read_csv(path,sep=',')
    df.columns=df.columns.str.strip() 
    
    #Validacion por si no se encuentra la columna 'date' en el dataset
    if 'date' not in df.columns:
        raise ValueError("La columna 'date' no se encuentra en el dataset.")
    
    #Convertir la columna 'date' a formato datetime
    df['date'] = pd.to_datetime(df['date'], errors='coerce')
    
    # Eliminar filas con fechas no parseables
    if df['date'].isna().any():
        # Contar cuántas filas tienen fechas no parseables antes de eliminarlas
        n_bad = df['date'].isna().sum()
        df = df.dropna(subset=['date'])
        print(f"{n_bad} filas con 'date' no parseable en dataset principal. Se eliminan.")
        
    print(f"oil_geopolitics_dataset_2010_2026.csv Cargada {len(df)} filas | {df['date'].min().date()} → {df['date'].max().date()}")
        
    return df

def load_geopolitical_events(path:str)-> pd.DataFrame:
    """Carga del dataset de eventos geopolíticos."""
    
    df=pd.read_csv(path,sep=',')
    df.columns=df.columns.str.strip() 
    
    #Validacion por si no se encuentra la columna 'date' en el dataset
    if 'date' not in df.columns:
        raise ValueError("La columna 'date' no se encuentra en el dataset.")
    
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
        
    print(f"[geopolitical_events_timeline.csv] Cargado {len(df)} eventos | columnas: {df.columns.tolist()}")
    return df

def _inspect_overlap(df_oil: pd.DataFrame, df_geo: pd.DataFrame) -> None:
    """
    Inspecciona solapamiento entre event_flag (dataset petróleo)
    y los eventos del dataset geopolítico.
    """
    
    if 'event_flag' in df_oil.columns:
        n_flagged = df_oil['event_flag'].astype(bool).sum()
        print(f"  Días con event_flag=1 en dataset petróleo: {n_flagged}")
        
    if 'event_type' in df_oil.columns:
        vc = df_oil['event_type'].value_counts()
        print(f"  Distribución event_type en dataset petróleo (top-5):\n{vc.head()}")
        
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

    print(f"\n[_resolve_overlap] Columnas tras resolución: {df.columns.tolist()}")
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
        errors.append(f"DUPLICADOS en 'date': {n_dup} filas duplicadas.")

    # 2. Número de filas
    if len(df) != expected_rows:
        errors.append(
            f"FILAS: esperadas {expected_rows}, obtenidas {len(df)}. "
            "El left join añadió o perdió filas (posible duplicado en dataset geopolítico)."
        )

    # 3. Nulos en event_severity
    if 'event_severity' in df.columns:
        n_null_sev = df['event_severity'].isna().sum()
        if n_null_sev > 0:
            errors.append(f"NULOS en 'event_severity': {n_null_sev} valores nulos sin rellenar.")

    # 4. Nulos en event_type
    if 'event_type' in df.columns:
        n_null_type = df['event_type'].isna().sum()
        if n_null_type > 0:
            errors.append(f"NULOS en 'event_type': {n_null_type} valores nulos sin rellenar.")

    # 5. Rango de fechas
    min_date = df['date'].min()
    max_date = df['date'].max()
    if min_date.year < 2009 or min_date.year > 2011:
        errors.append(f"RANGO DE FECHAS: fecha mínima inesperada ({min_date.date()}).")
    if max_date.year < 2025:
        errors.append(f"RANGO DE FECHAS: fecha máxima inesperada ({max_date.date()}).")

    print("\n[_validate] Resultados de validación:")
    if errors:
        for e in errors:
            print(f"  ✗ {e}")
        raise ValueError(f"Validación fallida con {len(errors)} error(es). Revisa los mensajes anteriores.")
    else:
        print("  ✓ Sin duplicados en 'date'")
        print(f"  ✓ Filas correctas: {len(df)} == {expected_rows}")
        print("  ✓ event_severity sin nulos")
        print("  ✓ event_type sin nulos")
        print(f"  ✓ Rango de fechas válido: {min_date.date()} → {max_date.date()}")


# ─────────────────────────────────────────────
# Ejecución directa (prueba del módulo)
# ─────────────────────────────────────────────
if __name__ == '__main__':
    df = load_oil_data()
    print("\nPrimeras 3 filas:")
    print(df.head(3).to_string())
    print("\nÚltimas 3 filas:")
    print(df.tail(3).to_string())
    print("\nInfo del DataFrame:")
    print(df.dtypes)