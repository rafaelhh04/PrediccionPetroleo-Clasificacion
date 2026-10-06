
"""
Pipeline de preprocesado para la predicción de la dirección del precio del Brent.

Orden de ejecución (importa):
    1. create_label          → genera label ANTES de cualquier filtrado
    2. engineer_features     → transformaciones y selección de features
    3. handle_nulls          → eliminación de filas con nulos
    4. split_temporal        → división cronológica train/val/test
    5. winsorize_features    → recorte de outliers (calculado sobre train)
    6. scale_features        → estandarización (ajustada solo sobre train)


Tras load_oil_data() las columnas event_type/event_severity/event_description
son las GEOPOLÍTICAS (post _resolve_overlap). Los originales del oil dataset
están en oil_event_*.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Tuple, List

from brent_forecast.config import PreprocessingSettings, SplitSettings


# ──────────────────────────────────────────────
# 1. GENERACIÓN DE LA VARIABLE OBJETIVO
# ──────────────────────────────────────────────

def create_label(df: pd.DataFrame) -> pd.DataFrame:
    """
    Genera la variable objetivo para clasificación binaria.

    label = 1 si brent_return(t+1) > 0, 0 en caso contrario.
    Se usa shift(-1) sobre brent_return para alinear cada fila con el retorno
    del día siguiente. La última fila queda con NaN y se elimina.

    IMPORTANTE: se llama ANTES de cualquier filtrado o reordenación
    para garantizar la alineación correcta entre features y label.
    """
    df = df.sort_values("date").reset_index(drop=True)
    df["label"] = (df["brent_return"].shift(-1) > 0).astype(float)

    # Verificación explícita de alineación sobre las primeras 100 filas
    for i in range(min(100, len(df) - 1)):
        retorno_siguiente = df.loc[i + 1, "brent_return"]
        label_actual = df.loc[i, "label"]
        assert (retorno_siguiente > 0) == (label_actual == 1.0), (
            f"Desalineación en fila {i}: retorno={retorno_siguiente:.4f}, label={label_actual}"
        )

    df = df.dropna(subset=["label"]).reset_index(drop=True)
    print(f"[label] Distribución de clases:\n{df['label'].value_counts(normalize=True).round(3)}")
    return df


# ──────────────────────────────────────────────
# 2. INGENIERÍA DE FEATURES
# ──────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Aplica transformaciones y eliminaciones sobre el DataFrame.

    Cambios aplicados:
    - P1: elimina wti_return (data leakage del mismo día que brent_return).
    - P2: convierte lags de precio a retornos logarítmicos (estacionariedad).
    - P3: sustituye gpr_index (constante semanal) por gpr_change (variación 21d).
    - P4: reemplaza columnas de evento dispersas por event_flag_binary y
          high_severity_flag (densas).
    - P5: elimina volatilidades WTI redundantes; añade vol_ratio.
    - Añade features de estacionalidad: day_of_week, month.

    Returns
    -------
    df : pd.DataFrame
        DataFrame con features transformadas.
    feature_cols : List[str]
        Nombres de las columnas que se usarán como features.
    """

    # P1 — Eliminar wti_return (leakage del mismo día)
    df = df.drop(columns=["wti_return"], errors="ignore")

    # P2 — Convertir lags de precio a retornos logarítmicos
    for n in [1, 3, 7]:
        df[f"lag_ret_{n}"] = np.log(df["brent_price"] / df[f"brent_lag_{n}"])

    lag_cols = [f"brent_lag_{n}" for n in [1, 3, 7]] + [f"wti_lag_{n}" for n in [1, 3, 7]]
    df = df.drop(columns=lag_cols, errors="ignore")

    # P3 — Sustituir gpr_index por gpr_change (21 días hábiles ≈ 1 mes)
    df["gpr_change"] = df["gpr_index"] - df["gpr_index"].shift(21)
    df = df.drop(columns=["gpr_index"], errors="ignore")

    # P4 — Features binarias densas a partir del event_severity geopolítico
    # event_severity aquí es la columna geopolítica (post _resolve_overlap en load_data)
    df["event_flag_binary"] = (df["event_severity"] > 0).astype(int)
    df["high_severity_flag"] = (df["event_severity"] >= 7).astype(int)

    # P5 — Eliminar volatilidades WTI redundantes (corr > 0.95 con Brent)
    df = df.drop(columns=["wti_volatility_7d", "wti_volatility_30d"], errors="ignore")
    df["vol_ratio"] = df["brent_volatility_7d"] / (df["brent_volatility_30d"] + 1e-10)

    # Estacionalidad
    df["day_of_week"] = df["date"].dt.dayofweek   # 0=Lunes … 4=Viernes
    df["month"]       = df["date"].dt.month        # 1–12

    # Set de columnas a EXCLUIR del feature set
    # - 'date' / 'label': metadatos
    # - 'brent_return': fuente del label, sería leakage
    # - 'brent_price' / 'wti_price': precios absolutos no estacionarios.
    #     Su distribución cambia entre 2010 ($75) y 2022 ($110): el modelo
    #     entrenado en un nivel ve el otro como outlier. La señal predictiva
    #     se mantiene vía lag_ret_n (retornos logarítmicos, sí estacionarios).
    # - event_*: textuales (geopolíticas) — ya capturadas por las features binarias
    # - oil_event_*: residuales del rename en load_data — descartar
    # - event_flag: implícito en event_flag_binary; evita duplicar señal
    exclude = {
        "date", "label",
        "brent_return",
        "brent_price", "wti_price",
        "event_type", "event_description", "event_severity",
        "oil_event_type", "oil_event_description", "oil_event_severity",
        "event_flag",
    }

    feature_cols = [
        c for c in df.columns
        if c not in exclude and df[c].dtype != object
    ]

    print(f"[features] Total features seleccionadas: {len(feature_cols)}")
    print(f"[features] {feature_cols}")
    return df, feature_cols


# ──────────────────────────────────────────────
# 3. TRATAMIENTO DE NULOS
# ──────────────────────────────────────────────

def handle_nulls(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    """
    Elimina o imputa filas con valores nulos en las features.

    Los lags generan nulos al inicio:
    - lag_ret_1: 1 fila, lag_ret_3: 3, lag_ret_7: 7.
    gpr_change genera 21 nulos al inicio (ventana de un mes hábil).

    Se eliminan a lo sumo 21 filas sobre ~4.000 (< 0.6%).
    """
    for col in ["vix", "dxy_index"]:
        if col in df.columns:
            df[col] = df[col].ffill()

    n_before = len(df)
    df = df.dropna(subset=feature_cols).reset_index(drop=True)
    n_dropped = n_before - len(df)
    print(f"[nulls] Filas eliminadas por nulos en features: {n_dropped} ({n_dropped/n_before*100:.1f}%)")
    return df


# ──────────────────────────────────────────────
# 3b. FILTRADO POR MULTICOLINEALIDAD (VIF)
# ──────────────────────────────────────────────

def filter_features_by_vif(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_end_date: pd.Timestamp,
    vif_threshold: float,
) -> List[str]:
    """
    Elimina iterativamente la feature con mayor VIF mientras alguna supere el umbral.

    El Variance Inflation Factor mide cuánto se infla la varianza del
    coeficiente de una feature en una regresión lineal por la presencia
    de las demás. VIF > 10 indica multicolinealidad fuerte: la feature
    es prácticamente combinación lineal de otras y aporta poca información
    nueva al modelo lineal (LogReg, SVM lineal) y degrada la interpretabilidad.

    IMPORTANTE: el VIF se calcula SOLO con los datos de train (fechas anteriores
    a `train_end_date`) para evitar data leakage. Val y test no influyen
    en la decisión de qué features sobreviven.
    """
    from statsmodels.stats.outliers_influence import variance_inflation_factor

    train_mask = df["date"] < train_end_date
    remaining = list(feature_cols)
    print(f"[vif] Inicio del filtrado | features iniciales: {len(remaining)} | umbral VIF > {vif_threshold}")

    while len(remaining) > 1:
        X = df.loc[train_mask, remaining].values.astype(float)

        vifs = []
        for i in range(X.shape[1]):
            try:
                v = float(variance_inflation_factor(X, i))
            except Exception:
                v = float("inf")
            vifs.append(v)

        max_vif = max(vifs)
        if max_vif <= vif_threshold:
            break

        worst_idx = vifs.index(max_vif)
        worst_name = remaining[worst_idx]
        vif_str = "inf" if not np.isfinite(max_vif) else f"{max_vif:.2f}"
        print(f"[vif]  Eliminada: {worst_name:<25} VIF = {vif_str}")
        remaining.pop(worst_idx)

    # Reporte final de VIFs de las features que sobreviven
    X_final = df.loc[train_mask, remaining].values.astype(float)
    print(f"[vif] Features finales ({len(remaining)}):")
    for i, name in enumerate(remaining):
        try:
            v = float(variance_inflation_factor(X_final, i))
            v_str = "inf" if not np.isfinite(v) else f"{v:.2f}"
        except Exception:
            v_str = "n/a"
        print(f"[vif]    {name:<25} VIF = {v_str}")

    return remaining


# ──────────────────────────────────────────────
# 4. DIVISIÓN TEMPORAL ESTRICTA
# ──────────────────────────────────────────────

def split_temporal(
    df: pd.DataFrame,
    feature_cols: List[str],
    train_end: pd.Timestamp,
    val_end: pd.Timestamp,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray]:
    """
    Divide el dataset respetando el orden cronológico estricto.

    Train:      date <  train_end            (por defecto 2010–2021)
    Validación: train_end <= date < val_end  (por defecto 2022–2023)
    Test:       date >= val_end              (por defecto 2024–2026)
    """
    train_mask = df["date"] < train_end
    val_mask   = (df["date"] >= train_end) & (df["date"] < val_end)
    test_mask  = df["date"] >= val_end

    X_train = df.loc[train_mask, feature_cols].values
    X_val   = df.loc[val_mask,   feature_cols].values
    X_test  = df.loc[test_mask,  feature_cols].values

    y_train = df.loc[train_mask, "label"].values
    y_val   = df.loc[val_mask,   "label"].values
    y_test  = df.loc[test_mask,  "label"].values

    print(f"[split] Train:      {X_train.shape[0]} filas")
    print(f"[split] Validación: {X_val.shape[0]} filas")
    print(f"[split] Test:       {X_test.shape[0]} filas")
    print(f"[split] Balance train — clase 1: {y_train.mean():.2%} | clase 0: {(1 - y_train.mean()):.2%}")
    return X_train, X_val, X_test, y_train, y_val, y_test


# ──────────────────────────────────────────────
# 5. WINSORIZACIÓN (anti-outliers)
# ──────────────────────────────────────────────

def winsorize_features(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
    feature_cols: List[str],
    lower: float,
    upper: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Limita los valores extremos de las features de retorno al rango [lower, upper]
    (percentiles; por defecto [1%, 99%]).

    Motivación: el colapso COVID (marzo 2020) generó retornos > 10 que
    distorsionan LogReg, MLP y SVM. Los percentiles se calculan SOLO sobre
    train para evitar data leakage.
    """
    retorno_idx = [i for i, c in enumerate(feature_cols) if "ret" in c or "return" in c]

    for idx in retorno_idx:
        lo = np.percentile(X_train[:, idx], lower * 100)
        hi = np.percentile(X_train[:, idx], upper * 100)
        X_train[:, idx] = np.clip(X_train[:, idx], lo, hi)
        X_val[:, idx]   = np.clip(X_val[:, idx],   lo, hi)
        X_test[:, idx]  = np.clip(X_test[:, idx],  lo, hi)

    print(f"[winsorize] Features de retorno winsorizadas: {len(retorno_idx)}")
    return X_train, X_val, X_test


# ──────────────────────────────────────────────
# 6. ESCALADO
# ──────────────────────────────────────────────

def scale_features(
    X_train: np.ndarray,
    X_val:   np.ndarray,
    X_test:  np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Estandariza features a media 0 y std 1.

    fit() SOLO sobre train. Los datos de val y test reciben solo transform()
    para no introducir data leakage.
    """
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_val_sc   = scaler.transform(X_val)
    X_test_sc  = scaler.transform(X_test)

    print(f"[scaler] Media post-escalado (train, primeras 3): {X_train_sc.mean(axis=0)[:3].round(4)}")
    print(f"[scaler] Std  post-escalado (train, primeras 3): {X_train_sc.std(axis=0)[:3].round(4)}")
    return X_train_sc, X_val_sc, X_test_sc, scaler


# ──────────────────────────────────────────────
# 7. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────

def preprocess(
    df: pd.DataFrame,
    split: SplitSettings,
    preprocessing: PreprocessingSettings,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray,
           np.ndarray, np.ndarray, np.ndarray,
           StandardScaler, List[str]]:
    """
    Pipeline completo: de DataFrame crudo a arrays listos para entrenar.

    Returns
    -------
    X_train, X_val, X_test : np.ndarray (escalados)
    y_train, y_val, y_test : np.ndarray
    scaler                 : StandardScaler ajustado
    feature_cols           : List[str] con los nombres de las features
    """
    print("\n" + "=" * 50)
    print("PIPELINE DE PREPROCESADO")
    print("=" * 50)

    df = create_label(df)
    df, feature_cols = engineer_features(df)
    df = handle_nulls(df, feature_cols)
    train_end = pd.Timestamp(split.train_end)
    val_end = pd.Timestamp(split.val_end)
    feature_cols = filter_features_by_vif(
        df, feature_cols, train_end, preprocessing.vif_threshold
    )

    X_train, X_val, X_test, y_train, y_val, y_test = split_temporal(
        df, feature_cols, train_end, val_end
    )
    X_train, X_val, X_test = winsorize_features(
        X_train, X_val, X_test, feature_cols,
        preprocessing.winsor_lower, preprocessing.winsor_upper,
    )
    X_train_sc, X_val_sc, X_test_sc, scaler = scale_features(X_train, X_val, X_test)

    print("\n[preprocesado]  Pipeline completado")
    print(f"[preprocesado] Shape X_train: {X_train_sc.shape}")
    print(f"[preprocesado] Shape X_val:   {X_val_sc.shape}")
    print(f"[preprocesado] Shape X_test:  {X_test_sc.shape}")
    print("=" * 50 + "\n")

    return X_train_sc, X_val_sc, X_test_sc, y_train, y_val, y_test, scaler, feature_cols
