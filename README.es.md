# Predicción del Precio del Brent — Clasificación Binaria

> Dirección del precio del Brent al día siguiente a partir de indicadores de mercado y eventos geopolíticos (2010–2026).

[![CI](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/actions/workflows/ci.yml/badge.svg)](https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/rafaelhh04/PrediccionPetroleo-Clasificacion/branch/main/graph/badge.svg)](https://codecov.io/gh/rafaelhh04/PrediccionPetroleo-Clasificacion)

*English version: [README.md](README.md).*

Clasificación binaria de si el Brent cerrará al alza mañana, comparando cuatro modelos — regresión
logística, SVM con kernel RBF, Random Forest y una **red neuronal MLP implementada desde cero en NumPy**
(backpropagation, dropout invertido, inicialización He, early stopping) — con una evaluación cronológica
estricta y consciente del *data leakage*.

| Clase | Condición | Significado |
|-------|-----------|-------------|
| `1` | `brent_return(t+1) > 0` | El precio sube |
| `0` | `brent_return(t+1) ≤ 0` | El precio baja o se mantiene |

El periodo cubierto (2010–2026) incluye eventos de alta relevancia como la pandemia de COVID-19 (2020), la
guerra de Ucrania (2022) o la crisis del Mar Rojo (2024).

---

## Instalación y ejecución

Requiere [uv](https://docs.astral.sh/uv/getting-started/installation/) (instala Python 3.12 si hace falta).

```bash
git clone https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion.git
cd PrediccionPetroleo-Clasificacion

uv sync                      # crea .venv a partir de uv.lock
uv run brent data download   # descarga el dataset de Kaggle en data/raw/ (requiere credenciales)
uv run brent train           # pipeline completa: preprocesado, tuning, entrenamiento y evaluación
```

Los resultados se generan en `results/` (ignorado por git): `metrics.json` (métricas train/val/test),
`run.log` (log completo) y `plots/` (curvas ROC, matrices de confusión, curvas de aprendizaje).

### Datos

Los CSV no se versionan. `brent data download` descarga
[kavyadhyani/global-oil-prices-andgeopolitical-events](https://www.kaggle.com/datasets/kavyadhyani/global-oil-prices-andgeopolitical-events)
con `kagglehub`, identifica cada fichero por su cabecera, lo copia a `data/raw/` con su nombre canónico y
verifica su SHA-256 contra `configs/data_checksums.json`.

| Fichero canónico | Contenido |
|------------------|-----------|
| `oil_geopolitics_dataset_2010_2026.csv` | ~4.000 filas diarias × 23 columnas: precios, retornos, lags, volatilidades, DXY, VIX, GPR, spread, eventos |
| `geopolitical_events_timeline.csv` | 35 eventos: `date`, `event_type`, `event_description`, `event_severity` (1–10) |

Los dos datasets se integran mediante **left join** sobre `date`; los días sin evento reciben
`event_type = 'none'` y `event_severity = 0`.

**Credenciales:** exporta `KAGGLE_USERNAME` y `KAGGLE_KEY` (o usa `~/.kaggle/kaggle.json`).

**Descarga manual:** descarga el ZIP desde Kaggle, copia los dos CSV a `data/raw/` con los nombres canónicos
y ejecuta `uv run brent data verify`. Si `configs/data_checksums.json` aún no tiene los hashes (valores
`null`), la primera verificación los registra: haz commit del fichero.

### Configuración

Todos los valores ajustables (rutas, semilla, fechas de split, folds de CV, umbrales de VIF y winsorización,
hiperparámetros y grids) están en [`configs/default.yaml`](configs/default.yaml) y se pueden sobrescribir con
variables de entorno `BRENT_*` (`__` como separador de anidamiento), p. ej.
`BRENT_SPLIT__TRAIN_END=2021-01-01 uv run brent train`. `uv run brent config show` muestra la configuración
resuelta.

---

## Metodología

- **División temporal estricta:** train `< 2022-01-01`, validación `2022–2023`, test `≥ 2024-01-01`.
- **Control de leakage:** se elimina `wti_return` (mismo día); el filtro VIF, los percentiles de
  winsorización y el `StandardScaler` se ajustan solo con train; el tuning usa `TimeSeriesSplit` (5 folds,
  ventana expansiva) solo sobre train; el test se evalúa una única vez.
- **Búsqueda de hiperparámetros:** `GridSearchCV` para los modelos sklearn y búsqueda manual para la MLP.
- **Métricas:** accuracy, precision / recall / F1 macro y AUC-ROC (métrica principal).

| Modelo | Implementación |
|--------|----------------|
| Regresión logística (L2) | scikit-learn |
| SVM (kernel RBF) | scikit-learn |
| Random Forest | scikit-learn |
| **MLP 64 → 32 → 1** | **NumPy puro** |

## Resultados (AUC-ROC)

Ejecución de referencia del proyecto original con los datos de Kaggle. Es anterior a las correcciones de
la Fase 3 (p. ej. el bug de la última etiqueta) y aún no se ha repetido con datos reales: Kaggle no es
accesible desde el entorno de desarrollo, así que las cifras posteriores se reproducen con el dataset
sintético de los tests.

| Modelo | Train | Validación | Test |
|--------|------:|-----------:|-----:|
| Logistic Regression | 0.5507 | 0.4800 | 0.4858 |
| SVM (RBF) | 0.5646 | 0.4700 | 0.4909 |
| Random Forest | 0.7342 | 0.4936 | 0.4927 |
| MLP NumPy | 0.5717 | 0.4933 | 0.5002 |

Ningún modelo supera al azar fuera de muestra, coherente con la hipótesis de mercado eficiente en su forma
débil para retornos diarios. Demostrarlo con rigor (baselines, intervalos de confianza, tests estadísticos y
backtest económico) es el objetivo de la siguiente fase del [roadmap](docs/ROADMAP.md).

---

## Estructura del proyecto

```text
├── configs/                    # default.yaml (configuración) y data_checksums.json
├── docs/ROADMAP.md             # plan de profesionalización y registro de progreso
├── src/brent_forecast/
│   ├── cli.py                  # CLI con Typer (`brent`)
│   ├── config.py               # configuración tipada (pydantic-settings)
│   ├── logging_config.py       # logging a consola y a fichero
│   ├── pipeline.py             # orquestación del entrenamiento
│   ├── data/                   # descarga de Kaggle + checksums, carga y merge
│   ├── features/               # label, features, VIF, split, winsorización, escalado
│   ├── models/                 # LogReg, SVM, RF, MLP NumPy y helpers de tuning
│   └── evaluation/             # métricas, gráficas, curvas de aprendizaje, evaluación final
├── tests/                      # tests unitarios, de leakage, gradient check y propiedades
├── .github/                    # CI, Dependabot, plantillas, CODEOWNERS, ruleset
├── Makefile
└── pyproject.toml / uv.lock
```

## Desarrollo

`make install` (dependencias + hooks), `make lint` (ruff, codespell, mypy estricto), `make test`,
`make coverage`, `make help`. ~130 tests con datos sintéticos, guardas de *data leakage*, *gradient check*
numérico del MLP y tests de propiedades (Hypothesis); CI en Python 3.12 y 3.13 en cada PR. Código,
docstrings, logs y commits en inglés; Conventional Commits; semilla `42`. Guía completa en
[CONTRIBUTING.md](CONTRIBUTING.md).

## Autor

Rafael Hernando Herias — proyecto individual de la asignatura de Aprendizaje Automático (curso 2025-26).

## Licencia

[MIT](LICENSE)
