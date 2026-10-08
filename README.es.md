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
uv run brent report          # informe estadístico de la última ejecución (IC, tests, calibración)
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
`BRENT_VALIDATION__MODE=rolling uv run brent train`. `uv run brent config show` muestra la configuración
resuelta.

---

## Metodología

- **Validación walk-forward con purga y embargo:** periodo de desarrollo `< 2024-01-01` para ajustar
  hiperparámetros (CV walk-forward de 5 folds) y periodo fuera de muestra `≥ 2024-01-01` evaluado por
  walk-forward, reentrenando cada 63 sesiones con todos los datos anteriores. Entre train y test hay un hueco
  de `purge` (1 día: la etiqueta de *t* se realiza en *t+1*) + `embargo` (5 días).
- **Control de leakage:** se elimina `wti_return` (mismo día); el filtro VIF, los percentiles de
  winsorización y el `StandardScaler` son pasos del `Pipeline` de cada modelo y se reajustan en cada fold y
  ventana; los tests prueban que corromper el futuro no cambia features, etiquetas de train ni predicciones
  anteriores. Cada pipeline final se guarda en `results/models/`.
- **Búsqueda de hiperparámetros:** `GridSearchCV` (también para la MLP, ahora estimador de sklearn).
- **Métricas:** accuracy, precision / recall / F1 macro y AUC-ROC (métrica principal).

| Modelo | Implementación |
|--------|----------------|
| Regresión logística (L2) | scikit-learn |
| SVM (kernel RBF, probabilidades calibradas) | scikit-learn |
| Random Forest | scikit-learn |
| **MLP 64 → 32 → 1** | **núcleo NumPy puro**, API de estimador scikit-learn |

Baselines evaluados con el mismo walk-forward: clase mayoritaria, persistencia (mañana = hoy), aleatorio
estratificado y buy & hold (siempre sube).

**Evaluación estadística** (`brent report`, lee `predictions.csv` y `metrics.json` sin reentrenar):
intervalos de confianza de AUC y accuracy con *block bootstrap* circular (bloques de 21 días, porque los
días no son independientes), test de DeLong de cada modelo contra el mejor baseline con corrección de Holm,
test binomial de accuracy frente a la *no-information rate*, Brier score, Brier skill score frente a la
previsión climatológica y diagrama de fiabilidad. El informe indica si los datos coinciden con los
checksums registrados de Kaggle o si su procedencia no está verificada.

**Backtest económico:** cada candidato se convierte en una estrategia long/flat (comprado si
`P(sube) >= 0.5`) sobre el retorno del día siguiente, con 5 pb de coste por operación. Se reportan CAGR,
volatilidad, Sharpe, drawdown máximo, hit ratio, exposición y número de operaciones, y un intervalo por
block bootstrap de la diferencia de Sharpe frente a buy & hold.

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
débil para retornos diarios.

**Informe estadístico con el dataset sintético** (no son datos reales; fuera de muestra 2024-01-01 a
2026-03-11, 573 días, IC 95 % por block bootstrap, baseline de referencia: persistencia, AUC 0.5073):

| Modelo | AUC OOS [IC 95 %] | ΔAUC vs persistencia [IC 95 %] | p (Holm) |
|--------|-------------------|--------------------------------|---------:|
| Logistic Regression | 0.508 [0.470, 0.548] | +0.001 [−0.054, +0.056] | 1.000 |
| SVM (RBF) | 0.492 [0.445, 0.541] | −0.015 [−0.065, +0.041] | 1.000 |
| Random Forest | 0.473 [0.424, 0.518] | −0.035 [−0.096, +0.025] | 0.932 |
| MLP NumPy | 0.467 [0.425, 0.512] | −0.040 [−0.101, +0.024] | 0.932 |

Todos los intervalos de AUC contienen 0.5, ningún modelo supera a la persistencia ni a la
*no-information rate* y ninguna estrategia bate a buy & hold tras costes (Sharpe 1.06 frente a 0.26 del mejor
modelo, la MLP): no hay capacidad predictiva que encontrar.

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
