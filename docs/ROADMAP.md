# Roadmap de profesionalización — Brent Direction Classifier

> Objetivo: convertir un proyecto académico de ML en un **proyecto de portfolio de nivel profesional**
> que demuestre las competencias que piden las ofertas de *ML Engineer / Data Scientist / MLOps*:
> ingeniería de software sólida, rigor experimental, reproducibilidad, despliegue y monitorización.

---

## 0. Diagnóstico del estado actual

| Área | Estado actual | Problema para un reclutador técnico |
|------|---------------|-------------------------------------|
| Resultados | AUC test ≈ 0.49–0.50 en los 4 modelos | Sin baselines ni tests estadísticos no se puede afirmar *nada*; parece un fallo, cuando en realidad es un resultado (eficiencia de mercado) mal contado |
| Reproducibilidad | CSVs fuera del repo, nombres de fichero en `load_data.py` ≠ README, no existe `environment.yml` ni `requirements` | `git clone && python main.py` falla |
| `.gitIgnore` | Mal nombrado (Git solo reconoce `.gitignore`); en Linux no se aplica, en Windows/macOS sí → comportamiento distinto por SO. Además ignora `/*.md` y `/docs` | Ficheros generados (`results/run_log.txt`) versionados |
| Calidad de código | Sin tests, sin linter, sin tipado verificado, `print` como logging, rutas y fechas *hardcodeadas* | No supera un *code review* profesional |
| Arquitectura | Scripts acoplados; preprocesado manual con arrays NumPy en vez de `Pipeline` de sklearn | Riesgo de *leakage* al reutilizar, imposible servir el modelo |
| MLOps | Ni tracking de experimentos, ni versionado de datos/modelos, ni CI/CD | Lo que más diferencia a un junior en 2026 |
| Despliegue | Inexistente | No hay demo que enseñar en una entrevista |
| Documentación | README en español, referencias a ficheros inexistentes (`memoria.pdf`, `environment.yml`, un documento de planificación citado en el MLP) | Mercado internacional → inglés |

**Puntos fuertes a conservar y destacar:** MLP implementado desde cero en NumPy (backprop, dropout, He init),
split temporal estricto, conciencia explícita de *data leakage* (VIF y winsorización ajustados solo en train).

---

## Reglas transversales (aplican a TODAS las fases)

1. **Autoría de commits:** `rafaelhh04 <132058207+rafaelhh04@users.noreply.github.com>`.
   Sin `Co-Authored-By`, sin trailers, sin ninguna mención a asistentes de IA en commits, código, docs ni PRs.
   Antes de commitear: `git config user.name "rafaelhh04" && git config user.email "132058207+rafaelhh04@users.noreply.github.com"`.
2. **Ramas con nombre de la funcionalidad**, prefijo según tipo, partiendo siempre de `main` actualizado:
   `feat/…`, `fix/…`, `refactor/…`, `test/…`, `ci/…`, `docs/…`, `chore/…`. Una rama = una funcionalidad = un PR.
3. **Conventional Commits** (`feat(data): add kaggle download script`), commits pequeños y atómicos.
4. **Ningún PR se mezcla con rojo:** lint, tipos y tests en verde antes de push.
5. **Idioma:** código, docstrings, commits y README principal en **inglés** (decisión de Fase 1); `README.es.md` como traducción.
6. **Al cerrar cada fase** se actualiza la sección *Registro de progreso* de este documento y se genera el
   *prompt de traspaso* para la siguiente fase (plantilla al final de cada fase).

---

## Fase 1 — Fundamentos del repositorio y reproducibilidad

**Objetivo:** que cualquiera pueda clonar e instalar el proyecto en < 5 min con un único comando.

| Rama | Contenido |
|------|-----------|
| `chore/project-packaging` | `pyproject.toml` (PEP 621) gestionado con **uv**, *lockfile* `uv.lock`, Python 3.12+. Layout `src/brent_forecast/` (`data/`, `features/`, `models/`, `evaluation/`, `cli.py`). Punto de entrada CLI con **Typer** (`brent train`, `brent evaluate`). |
| `chore/gitignore-cleanup` | Renombrar `.gitIgnore` → `.gitignore`, quitar `/*.md` y `/docs`, des-versionar `results/run_log.txt`. |
| `feat/data-download` | Script `brent data download` con `kagglehub` (credenciales vía variables de entorno), checksums SHA-256 de los CSV, nombres de fichero centralizados. |
| `feat/config-management` | Configuración tipada con **pydantic-settings** + YAML (`configs/default.yaml`): fechas de split, semillas, rutas, grids. Cero *magic numbers* en el código. |
| `refactor/structured-logging` | Sustituir `print`/`Tee` por `logging` (o `structlog`) con niveles y salida a fichero. |
| `chore/dev-tooling` | **pre-commit** con `ruff` (lint + format), `mypy --strict` en `src/`, `codespell`; `Makefile`/`justfile` (`make install`, `make lint`, `make test`, `make train`). `LICENSE` (MIT), `.editorconfig`. |
| `docs/remove-stale-references` | Eliminar referencias a ficheros inexistentes (documento de planificación citado en el MLP, `memoria.pdf`, `environment.yml`). |

**Definition of Done:** `uv sync && uv run brent data download && uv run brent train` reproduce los resultados actuales (mismas métricas ±0.001); `pre-commit run --all-files` en verde.

**Por qué:** `uv` es hoy el estándar de facto (rápido, lockfile reproducible); el *src layout* evita importar
código sin instalar y obliga a empaquetar bien; la config externa es requisito para tracking de experimentos (Fase 4).

---

## Fase 2 — Calidad: testing y CI

**Objetivo:** red de seguridad automatizada antes de tocar la lógica de ML.

| Rama | Contenido |
|------|-----------|
| `test/unit-test-suite` | **pytest** + `pytest-cov`. Tests de carga/merge (fixtures sintéticas, sin datos reales), de `create_label` (alineación t+1), de split temporal (no solapamiento, orden). Cobertura ≥ 85 % en `src/`. |
| `test/leakage-guards` | Tests que **fallan si hay leakage**: el scaler/winsorizer/VIF solo ven train; ninguna feature usa información de t+1 (test de permutación del futuro: alterar filas futuras no cambia features pasadas). |
| `test/mlp-gradient-check` | *Gradient checking* numérico del MLP NumPy (diferencias finitas vs backprop, error relativo < 1e-6) y tests de propiedades con **Hypothesis**. Muy vistoso en entrevista. |
| `ci/github-actions` | Workflow CI: matriz Python 3.12/3.13, `uv sync --frozen`, ruff, mypy, pytest con cobertura, subida a Codecov. Caché de uv. Badge en README. |
| `chore/repo-governance` | Plantillas de PR e issues, `CODEOWNERS`, `CONTRIBUTING.md`, Dependabot/Renovate, protección de `main` (PR obligatorio + CI verde). |

**DoD:** CI verde en `main`, cobertura ≥ 85 %, badge visible, `main` protegida.

---

## Fase 3 — Rigor de Machine Learning (el corazón del portfolio)

**Objetivo:** pasar de "mis modelos dan 0.50" a "demuestro con rigor estadístico qué señal hay y cuánto vale".

| Rama | Contenido |
|------|-----------|
| `refactor/sklearn-pipelines` | Todo el preprocesado como `Pipeline`/`ColumnTransformer` con *transformers* propios (`Winsorizer`, `VIFSelector`) que implementan `fit/transform`. El MLP NumPy envuelto como estimador compatible con sklearn (`BaseEstimator`, `ClassifierMixin`) → `check_estimator` pasa. Un único artefacto serializable `pipeline.joblib`. |
| `feat/walk-forward-validation` | Validación *walk-forward* con **purga y embargo** (López de Prado) en lugar del split fijo único; métricas por ventana. |
| `feat/baselines` | Baselines obligatorios: clase mayoritaria, *persistence* (mañana = hoy), aleatorio estratificado, *buy & hold*. |
| `feat/statistical-evaluation` | IC al 95 % por *bootstrap* (block bootstrap para series temporales), test de DeLong entre AUCs, test binomial contra 50 %, calibración (Brier, *reliability diagram*). |
| `feat/financial-backtest` | Backtest económico: estrategia long/flat según probabilidad, costes de transacción, Sharpe, max drawdown, hit ratio. La métrica de negocio importa más que la AUC. |
| `feat/feature-engineering-v2` | Indicadores técnicos (RSI, MACD, Bollinger, momentum multi-horizonte), *term structure*/spread, features de calendario cíclicas (sin/cos), *lagged* VIX/DXY. Cada feature con test de no-leakage. |
| `feat/gradient-boosting-optuna` | LightGBM/XGBoost + búsqueda bayesiana con **Optuna** (pruning), sustituyendo GridSearch. |
| `feat/model-explainability` | **SHAP** (global y local), permutation importance en validación. |
| `docs/model-card` | `MODEL_CARD.md` (Mitchell et al.): uso previsto, datos, métricas con IC, limitaciones, riesgos. |

**DoD:** informe reproducible (`brent report`) que compara todos los modelos contra baselines con IC y p-values,
y concluye honestamente. Si no hay señal, se documenta como hallazgo (hipótesis de mercado eficiente) — esto
**es** un resultado profesional valioso.

---

## Fase 4 — MLOps: tracking, versionado y validación de datos

| Rama | Contenido |
|------|-----------|
| `feat/experiment-tracking` | **MLflow**: parámetros, métricas, artefactos (plots, pipeline), tags de commit git. Model Registry con *aliases* (`champion`/`challenger`). |
| `feat/data-versioning` | **DVC** con *remote* (GCS/S3 o local) y `dvc.yaml` con etapas `download → validate → featurize → train → evaluate`; `dvc repro` reproduce todo. |
| `feat/data-validation` | **Pandera** (o Great Expectations) para esquemas de entrada: tipos, rangos, fechas monotónicas, nulos. Falla rápido. |
| `feat/data-ingestion-live` | Ingesta incremental de precios recientes (p. ej. `yfinance` para `BZ=F`, FRED para DXY/VIX) para poder predecir sobre datos actuales. |

**DoD:** `dvc repro` reproduce el pipeline entero; cada ejecución queda registrada en MLflow; el modelo campeón se carga por alias.

---

## Fase 5 — Serving: API y contenedores

| Rama | Contenido |
|------|-----------|
| `feat/inference-api` | **FastAPI** + Pydantic v2: `POST /predict`, `GET /health`, `GET /model-info`. Carga del pipeline desde el registry. Validación estricta de entrada, OpenAPI documentada. |
| `test/api-tests` | Tests con `httpx`/`TestClient`, test de contrato del esquema, test de carga con **Locust** (latencia p95 documentada). |
| `feat/docker` | `Dockerfile` *multi-stage* con uv, imagen *slim*, usuario no-root, `HEALTHCHECK`; `docker-compose.yml` (API + MLflow). Escaneo con **Trivy** en CI. |
| `feat/observability` | Métricas Prometheus (`prometheus-fastapi-instrumentator`), logs JSON, *request id*. |
| `feat/demo-dashboard` | Demo en **Streamlit**: histórico, predicción del día, explicación SHAP, resultados del backtest. |

**DoD:** `docker compose up` levanta API + dashboard; CI construye y escanea la imagen.

---

## Fase 6 — Cloud y CD

| Rama | Contenido |
|------|-----------|
| `feat/infra-terraform` | **Terraform** para GCP (Cloud Run, Artifact Registry, GCS bucket para DVC/MLflow, Secret Manager). Estado remoto. |
| `ci/continuous-deployment` | GitHub Actions con **OIDC / Workload Identity Federation** (sin claves en secretos): build → push → deploy a Cloud Run en merge a `main`; entorno *staging* previo. |
| `feat/scheduled-retraining` | Reentrenamiento programado (Cloud Run Job + Cloud Scheduler o GitHub Actions cron) con promoción automática solo si el *challenger* supera al *champion* con significancia. |
| `feat/drift-monitoring` | **Evidently** para *data drift* y *prediction drift*; informe publicado y alerta. |

**DoD:** URL pública de la API y el dashboard; despliegue automático; coste estimado documentado (capa gratuita).

---

## Fase 7 — Portfolio y empleabilidad

| Rama | Contenido |
|------|-----------|
| `docs/readme-portfolio` | README en inglés orientado a reclutador: TL;DR con resultado clave, badges (CI, coverage, Python, license), diagrama de arquitectura en **Mermaid**, GIF de la demo, *quickstart* en 3 comandos, sección "Key engineering decisions". |
| `docs/mkdocs-site` | Documentación con **MkDocs Material** + `mkdocstrings` (API reference) publicada en GitHub Pages; ADRs (*Architecture Decision Records*) en `docs/adr/`. |
| `docs/technical-writeup` | Artículo técnico (blog/Medium/dev.to): "Why 4 models can't beat a coin flip on Brent — and how to prove it". |
| `chore/release-v1` | `CHANGELOG.md` generado (release-please), tag `v1.0.0`, GitHub Release. |

**Entregables para el CV:** 3–4 *bullets* cuantificados, p. ej.:
- *Built an end-to-end ML system (data → MLflow/DVC → FastAPI on Cloud Run) with CI/CD, 90 % test coverage and IaC (Terraform).*
- *Implemented a neural network from scratch in NumPy, validated by numerical gradient checking.*
- *Designed a leakage-safe walk-forward evaluation with block-bootstrap CIs and an economic backtest.*

---

## Orden y estimación

| Fase | Esfuerzo orientativo | Dependencias |
|------|----------------------|--------------|
| 1 Fundamentos | 1 semana | — |
| 2 Calidad | 1 semana | 1 |
| 3 Rigor ML | 2–3 semanas | 2 |
| 4 MLOps | 1–2 semanas | 3 |
| 5 Serving | 1–2 semanas | 3 (4 recomendable) |
| 6 Cloud | 1–2 semanas | 5 |
| 7 Portfolio | continuo, cierre 1 semana | todas |

Si el tiempo es limitado, el **mínimo viable de portfolio** es: Fases 1 + 2 + 3 + `feat/inference-api` + `feat/docker` + README de la Fase 7.

---

## Registro de progreso

| Fase | Estado | Ramas mergeadas | Decisiones / desviaciones |
|------|--------|-----------------|---------------------------|
| 0 Plan | ✅ | `docs/roadmap-profesionalizacion` | Plan inicial creado |
| 1 Fundamentos | ✅ (PRs abiertos, encadenados) | `chore/gitignore-cleanup`, `chore/project-packaging`, `feat/data-download`, `feat/config-management`, `refactor/structured-logging`, `chore/dev-tooling`, `docs/remove-stale-references` | uv + PEP 621 + src layout; CLI Typer; pydantic-settings + YAML + `BRENT_*`; logging estándar; ruff + mypy estricto + pre-commit. Ver notas de cierre ↓ |
| 2 Calidad | ✅ | `test/unit-test-suite` (#9), `test/leakage-guards` (#10), `test/mlp-gradient-check` (#11), `ci/github-actions` (#12), `chore/repo-governance` | 130 tests, 98 % cobertura de ramas, gradient check ≤ 2.5e-9, CI 3.12/3.13 verde. Protección de `main` versionada como ruleset (requiere importarla). Ver notas de cierre ↓ |
| 3 Rigor ML | ✅ | `fix/label-last-row` (#15), `refactor/sklearn-pipelines` (#16), `feat/walk-forward-validation` (#17), `feat/baselines` (#18), `feat/statistical-evaluation` (#19), `feat/financial-backtest` (#20), `feat/feature-engineering-v2` (#21), `feat/gradient-boosting-optuna` (#22), `feat/model-explainability` (#23), `docs/model-card` (#24) | Pipelines sklearn + walk-forward purgado (10 ventanas OOS) + 4 baselines + `brent report` (IC block bootstrap, DeLong+Holm, binomial, Brier, backtest) + LightGBM/Optuna + `brent explain` (SHAP, permutation). Conclusión: sin señal. 422 tests, 99 % cobertura. Ver notas de cierre ↓ |
| 4 MLOps | ✅ | `feat/experiment-tracking` (#25), `feat/data-versioning` (#26), `feat/data-validation` (#27), `feat/data-ingestion-live` (#28), `fix/dvc-reproducible-lock` (#29), `docs/phase-4-closing` (#30) | MLflow (runs anidados, artefactos, skops, registry con alias `champion`/`challenger` y regla de promoción testeada) + DVC (6 etapas, `dvc.lock`, remote local) + Pandera (5 contratos) + ingesta Yahoo/FRED + `brent predict`. Campeón = baseline de persistencia (ningún modelo la supera). 516 tests, 98.5 % cobertura. Ver notas de cierre ↓ |
| 5 | ⏳ pendiente | | |
| 6 | ⏳ pendiente | | |
| 7 | ⏳ pendiente | | |


### Notas de cierre — Fase 1

**Verificación.** Kaggle no era accesible desde el entorno de desarrollo (proxy 403, sin credenciales), así que la
equivalencia se demostró sobre un dataset sintético con el mismo esquema: el `main.py` original y `brent train`
producen las 60 métricas impresas idénticas y un `results/metrics.json` idéntico byte a byte en cada rama. Con los
datos reales hay que ejecutar una vez `brent data download` (o la descarga manual + `brent data verify`), comprobar
la tabla de referencia (AUC test 0.4858 / 0.4909 / 0.4927 / 0.5002) y hacer commit de `configs/data_checksums.json`,
que hoy contiene `null` (se rellena en el primer uso).

**Decisiones y desviaciones respecto al plan.**
- `main.py` eliminado: la orquestación vive en `brent_forecast.pipeline` y solo se invoca con `brent train`
  (o `python -m brent_forecast`). No se creó `brent evaluate` (la evaluación sobre test forma parte de `train`).
- `brent data download` identifica cada CSV por su cabecera (no confía en los nombres de Kaggle) y añade
  `brent data verify` para la descarga manual. Checksums con *trust on first use* + `--update-checksums`.
- Configuración: `configs/default.yaml` es la única fuente de valores por defecto (los modelos pydantic solo
  declaran tipos y validaciones). Precedencia: argumentos > entorno (`BRENT_`, separador `__`) > YAML.
  La semilla se inyecta como `random_state` en los 4 modelos (también en el MLP).
- Logging: `results/run.log` sustituye a `results/run_log.txt`; opción global `--log-level`.
- Hooks de ruff, codespell y mypy como hooks *locales* (`uv run --frozen`), para que sus versiones sean las
  de `uv.lock`. mypy `--strict` pasa sin ningún `type: ignore`; solo sklearn/statsmodels/kagglehub ignoran
  imports sin tipos. Ruff ignora N803/N806 (convención `X_train` de scikit-learn).
- Refactor sin cambio de lógica: helpers comunes `grid_search()` y `fit_and_evaluate()` para los modelos sklearn;
  `predict_proba` del MLP ya no recibe `dropout_p` (no tenía efecto en inferencia).
- Se adelantaron 13 tests (descarga con *downloader* inyectado y configuración) como semilla de la Fase 2.

**Deuda técnica detectada (no corregida para no alterar métricas).**
- `create_label`: `NaN > 0` es `False`, así que la última fila se queda con `label = 0` en vez de eliminarse
  (afecta a 1 fila del test). Corregir en la Fase 3 y re-baselinar.
- `create_label` valida la alineación con `assert` (se desactiva con `python -O`).
- `SVC(probability=True)` emite `FutureWarning` en scikit-learn 1.9 (deprecado, se elimina en 1.11): migrar a
  `CalibratedClassifierCV` en la Fase 3.
- `_validate` en `data/load.py` tiene rangos de fechas fijos (2009–2011 / ≥ 2025).


### Notas de cierre — Fase 2

**Resultado.** 130 tests (+1 `slow`, +1 `xfail` estricto) en ~30 s, **98 % de cobertura de ramas** (umbral
85 % en `pyproject.toml`). CI (`.github/workflows/ci.yml`) en verde: lint + `mypy --strict` + codespell y
tests en Python 3.12 y 3.13; job `slow` semanal y manual. Métricas de ML sin cambios: `brent train` sobre el
dataset sintético de la Fase 1 produce un `metrics.json` idéntico byte a byte al del `main.py` original.

**Qué cubren los tests.**
- `tests/conftest.py`: datos sintéticos con semilla y el esquema de Kaggle (calendario diario completo y
  submuestreo semanal para e2e rápidos), `settings` en `tmp_path`, `fast_settings` con grids mínimos,
  aislamiento automático de `BRENT_*` y del root logger. Sin datos reales ni red.
- `test_leakage.py`: lo ajustado (VIF, winsorización, scaler) no cambia al corromper val/test; reescribir
  el futuro no cambia features pasadas (4 cortes); la label solo depende de t+1; espías en la pipeline
  prueban que `X_test` no llega a tuning/entrenamiento/curvas y que se evalúa una sola vez. Validado por
  mutación (3 fugas inyectadas → la suite falla).
- `test_mlp_gradients.py`: diferencias centrales vs `backward()` para todos los parámetros, sin dropout y
  con máscara fija; error relativo máx. 2.5e-9 (< 1e-6); control negativo; propiedades con Hypothesis.

**Decisiones y desviaciones respecto al plan.**
- `pytest-xdist` evaluado y descartado: sin mejora, los modelos ya usan todos los cores (`n_jobs=-1`).
- La cobertura vive en `[tool.coverage]` (branch, `fail_under = 85`) y se activa con `--cov` (`make
  coverage`, CI), para que ejecutar un solo fichero de tests no falle por umbral.
- CI usa `UV_PYTHON=<matrix>`: sin él, `uv run` recrea en silencio un entorno 3.12 por `.python-version`.
- Codecov: subida con el secret `CODECOV_TOKEN` y `fail_ci_if_error: false` (el umbral lo impone pytest).
  **Pendiente del usuario:** dar de alta el secret (pasos en `CONTRIBUTING.md`); sin él la subida se rechaza
  y el badge no muestra datos.
- Protección de `main`: las herramientas disponibles no permiten configurarla; se versiona
  `.github/rulesets/protect-main.json` (PR obligatorio, 3 checks requeridos, solo merge commit, sin
  force-push ni borrado). **Pendiente del usuario:** importarla (*Settings → Rules → Rulesets → Import*).
- Dependabot semanal agrupado para `uv` y GitHub Actions; CODEOWNERS, plantillas de PR/issues y
  `CONTRIBUTING.md`.

**Hallazgos (no corregidos: son lógica de ML, Fase 3).**
- La label del último día de train es la dirección del primer día de validación (no hay purga/embargo):
  los tests de leakage la excluyen explícitamente. Resolver con walk-forward + embargo.
- Los overrides de configuración se fusionan en profundidad (*deep merge*): sobrescribir un `grid` desde
  entorno o kwargs conserva las claves no mencionadas. Documentado en `tests/conftest.py`.

- Reproducibilidad entre máquinas (#29): la primera verificación en clon limpio dio modelos y métricas
  idénticos pero un `dvc.lock` distinto. Causas y arreglos: `__pycache__` dentro de los directorios de
  código que son dependencias (→ `.dvcignore`); `mlflow_run.json` y `promotion.json` dependen del store de
  MLflow (→ ya no son salidas DVC); el random forest con `n_jobs=-1` suma probabilidades en orden no
  determinista (diferencias de 1 ulp → bosque monohilo y búsqueda en paralelo, mismo tiempo).

**Lecciones de proceso.** `pre-commit run --all-files` solo ve ficheros versionados: hacer `git add` antes
de ejecutar los hooks (documentado en `CONTRIBUTING.md`). Los tests de CLI normalizan la salida (ANSI),
porque CI fuerza colores.


### Notas de cierre — Fase 3

**Resultado.** `brent train` → `brent report` → `brent explain` forman un flujo reproducible (semilla 42,
LightGBM determinista, TPE con semilla) que compara 5 modelos con 4 baselines fuera de muestra
(2024-01-01 → 2026-03-11, 10 ventanas walk-forward de 63 días, purga 1 + embargo 5) con IC al 95 % por
*block bootstrap*, DeLong con corrección de Holm, test binomial, calibración y backtest con costes. La
conclusión generada es honesta: **ningún modelo supera a la persistencia ni a buy & hold; todos los IC de
AUC contienen 0.5**. Todas las cifras son del dataset **sintético** (Kaggle inaccesible desde el entorno;
`configs/data_checksums.json` sigue con `null` y el informe lo declara "unverified").

**Métricas clave (sintético, OOS).**

| Modelo | AUC [IC 95 %] | Sharpe (5 pb) | ΔSharpe vs B&H [IC 95 %] |
|---|---|---|---|
| Logistic Regression | 0.536 [0.489, 0.582] | 1.11 | +0.06 [−1.09, +1.18] |
| SVM (RBF) | 0.491 [0.442, 0.543] | 0.79 | −0.27 [−1.84, +1.21] |
| Random Forest | 0.510 [0.465, 0.552] | −0.27 | −1.33 [−2.52, −0.21] |
| MLP NumPy | 0.501 [0.453, 0.549] | 1.00 | −0.06 [−1.33, +1.15] |
| LightGBM | 0.450 [0.401, 0.497] | −0.24 | −1.30 [−2.60, −0.07] |
| Persistencia (mejor baseline) | 0.507 [0.474, 0.538] | 0.90 | −0.16 [−1.04, +0.72] |
| Buy & hold | 0.500 | 1.06 | — |

Calidad: 422 tests (+1 `slow` e2e train → report → explain), **98.8 % de cobertura de ramas**, `mypy --strict`
sin `type: ignore` en `src/`, gradient check del MLP < 1e-6 y `check_estimator` en verde, CI 3.12/3.13 verde.

**Decisiones y desviaciones respecto al plan.**
- Bug de `create_label` corregido (la última fila sin retorno futuro se elimina en vez de etiquetarse 0);
  la validación de alineación lanza `ValueError` (no `assert`). Los tests de leakage ya no excluyen nada.
- Split fijo train/val/test sustituido por: tuning con CV walk-forward purgada (5 folds) en el periodo de
  desarrollo `< 2024-01-01` + evaluación walk-forward OOS reentrenando cada 63 sesiones (expanding por
  defecto, `rolling` configurable). `PurgedWalkForwardSplit` es un splitter de sklearn (sirve en
  `GridSearchCV`, `cross_val_score`, `learning_curve`).
- `SVC(probability=True)` → `CalibratedClassifierCV(sigmoid, ensemble=False)`; en la búsqueda se usa la
  SVC desnuda (AUC solo necesita `decision_function`, ~5× más rápido).
- MLP: early stopping sobre un *hold-out* cronológico interno (último 10 % del train) monitorizando la
  pérdida; antes usaba el propio conjunto de validación (AUC optimista). Ya no hay `X_val` en la API.
- Artefactos: un `.joblib` por modelo (pipeline completo reentrenado con todos los datos) en vez de un único
  `pipeline.joblib`, porque se comparan 5 modelos; el campeón se elegirá en la Fase 4 (registry).
- `brent report` es un comando aparte (lee `predictions.csv` + `metrics.json`, no reentrena); `brent
  explain` reentrena en desarrollo y explica en OOS para no explicar memorización.
- "Test binomial contra 50 %" → contra la *no-information rate* (más exigente y estándar, p. ej. caret).
- Feature v2: 29 candidatas (antes 14); `day_of_week`/`month` crudos sustituidos por sin/cos. `next_return`
  se guarda en el dataset solo para el backtest y está excluido explícitamente de las features (test).
- Solo LightGBM (no XGBoost): una librería de boosting basta para el objetivo; Optuna solo para modelos con
  `space` en el YAML (el resto mantiene grid, espacios pequeños y discretos).

**Deuda técnica / pendientes.**
- **Ejecutar con datos reales** (`brent data download` o `data verify`), commitear los checksums y
  sustituir las tablas sintéticas de README y `docs/MODEL_CARD.md`.
- `brent train` tarda ~11 min en 4 cores (Optuna 50 trials + SVM/RF con 29 features). Candidatos: cachear
  el preprocesado por fold, `n_jobs` en Optuna con sampler determinista o menos trials en CI.
- Optimismo del tuning sin medir (LightGBM: 0.517 en CV → 0.450 OOS): CV anidada o ventana de validación
  de tuning separada.
- El backtest ignora roll de futuros, financiación y slippage variable.
- `pipeline.run` concentra orquestación + IO; la Fase 4 (MLflow/DVC) es buen momento para dividirlo en
  etapas (`featurize`, `train`, `evaluate`).
- Dependabot #14 (bump de `uv_build`) sigue abierto para revisión del usuario.

- Reproducibilidad entre máquinas (#29): la primera verificación en clon limpio dio modelos y métricas
  idénticos pero un `dvc.lock` distinto. Causas y arreglos: `__pycache__` dentro de los directorios de
  código que son dependencias (→ `.dvcignore`); `mlflow_run.json` y `promotion.json` dependen del store de
  MLflow (→ ya no son salidas DVC); el random forest con `n_jobs=-1` suma probabilidades en orden no
  determinista (diferencias de 1 ulp → bosque monohilo y búsqueda en paralelo, mismo tiempo).

**Lecciones de proceso.** El servidor de PRs añade un pie automático a la descripción: se crea el PR con un
cuerpo mínimo y se sustituye con `update`. Para el merge con `expectedHeadSha` hace falta el SHA de 40
caracteres. Los indicadores con ventanas largas (momentum 63) amplían el *warm-up*: ajustar los tests que
cuentan filas descartadas.

### Notas de cierre — Fase 4

**Resultado.** El ciclo de vida completo es reproducible y trazable:
`uv run dvc repro` (download → validate → featurize → train → evaluate → explain) re-ejecuta solo lo que
cambió; cada `train` y `evaluate` queda en MLflow (run padre con protocolo, commit, SHA-256 de los datos,
métricas, predicciones, configuración y gráficos + un run anidado por candidato con su pipeline); `brent
report` aplica la regla de promoción y registra `champion`/`challenger` en el Model Registry; `brent
predict` carga `models:/brent-direction-classifier@champion`, construye las features del último día y las
valida con Pandera antes de predecir. Las métricas de ML no cambian (datos **sintéticos**, sin señal): el
campeón es el **baseline de persistencia** y la regresión logística queda como challenger.

**Verificación del DoD.**
- Clon limpio + `dvc pull` del remote local + `dvc repro --force --downstream validate`: las cinco etapas
  posteriores a `download` se reproducen (12 min), `dvc status` queda limpio y **`dvc.lock` sale idéntico
  byte a byte** (#29). `download` no puede ejecutarse en el entorno de desarrollo (la política de red
  bloquea Kaggle).
- MLflow: un run padre por entrenamiento (etapas `train` → `evaluate`) con 9 runs hijos; versiones del
  modelo registradas con alias y la razón de la promoción como tag.
- Campeón por alias → predicción validada: test e2e (`test_tracking.py`) y `brent predict` en CLI.
- Pandera: 23 tests de datos corruptos (`test_schemas.py`) + 21 de carga.
- CI verde en 3.12 y 3.13; 516 tests, cobertura de ramas 98.5 %; `mypy --strict` sin `type: ignore` en
  `src/`; `pre-commit` verde; `git grep -i claude` vacío; 133 commits, todos de rafaelhh04, sin trailers.

**Decisiones y desviaciones respecto al plan.**
- Regla de promoción explícita (`select_champion`): el mejor modelo por AUC OOS solo es `champion` si supera
  al mejor baseline con significancia (Holm-DeLong + IC bootstrap de ΔAUC > 0); si no, el campeón es el mejor
  **baseline** y el modelo queda como `challenger`. Por eso los baselines también se reentrenan y guardan.
- MLflow local por defecto (SQLite en `mlruns/mlflow.db` + artefactos en `mlruns/artifacts`), URI
  configurable. Modelos serializados con **skops** (no pickle) y una lista explícita de tipos de confianza,
  con un test que obliga a ampliarla si un modelo nuevo la necesita. `pip_requirements` fijados a mano
  (evita la inferencia lenta). MLflow 3 registra *logged models* (`models:/m-…`): se guardan sus URIs.
- DVC: remote por defecto local fuera del repo (`../brent-dvc-storage`); GCS/S3 documentados. La etapa
  `download` depende de parámetros (si no, sería una *callback stage* que se ejecuta siempre). Los plots se
  declaran fichero a fichero porque `results/plots/` lo escriben tres etapas. Fechas del YAML entre comillas
  (DVC guarda los parámetros como JSON). `brent train --features` entrena sobre el snapshot versionado.
- Las métricas son `cache: false` en `results/` (ignorado por git) y no se versionan en git (son artefactos
  generados): `dvc metrics diff` entre commits no tiene columna base; la comparación histórica vive en MLflow.
- Pandera sustituye las validaciones ad hoc de `data/load.py`; validación *lazy* con `DataValidationError`
  legible. Rangos de plausibilidad laxos; el WTI no se acota (cotizó a −37 USD el 2020-04-20).
- Ingesta en vivo con la API chart de Yahoo Finance (Brent `BZ=F`, WTI `CL=F`, DXY `DX-Y.NYB`) y FRED
  (`VIXCLS`) vía `urllib` (sin `yfinance`: una dependencia menos y un cliente inyectable trivial de testear);
  se escribe en `data/live/` y nunca toca el snapshot versionado. DXY de Yahoo, no `DTWEXBGS` de FRED (otro
  índice y otra escala).

**Deuda técnica / pendientes.**
- **Datos reales**: la política de red del entorno bloquea Kaggle, Yahoo Finance y FRED (proxy 403). Con
  acceso: `brent data download`, commitear checksums, `dvc repro`, sustituir tablas sintéticas y probar
  `brent data ingest` contra las fuentes reales (los formatos están cubiertos con respuestas simuladas).
- Sin fuente en vivo para el GPR ni los eventos (se arrastra el último GPR; días nuevos sin evento).
- `derivation_gap` asume que las columnas derivadas de Kaggle siguen las definiciones del dataset sintético
  (retornos en %, volatilidad = std de 7/30 retornos); comprobarlo con los datos reales (avisa si difieren).
- `brent train` sigue tardando ~11–12 min (+25 s de tracking); `dvc repro` evita repetirlo si nada cambió.
- El registro crea versiones nuevas en cada `brent report` (también si el campeón no cambia).
- Pendientes del usuario: secret `CODECOV_TOKEN`, importar `.github/rulesets/protect-main.json`, Dependabot
  #14, y permitir los dominios de Kaggle/Yahoo/FRED en la red del entorno si se quieren datos reales aquí.

- Reproducibilidad entre máquinas (#29): la primera verificación en clon limpio dio modelos y métricas
  idénticos pero un `dvc.lock` distinto. Causas y arreglos: `__pycache__` dentro de los directorios de
  código que son dependencias (→ `.dvcignore`); `mlflow_run.json` y `promotion.json` dependen del store de
  MLflow (→ ya no son salidas DVC); el random forest con `n_jobs=-1` suma probabilidades en orden no
  determinista (diferencias de 1 ulp → bosque monohilo y búsqueda en paralelo, mismo tiempo).

**Lecciones de proceso.** Los comandos que imprimen JSON se prueban con `--log-level WARNING` (CliRunner
mezcla stderr). `dvc.lock` se excluye de los hooks de espacios en blanco (lo genera DVC). Cambiar un módulo
listado como dependencia invalida la etapa aunque el resultado sea idéntico: DVC re-ejecuta esa etapa pero
salta las siguientes si el hash de su salida no cambia.

---

## Plantilla de prompt de traspaso entre fases

Al terminar cada fase, el agente que la ejecutó debe rellenar y entregar esto:

```text
Contexto: repo rafaelhh04/PrediccionPetroleo-Clasificacion. Sigue docs/ROADMAP.md.
Ejecuta la FASE <N+1> — <nombre>.

Progreso hasta ahora:
- Fases completadas: <lista> (ramas mergeadas: <lista>)
- Estado de CI / cobertura / métricas clave: <valores>

Decisiones tomadas (y cambios respecto al plan):
- <decisión> — motivo: <por qué>

Pendientes / deuda técnica arrastrada:
- <item>

A tener en cuenta:
- <gotchas: rutas, comandos, variables de entorno, tests lentos, etc.>

Reglas obligatorias:
- Commits como rafaelhh04 <132058207+rafaelhh04@users.noreply.github.com>; sin Co-Authored-By
  ni ninguna referencia a asistentes de IA en commits, código, docs ni PRs.
- Una rama por funcionalidad con prefijo (feat/, fix/, refactor/, test/, ci/, docs/, chore/) desde main.
- Conventional Commits. Lint, tipos y tests en verde antes de cada push.
- Al acabar: actualiza "Registro de progreso" en docs/ROADMAP.md y entrega el prompt de la fase siguiente
  con esta misma plantilla.
```
