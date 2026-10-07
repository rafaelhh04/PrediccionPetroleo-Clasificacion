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
| 3 | ⏳ pendiente | | |
| 4 | ⏳ pendiente | | |
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

**Lecciones de proceso.** `pre-commit run --all-files` solo ve ficheros versionados: hacer `git add` antes
de ejecutar los hooks (documentado en `CONTRIBUTING.md`). Los tests de CLI normalizan la salida (ANSI),
porque CI fuerza colores.

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
