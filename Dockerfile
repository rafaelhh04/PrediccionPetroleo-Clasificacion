# syntax=docker/dockerfile:1
# Inference image of the Brent direction classifier (`brent serve`).
#
#   docker build -t brent-forecast .
#   docker run --rm -p 8000:8000 -v "$PWD/mlruns:/app/mlruns" -v "$PWD/data:/app/data:ro" brent-forecast
#
# Two stages: `builder` resolves the locked runtime dependencies with uv (no dev or
# load-test groups) into /app/.venv; `runtime` copies only that environment, the
# configuration and nothing else, and runs as an unprivileged user.

# Base images are pinned in the FROM lines so Dependabot can update them.
FROM ghcr.io/astral-sh/uv:0.11.32 AS uv

# ── builder ───────────────────────────────────────────────────
FROM python:3.12-slim-trixie AS builder
COPY --from=uv /uv /usr/local/bin/uv
ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    UV_PROJECT_ENVIRONMENT=/app/.venv
WORKDIR /src

# Dependencies first: this layer is reused until uv.lock changes.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-dev --no-install-project

# Then the project itself, installed as a regular (non-editable) package.
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev --no-editable

# ── runtime ───────────────────────────────────────────────────
FROM python:3.12-slim-trixie AS runtime

LABEL org.opencontainers.image.title="brent-forecast" \
      org.opencontainers.image.description="Next-day Brent crude oil direction classifier (inference API)" \
      org.opencontainers.image.source="https://github.com/rafaelhh04/PrediccionPetroleo-Clasificacion" \
      org.opencontainers.image.licenses="MIT"

# libgomp1: OpenMP runtime required by LightGBM (the challenger or a future champion).
RUN apt-get update \
    && apt-get install --yes --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd --system --gid 10001 brent \
    && useradd --system --uid 10001 --gid brent --home-dir /app --no-create-home brent \
    && mkdir -p /app/mlruns /app/results /app/data \
    && chown -R brent:brent /app

WORKDIR /app
COPY --from=builder --chown=brent:brent /app/.venv /app/.venv
COPY --chown=brent:brent configs ./configs

ENV PATH=/app/.venv/bin:$PATH \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MLFLOW_DISABLE_AGENT_HINT=1 \
    GIT_PYTHON_REFRESH=quiet \
    MPLCONFIGDIR=/tmp/matplotlib

USER brent:brent
EXPOSE 8000
VOLUME ["/app/mlruns", "/app/results"]

# Readiness: 200 only once the registered model is loaded (503 before the first `brent report`).
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4)"]

ENTRYPOINT ["brent"]
CMD ["serve", "--host", "0.0.0.0", "--port", "8000"]
