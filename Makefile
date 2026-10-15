# Developer shortcuts. Every target runs inside the uv-managed environment.
.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test coverage test-all data demo train retrain drift serve dashboard load-test docker-build docker-train docker-up infra-check docs docs-serve changelog clean

help: ## Show this help
	@grep -E '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-10s\033[0m %s\n", $$1, $$2}'

install: ## Install dependencies (from uv.lock) and the git hooks
	uv sync --frozen
	uv run pre-commit install

lint: ## Run every pre-commit hook on the whole repository
	uv run pre-commit run --all-files

format: ## Auto-format and auto-fix with ruff
	uv run ruff format src tests
	uv run ruff check --fix src tests

typecheck: ## Static type checking (mypy --strict on src/)
	uv run mypy

test: ## Run the test suite (slow end-to-end tests excluded)
	uv run pytest

coverage: ## Run the tests with branch coverage (fails under 85 %) and an HTML report
	uv run pytest --cov --cov-report=term-missing --cov-report=html

test-all: ## Run every test, including the slow end-to-end ones
	uv run pytest -m "slow or not slow"

data: ## Download the Kaggle dataset and verify checksums
	uv run brent data download

DEMO_CONFIG := --config configs/default.yaml --config configs/demo.yaml
OIL_CSV := data/raw/oil_geopolitics_dataset_2010_2026.csv

demo: ## End-to-end demo on synthetic data: install, data, train (fast profile), report + champion, explain
	uv sync --frozen
	@if [ -e $(OIL_CSV) ]; then echo "$(OIL_CSV) exists: using it"; else uv run brent data synthetic; fi
	uv run brent $(DEMO_CONFIG) train
	uv run brent $(DEMO_CONFIG) report
	uv run brent $(DEMO_CONFIG) explain
	@echo "Done. Next: 'make serve' (API, :8000/docs) and, in another shell, 'make dashboard' (:8501)."

train: ## Run the full training pipeline
	uv run brent train

retrain: ## Scheduled retraining on local data: train, report + promotion, explain (skips if unchanged)
	uv run brent retrain --no-ingest

drift: ## Data and prediction drift report (Evidently) in results/drift/
	uv run brent monitor drift

API_URL ?= http://127.0.0.1:8000

serve: ## Serve the registered champion over HTTP (docs at /docs)
	uv run brent serve --port 8000

dashboard: ## Run the Streamlit demo on :8501 (needs `make serve` for the prediction)
	uv run brent dashboard --port 8501

load-test: ## Load-test a running API (`make serve`) with Locust; CSVs in results/load/
	uv run --group load locust -f tests/load/locustfile.py --headless --host $(API_URL) \
		--users 10 --spawn-rate 5 --run-time 60s --csv results/load/locust --only-summary

docker-build: ## Build the inference image (brent-forecast:latest)
	docker compose build

docker-train: ## Train, report and register the champion inside the compose stack
	docker compose --profile train run --rm trainer

docker-up: ## Start the API (:8000), the dashboard (:8501) and MLflow (:5000) in the background
	docker compose up -d

TERRAFORM ?= terraform

infra-check: ## Terraform: fmt check, validate both stacks (no backend) and plan-level tests
	$(TERRAFORM) -chdir=infra fmt -recursive -check
	$(TERRAFORM) -chdir=infra/bootstrap init -backend=false -input=false
	$(TERRAFORM) -chdir=infra/bootstrap validate
	$(TERRAFORM) -chdir=infra/bootstrap test
	$(TERRAFORM) -chdir=infra init -backend=false -input=false
	$(TERRAFORM) -chdir=infra validate
	$(TERRAFORM) -chdir=infra test

docs: ## Build the documentation site with warnings as errors (site/)
	uv run --group docs mkdocs build --strict

docs-serve: ## Serve the documentation site with live reload on :8001
	uv run --group docs mkdocs serve --dev-addr 127.0.0.1:8001

GIT_CLIFF ?= uvx git-cliff@2.14.2

changelog: ## Regenerate CHANGELOG.md from the Conventional Commits (VERSION=X.Y.Z for a release)
	$(GIT_CLIFF) $(if $(VERSION),--tag v$(VERSION)) --output CHANGELOG.md

clean: ## Remove caches and generated results
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage htmlcov results site
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
