# Developer shortcuts. Every target runs inside the uv-managed environment.
.DEFAULT_GOAL := help
.PHONY: help install lint format typecheck test coverage test-all data train serve load-test clean

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

train: ## Run the full training pipeline
	uv run brent train

API_URL ?= http://127.0.0.1:8000

serve: ## Serve the registered champion over HTTP (docs at /docs)
	uv run brent serve --port 8000

load-test: ## Load-test a running API (`make serve`) with Locust; CSVs in results/load/
	uv run --group load locust -f tests/load/locustfile.py --headless --host $(API_URL) \
		--users 10 --spawn-rate 5 --run-time 60s --csv results/load/locust --only-summary

clean: ## Remove caches and generated results
	rm -rf .ruff_cache .mypy_cache .pytest_cache .coverage htmlcov results
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
