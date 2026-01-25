.PHONY: help install test test-cov lint format type-check pre-commit clean benchmark

help:  ## Show this help message
	@echo "Available commands:"
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'

install:  ## Install dependencies with uv
	uv sync

install-dev:  ## Install development dependencies
	uv sync
	uv pip install pre-commit
	pre-commit install

test:  ## Run tests (excluding benchmarks)
	uv run pytest test/ -v -m "not benchmark"

test-cov:  ## Run tests with coverage report
	uv run pytest test/ -v --cov=source --cov-report=term-missing --cov-report=html -m "not benchmark"

benchmark:  ## Run performance benchmarks
	uv run pytest test/ -m benchmark --benchmark-only

lint:  ## Run linter (ruff)
	uv run ruff check source/ test/

format:  ## Format code with ruff
	uv run ruff format source/ test/

format-check:  ## Check code formatting without modifying
	uv run ruff format --check source/ test/

type-check:  ## Run type checker (mypy)
	uv run mypy source/ --ignore-missing-imports

pre-commit:  ## Run all pre-commit hooks
	pre-commit run --all-files

clean:  ## Remove cache and build artifacts
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".mypy_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".ruff_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete
	rm -rf htmlcov/ .coverage coverage.xml

ci:  ## Run CI checks locally (lint, type-check, test)
	@echo "Running CI checks locally..."
	@make lint
	@make type-check
	@make test-cov
	@echo "✅ All CI checks passed!"
