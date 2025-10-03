# Makefile for EnvTorch development and testing

.PHONY: help test test-unit test-integration test-performance test-edge test-quick test-coverage lint format clean install dev-install validate

# Default target
help:
	@echo "EnvTorch Development Commands"
	@echo "============================="
	@echo ""
	@echo "Testing:"
	@echo "  test              Run all tests"
	@echo "  test-unit         Run unit tests only"
	@echo "  test-integration  Run integration tests only"
	@echo "  test-performance  Run performance tests only"
	@echo "  test-edge         Run edge case tests only"
	@echo "  test-quick        Run quick tests (exclude slow tests)"
	@echo "  test-coverage     Run tests with coverage reporting"
	@echo ""
	@echo "Code Quality:"
	@echo "  lint              Run linting (flake8)"
	@echo "  format            Format code (if black is installed)"
	@echo "  validate          Run full validation suite"
	@echo ""
	@echo "Setup:"
	@echo "  install           Install package in current environment"
	@echo "  dev-install       Install package in development mode"
	@echo "  clean             Clean build artifacts"

# Testing targets
test:
	python tests/test_runner.py all

test-unit:
	python tests/test_runner.py unit

test-integration:
	python tests/test_runner.py integration

test-performance:
	python tests/test_runner.py performance

test-edge:
	python tests/test_runner.py edge-cases

test-quick:
	python tests/test_runner.py quick

test-coverage:
	python tests/test_runner.py coverage

# Code quality targets
lint:
	python tests/test_runner.py lint

format:
	@command -v black >/dev/null 2>&1 && black src/ tests/ || echo "black not installed, skipping formatting"
	@command -v isort >/dev/null 2>&1 && isort src/ tests/ || echo "isort not installed, skipping import sorting"

validate:
	python tests/test_runner.py validate

# Setup targets
install:
	pip install -e .

dev-install:
	pip install -e ".[dev]"

clean:
	rm -rf build/
	rm -rf dist/
	rm -rf *.egg-info/
	rm -rf .coverage
	rm -rf htmlcov/
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete

# Quick development workflow
dev: clean dev-install lint test-quick

# CI/CD workflow
ci: clean install validate