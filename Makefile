.PHONY: checks, fmt, config, type, tests

# Run pre-commit checks
# ------------------------------------------------------------------------------
checks:
	uvx pre-commit run --all-files

fmt:
	uv run ruff format src tests

lint:
	uv run ruff check --fix src tests

type:
	uv run mypy src --install-types --non-interactive --show-traceback

tests:
	uv run pytest --cov=imilia --cov-report=term-missing tests/ -s -vv

install-uv: ## Install uv
	curl -LsSf https://astral.sh/uv/install.sh | sh

install: ## Install all package and development dependencies for testing to the active Python's site-packages
	uv sync --all-extras

install-all: install-uv install ## Install uv along with all package and development dependencies
