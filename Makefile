.PHONY: checks, docs-serve, docs-build, fmt, config, type, tests

# Run pre-commit checks
# ------------------------------------------------------------------------------
checks:
	uvx pre-commit run --all-files

docs-serve:
	uv run mkdocs serve

docs-build:
	uv run mkdocs build

fmt:
	uv run ruff format src tests

lint:
	uv run ruff check --fix src tests

type:
	uv run mypy src --install-types --non-interactive --show-traceback

tests:
	uv run pytest --cov=REPLACE_PACKAGE_NAME --cov-report=term-missing tests/ -s -vv

config: ## Configure .netrc with Owkin's codeartifact credentials
	$(eval CODEARTIFACT_PASSWORD ?= $(shell bash -c 'aws codeartifact get-authorization-token --domain abstra --domain-owner 058264397262 --query authorizationToken --output text'))
	@if [ -z "$(CODEARTIFACT_PASSWORD)" ]; then \
		echo "Error: CODEARTIFACT_PASSWORD must be set"; \
		exit 1; \
	fi
	@if [ ! -f ~/.netrc ]; then touch ~/.netrc; fi
	@chmod 600 ~/.netrc
	@if grep -q "machine abstra-058264397262.d.codeartifact.eu-west-1.amazonaws.com" ~/.netrc; then \
		sed -i "/machine abstra-058264397262.d.codeartifact.eu-west-1.amazonaws.com/,+2d" ~/.netrc; \
	fi
	@echo "machine abstra-058264397262.d.codeartifact.eu-west-1.amazonaws.com" >> ~/.netrc
	@echo "login aws" >> ~/.netrc
	@echo "password $(CODEARTIFACT_PASSWORD)" >> ~/.netrc
