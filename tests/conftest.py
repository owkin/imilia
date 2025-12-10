"""Fixtures module for the tests."""

import pytest


@pytest.fixture()
def python_version() -> str:
    """Return the content of the .python-version file."""
    with open(".python-version", "r") as file:
        return file.read().strip()


@pytest.fixture()
def pyproject_content() -> str:
    """Return the content of the pyproject.toml file."""
    with open("pyproject.toml", "r") as file:
        return file.read()


@pytest.fixture()
def readme_content() -> str:
    """Return the content of the README.md file."""
    with open("README.md", "r") as file:
        return file.read()
