"""Test the integrity of the package files."""

import re
from pathlib import Path

import pytest
from packaging import version


class TestPackageIntegrity:
    """Test the integrity of the package files."""

    ROOT = Path(__file__).parent.parent

    @pytest.mark.parametrize(
        "filepath",
        [
            ".gitattributes",
            ".gitignore",
            ".pre-commit-config.yaml",
            ".python-version",
            "assets",
            "CONTRIBUTING.md",
            "docs",
            "Makefile",
            "mkdocs.yml",
            "pyproject.toml",
            "README.md",
            "src",
            "tests",
            "uv.lock",
        ],
    )
    def test_important_files(self, filepath: str) -> None:
        """Test that important files are present."""
        assert self.ROOT.joinpath(filepath).exists()

    @pytest.mark.usefixtures("readme_content")
    def test_versions_are_aligned(self, readme_content) -> None:
        """Test that the versions are aligned."""
        from REPLACE_PACKAGE_NAME import __version__

        match = re.search(r'<img src="https://img\.shields\.io/badge/version-([0-9.]+)-[a-z]+\.svg"', readme_content)
        if match:
            version_readme = match.group(1)
        else:
            raise ValueError("Version badge not found in README.md")

        assert __version__ == version_readme

    @pytest.mark.usefixtures("pyproject_content", "python_version")
    def test_pinned_python_version(self, pyproject_content, python_version) -> None:
        """Test that the python versions are aligned."""
        python_version_pyproject = pyproject_content.split('python = "')[1].split('"')[0].split(",")[0]

        if (
            python_version_pyproject.startswith(">=")
            or python_version_pyproject.startswith("==")
            or python_version_pyproject.startswith("<=")
            or python_version_pyproject.startswith("~=")
        ):
            python_version_pyproject = python_version_pyproject[2:]
        elif (
            python_version_pyproject.startswith(">")
            or python_version_pyproject.startswith("<")
            or python_version_pyproject.startswith("~")
            or python_version_pyproject.startswith("^")
        ):
            python_version_pyproject = python_version_pyproject[1:]

        assert version.parse(python_version) >= version.parse(python_version_pyproject)
