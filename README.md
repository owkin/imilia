<h1 align="center">REPLACE_PACKAGE_NAME</h1>

<div align="center">
    <img src="https://img.shields.io/badge/version-0.0.1-orange.svg" />
    <img src="https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue?logo=python" />
    <a href="https://github.com/owkin/REPLACE_PACKAGE_NAME/actions?workflow=docs" target="_blank">
        <img src="https://github.com/owkin/REPLACE_PACKAGE_NAME/workflows/docs/badge.svg" />
    </a>
    <a href="https://github.com/owkin/REPLACE_PACKAGE_NAME/actions?workflow=ci-cd" target="_blank">
        <img src="https://github.com/owkin/REPLACE_PACKAGE_NAME/workflows/ci-cd/badge.svg" />
    </a>
    <img src="assets/cov_badge.svg"/>
    <a href="https://docs.astral.sh/uv/" target="_blank">
        <img src="https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/refs/heads/main/assets/badge/v0.json" />
    </a>
</div>

<p align="center"><em>REPLACE_PACKAGE_DESCRIPTION</em></p>

---

## ✨ Features

* Package configuration with `pyproject.toml` built with `uv`
* Code formatting and linting with `ruff`
* `pre-commit` configuration file
* CI-CD Pipelines with GitHub Actions
* Publish package on Owkin's registry
* Basic `pytest` set-up for unit tests
* Auto-generated docs with `mkdocs` and `mkdocs-material` (check it [here](https://symmetrical-guide-ev6vn8m.pages.github.io/))

## 🚚 Replacements

* `REPLACE_PACKAGE_NAME`: name of the package (usually the same name as the repository in which it's hosted).
* `REPLACE_PACKAGE_DESCRIPTION`: description of the package.
* `REPLACE_FULL_NAME`: user's full name.
* `REPLACE_EMAIL_NAME`: user's email name.

## 📦 Installation

Use the `Use this template` button on the top right of the repository to create a new repository with the same structure.

> [!WARNING]
> If your package name has more than one word, replace the underscores (`_`) with hyphens (`-`) in the `pyproject.toml` file.

## 🚀 Publishing

In order to publish on the CodeArtifact repository, you will need specific credentials.

> [!IMPORTANT]
> Reach out to the Platform team through Zendesk to have a role provisioned for your repository. You will need to provide the name of the package you want to publish, this name will be the same as the `project.name` in the `pyproject.toml` file.
