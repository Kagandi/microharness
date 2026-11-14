# AGENTS.md

Instructions and project conventions for AI agents operating on `microharness`.

## Guiding Principles
- **Zero Runtime Dependencies**: The core harness must only rely on Python's standard library. Do not add runtime dependencies to `pyproject.toml`.
- **Pluggability**: Components must implement minimal protocols (`Model`, `Tool`, `Evaluator`). Never introduce provider-specific subclasses or hardcoded branching.
- **Testing Standard**: Every feature must include rigorous unit tests in `tests/`. Never write tautological "pass-through" assertions (`assertIn(result, [A, B, C])`). Always use strict equality and test negative failure cases.
- **Verification**: Run `uv run pytest` and verify formatting/linting with `uv run ruff check .` and `uv run ruff format --check .` (or `uv run pre-commit run --all-files`) before concluding any changes.

## CI & Quality Automation

- **PR and Push CI (`.github/workflows/ci.yml`)**:
  - Automatically runs on pull requests targeting `main` and pushes to `main`.
  - Jobs:
    - `lint`: Validates style and formatting with `ruff check .` and `ruff format --check .`.
    - `test`: Executes `pytest` across Python versions (`3.11`, `3.12`, `3.13`).
    - `ci-status`: Umbrella status check aggregating `lint` and `test` results to configure as a required status check in GitHub branch protection rulesets.
- **Security Scanning (`.github/workflows/codeql.yml`)**:
  - Runs GitHub CodeQL static analysis for Python on pushes to `main`, PRs to `main`, and scheduled weekly scans.
- **Pre-commit Hooks (`.pre-commit-config.yaml`)**:
  - Configured with file integrity checks (trailing whitespace, end of file, YAML, TOML) and Astral Ruff linting + formatting.
  - Run locally across all files using `uv run pre-commit run --all-files`.

## Packaging & Publishing to PyPI

- **Packaging Setup**:
  - The project uses `hatchling` as the build backend via PEP 621.
  - The wheel target must include packages `["core", "benchmarks", "microharness"]` and force-include `"main.py"` so CLI entrypoints (`microharness`, `mini-harness`) and top-level imports resolve cleanly.
  - Test package builds using `uv build` and verify artifacts in clean environments.

- **Automated Publishing via GitHub Actions & PyPI Trusted Publishing (OIDC)**:
  - Workflow path: `.github/workflows/publish.yml`
  - Uses OIDC Trusted Publishing with zero stored secrets (`permissions: id-token: write` and `environment: pypi`).
  - **PyPI Trusted Publisher Configuration** (configured at https://pypi.org/manage/project/microharness/settings/publishing/):
    - **Owner**: `Kagandi`
    - **Repository name**: `microharness`
    - **Workflow name**: `publish.yml`
    - **Environment name**: `pypi`
  - **Release Triggers**:
    - Pushing a git tag matching `v*` (e.g., `git tag v0.1.1 && git push origin v0.1.1`).
    - Publishing a GitHub Release.
    - Manual trigger via GitHub Actions `workflow_dispatch`.
  - The pipeline automatically runs the test suite (`uv run pytest`) and linter (`uv run ruff check .`) before building and publishing.
