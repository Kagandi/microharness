# AGENTS.md

Instructions and project conventions for AI agents operating on `microharness`.

## Guiding Principles
- **Zero Runtime Dependencies**: The core harness must only rely on Python's standard library. Do not add runtime dependencies to `pyproject.toml`.
- **Pluggability**: Components must implement minimal protocols (`Model`, `Tool`, `Evaluator`). Never introduce provider-specific subclasses or hardcoded branching.
- **Testing Standard**: Every feature must include rigorous unit tests in `tests/`. Never write tautological "pass-through" assertions (`assertIn(result, [A, B, C])`). Always use strict equality and test negative failure cases.
- **Verification**: Run `uv run pytest` before concluding any changes.

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
