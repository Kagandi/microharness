# AGENTS.md

Instructions and project conventions for AI agents operating on `mini-harness`.

## Guiding Principles
- **Zero Runtime Dependencies**: The core harness must only rely on Python's standard library. Do not add runtime dependencies to `pyproject.toml`.
- **Pluggability**: Components must implement minimal protocols (`Model`, `Tool`, `Evaluator`). Never introduce provider-specific subclasses or hardcoded branching.
- **Testing Standard**: Every feature must include rigorous unit tests in `tests/`. Never write tautological "pass-through" assertions (`assertIn(result, [A, B, C])`). Always use strict equality and test negative failure cases.
- **Verification**: Run `uv run pytest` before concluding any changes.
