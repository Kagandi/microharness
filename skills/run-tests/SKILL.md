# Test Execution & Verification

Run the test suite and verify code quality for mini-harness.

## Commands
1. Run all unit & integration tests:
   ```bash
   uv run pytest -v
   ```
2. Run standard library unittest runner (zero install check):
   ```bash
   python3 -m unittest discover -s tests -v
   ```
3. Format and lint:
   ```bash
   uv run ruff check .
   ```
