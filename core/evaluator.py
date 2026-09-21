from __future__ import annotations

from typing import Protocol

from core.environment import Environment
from core.types import EvalResult, StepTrace


class Evaluator(Protocol):
    """Protocol for scoring and assessing agent performance."""

    def evaluate(self, env: Environment, steps: list[StepTrace]) -> EvalResult:
        """Inspect environment or trajectory and return assessment."""
        ...


class CommandTestEvaluator:
    """Evaluates task completion by running a verification command (e.g. pytest)."""

    def __init__(
        self,
        command: str = "pytest",
        expected_exit_code: int = 0,
        timeout: float = 30.0,
    ):
        self.command = command
        self.expected_exit_code = expected_exit_code
        self.timeout = timeout

    def evaluate(self, env: Environment, steps: list[StepTrace]) -> EvalResult:
        exit_code, output = env.execute_bash(self.command, timeout=self.timeout)
        passed = exit_code == self.expected_exit_code

        feedback = (
            f"Command '{self.command}' passed (exit {exit_code}).\nOutput:\n{output.strip()}"
            if passed
            else f"Command '{self.command}' failed with exit {exit_code}.\nOutput:\n{output.strip()}"
        )

        return EvalResult(
            passed=passed,
            score=1.0 if passed else 0.0,
            feedback=feedback,
        )


class FileContentEvaluator:
    """Evaluates task completion by checking file presence and content patterns."""

    def __init__(
        self,
        file_path: str,
        must_contain: list[str] | None = None,
        must_not_contain: list[str] | None = None,
    ):
        self.file_path = file_path
        self.must_contain = must_contain or []
        self.must_not_contain = must_not_contain or []

    def evaluate(self, env: Environment, steps: list[StepTrace]) -> EvalResult:
        try:
            full_path = env.resolve_path(self.file_path)
            if not full_path.exists():
                return EvalResult(
                    passed=False,
                    score=0.0,
                    feedback=f"Target file '{self.file_path}' does not exist.",
                )

            content = full_path.read_text(encoding="utf-8")
        except Exception as exc:
            return EvalResult(
                passed=False,
                score=0.0,
                feedback=f"Error accessing file '{self.file_path}': {exc}",
            )

        for required in self.must_contain:
            if required not in content:
                return EvalResult(
                    passed=False,
                    score=0.0,
                    feedback=f"File '{self.file_path}' is missing required string: '{required}'",
                )

        for forbidden in self.must_not_contain:
            if forbidden in content:
                return EvalResult(
                    passed=False,
                    score=0.0,
                    feedback=f"File '{self.file_path}' contains forbidden string: '{forbidden}'",
                )

        return EvalResult(
            passed=True,
            score=1.0,
            feedback=f"File '{self.file_path}' satisfied all content assertions.",
        )
