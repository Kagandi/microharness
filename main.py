"""mini-harness: Educational demonstration of an AI agent harness.

Demonstrates:
1. Boundary between Harness (chassis) and AgentLoop (engine).
2. Workspace isolation & sandboxing.
3. Real bash and file tool execution.
4. Large log truncation + spill to disk (.logs/).
5. Context memory compaction.
6. Structured tracing (trace.json and trace.md).
7. Automated post-run evaluation.
8. True pluggability: plug in any Model, Tools, or Evaluators without modifying the harness.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.evaluator import CommandTestEvaluator
from core.harness import Harness
from core.model import HttpChatModel, MockModel, Model
from core.types import Task, ToolCall


def load_dotenv(path: str = ".env") -> None:
    """Minimal zero-dependency .env file loader."""
    env_file = Path(path)
    if not env_file.is_file():
        return
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        val = val.strip().strip("'\"")
        # Existing environment variables take precedence over .env file
        if key and key not in os.environ:
            os.environ[key] = val


def create_demo_task() -> Task:
    """Creates a sample software engineering task with an intentional bug."""
    initial_calculator_code = (
        "def add(a: float, b: float) -> float:\n"
        "    return a + b\n\n"
        "def divide(a: float, b: float) -> float:\n"
        "    # BUG: Multiplying instead of dividing, and missing ZeroDivision handling\n"
        "    return a * b\n"
    )

    initial_test_code = (
        "import unittest\n"
        "from calculator import add, divide\n\n"
        "class TestCalculator(unittest.TestCase):\n"
        "    def test_add(self):\n"
        "        self.assertEqual(add(2, 3), 5)\n\n"
        "    def test_divide(self):\n"
        "        self.assertEqual(divide(10, 2), 5)\n"
        "        with self.assertRaises(ValueError):\n"
        "            divide(5, 0)\n\n"
        "if __name__ == '__main__':\n"
        "    unittest.main()\n"
    )

    return Task(
        id="fix_calculator_bug",
        instruction=(
            "The calculator suite is failing. Run the unit tests using `python -m unittest test_calculator.py`.\n"
            "Investigate the failure, fix `calculator.py` so all tests pass (including raising ValueError on zero division),\n"
            "and verify that the tests succeed."
        ),
        initial_files={
            "calculator.py": initial_calculator_code,
            "test_calculator.py": initial_test_code,
        },
        max_steps=10,
        timeout_seconds=60.0,
    )


def build_mock_agent_script() -> MockModel:
    """Deterministic scripted model demonstrating the full agent trajectory."""
    fixed_code = (
        "def add(a: float, b: float) -> float:\n"
        "    return a + b\n\n"
        "def divide(a: float, b: float) -> float:\n"
        "    if b == 0:\n"
        "        raise ValueError('Cannot divide by zero')\n"
        "    return a / b\n"
    )

    script = [
        (
            "I will first run the tests to inspect what fails.",
            [
                ToolCall(
                    id="call_01",
                    name="bash",
                    arguments={"command": "python -m unittest test_calculator.py"},
                )
            ],
        ),
        (
            "Let me print a verbose diagnostic report to see system details.",
            [
                ToolCall(
                    id="call_02",
                    name="bash",
                    arguments={
                        "command": "python -c \"for i in range(120): print(f'DEBUG [Log Entry {i:03d}]: Diagnostic check ok. Testing buffer overflow handling.')\""
                    },
                )
            ],
        ),
        (
            "Now let's inspect the calculator implementation.",
            [
                ToolCall(
                    id="call_03",
                    name="read_file",
                    arguments={"path": "calculator.py"},
                )
            ],
        ),
        (
            "I see the bug: `divide` performs multiplication and does not raise `ValueError` on 0. Fixing it now.",
            [
                ToolCall(
                    id="call_04",
                    name="write_file",
                    arguments={"path": "calculator.py", "content": fixed_code},
                )
            ],
        ),
        (
            "Now re-running unit tests to verify the fix.",
            [
                ToolCall(
                    id="call_05",
                    name="bash",
                    arguments={"command": "python -m unittest test_calculator.py"},
                )
            ],
        ),
        (
            "The tests now pass completely. The bug in `calculator.py` has been resolved.",
            [],
        ),
    ]

    return MockModel(script=script)


def main():
    # Load .env file automatically if present
    load_dotenv()

    parser = argparse.ArgumentParser(description="Run the mini-harness educational demonstration.")
    parser.add_argument(
        "--endpoint",
        default=os.getenv("MODEL_ENDPOINT"),
        help="HTTP chat completions endpoint (env: MODEL_ENDPOINT)",
    )
    parser.add_argument(
        "--model",
        default=os.getenv("MODEL_NAME"),
        help="Model identifier (env: MODEL_NAME)",
    )
    parser.add_argument(
        "--api-key",
        default=os.getenv("MODEL_API_KEY", os.getenv("OPENROUTER_API_KEY")),
        help="API key (env: MODEL_API_KEY or OPENROUTER_API_KEY)",
    )
    args = parser.parse_args()

    # 1. Select Model: If an endpoint is configured, connect via HttpChatModel; else use MockModel.
    if args.endpoint:
        model_name = args.model or "qwen2.5-coder"
        print(f"🔌 Connected to: {args.endpoint} (model: {model_name})")
        model: Model = HttpChatModel(
            base_url=args.endpoint,
            model=model_name,
            api_key=args.api_key,
        )
    else:
        print("🎓 Running with deterministic MockModel (offline, no endpoint needed).")
        print("Tip: Add MODEL_ENDPOINT to .env or pass --endpoint to connect a live model.\n")
        model = build_mock_agent_script()

    # 2. Define Evaluator (tests must pass with exit code 0)
    evaluator = CommandTestEvaluator(
        command="python -m unittest test_calculator.py",
        expected_exit_code=0,
    )

    # 3. Initialize Harness (plugs in the chosen model, evaluator, and defaults)
    harness = Harness(
        model=model,
        evaluator=evaluator,
        max_tool_chars=1200,    # Truncation demonstration threshold
        max_context_chars=3500,  # Compaction demonstration threshold
        verbose=True,
    )

    # 4. Run Task
    task = create_demo_task()
    result = harness.run(task)

    print("=" * 60)
    print(f"Outcome: {'PASSED ✅' if result.passed else 'FAILED ❌'}")
    print(f"Score: {result.score}")
    print(f"Total Steps: {result.total_steps}")
    print(f"Workspace: {result.workspace_path}")
    print(f"JSON Trace: {result.trace_path}")
    print("=" * 60)


if __name__ == "__main__":
    main()
