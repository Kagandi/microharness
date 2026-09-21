"""mini-harness: Educational demonstration of an AI agent harness.

Demonstrates:
1. Boundary between Harness (chassis) and AgentLoop (engine).
2. Workspace isolation & sandboxing.
3. Real bash and file tool execution.
4. Large log truncation + spill to disk (.logs/).
5. Context memory compaction.
6. Structured tracing (trace.json and trace.md).
7. Automated post-run evaluation.
8. Pluggable models: MockModel (offline), OpenRouter, or Local endpoint (Ollama/vLLM/LM Studio).
"""

from __future__ import annotations

import argparse
import sys

from core.evaluator import CommandTestEvaluator
from core.harness import Harness
from core.model import LocalEndpointModel, MockModel, Model, OpenRouterModel
from core.types import Task, ToolCall


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
    """Builds a deterministic scripted model demonstrating the full agent trajectory."""
    fixed_code = (
        "def add(a: float, b: float) -> float:\n"
        "    return a + b\n\n"
        "def divide(a: float, b: float) -> float:\n"
        "    if b == 0:\n"
        "        raise ValueError('Cannot divide by zero')\n"
        "    return a / b\n"
    )

    script = [
        # Step 1: Initial test run
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
        # Step 2: Verbose diagnostic run (demonstrating log truncation)
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
        # Step 3: Inspect code
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
        # Step 4: Write fix
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
        # Step 5: Verification test run
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
        # Step 6: Final answer
        (
            "The tests now pass completely. The bug in `calculator.py` has been resolved.",
            [],
        ),
    ]

    return MockModel(script=script)


def get_model(provider: str, model_name: str | None, base_url: str | None) -> Model:
    """Instantiates the requested pluggable model."""
    if provider == "mock":
        print("🎓 Running with deterministic MockModel (offline, no API key needed).")
        return build_mock_agent_script()

    elif provider == "openrouter":
        chosen_model = model_name or "anthropic/claude-3.5-sonnet"
        print(f"🌐 Running with OpenRouter | Model: {chosen_model}")
        try:
            return OpenRouterModel(model_name=chosen_model)
        except ValueError as exc:
            print(f"❌ Configuration error: {exc}")
            print("Tip: export OPENROUTER_API_KEY='sk-or-...'")
            sys.exit(1)

    elif provider == "local":
        endpoint = base_url or "http://localhost:11434/v1"
        chosen_model = model_name or "qwen2.5-coder"
        print(f"💻 Running with Local Endpoint | URL: {endpoint} | Model: {chosen_model}")
        return LocalEndpointModel(base_url=endpoint, model_name=chosen_model)

    else:
        print(f"❌ Unknown provider '{provider}'. Choose from: mock, openrouter, local.")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(description="Run the mini-harness educational demonstration.")
    parser.add_argument(
        "--provider",
        choices=["mock", "openrouter", "local"],
        default="mock",
        help="LLM provider: 'mock' (offline), 'openrouter', or 'local' (default: mock)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (e.g. 'anthropic/claude-3.5-sonnet', 'deepseek/deepseek-chat', 'qwen2.5-coder')",
    )
    parser.add_argument(
        "--base-url",
        default=None,
        help="Custom base URL for local endpoint (e.g. 'http://localhost:11434/v1' for Ollama, 'http://localhost:8000/v1' for vLLM)",
    )
    args = parser.parse_args()

    # 1. Initialize Model
    model = get_model(provider=args.provider, model_name=args.model, base_url=args.base_url)

    # 2. Define Evaluator (tests must pass with exit code 0)
    evaluator = CommandTestEvaluator(
        command="python -m unittest test_calculator.py",
        expected_exit_code=0,
    )

    # 3. Initialize Harness
    harness = Harness(
        model=model,
        evaluator=evaluator,
        max_tool_chars=1200,    # Triggers truncation demo on long logs
        max_context_chars=3500,  # Triggers compaction demo on large history
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
