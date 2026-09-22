"""mini-harness: Educational demonstration of an AI agent harness.

Demonstrates:
1. Boundary between Harness (chassis) and AgentLoop (engine).
2. Project directory operation (edits real projects) vs. isolated sandboxing.
3. Real bash and file tool execution.
4. Large log truncation + spill to disk (.logs/).
5. Context memory compaction across turns.
6. Structured tracing (trace.json and trace.md).
7. Automated post-run evaluation.
8. Interactive multi-turn chat sessions + batch benchmark mode.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from core.evaluator import CommandTestEvaluator, Evaluator, FileContentEvaluator
from core.harness import DEFAULT_GLOBAL_DIR, Harness, load_global_config
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
        if key and key not in os.environ:
            os.environ[key] = val


def create_calculator_demo() -> tuple[Task, Evaluator]:
    """Demo SWE task: Failing test suite with a bug in calculator.py."""
    task = Task(
        id="fix_calculator_bug",
        instruction=(
            "The calculator suite is failing. Run the unit tests using `python -m unittest test_calculator.py`.\n"
            "Investigate the failure, fix `calculator.py` so all tests pass (including raising ValueError on zero division),\n"
            "and verify that the tests succeed."
        ),
        initial_files={
            "calculator.py": (
                "def add(a: float, b: float) -> float:\n    return a + b\n\n"
                "def divide(a: float, b: float) -> float:\n    # BUG: Multiplying instead of dividing\n    return a * b\n"
            ),
            "test_calculator.py": (
                "import unittest\nfrom calculator import add, divide\n\n"
                "class TestCalculator(unittest.TestCase):\n"
                "    def test_add(self):\n        self.assertEqual(add(2, 3), 5)\n\n"
                "    def test_divide(self):\n        self.assertEqual(divide(10, 2), 5)\n"
                "        with self.assertRaises(ValueError):\n            divide(5, 0)\n\n"
                "if __name__ == '__main__':\n    unittest.main()\n"
            ),
        },
        max_steps=10,
        timeout_seconds=60.0,
    )
    evaluator = CommandTestEvaluator(
        command="python -m unittest test_calculator.py",
        expected_exit_code=0,
    )
    return task, evaluator


def create_greeting_demo() -> tuple[Task, Evaluator]:
    """Lightweight 2-step demo task."""
    task = Task(
        id="write_greeting",
        instruction="Create a file named greeting.txt with the content Hello from Harness using the write_file tool.",
        max_steps=4,
        timeout_seconds=30.0,
    )
    evaluator = FileContentEvaluator(
        file_path="greeting.txt",
        must_contain=["Hello from Harness"],
    )
    return task, evaluator


def build_mock_agent_script() -> MockModel:
    """Deterministic scripted model for offline testing without API credentials."""
    fixed_code = (
        "def add(a: float, b: float) -> float:\n    return a + b\n\n"
        "def divide(a: float, b: float) -> float:\n"
        "    if b == 0:\n        raise ValueError('Cannot divide by zero')\n    return a / b\n"
    )
    script = [
        (
            "I will first run the tests to inspect what fails.",
            [
                ToolCall(
                    id="c1",
                    name="bash",
                    arguments={"command": "python -m unittest test_calculator.py"},
                )
            ],
        ),
        (
            "Let me print a verbose diagnostic report to see system details.",
            [
                ToolCall(
                    id="c2",
                    name="bash",
                    arguments={
                        "command": "python -c \"for i in range(120): print(f'DEBUG Log {i}')\""
                    },
                )
            ],
        ),
        (
            "Now let's inspect the calculator implementation.",
            [ToolCall(id="c3", name="read_file", arguments={"path": "calculator.py"})],
        ),
        (
            "Fixing calculator.py now.",
            [
                ToolCall(
                    id="c4",
                    name="write_file",
                    arguments={"path": "calculator.py", "content": fixed_code},
                )
            ],
        ),
        (
            "Re-running unit tests.",
            [
                ToolCall(
                    id="c5",
                    name="bash",
                    arguments={"command": "python -m unittest test_calculator.py"},
                )
            ],
        ),
        ("The tests now pass completely.", []),
    ]
    return MockModel(script=script)


def main():
    load_dotenv()

    parser = argparse.ArgumentParser(
        description="mini-harness: Interactive AI coding agent harness for existing projects or benchmarks.",
        usage="%(prog)s [instruction] [-d DIR] [--sandbox] [--batch] [--demo {calculator,greeting}] [options]",
    )
    parser.add_argument(
        "instruction",
        nargs="?",
        default=None,
        help="Instruction for the agent (e.g. 'Fix the bug in auth.py', 'Add tests')",
    )
    parser.add_argument(
        "-d",
        "--dir",
        default=".",
        help="Target project directory where the agent operates (default: current directory '.')",
    )
    parser.add_argument(
        "--sandbox",
        action="store_true",
        help="Run in an isolated disposable scratch directory instead of your real project directory",
    )
    parser.add_argument(
        "--global-dir",
        default=None,
        help="Global configuration and skills directory (default: ~/.mini-harness)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run single one-shot task and exit immediately without entering interactive chat",
    )
    parser.add_argument(
        "--demo",
        choices=["calculator", "greeting"],
        default=None,
        help="Run a built-in educational benchmark demo (runs sandboxed with evaluation)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="HTTP chat completions endpoint (env: MODEL_ENDPOINT or ~/.mini-harness/config.json)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (env: MODEL_NAME or ~/.mini-harness/config.json)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (env: MODEL_API_KEY, OPENROUTER_API_KEY, or ~/.mini-harness/config.json)",
    )
    args = parser.parse_args()

    # Resolve global config (~/.mini-harness/config.json)
    global_dir = Path(args.global_dir).resolve() if args.global_dir else DEFAULT_GLOBAL_DIR
    global_config = load_global_config(global_dir)

    # 1. Select Model (Precedence: CLI Flag > Environment Variable / .env > global config.json > Fallback)
    endpoint = args.endpoint or os.getenv("MODEL_ENDPOINT") or global_config.get("endpoint")
    model_name = (
        args.model
        or os.getenv("MODEL_NAME")
        or global_config.get("model")
        or "cohere/north-mini-code:free"
    )
    api_key = (
        args.api_key
        or os.getenv("MODEL_API_KEY")
        or os.getenv("OPENROUTER_API_KEY")
        or global_config.get("api_key")
    )

    if endpoint:
        print(f"🔌 Connected to: {endpoint} (model: {model_name})")
        model: Model = HttpChatModel(
            base_url=endpoint,
            model=model_name,
            api_key=api_key,
        )
    else:
        print("🎓 Running with deterministic MockModel (offline, no endpoint configured).")
        print(
            "Tip: Add MODEL_ENDPOINT to .env, pass --endpoint, or configure ~/.mini-harness/config.json.\n"
        )
        model = build_mock_agent_script()

    # 2. If --demo is passed, run Benchmark Mode (isolated sandbox with evaluator)
    if args.demo:
        task, evaluator = (
            create_calculator_demo() if args.demo == "calculator" else create_greeting_demo()
        )
        harness = Harness(
            model=model,
            global_dir=global_dir,
            sandbox=True,
            evaluator=evaluator,
            verbose=True,
        )
        result = harness.run(task)
        print("=" * 60)
        print(f"Outcome: {'PASSED ✅' if result.passed else 'FAILED ❌'} (Score: {result.score})")
        print(f"Total Steps: {result.total_steps}")
        print(f"Workspace: {result.workspace_path}")
        print(f"Trace: {Path(result.trace_path).parent / 'trace.md'}")
        if result.patch_path:
            print(f"Patch: {result.patch_path}")
        print("=" * 60)
        return

    # 3. Resolve Target Workspace and Sandbox Flag
    target_dir = Path(args.dir).resolve()
    sandbox = args.sandbox or bool(global_config.get("sandbox", False))
    harness = Harness(
        model=model,
        workspace_dir=target_dir,
        global_dir=global_dir,
        sandbox=sandbox,
        verbose=True,
    )

    # 4. Batch Mode vs Interactive Mode
    if args.batch and args.instruction:
        task = Task(id="user_task", instruction=args.instruction.strip())
        result = harness.run(task)
        print("=" * 60)
        print(f"Outcome: {'PASSED ✅' if result.passed else 'FINISHED 🏁'}")
        print(f"Total Steps: {result.total_steps}")
        print(f"Workspace: {result.workspace_path}")
        print(f"Trace: {Path(result.trace_path).parent / 'trace.md'}")
        if result.patch_path:
            print(f"Patch: {result.patch_path}")
        print("=" * 60)
        return

    # Default: Interactive Mode in the target project directory
    harness.run_interactive(initial_instruction=args.instruction)


if __name__ == "__main__":
    main()
