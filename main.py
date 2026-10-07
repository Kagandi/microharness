"""microharness: Clean, zero-dependency AI agent microframework and coding harness.

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
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.config import ConfigLayers, resolve_project_tool_trust
from core.evaluator import CommandTestEvaluator, Evaluator, FileContentEvaluator
from core.harness import Harness, get_default_global_dir
from core.model import (
    MockModel,
    Model,
    ResettableModel,
    seed_user_providers,
)
from core.providers import (
    build_model_from_spec,
    get_available_models_from_config,
    load_provider_registry,
    resolve_connection,
)
from core.types import RunResult, Task, ToolCall


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


def _add_run_arguments(parser: argparse.ArgumentParser) -> None:
    """Where the agent works and how the session is driven."""
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
        help="Global configuration and skills directory (default: ~/.microharness)",
    )
    parser.add_argument(
        "--runs-dir",
        default=None,
        help="Directory to store runs, logs, and traces (default: ~/.microharness/runs)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run single one-shot task and exit immediately without entering interactive chat",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch interactive mode in full-screen TUI (default)",
    )
    parser.add_argument(
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume previous session or benchmark execution (optional: run_id to resume specific session)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum steps/actions the agent can take for a task (e.g. 50, 100, 500)",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=None,
        help="Timeout in seconds for task execution (e.g. 3600, 7200)",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Which model answers, and how it is tuned."""
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider preset (e.g. 'google', 'openrouter', 'openai', 'anthropic', 'deepseek', 'ollama', 'groq', or custom from providers.json)",
    )
    parser.add_argument(
        "--endpoint",
        default=None,
        help="HTTP chat completions endpoint (env: MODEL_ENDPOINT or ~/.microharness/config.json)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Model identifier (env: MODEL_NAME or ~/.microharness/config.json)",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="API key (env: MODEL_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, or ~/.microharness/config.json)",
    )
    parser.add_argument(
        "--cache-ttl",
        choices=["5m", "1h", "off"],
        default=None,
        help="Prompt cache duration: 5m (default ephemeral), 1h (extended beta), or off (env: CACHE_TTL or config.json)",
    )
    parser.add_argument(
        "--reasoning",
        default=None,
        help="Reasoning/thinking effort level: 'low', 'medium', 'high', numeric token budget, or 'off' (env: REASONING_EFFORT or config.json)",
    )


def _add_benchmark_arguments(parser: argparse.ArgumentParser) -> None:
    """Benchmark suites and built-in demos."""
    parser.add_argument(
        "--benchmark",
        choices=["mini-swe", "calculator", "all", "deepswe-mini"],
        default=None,
        help="Run a benchmark suite (e.g. 'mini-swe', 'deepswe-mini')",
    )
    parser.add_argument(
        "--task",
        default=None,
        help="Filter to a specific task ID in the benchmark suite (e.g. 'rate_limiter')",
    )
    parser.add_argument(
        "--demo",
        choices=["calculator", "greeting"],
        default=None,
        help="Run a built-in educational benchmark demo (runs sandboxed with evaluation)",
    )


def _add_sandbox_arguments(parser: argparse.ArgumentParser) -> None:
    """Docker isolation and the project-tool trust boundary."""
    parser.add_argument(
        "--docker",
        action="store_true",
        default=None,
        help="Execute actions inside an isolated Docker container sandbox (e.g. via OrbStack)",
    )
    parser.add_argument(
        "--docker-image",
        default=None,
        help="Docker image for container sandboxing (default: python:3.11 or from config)",
    )
    parser.add_argument(
        "--trust-project-tools",
        action="store_true",
        default=None,
        help="Allow executing custom tool plugins defined in the project workspace (default: false)",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="microharness: Interactive AI coding agent microframework for existing projects or benchmarks.",
        usage="%(prog)s [instruction] [-d DIR] [--sandbox] [--batch] [--demo {calculator,greeting}] [options]",
    )
    _add_run_arguments(parser)
    _add_model_arguments(parser)
    _add_benchmark_arguments(parser)
    _add_sandbox_arguments(parser)
    return parser


def _warn_unrecognized_model_flag(
    requested_model: str | None,
    config: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> None:
    """Warns when ``--model`` matches no configured alias, model id, or known provider.

    A mismatch is almost always a typo, but the flag stays a usable escape hatch
    for endpoints the registry does not know yet, so this warns instead of
    refusing and surfaces the problem here rather than as an opaque HTTP error.
    """
    if not requested_model:
        return
    try:
        known = get_available_models_from_config(config, registry)
        names = {m["name"].lower() for m in known}
        ids = {m.get("model", "").lower() for m in known if m.get("model")}
        providers = {name.lower() for name in registry}
        head = requested_model.split("/", 1)[0].lower()
        lowered = requested_model.strip().lower()
        if lowered in names or lowered in ids or lowered in providers or head in providers:
            return
        available = ", ".join(m["name"] for m in known) or "none configured"
        print(
            f"⚠️  Warning: --model '{requested_model}' does not match any configured "
            f"model or known provider. Available: {available}",
            file=sys.stderr,
        )
    except Exception:
        pass


def _resolve_model(
    args: argparse.Namespace,
    config: dict[str, Any],
    cache_ttl: str,
    reasoning_effort: str | None = None,
    global_dir: Path | None = None,
    target_dir: Path | None = None,
) -> Model:
    """Builds the startup model from CLI flags, environment, and the provider registry.

    Flag precedence (flag > environment > config file) is applied when the spec is
    assembled here; the registry itself only ever fills in what is still missing.
    """
    registry = load_provider_registry(
        global_dir=global_dir,
        project_dir=target_dir,
        inline_providers=config.get("providers"),
    )
    spec: dict[str, Any] = {
        "model": args.model or os.getenv("MODEL_NAME") or config.get("model"),
        "provider": args.provider or config.get("provider"),
        "endpoint": args.endpoint or os.getenv("MODEL_ENDPOINT") or config.get("endpoint"),
        "api_key": args.api_key or os.getenv("MODEL_API_KEY"),
        "headers": config.get("headers"),
    }
    connection = resolve_connection(spec, config, registry)

    if connection.is_offline:
        if args and (getattr(args, "demo", None) or getattr(args, "benchmark", None)):
            print("🎓 Running demo with deterministic MockModel (offline, no endpoint configured).\n")
            return build_mock_agent_script()
        print(
            "❌ No API key or model endpoint configured.\n\n"
            "To configure a provider, choose one of:\n"
            "  1. Add credentials to .env (e.g. GEMINI_API_KEY=..., OPENROUTER_API_KEY=...)\n"
            "  2. Pass CLI flags: --provider <name> --api-key <key>\n"
            "  3. Use a local provider: --provider lmstudio (or ollama, vllm, llamacpp)\n"
            "  4. Configure ~/.microharness/config.json with default_model and providers.\n\n"
            "Tip: Run with --demo calculator to test offline without API credentials.\n",
            file=sys.stderr,
        )
        sys.exit(1)

    cache_label = cache_ttl.upper() if cache_ttl and cache_ttl.lower() != "off" else "OFF"
    reasoning_label = f", reasoning: {reasoning_effort.upper()}" if reasoning_effort else ""
    provider_label = f" [{connection.provider}]" if connection.provider else ""
    print(
        f"🔌 Connected to{provider_label}: {connection.endpoint} "
        f"(model: {connection.model}, cache: {cache_label}{reasoning_label})"
    )

    _warn_unrecognized_model_flag(args.model, config, registry)
    return build_model_from_spec(
        spec,
        config=config,
        registry=registry,
        cache_ttl=cache_ttl,
        reasoning_effort=reasoning_effort,
    )


def _print_run_summary(result: RunResult) -> None:
    print("=" * 60)
    if result.eval_result:
        print(f"Outcome: {'PASSED ✅' if result.passed else 'FAILED ❌'} (Score: {result.score})")
    else:
        print(f"Outcome: {'PASSED ✅' if result.passed else 'FINISHED 🏁'}")
    print(f"Total Steps: {result.total_steps}")
    if result.prompt_tokens or result.completion_tokens:
        cached_info = (
            f" ({result.cached_tokens:,} cached - {result.cache_hit_rate_pct:.1f}% hit rate)"
            if result.cached_tokens
            else ""
        )
        print(f"Tokens: {result.prompt_tokens:,} in{cached_info}, {result.completion_tokens:,} out")
    print(f"Workspace: {result.workspace_path}")
    print(f"Trace: {Path(result.trace_path).parent / 'trace.md'}")
    if result.patch_path:
        print(f"Patch: {result.patch_path}")
    print("=" * 60)


@dataclass(frozen=True)
class CliContext:
    """Everything the CLI resolved once and hands to each run mode."""

    model: Model
    global_dir: Path
    runs_dir: Path
    target_dir: Path | None = None
    cache_ttl: str = "5m"
    reasoning_effort: str | None = None
    use_docker: bool = False
    docker_image: str = "python:3.11"
    trust_project_tools: bool = False
    verbose: bool = True
    args: argparse.Namespace | None = None
    config: dict[str, Any] = field(default_factory=dict)

    def harness(self, evaluator: Evaluator | None = None, sandbox: bool = True) -> Harness:
        """Builds a harness for this context, resetting a reusable model between tasks."""
        if isinstance(self.model, ResettableModel):
            self.model.reset()
        return Harness(
            model=self.model,
            workspace_dir=self.target_dir,
            global_dir=self.global_dir,
            runs_dir=self.runs_dir,
            sandbox=sandbox,
            docker=self.use_docker,
            docker_image=self.docker_image,
            evaluator=evaluator,
            cache_ttl=self.cache_ttl,
            reasoning_effort=self.reasoning_effort,
            trust_project_tools=self.trust_project_tools,
            verbose=self.verbose,
        )


def create_benchmark_harness_factory(
    model: Model,
    global_dir: Path,
    runs_dir: Path,
    use_docker: bool = False,
    docker_image: str | None = None,
    cache_ttl: str | None = None,
    reasoning_effort: str | None = None,
    trust_project_tools: bool = False,
    verbose: bool = True,
) -> Callable[[Evaluator], Harness]:
    """Creates a harness factory for benchmark suites that resets ResettableModel between tasks."""
    context = CliContext(
        model=model,
        global_dir=global_dir,
        runs_dir=runs_dir,
        cache_ttl=cache_ttl or "5m",
        reasoning_effort=reasoning_effort,
        use_docker=use_docker,
        docker_image=docker_image or "python:3.11",
        trust_project_tools=trust_project_tools,
        verbose=verbose,
    )
    return lambda evaluator: context.harness(evaluator)


def _benchmark_metadata(context: CliContext) -> dict[str, Any]:
    """The model/limits summary recorded alongside a benchmark's scorecard."""
    model = context.model
    return {
        "provider": getattr(context.args, "provider", None) or context.config.get("provider"),
        "model": getattr(model, "model", None) or context.config.get("model"),
        "endpoint": getattr(model, "endpoint", None) or context.config.get("endpoint"),
        "cache_ttl": context.cache_ttl,
        "reasoning_effort": context.reasoning_effort,
        "max_steps": context.args.max_steps or int(context.config.get("max_steps", 30)),
    }


def _resolve_deepswe_dataset(target_dir: Path) -> Path | None:
    """Locates the DeepSWE task directory, preferring one inside the target project."""
    candidate = target_dir / "datasets" / "deep-swe" / "tasks"
    if not candidate.is_dir():
        candidate = Path("datasets/deep-swe/tasks").resolve()
    return candidate if candidate.is_dir() else None


def _run_mini_swe(context: CliContext) -> None:
    """Runs the bundled Mini-SWE suite."""
    from benchmarks.mini_swe import get_mini_swe_suite
    from benchmarks.runner import run_benchmark_suite

    suite = get_mini_swe_suite(task_filter=context.args.task)
    if not suite:
        print(f"❌ No benchmark tasks found matching filter: '{context.args.task}'")
        return

    if context.args.max_steps or context.args.timeout:
        for task, _ in suite:
            if context.args.max_steps:
                task.max_steps = context.args.max_steps
            if context.args.timeout:
                task.timeout_seconds = context.args.timeout

    run_benchmark_suite(
        harness_factory=lambda evaluator: context.harness(evaluator),
        tasks=suite,
        suite_name="Mini-SWE",
        output_dir=context.runs_dir,
        verbose=True,
        resume=bool(context.args.resume),
        config=_benchmark_metadata(context),
    )


def _run_deepswe_mini(context: CliContext) -> None:
    """Runs the DeepSWE-mini suite from the on-disk dataset."""
    from benchmarks.harbor import get_deepswe_mini_suite
    from benchmarks.runner import run_benchmark_suite

    dataset_dir = _resolve_deepswe_dataset(context.target_dir)
    if dataset_dir is None:
        print("❌ DeepSWE dataset not found at datasets/deep-swe/tasks")
        print(
            "Tip: Run `git clone --depth 1 https://github.com/datacurve-ai/deep-swe "
            "datasets/deep-swe` to download tasks."
        )
        return

    max_steps = context.args.max_steps or int(context.config.get("max_steps", 30))
    timeout_seconds = context.args.timeout or (max_steps * 30.0 if max_steps > 30 else 900.0)
    suite = get_deepswe_mini_suite(
        dataset_dir=dataset_dir,
        task_filter=context.args.task,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
    )
    if not suite:
        print(f"❌ No DeepSWE tasks found matching filter: '{context.args.task}'")
        return

    run_benchmark_suite(
        harness_factory=lambda evaluator: context.harness(evaluator),
        tasks=suite,
        suite_name="DeepSWE-mini",
        output_dir=context.runs_dir,
        verbose=True,
        resume=bool(context.args.resume),
        config=_benchmark_metadata(context),
    )


def _run_demo(context: CliContext, name: str) -> None:
    """Runs one of the built-in single-task demos."""
    task, evaluator = create_calculator_demo() if name == "calculator" else create_greeting_demo()
    _print_run_summary(context.harness(evaluator).run(task))


BENCHMARK_RUNNERS: dict[str, Callable[[CliContext], None]] = {
    "mini-swe": _run_mini_swe,
    "all": _run_mini_swe,
    "deepswe-mini": _run_deepswe_mini,
    "calculator": lambda ctx: _run_demo(ctx, "calculator"),
    "greeting": lambda ctx: _run_demo(ctx, "greeting"),
}


def _dispatch_benchmark(context: CliContext) -> bool:
    """Runs the requested benchmark suite or demo. Returns False when none was requested."""
    choice = context.args.benchmark or context.args.demo
    runner = BENCHMARK_RUNNERS.get(choice) if choice else None
    if runner is None:
        return False
    runner(context)
    return True


def _resolve_cli_context(args: argparse.Namespace) -> CliContext:
    """Resolves directories, configuration layers, and the model, in precedence order.

    Explicit flags beat environment variables, which beat the project config file,
    which beats the user's global config file. The harness re-applies the same
    cascade to its own settings, so this only has to resolve what the CLI needs.
    """
    target_dir = Path(args.dir).resolve()
    global_dir = Path(args.global_dir).resolve() if args.global_dir else get_default_global_dir()
    seed_user_providers(global_dir)

    layers = ConfigLayers.load(global_dir, target_dir)
    config = layers.merged

    configured_runs_dir = args.runs_dir or config.get("runs_dir")
    runs_dir = (
        Path(configured_runs_dir).resolve()
        if configured_runs_dir
        else (global_dir / "runs").resolve()
    )

    # Trust is a security boundary: it may come from the flag or the user's own
    # global config, never from the project config. Harness reports the rejection
    # once, when it builds itself, so this stays silent.
    trust_project_tools, _ = resolve_project_tool_trust(args.trust_project_tools, layers)

    cache_ttl = _resolve_cache_ttl(args, config)
    reasoning_effort = _resolve_reasoning_effort(args, config)

    return CliContext(
        args=args,
        config=config,
        target_dir=target_dir,
        global_dir=global_dir,
        runs_dir=runs_dir,
        model=_resolve_model(
            args,
            config,
            cache_ttl,
            reasoning_effort,
            global_dir=global_dir,
            target_dir=target_dir,
        ),
        cache_ttl=cache_ttl,
        reasoning_effort=reasoning_effort,
        use_docker=args.docker if args.docker is not None else bool(config.get("docker", False)),
        docker_image=args.docker_image or config.get("docker_image", "python:3.11"),
        trust_project_tools=trust_project_tools,
    )


def _resolve_cache_ttl(args: argparse.Namespace, config: dict[str, Any]) -> str:
    """Prompt cache duration: flag, then environment, then config."""
    return args.cache_ttl or os.getenv("CACHE_TTL") or config.get("cache_ttl", "5m")


def _resolve_reasoning_effort(args: argparse.Namespace, config: dict[str, Any]) -> str | None:
    """Reasoning effort: flag, then either environment variable, then either config key."""
    return (
        args.reasoning
        or os.getenv("REASONING_EFFORT")
        or os.getenv("MODEL_REASONING_EFFORT")
        or config.get("reasoning_effort")
        or config.get("reasoning")
    )


def _run_single_task(context: CliContext, parser: argparse.ArgumentParser) -> None:
    """One-shot batch mode: run the instruction and print the summary."""
    args = context.args
    if not args.instruction:
        parser.error("--batch requires an instruction argument.")
    task = Task(
        id="user_task",
        instruction=args.instruction.strip(),
        max_steps=args.max_steps or 15,
        timeout_seconds=args.timeout or 60.0,
    )
    harness = context.harness(sandbox=bool(args.sandbox or context.config.get("sandbox", False)))
    _print_run_summary(harness.run(task))


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)
    context = _resolve_cli_context(args)

    if _dispatch_benchmark(context):
        return

    if args.batch:
        _run_single_task(context, parser)
        return

    if not sys.stdin.isatty():
        print(
            "❌ Interactive TUI requires a terminal. Use --batch for scripted/headless execution.",
            file=sys.stderr,
        )
        sys.exit(1)

    sandbox = args.sandbox or bool(context.config.get("sandbox", False))
    harness = Harness(
        model=context.model,
        workspace_dir=context.target_dir,
        global_dir=context.global_dir,
        runs_dir=context.runs_dir,
        sandbox=sandbox,
        docker=context.use_docker,
        docker_image=context.docker_image,
        cache_ttl=context.cache_ttl,
        reasoning_effort=context.reasoning_effort,
        trust_project_tools=context.trust_project_tools,
        verbose=True,
    )

    harness.run_tui(initial_instruction=args.instruction, resume=args.resume)


if __name__ == "__main__":
    main()
