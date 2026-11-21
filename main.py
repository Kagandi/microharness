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
import getpass
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.auth_store import get_auth_store_path
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
    PROVIDER_ALIASES,
    Connection,
    build_model_from_spec,
    format_provider_auth_line,
    get_available_models_from_config,
    inspect_provider_auth_statuses,
    load_provider_registry,
    login_provider_credential,
    logout_provider_credential,
    parse_model_spec,
    resolve_auth_provider,
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
    parser.add_argument(
        "--bash-timeout",
        type=float,
        default=None,
        help="Per-command timeout in seconds for the bash tool (default: 120.0 or from config)",
    )


def _add_model_arguments(parser: argparse.ArgumentParser) -> None:
    """Which model answers, and how it is tuned."""
    parser.add_argument(
        "--provider",
        default=None,
        help="Provider preset (e.g. 'google-ai-studio', 'openrouter', 'vercel', 'openai', 'anthropic', 'deepseek', 'ollama', 'groq', or custom from providers.json)",
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
        help="API key (env: MODEL_API_KEY, GEMINI_API_KEY, OPENROUTER_API_KEY, AI_GATEWAY_API_KEY, or ~/.microharness/config.json)",
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
        help="Reasoning/thinking effort level: 'low', 'medium', 'high', 'xhigh', 'max', numeric token budget, or 'off' (env: REASONING_EFFORT or config.json)",
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
        "--docker-network",
        default=None,
        help="Docker network mode for container execution (e.g. 'none' to block outbound network)",
    )
    parser.add_argument(
        "--no-network-tools",
        action="store_true",
        default=None,
        help="Disable outbound network tool plugins (such as web_fetch)",
    )
    parser.add_argument(
        "--trust-project-tools",
        action="store_true",
        default=None,
        help="Allow executing custom tool plugins defined in the project workspace (default: false)",
    )
    parser.add_argument(
        "--permission-mode",
        choices=["auto", "ask", "read-only"],
        default=None,
        help="Host execution permission mode: 'auto' (default), 'ask', or 'read-only'",
    )
    parser.add_argument(
        "--docker-memory",
        default=None,
        help="Optional Docker container memory limit (e.g. '2g', '512m')",
    )
    parser.add_argument(
        "--docker-cpus",
        type=float,
        default=None,
        help="Optional Docker container CPU limit (e.g. 2.0)",
    )
    parser.add_argument(
        "--docker-pids-limit",
        type=int,
        default=None,
        help="Optional Docker container PIDs limit (e.g. 256)",
    )


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="microharness: Interactive AI coding agent microframework for existing projects or benchmarks.",
        usage="%(prog)s [instruction] [-d DIR] [--sandbox] [--batch] [--demo {calculator,greeting}] [options]",
        epilog="Credential management: %(prog)s auth {login,list,logout} --help",
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
        lowered = requested_model.strip().lower()
        head = lowered.split("/", 1)[0]
        if (
            lowered in names
            or lowered in ids
            or lowered in providers
            or head in providers
            or PROVIDER_ALIASES.get(lowered) in providers
            or PROVIDER_ALIASES.get(head) in providers
        ):
            return
        available = ", ".join(m["name"] for m in known) or "none configured"
        print(
            f"⚠️  Warning: --model '{requested_model}' does not match any configured "
            f"model or known provider. Available: {available}",
            file=sys.stderr,
        )
    except Exception:
        pass


def _startup_model_spec(
    args: argparse.Namespace,
    config: dict[str, Any],
    registry: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Assembles the startup model spec from flags/env (config defaults stay fallbacks)."""
    spec: dict[str, Any] = {
        "model": args.model or os.getenv("MODEL_NAME") or config.get("model"),
        "provider": args.provider,
        "endpoint": args.endpoint or os.getenv("MODEL_ENDPOINT"),
        "api_key": args.api_key or os.getenv("MODEL_API_KEY"),
        "headers": None,
    }
    if (
        args.model
        and spec["provider"] is None
        and spec["endpoint"] is None
        and registry is not None
    ):
        cfg_models = config.get("models")
        if not (isinstance(cfg_models, dict) and args.model in cfg_models):
            parsed = parse_model_spec(args.model, config, registry)
            if parsed.provider:
                spec["model"] = parsed.model
                spec["provider"] = parsed.provider
    return spec


def _resolve_model(
    args: argparse.Namespace,
    config: dict[str, Any],
    cache_ttl: str,
    reasoning_effort: str | None = None,
    global_dir: Path | None = None,
    target_dir: Path | None = None,
    registry: dict[str, dict[str, Any]] | None = None,
) -> Model:
    """Builds the startup model from CLI flags, environment, and the provider registry.

    Flag precedence (flag > environment > config file) is applied when the spec is
    assembled here; the registry itself only ever fills in what is still missing.
    """
    if registry is None:
        registry = load_provider_registry(
            global_dir=global_dir,
            project_dir=target_dir,
            inline_providers=config.get("providers"),
        )
    spec: dict[str, Any] = _startup_model_spec(args, config, registry)
    connection = resolve_connection(spec, config, registry, global_dir=global_dir)

    if connection.is_offline:
        print("🎓 Running with deterministic MockModel (offline, no endpoint configured).")
        print(
            "Tip: Add MODEL_ENDPOINT to .env, pass --endpoint, or "
            "configure ~/.microharness/config.json.\n"
        )
        return build_mock_agent_script()

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
        global_dir=global_dir,
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
    docker_network: str | None = None
    allow_network_tools: bool | None = None
    bash_timeout: float | None = None
    trust_project_tools: bool = False
    verbose: bool = True
    args: argparse.Namespace | None = None
    config: dict[str, Any] = field(default_factory=dict)
    model_spec: dict[str, Any] | None = None
    permission_mode: str | None = None
    docker_memory: str | None = None
    docker_cpus: float | None = None
    docker_pids_limit: int | None = None

    def harness(self, evaluator: Evaluator | None = None, sandbox: bool = True) -> Harness:
        """Builds a harness for this context, resetting a reusable model between tasks."""
        if isinstance(self.model, ResettableModel):
            self.model.reset()
        harness = Harness(
            model=self.model,
            workspace_dir=self.target_dir,
            global_dir=self.global_dir,
            runs_dir=self.runs_dir,
            sandbox=sandbox,
            docker=self.use_docker,
            docker_image=self.docker_image,
            docker_network=self.docker_network,
            allow_network_tools=self.allow_network_tools,
            bash_timeout=self.bash_timeout,
            evaluator=evaluator,
            cache_ttl=self.cache_ttl,
            reasoning_effort=self.reasoning_effort,
            trust_project_tools=self.trust_project_tools,
            verbose=self.verbose,
            permission_mode=self.permission_mode,
            docker_memory=self.docker_memory,
            docker_cpus=self.docker_cpus,
            docker_pids_limit=self.docker_pids_limit,
        )
        if self.model_spec is not None:
            harness.apply_model_context_tokens(self.model_spec)
        return harness


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
    docker_network: str | None = None,
    allow_network_tools: bool = False,
    bash_timeout: float | None = None,
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
        docker_network=docker_network,
        allow_network_tools=allow_network_tools,
        bash_timeout=bash_timeout,
        trust_project_tools=trust_project_tools,
        verbose=verbose,
    )
    return lambda evaluator: context.harness(evaluator)


def _task_requires_finish_tool(context: CliContext) -> bool:
    """Requires explicit finish-tool termination in autonomous runs for real models."""
    return not isinstance(context.model, MockModel)


def _benchmark_metadata(context: CliContext) -> dict[str, Any]:
    """The model/limits summary recorded alongside a benchmark's scorecard."""
    model = context.model
    return {
        "provider": (
            getattr(context.args, "provider", None)
            or getattr(model, "provider", None)
            or context.config.get("provider")
        ),
        "model": getattr(model, "model", None) or context.config.get("model"),
        "endpoint": getattr(model, "endpoint", None) or context.config.get("endpoint"),
        "cache_ttl": context.cache_ttl,
        "reasoning_effort": context.reasoning_effort,
        "max_steps": getattr(context.args, "max_steps", None)
        or int(context.config.get("max_steps", 30)),
    }


def _resolve_deepswe_dataset(target_dir: Path) -> Path | None:
    """Locates the DeepSWE task directory, preferring one inside the target project."""
    candidate = target_dir / "datasets" / "deep-swe" / "tasks"
    if not candidate.is_dir():
        candidate = Path("datasets/deep-swe/tasks").resolve()
    return candidate if candidate.is_dir() else None


def _normalize_optional_str(value: Any) -> str | None:
    """Strips optional string inputs and turns empty/whitespace strings into None."""
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _run_mini_swe(context: CliContext) -> None:
    """Runs the bundled Mini-SWE suite."""
    from benchmarks.mini_swe import get_mini_swe_suite
    from benchmarks.runner import run_benchmark_suite

    assert context.args is not None
    args = context.args
    suite = get_mini_swe_suite(task_filter=args.task)
    if not suite:
        print(f"❌ No benchmark tasks found matching filter: '{args.task}'")
        return

    require_finish = _task_requires_finish_tool(context)
    explicit_docker_net = _normalize_optional_str(getattr(args, "docker_network", None))
    requested_net = explicit_docker_net or context.docker_network
    docker_active_any = context.use_docker or any(bool(task.docker_image) for task, _ in suite)
    if requested_net and not docker_active_any:
        print(
            "⚠️ Warning: --docker-network was specified without --docker; bash commands will run on "
            "the host without container network isolation (network tools remain disabled).",
            file=sys.stderr,
        )

    for task, _ in suite:
        if args.max_steps:
            task.max_steps = args.max_steps
        if args.timeout:
            task.timeout_seconds = args.timeout
        if require_finish:
            task.require_finish_tool = True
        task.allow_network_tools = False
        if explicit_docker_net:
            task.docker_network = explicit_docker_net
        elif context.use_docker or task.docker_image:
            task.docker_network = task.docker_network or context.docker_network or "none"
        if context.bash_timeout is not None:
            task.bash_timeout = context.bash_timeout

    run_benchmark_suite(
        harness_factory=lambda evaluator: context.harness(evaluator),
        tasks=suite,
        suite_name="Mini-SWE",
        output_dir=context.runs_dir,
        verbose=True,
        resume=bool(args.resume),
        config=_benchmark_metadata(context),
    )


def _run_deepswe_mini(context: CliContext) -> None:
    """Runs the DeepSWE-mini suite from the on-disk dataset."""
    from benchmarks.harbor import get_deepswe_mini_suite
    from benchmarks.runner import run_benchmark_suite

    assert context.args is not None
    args = context.args
    dataset_dir = _resolve_deepswe_dataset(context.target_dir or Path.cwd())
    if dataset_dir is None:
        print("❌ DeepSWE dataset not found at datasets/deep-swe/tasks")
        print(
            "Tip: Run `git clone --depth 1 https://github.com/datacurve-ai/deep-swe "
            "datasets/deep-swe` to download tasks."
        )
        return

    max_steps = args.max_steps or int(context.config.get("max_steps", 30))
    timeout_seconds = args.timeout or (max_steps * 30.0 if max_steps > 30 else 900.0)
    suite = get_deepswe_mini_suite(
        dataset_dir=dataset_dir,
        task_filter=args.task,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
    )
    if not suite:
        print(f"❌ No DeepSWE tasks found matching filter: '{args.task}'")
        return

    require_finish = _task_requires_finish_tool(context)
    explicit_docker_net = _normalize_optional_str(getattr(args, "docker_network", None))
    for task, _ in suite:
        if require_finish:
            task.require_finish_tool = True
        task.allow_network_tools = False
        if explicit_docker_net:
            task.docker_network = explicit_docker_net
        elif not task.docker_network:
            task.docker_network = context.docker_network or "none"
        if context.bash_timeout is not None:
            task.bash_timeout = context.bash_timeout

    run_benchmark_suite(
        harness_factory=lambda evaluator: context.harness(evaluator),
        tasks=suite,
        suite_name="DeepSWE-mini",
        output_dir=context.runs_dir,
        verbose=True,
        resume=bool(args.resume),
        config=_benchmark_metadata(context),
    )


def _run_demo(context: CliContext, name: str) -> None:
    """Runs one of the built-in single-task demos."""
    task, evaluator = create_calculator_demo() if name == "calculator" else create_greeting_demo()
    if _task_requires_finish_tool(context):
        task.require_finish_tool = True
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
    assert context.args is not None
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
    registry = load_provider_registry(
        global_dir=global_dir,
        project_dir=target_dir,
        inline_providers=config.get("providers"),
    )

    docker_network = _normalize_optional_str(
        getattr(args, "docker_network", None)
    ) or _normalize_optional_str(config.get("docker_network"))
    allow_network_tools: bool | None = False if getattr(args, "no_network_tools", None) else None
    raw_bash_timeout = getattr(args, "bash_timeout", None)
    bash_timeout = (
        float(raw_bash_timeout)
        if raw_bash_timeout is not None
        else (float(config["bash_timeout"]) if config.get("bash_timeout") is not None else None)
    )
    permission_mode = _normalize_optional_str(
        getattr(args, "permission_mode", None)
    ) or _normalize_optional_str(config.get("permission_mode"))
    docker_memory = _normalize_optional_str(
        getattr(args, "docker_memory", None)
    ) or _normalize_optional_str(config.get("docker_memory"))
    raw_cpus = getattr(args, "docker_cpus", None)
    docker_cpus = float(raw_cpus) if raw_cpus is not None else None
    raw_pids = getattr(args, "docker_pids_limit", None)
    docker_pids_limit = int(raw_pids) if raw_pids is not None else None

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
            registry=registry,
        ),
        cache_ttl=cache_ttl,
        reasoning_effort=reasoning_effort,
        use_docker=args.docker if args.docker is not None else bool(config.get("docker", False)),
        docker_image=args.docker_image or config.get("docker_image", "python:3.11"),
        docker_network=docker_network,
        allow_network_tools=allow_network_tools,
        bash_timeout=bash_timeout,
        trust_project_tools=trust_project_tools,
        model_spec=_startup_model_spec(args, config, registry),
        permission_mode=permission_mode,
        docker_memory=docker_memory,
        docker_cpus=docker_cpus,
        docker_pids_limit=docker_pids_limit,
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
    assert args is not None
    if not args.instruction:
        parser.error("--batch requires an instruction argument.")
    task = Task(
        id="user_task",
        instruction=args.instruction.strip(),
        max_steps=args.max_steps or 15,
        timeout_seconds=args.timeout or 60.0,
        require_finish_tool=_task_requires_finish_tool(context),
    )
    harness = context.harness(sandbox=bool(args.sandbox or context.config.get("sandbox", False)))
    _print_run_summary(harness.run(task))


def _build_auth_arg_parser() -> argparse.ArgumentParser:
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--global-dir",
        default=argparse.SUPPRESS,
        help="Global configuration and credentials directory (default: ~/.microharness)",
    )
    common.add_argument(
        "-d",
        "--dir",
        default=argparse.SUPPRESS,
        help="Target project directory for project providers.json (default: '.')",
    )

    parser = argparse.ArgumentParser(
        prog="microharness auth",
        description="Manage global per-provider API credentials in ~/.microharness/auth.json.",
        parents=[common],
    )
    subparsers = parser.add_subparsers(dest="auth_command", metavar="command")

    login_parser = subparsers.add_parser(
        "login",
        parents=[common],
        help="Store or update an API key for a provider in auth.json",
    )
    login_parser.add_argument(
        "provider",
        nargs="?",
        default=None,
        help="Provider name or alias (e.g. openrouter, vercel, google-ai-studio, openai, anthropic)",
    )

    subparsers.add_parser(
        "list",
        parents=[common],
        help="List providers and credential status/source without revealing secrets",
    )

    logout_parser = subparsers.add_parser(
        "logout",
        parents=[common],
        help="Remove stored credentials for a provider from auth.json",
    )
    logout_parser.add_argument(
        "provider",
        help="Provider name or alias to remove from auth.json",
    )
    return parser


def _auth_login(
    provider_arg: str | None,
    global_dir: Path,
    registry: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> int:
    statuses = inspect_provider_auth_statuses(registry, config, global_dir)
    if not provider_arg:
        print("Available providers:")
        for index, entry in enumerate(statuses, start=1):
            print(format_provider_auth_line(entry, index=index))
        try:
            chosen = input("Select provider (name or number): ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nAborted.", file=sys.stderr)
            return 1
        if not chosen:
            print("❌ Aborted: no provider selected.", file=sys.stderr)
            return 1
        provider_arg = chosen

    ok, canonical, error = resolve_auth_provider(
        provider_arg, registry, config, global_dir, statuses=statuses
    )
    if not ok:
        print(f"❌ {error}", file=sys.stderr)
        return 1

    try:
        api_key = getpass.getpass(f"API key for {canonical}: ")
    except (EOFError, KeyboardInterrupt):
        print("\nAborted.", file=sys.stderr)
        return 1

    saved, _, message = login_provider_credential(
        canonical,
        api_key,
        registry=registry,
        config=config,
        global_dir=global_dir,
    )
    if not saved:
        print(f"❌ {message}", file=sys.stderr)
        return 1
    print(f"✅ {message}")
    return 0


def _auth_list(
    global_dir: Path,
    registry: dict[str, dict[str, Any]],
    config: dict[str, Any],
) -> int:
    auth_path = get_auth_store_path(global_dir)
    exists_label = "exists" if auth_path and auth_path.is_file() else "not created yet"
    print(f"Credential store: {auth_path} ({exists_label})")
    for entry in inspect_provider_auth_statuses(registry, config, global_dir):
        print(format_provider_auth_line(entry))
    return 0


def _auth_logout(
    provider_arg: str,
    global_dir: Path,
    registry: dict[str, dict[str, Any]],
    config: dict[str, Any] | None = None,
) -> int:
    ok, removed, _, message = logout_provider_credential(
        provider_arg,
        registry=registry,
        config=config,
        global_dir=global_dir,
    )
    if not ok:
        print(f"❌ {message}", file=sys.stderr)
        return 1
    prefix = "✅" if removed else "ℹ️ "
    print(f"{prefix} {message}")
    return 0


def _run_auth_cli(argv: list[str]) -> int:
    parser = _build_auth_arg_parser()
    args = parser.parse_args(argv)
    if not args.auth_command:
        parser.print_help()
        return 0

    raw_global = getattr(args, "global_dir", None)
    raw_dir = getattr(args, "dir", ".")
    global_dir = Path(raw_global).resolve() if raw_global else get_default_global_dir()
    target_dir = Path(raw_dir).resolve()

    layers = ConfigLayers.load(global_dir, target_dir)
    config = layers.merged
    registry = load_provider_registry(
        global_dir=global_dir,
        project_dir=target_dir,
        inline_providers=config.get("providers"),
    )

    if args.auth_command == "login":
        return _auth_login(args.provider, global_dir, registry, config)
    if args.auth_command == "list":
        return _auth_list(global_dir, registry, config)
    if args.auth_command == "logout":
        return _auth_logout(args.provider, global_dir, registry, config)
    parser.print_help()
    return 1


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    raw_argv = sys.argv[1:] if argv is None else list(argv)
    if raw_argv and raw_argv[0] == "auth":
        exit_code = _run_auth_cli(raw_argv[1:])
        if exit_code != 0:
            sys.exit(exit_code)
        return

    parser = _build_arg_parser()
    args = parser.parse_args(raw_argv)
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

    from core.model import HttpChatModel, MockModel

    conn = (
        Connection(
            endpoint=context.model.endpoint,
            api_key=context.model.api_key,
            model=getattr(context.model, "model", "unknown"),
            auth_scheme=getattr(context.model, "auth_scheme", "bearer") or "bearer",
            auth_options=dict(getattr(context.model, "auth_options", None) or {}),
            auth_error=getattr(context.model, "auth_error", None),
        )
        if isinstance(context.model, HttpChatModel)
        else Connection(model="offline")
    )

    if isinstance(context.model, MockModel) or conn.missing_credentials:
        reason = (
            f"No API key configured for model '{conn.model}'."
            if conn.missing_credentials
            else "No API key or model endpoint configured."
        )
        print(
            f"❌ {reason}\n\n"
            "To configure a provider, choose one of:\n"
            "  1. Run `microharness auth login [provider]` (or `/login` inside a session)\n"
            "  2. Add credentials to .env (e.g. GEMINI_API_KEY=..., OPENROUTER_API_KEY=..., AI_GATEWAY_API_KEY=...)\n"
            "  3. Pass CLI flags: --provider <name> --api-key <key>\n"
            "  4. Use a local provider: --provider lmstudio (or ollama, vllm, llamacpp)\n"
            "  5. Configure ~/.microharness/config.json with default_model and providers.\n\n"
            "Tip: Run with --demo calculator to test offline without API credentials.\n",
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
        docker_network=context.docker_network,
        allow_network_tools=context.allow_network_tools,
        bash_timeout=context.bash_timeout,
        cache_ttl=context.cache_ttl,
        reasoning_effort=context.reasoning_effort,
        trust_project_tools=context.trust_project_tools,
        permission_mode=context.permission_mode,
        docker_memory=context.docker_memory,
        docker_cpus=context.docker_cpus,
        docker_pids_limit=context.docker_pids_limit,
        verbose=True,
    )
    if context.model_spec is not None:
        harness.apply_model_context_tokens(context.model_spec)

    harness.run_tui(initial_instruction=args.instruction, resume=args.resume)


if __name__ == "__main__":
    main()
