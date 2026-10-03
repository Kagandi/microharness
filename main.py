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
from pathlib import Path
from typing import Any, Callable

from core.evaluator import CommandTestEvaluator, Evaluator, FileContentEvaluator
from core.harness import (
    Harness,
    get_default_global_dir,
    load_global_config,
    load_project_config,
)
from core.model import (
    EndpointAwareModel,
    HttpChatModel,
    MockModel,
    Model,
    NamedModel,
    ResettableModel,
    find_provider_by_endpoint,
    get_available_models_from_config,
    load_provider_registry,
    resolve_provider,
    seed_user_providers,
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


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="microharness: Interactive AI coding agent microframework for existing projects or benchmarks.",
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
        help="Global configuration and skills directory (default: ~/.microharness)",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Run single one-shot task and exit immediately without entering interactive chat",
    )
    parser.add_argument(
        "--tui",
        action="store_true",
        help="Launch interactive mode in full-screen TUI with bottom-anchored chat",
    )
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
        "--resume",
        nargs="?",
        const=True,
        default=False,
        help="Resume previous session or benchmark execution (optional: run_id to resume specific session)",
    )
    parser.add_argument(
        "--demo",
        choices=["calculator", "greeting"],
        default=None,
        help="Run a built-in educational benchmark demo (runs sandboxed with evaluation)",
    )
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
        "--runs-dir",
        default=None,
        help="Directory to store runs, logs, and traces (default: ~/.microharness/runs)",
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
        "--trust-project-tools",
        action="store_true",
        default=None,
        help="Allow executing custom tool plugins defined in the project workspace (default: false)",
    )
    return parser


def _resolve_model(
    args: argparse.Namespace,
    config: dict[str, Any],
    cache_ttl: str,
    reasoning_effort: str | None = None,
    global_dir: Path | None = None,
    target_dir: Path | None = None,
) -> Model:
    registry = load_provider_registry(
        global_dir=global_dir,
        project_dir=target_dir,
        inline_providers=config.get("providers"),
    )
    provider_arg = getattr(args, "provider", None) or config.get("provider")
    resolved_name, profile = resolve_provider(registry, provider_arg)

    # If provider wasn't explicitly named, attempt to detect by endpoint
    endpoint_candidate = (
        getattr(args, "endpoint", None)
        or os.getenv("MODEL_ENDPOINT")
        or config.get("endpoint")
    )
    if not profile and endpoint_candidate:
        detected_name, detected_profile = find_provider_by_endpoint(registry, endpoint_candidate)
        if detected_profile:
            resolved_name = detected_name
            profile = detected_profile

    default_endpoint = profile.get("endpoint")
    default_model = profile.get("default_model", "cohere/north-mini-code:free")

    endpoint = (
        getattr(args, "endpoint", None)
        or os.getenv("MODEL_ENDPOINT")
        or config.get("endpoint")
        or default_endpoint
    )
    model_name = (
        getattr(args, "model", None)
        or os.getenv("MODEL_NAME")
        or config.get("model")
        or default_model
    )

    api_key = getattr(args, "api_key", None) or os.getenv("MODEL_API_KEY")
    if not api_key and profile.get("env_keys"):
        for env_k in profile["env_keys"]:
            if val := os.getenv(env_k):
                api_key = val
                break
    if not api_key:
        fallback_env_keys = [
            "GEMINI_API_KEY",
            "GOOGLE_API_KEY",
            "OPENROUTER_API_KEY",
            "OPENAI_API_KEY",
            "ANTHROPIC_API_KEY",
            "DEEPSEEK_API_KEY",
            "GROQ_API_KEY",
        ]
        for env_k in fallback_env_keys:
            if val := os.getenv(env_k):
                api_key = val
                break
    if not api_key:
        api_key = config.get("api_key")

    headers = dict(profile.get("headers") or {})
    if isinstance(config.get("headers"), dict):
        headers.update(config["headers"])

    reasoning_format = profile.get("reasoning_format", "standard")
    cache_strategy = profile.get("cache_strategy", "implicit")

    if endpoint:
        cache_lbl = cache_ttl.upper() if cache_ttl and cache_ttl.lower() != "off" else "OFF"
        reasoning_lbl = f", reasoning: {reasoning_effort.upper()}" if reasoning_effort else ""
        prov_lbl = f" [{resolved_name}]" if resolved_name else ""
        print(f"🔌 Connected to{prov_lbl}: {endpoint} (model: {model_name}, cache: {cache_lbl}{reasoning_lbl})")

        # An explicit --model that matches no configured alias, model id, or known
        # provider is almost always a typo. We warn rather than refuse so the flag
        # stays a usable escape hatch for providers the registry does not know yet,
        # but the mismatch is surfaced here instead of as an opaque HTTP error later.
        requested_model = getattr(args, "model", None)
        if requested_model:
            try:
                registry = load_provider_registry(
                    global_dir=global_dir, project_dir=target_dir
                )
                known = get_available_models_from_config(config, registry)
                names = {m["name"].lower() for m in known}
                ids = {m.get("model", "").lower() for m in known if m.get("model")}
                reg_names = {n.lower() for n in registry}
                head = requested_model.split("/", 1)[0].lower()
                if (
                    requested_model.strip().lower() not in names
                    and requested_model.strip().lower() not in ids
                    and requested_model.strip().lower() not in reg_names
                    and head not in reg_names
                ):
                    available = ", ".join(m["name"] for m in known) or "none configured"
                    print(
                        f"⚠️  Warning: --model '{requested_model}' does not match any configured "
                        f"model or known provider. Available: {available}",
                        file=sys.stderr,
                    )
            except Exception:
                pass

        return HttpChatModel(
            base_url=endpoint,
            model=model_name,
            api_key=api_key,
            headers=headers,
            cache_ttl=cache_ttl,
            reasoning_effort=reasoning_effort,
            reasoning_format=reasoning_format,
            cache_strategy=cache_strategy,
        )

    print("🎓 Running with deterministic MockModel (offline, no endpoint configured).")
    print(
        "Tip: Add MODEL_ENDPOINT to .env, pass --endpoint, or configure ~/.microharness/config.json.\n"
    )
    return build_mock_agent_script()


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

    def factory(ev: Evaluator) -> Harness:
        if isinstance(model, ResettableModel):
            model.reset()
        return Harness(
            model=model,
            global_dir=global_dir,
            runs_dir=runs_dir,
            sandbox=True,
            docker=use_docker,
            docker_image=docker_image,
            evaluator=ev,
            cache_ttl=cache_ttl,
            reasoning_effort=reasoning_effort,
            trust_project_tools=trust_project_tools,
            verbose=verbose,
        )

    return factory


def _dispatch_benchmark(
    args: argparse.Namespace,
    config: dict[str, Any],
    model: Model,
    runs_dir: Path,
    global_dir: Path,
    target_dir: Path,
    cache_ttl: str,
    reasoning_effort: str | None,
    use_docker: bool,
    docker_image: str,
    trust_project_tools: bool = False,
) -> bool:
    bench_choice = args.benchmark or args.demo
    if not bench_choice:
        return False

    factory = create_benchmark_harness_factory(
        model=model,
        global_dir=global_dir,
        runs_dir=runs_dir,
        use_docker=use_docker,
        docker_image=docker_image,
        cache_ttl=cache_ttl,
        reasoning_effort=reasoning_effort,
        trust_project_tools=trust_project_tools,
        verbose=True,
    )

    model_name = (
        model.model
        if isinstance(model, NamedModel)
        else getattr(model, "model", None)
    )
    endpoint_url = (
        model.endpoint
        if isinstance(model, EndpointAwareModel)
        else getattr(model, "endpoint", None)
    )
    benchmark_metadata = {
        "provider": getattr(args, "provider", None) or config.get("provider"),
        "model": model_name or getattr(args, "model", None) or config.get("model"),
        "endpoint": endpoint_url or getattr(args, "endpoint", None) or config.get("endpoint"),
        "cache_ttl": cache_ttl,
        "reasoning_effort": reasoning_effort,
        "max_steps": args.max_steps or int(config.get("max_steps", 30)),
    }

    if bench_choice in ("mini-swe", "all"):
        from benchmarks.mini_swe import get_mini_swe_suite
        from benchmarks.runner import run_benchmark_suite

        suite = get_mini_swe_suite(task_filter=args.task)
        if not suite:
            print(f"❌ No benchmark tasks found matching filter: '{args.task}'")
            return True

        if args.max_steps or args.timeout:
            for t, _ in suite:
                if args.max_steps:
                    t.max_steps = args.max_steps
                if args.timeout:
                    t.timeout_seconds = args.timeout

        run_benchmark_suite(
            harness_factory=factory,
            tasks=suite,
            suite_name="Mini-SWE",
            output_dir=runs_dir,
            verbose=True,
            resume=bool(args.resume),
            config=benchmark_metadata,
        )
        return True

    elif bench_choice == "deepswe-mini":
        from benchmarks.harbor import get_deepswe_mini_suite
        from benchmarks.runner import run_benchmark_suite

        dataset_dir = target_dir / "datasets" / "deep-swe" / "tasks"
        if not dataset_dir.is_dir():
            dataset_dir = Path("datasets/deep-swe/tasks").resolve()

        if not dataset_dir.is_dir():
            print("❌ DeepSWE dataset not found at datasets/deep-swe/tasks")
            print(
                "Tip: Run `git clone --depth 1 https://github.com/datacurve-ai/deep-swe datasets/deep-swe` to download tasks."
            )
            return True

        max_steps = args.max_steps or int(config.get("max_steps", 30))
        timeout_seconds = args.timeout or (max_steps * 30.0 if max_steps > 30 else 900.0)

        suite = get_deepswe_mini_suite(
            dataset_dir=dataset_dir,
            task_filter=args.task,
            max_steps=max_steps,
            timeout_seconds=timeout_seconds,
        )
        if not suite:
            print(f"❌ No DeepSWE tasks found matching filter: '{args.task}'")
            return True

        run_benchmark_suite(
            harness_factory=factory,
            tasks=suite,
            suite_name="DeepSWE-mini",
            output_dir=runs_dir,
            verbose=True,
            resume=bool(args.resume),
            config=benchmark_metadata,
        )
        return True

    elif bench_choice in ("calculator", "greeting"):
        task, evaluator = (
            create_calculator_demo() if bench_choice == "calculator" else create_greeting_demo()
        )
        harness = factory(evaluator)
        result = harness.run(task)
        _print_run_summary(result)
        return True

    return False


def main(argv: list[str] | None = None) -> None:
    load_dotenv()
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # Resolve target directory, global config (~/.microharness/config.json), and project config (.microharness/config.json)
    target_dir = Path(args.dir).resolve()
    global_dir = Path(args.global_dir).resolve() if args.global_dir else get_default_global_dir()
    seed_user_providers(global_dir)
    global_config = load_global_config(global_dir)
    project_config = load_project_config(target_dir)
    config = {**global_config, **project_config}

    configured_runs_dir = args.runs_dir or config.get("runs_dir")
    runs_dir = (
        Path(configured_runs_dir).resolve()
        if configured_runs_dir
        else (global_dir / "runs").resolve()
    )
    cache_ttl = args.cache_ttl or os.getenv("CACHE_TTL") or config.get("cache_ttl", "5m")
    reasoning_effort = (
        getattr(args, "reasoning", None)
        or os.getenv("REASONING_EFFORT")
        or os.getenv("MODEL_REASONING_EFFORT")
        or config.get("reasoning_effort")
        or config.get("reasoning")
    )
    use_docker = args.docker if args.docker is not None else bool(config.get("docker", False))
    docker_image = args.docker_image or config.get("docker_image", "python:3.11")

    # Trust boundary: project tools must be explicitly trusted via CLI flag or user global config.
    # Project config is intentionally prohibited from self-granting trust.
    trust_project_tools = (
        args.trust_project_tools
        if args.trust_project_tools is not None
        else bool(global_config.get("trust_project_tools", False))
    )
    if project_config.get("trust_project_tools") and not trust_project_tools:
        print(
            "⚠️ Notice: Project config attempted to enable 'trust_project_tools'. "
            "Project tools can only be trusted via CLI flag (--trust-project-tools) "
            "or global config (~/.microharness/config.json).",
            file=sys.stderr,
        )

    # 1. Select Model
    model = _resolve_model(
        args,
        config,
        cache_ttl,
        reasoning_effort,
        global_dir=global_dir,
        target_dir=target_dir,
    )

    # 2. Benchmark Suite Mode (e.g. --benchmark mini-swe or --demo calculator)
    if args.benchmark or args.demo:
        _dispatch_benchmark(
            args=args,
            config=config,
            model=model,
            runs_dir=runs_dir,
            global_dir=global_dir,
            target_dir=target_dir,
            cache_ttl=cache_ttl,
            reasoning_effort=reasoning_effort,
            use_docker=use_docker,
            docker_image=docker_image,
            trust_project_tools=trust_project_tools,
        )
        return

    # 3. Resolve Sandbox Flag and Instantiate Harness
    sandbox = args.sandbox or bool(config.get("sandbox", False))
    harness = Harness(
        model=model,
        workspace_dir=target_dir,
        global_dir=global_dir,
        runs_dir=runs_dir,
        sandbox=sandbox,
        docker=use_docker,
        docker_image=docker_image,
        cache_ttl=cache_ttl,
        reasoning_effort=reasoning_effort,
        trust_project_tools=trust_project_tools,
        verbose=True,
    )

    # 4. Batch Mode vs Interactive Mode
    if args.batch:
        if not args.instruction:
            parser.error("--batch requires an instruction argument.")
        task = Task(
            id="user_task",
            instruction=args.instruction.strip(),
            max_steps=args.max_steps or 15,
            timeout_seconds=args.timeout or 60.0,
        )
        result = harness.run(task)
        _print_run_summary(result)
        return

    # Default: Interactive Mode in the target project directory
    if args.tui:
        harness.run_tui(initial_instruction=args.instruction, resume=args.resume)
    else:
        harness.run_interactive(initial_instruction=args.instruction, resume=args.resume)


if __name__ == "__main__":
    main()
