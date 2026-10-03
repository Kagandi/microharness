from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib  # type: ignore

from core.environment import ContainerEnvironment, Environment
from core.types import EvalResult, StepTrace, Task

DEEPSWE_MINI_TASKS: list[str] = [
    "aiomonitor-task-snapshots-diff",
    "bandit-structured-nosec-directives",
    "dasel-html-document-format",
    "dynamodb-toolbox-lazy-recursive-schemas",
    "expr-try-catch-errors",
    "ipython-session-bundle-replay",
    "koota-composite-trait-aspects",
    "narwhals-rolling-window-suite",
    "ofetch-per-origin-circuit-breaker",
    "prometheus-transactional-reload-status",
    "prometheus-typed-label-sorting",
    "scc-bounded-memory-spilling",
    "termenv-preserve-ansi-resets",
    "textual-kitty-key-phases",
    "valibot-recursive-schema-composition",
    "yaegi-go-embed-directives",
]


@dataclass
class HarborTaskDef:
    task_id: str
    display_title: str
    language: str
    repository_url: str
    base_commit: str
    docker_image: str
    instruction: str
    task_dir: Path


class HarborEvaluator:
    """Evaluates Harbor and DeepSWE tasks inside an isolated Docker sandbox."""

    def __init__(
        self,
        task_dir: Path | str,
        base_commit: str,
        collect_cmd: str | None = None,
        timeout: float = 600.0,
    ):
        self.task_dir = Path(task_dir).resolve()
        self.base_commit = base_commit
        self.collect_cmd = collect_cmd
        self.timeout = timeout

    def evaluate(self, env: Environment, steps: list[StepTrace]) -> EvalResult:
        if not isinstance(env, ContainerEnvironment) or not env.is_running:
            return EvalResult(
                passed=False,
                score=0.0,
                feedback="Harbor evaluation requires an active DockerEnvironment (ContainerEnvironment).",
            )

        container = env.container_name
        tests_dir = self.task_dir / "tests"

        # 1. Dynamically inject pristine tests into container at /tests/
        if tests_dir.is_dir():
            subprocess.run(
                ["docker", "cp", f"{tests_dir}/.", f"{container}:/tests/"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )

        # 2. Collect patch against base_commit so both staged & unstaged changes are preserved
        collect_core = self.collect_cmd or (
            f"git diff --binary {self.base_commit} HEAD > /logs/artifacts/model.patch"
        )
        collect_script = (
            "mkdir -p /logs/artifacts /logs/verifier && cd /app && "
            "git config --global --add safe.directory /app >/dev/null 2>&1 || true && "
            "git add -A >/dev/null 2>&1 || true && "
            "git -c user.name='Verifier' -c user.email='verifier@eval' commit -m 'eval_staging' >/dev/null 2>&1 || true && "
            f"{collect_core} && "
            f"git checkout -f {self.base_commit} >/dev/null 2>&1 || true && "
            "git clean -fd >/dev/null 2>&1 || true"
        )
        env.execute_bash(collect_script, timeout=60.0)

        # 3. Execute verifier script /tests/test.sh inside container
        exit_code, test_output = env.execute_bash("bash /tests/test.sh", timeout=self.timeout)

        # 4. Read reward artifact /logs/verifier/reward.json
        _, reward_raw = env.execute_bash("cat /logs/verifier/reward.json", timeout=10.0)

        reward_data: dict[str, Any] = {}
        if reward_raw.strip():
            try:
                reward_data = json.loads(reward_raw.strip())
            except Exception:
                pass

        reward = reward_data.get("reward", 0)
        passed = reward == 1
        partial = reward_data.get("partial")
        score = float(partial) if partial is not None else float(reward)
        f2p = reward_data.get("f2p_passed", 0)
        f2p_tot = reward_data.get("f2p_total", 0)
        p2p = reward_data.get("p2p_passed", 0)
        p2p_tot = reward_data.get("p2p_total", 0)
        partial_val = float(partial) if partial is not None else float(reward)

        output_preview = test_output.strip()[-2000:] if test_output.strip() else "(no output)"
        feedback = (
            f"Harbor/DeepSWE Verification {'PASSED ✅' if passed else 'FAILED ❌'} (exit {exit_code})\n"
            f"Score: {score:.2f} | Partial: {partial_val:.2f} | F2P: {f2p}/{f2p_tot} | P2P: {p2p}/{p2p_tot}\n"
            f"Verifier Output:\n{output_preview}"
        )
        return EvalResult(passed=passed, score=score, feedback=feedback)


def load_harbor_task(
    task_dir: Path | str,
    max_steps: int = 30,
    timeout_seconds: float = 900.0,
) -> tuple[Task, HarborEvaluator]:
    """Loads a Harbor/DeepSWE task definition from its directory."""
    path = Path(task_dir).resolve()
    toml_file = path / "task.toml"
    instruction_file = path / "instruction.md"

    if not toml_file.is_file():
        raise FileNotFoundError(f"Missing task.toml in {path}")
    if not instruction_file.is_file():
        raise FileNotFoundError(f"Missing instruction.md in {path}")

    with open(toml_file, "rb") as f:
        cfg = tomllib.load(f)

    meta = cfg.get("metadata", {})
    task_id = meta.get("task_id") or path.name
    base_commit = meta.get("base_commit_hash", "HEAD~1")
    docker_image = cfg.get("environment", {}).get("docker_image")
    if not docker_image:
        raise ValueError(f"Task '{task_id}' has no docker_image in task.toml [environment].")

    instruction = instruction_file.read_text(encoding="utf-8", errors="replace").strip()

    collect_cmd = None
    collect_list = cfg.get("verifier", {}).get("collect", [])
    if isinstance(collect_list, list) and len(collect_list) > 0:
        collect_cmd = collect_list[0].get("command")

    task = Task(
        id=task_id,
        instruction=instruction,
        max_steps=max_steps,
        timeout_seconds=timeout_seconds,
        docker_image=docker_image,
        init_workdir="/app",
    )
    evaluator = HarborEvaluator(task_dir=path, base_commit=base_commit, collect_cmd=collect_cmd)
    return task, evaluator


def get_deepswe_mini_suite(
    dataset_dir: Path | str | None = None,
    task_filter: str | None = None,
    max_steps: int = 30,
    timeout_seconds: float = 900.0,
) -> list[tuple[Task, HarborEvaluator]]:
    """Returns list of (Task, HarborEvaluator) for the 16 DeepSWE-mini tasks."""
    base_path = (
        Path(dataset_dir).resolve() if dataset_dir else Path("datasets/deep-swe/tasks").resolve()
    )
    if not base_path.is_dir():
        # Fallback to current project root/datasets/deep-swe/tasks
        candidate = Path(__file__).resolve().parent.parent / "datasets" / "deep-swe" / "tasks"
        if candidate.is_dir():
            base_path = candidate

    if not base_path.is_dir():
        return []

    suite: list[tuple[Task, HarborEvaluator]] = []
    for tid in DEEPSWE_MINI_TASKS:
        if task_filter:
            tf = task_filter.lower().strip()
            if tf not in tid.lower() and tid.lower() not in tf:
                continue

        task_path = base_path / tid
        if not task_path.is_dir():
            continue

        try:
            task, evaluator = load_harbor_task(
                task_path,
                max_steps=max_steps,
                timeout_seconds=timeout_seconds,
            )
            suite.append((task, evaluator))
        except Exception as exc:
            print(f"⚠️ Warning: Failed to load DeepSWE task '{tid}': {exc}")

    return suite
