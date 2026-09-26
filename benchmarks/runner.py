from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from core.evaluator import Evaluator
from core.harness import Harness
from core.types import RunResult, Task


@dataclass
class TaskReport:
    task_id: str
    passed: bool
    score: float
    total_steps: int
    duration_s: float
    finish_reason: str
    trace_path: str
    workspace_path: str
    patch_path: str | None = None
    feedback: str = ""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_hit_rate_pct: float = 0.0


@dataclass
class BenchmarkReport:
    suite_name: str
    total_tasks: int
    passed_tasks: int
    pass_rate: float
    mean_score: float
    total_duration_s: float
    tasks: list[TaskReport]
    total_prompt_tokens: int = 0
    total_completion_tokens: int = 0
    total_cached_tokens: int = 0
    cache_hit_rate_pct: float = 0.0
    config: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def print_scorecard(self) -> None:
        width = 108
        print("\n" + "=" * width)
        print(f"📊 {self.suite_name.upper()} BENCHMARK SCORECARD".center(width))
        print("=" * width)
        if self.config:
            cfg_parts = []
            if self.config.get("model"):
                cfg_parts.append(f"Model: {self.config['model']}")
            if self.config.get("provider"):
                cfg_parts.append(f"Provider: {self.config['provider']}")
            if self.config.get("reasoning_effort"):
                cfg_parts.append(f"Reasoning: {str(self.config['reasoning_effort']).upper()}")
            if self.config.get("max_steps"):
                cfg_parts.append(f"Max Steps: {self.config['max_steps']}")
            if cfg_parts:
                print(("Config: " + " | ".join(cfg_parts)).center(width))
                print("-" * width)
        header = (
            f"{'Task ID':<30} {'Status':<10} {'Score':<7} {'Steps':<6} {'Duration':<9} {'Tokens (In / Out)':<22} {'Cache Hit':<10} {'Trace'}"
        )
        print(header)
        print("-" * width)
        for t in self.tasks:
            status_str = "PASSED ✅" if t.passed else "FAILED ❌"
            trace_short = Path(t.trace_path).name if t.trace_path else "-"
            tokens_str = f"{t.prompt_tokens:,} / {t.completion_tokens:,}"
            hit_str = f"{t.cache_hit_rate_pct:.1f}%" if t.prompt_tokens > 0 else "-"
            line = (
                f"{t.task_id:<30} {status_str:<10} {t.score:<7.2f} {t.total_steps:<6} {t.duration_s:<8.1f}s "
                f"{tokens_str:<22} {hit_str:<10} {trace_short}"
            )
            print(line)
        print("-" * width)
        summary = (
            f"Result: {self.passed_tasks}/{self.total_tasks} passed ({self.pass_rate:.1f}%) "
            f"| Mean Score: {self.mean_score:.2f} | Total Time: {self.total_duration_s:.1f}s"
        )
        print(summary.center(width))
        if self.total_prompt_tokens > 0 or self.total_completion_tokens > 0:
            token_summary = (
                f"Tokens: {self.total_prompt_tokens:,} in ({self.total_cached_tokens:,} cached - "
                f"{self.cache_hit_rate_pct:.1f}% hit rate) | {self.total_completion_tokens:,} out"
            )
            print(token_summary.center(width))
        print("=" * width + "\n")


def _build_report(
    suite_name: str,
    total_tasks: int,
    task_reports: list[TaskReport],
    duration_s: float,
    config: dict[str, Any] | None = None,
) -> BenchmarkReport:
    """Builds a BenchmarkReport with consistent denominators and zero-division guards."""
    passed = sum(1 for t in task_reports if t.passed)
    denominator = total_tasks if total_tasks > 0 else len(task_reports)
    pass_rate = (passed / denominator * 100.0) if denominator > 0 else 0.0
    mean_score = (sum(t.score for t in task_reports) / denominator) if denominator > 0 else 0.0

    total_prompt = sum(t.prompt_tokens for t in task_reports)
    total_comp = sum(t.completion_tokens for t in task_reports)
    total_cached = sum(t.cached_tokens for t in task_reports)
    overall_hit_rate = (
        round(total_cached / total_prompt * 100.0, 1) if total_prompt > 0 else 0.0
    )

    return BenchmarkReport(
        suite_name=suite_name,
        total_tasks=total_tasks,
        passed_tasks=passed,
        pass_rate=pass_rate,
        mean_score=mean_score,
        total_duration_s=duration_s,
        tasks=list(task_reports),
        total_prompt_tokens=total_prompt,
        total_completion_tokens=total_comp,
        total_cached_tokens=total_cached,
        cache_hit_rate_pct=overall_hit_rate,
        config=dict(config or {}),
    )


def _save_report_json(report: BenchmarkReport, output_dir: Path | str, suite_name: str) -> Path:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    report_file = out_path / f"{suite_name.lower().replace('-', '_')}_report.json"
    report_file.write_text(json.dumps(report.to_dict(), indent=2), encoding="utf-8")
    return report_file


def run_benchmark_suite(
    harness_factory: Callable[[Evaluator], Harness],
    tasks: list[tuple[Task, Evaluator]],
    suite_name: str = "Mini-SWE",
    output_dir: Path | str | None = None,
    verbose: bool = True,
    resume: bool = False,
    config: dict[str, Any] | None = None,
) -> BenchmarkReport:
    """Executes a list of (Task, Evaluator) instances sequentially and aggregates results."""
    start_time = time.time()
    task_reports: list[TaskReport] = []
    effective_config: dict[str, Any] = dict(config or {})

    retained_tasks_map: dict[str, TaskReport] = {}
    if resume and output_dir:
        report_file = Path(output_dir) / f"{suite_name.lower().replace('-', '_')}_report.json"
        if report_file.is_file():
            try:
                prev_data = json.loads(report_file.read_text(encoding="utf-8"))
                prev_cfg = prev_data.get("config", {})
                if isinstance(prev_cfg, dict):
                    effective_config = {**prev_cfg, **effective_config}

                for t_dict in prev_data.get("tasks", []):
                    finish_reason = str(t_dict.get("finish_reason", "completed"))
                    is_passed = bool(t_dict.get("passed", False))
                    total_steps = int(t_dict.get("total_steps", 0))

                    # Rerun tasks that failed due to infrastructure/API errors or rate limits,
                    # or aborted before executing any steps.
                    # Retain tasks that passed, OR completed normal evaluation without an error
                    # (do not re-run legitimate attempts just because the agent did not pass).
                    is_error = finish_reason in ("error", "rate_limited") or (
                        not is_passed and total_steps == 0
                    )
                    if is_passed or not is_error:
                        retained_tasks_map[t_dict["task_id"]] = TaskReport(
                            task_id=t_dict["task_id"],
                            passed=is_passed,
                            score=float(t_dict.get("score", 1.0 if is_passed else 0.0)),
                            total_steps=total_steps,
                            duration_s=float(t_dict.get("duration_s", 0.0)),
                            finish_reason=finish_reason,
                            trace_path=str(t_dict.get("trace_path", "")),
                            workspace_path=str(t_dict.get("workspace_path", "")),
                            patch_path=t_dict.get("patch_path"),
                            feedback=str(t_dict.get("feedback", "")),
                            prompt_tokens=int(t_dict.get("prompt_tokens", 0)),
                            completion_tokens=int(t_dict.get("completion_tokens", 0)),
                            cached_tokens=int(t_dict.get("cached_tokens", 0)),
                            cache_hit_rate_pct=float(t_dict.get("cache_hit_rate_pct", 0.0)),
                        )
                if verbose and retained_tasks_map:
                    print(
                        f"🔄 Resuming {suite_name}: {len(retained_tasks_map)} task(s) already completed in previous run (retaining results)."
                    )
            except Exception as exc:
                if verbose:
                    print(
                        f"⚠️ Could not load previous report for resume ({exc}). Running all tasks."
                    )

    if verbose:
        print(f"\n🏁 Launching {suite_name} Benchmark ({len(tasks)} tasks)...")

    for idx, (task, evaluator) in enumerate(tasks, start=1):
        if task.id in retained_tasks_map:
            retained = retained_tasks_map[task.id]
            if verbose:
                status_str = "PASSED" if retained.passed else f"COMPLETED ({retained.finish_reason})"
                print(
                    f"\n[{idx}/{len(tasks)}] ⏭️ Task '{task.id}' already {status_str} in previous run (score={retained.score:.2f}). Skipping."
                )
            task_reports.append(retained)
            continue

        if verbose:
            print(f"\n[{idx}/{len(tasks)}] Running Task: {task.id}")

        harness = harness_factory(evaluator)
        run_result: RunResult = harness.run(task)

        task_reports.append(
            TaskReport(
                task_id=task.id,
                passed=run_result.passed,
                score=run_result.score,
                total_steps=run_result.total_steps,
                duration_s=run_result.duration_s,
                finish_reason=run_result.finish_reason,
                trace_path=run_result.trace_path,
                workspace_path=run_result.workspace_path,
                patch_path=run_result.patch_path,
                feedback=run_result.eval_result.feedback if run_result.eval_result else "",
                prompt_tokens=run_result.prompt_tokens,
                completion_tokens=run_result.completion_tokens,
                cached_tokens=run_result.cached_tokens,
                cache_hit_rate_pct=run_result.cache_hit_rate_pct,
            )
        )

        # Incrementally persist benchmark report after each task
        if output_dir:
            cur_report = _build_report(
                suite_name=suite_name,
                total_tasks=len(tasks),
                task_reports=task_reports,
                duration_s=time.time() - start_time,
                config=effective_config,
            )
            _save_report_json(cur_report, output_dir, suite_name)

        if run_result.finish_reason == "rate_limited":
            # Preserve any remaining retained tasks from previous run so they are not lost
            seen_ids = {t.task_id for t in task_reports}
            for rem_task, _ in tasks[idx:]:
                if rem_task.id in retained_tasks_map and rem_task.id not in seen_ids:
                    task_reports.append(retained_tasks_map[rem_task.id])
            if output_dir:
                cur_report = _build_report(
                    suite_name=suite_name,
                    total_tasks=len(tasks),
                    task_reports=task_reports,
                    duration_s=time.time() - start_time,
                    config=effective_config,
                )
                _save_report_json(cur_report, output_dir, suite_name)

            if verbose:
                print(
                    f"\n🛑 Benchmark halted due to rate limits on task '{task.id}'.\n"
                    f"Saved {len(task_reports)} task result(s) to report.\n"
                    f"Tip: Run with --resume to continue remaining tasks once your rate limits reset."
                )
            break

    report = _build_report(
        suite_name=suite_name,
        total_tasks=len(tasks),
        task_reports=task_reports,
        duration_s=time.time() - start_time,
        config=effective_config,
    )

    if verbose:
        report.print_scorecard()

    if output_dir:
        report_file = _save_report_json(report, output_dir, suite_name)
        if verbose:
            print(f"📄 Benchmark report saved to: {report_file}")

    return report
