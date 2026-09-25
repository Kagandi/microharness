from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
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


@dataclass
class BenchmarkReport:
    suite_name: str
    total_tasks: int
    passed_tasks: int
    pass_rate: float
    mean_score: float
    total_duration_s: float
    tasks: list[TaskReport]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def print_scorecard(self) -> None:
        width = 96
        print("\n" + "=" * width)
        print(f"📊 {self.suite_name.upper()} BENCHMARK SCORECARD".center(width))
        print("=" * width)
        header = (
            f"{'Task ID':<32} {'Status':<10} {'Score':<8} {'Steps':<7} {'Duration':<10} {'Trace'}"
        )
        print(header)
        print("-" * width)
        for t in self.tasks:
            status_str = "PASSED ✅" if t.passed else "FAILED ❌"
            trace_short = Path(t.trace_path).name if t.trace_path else "-"
            line = f"{t.task_id:<32} {status_str:<10} {t.score:<8.2f} {t.total_steps:<7} {t.duration_s:<9.1f}s {trace_short}"
            print(line)
        print("-" * width)
        summary = (
            f"Result: {self.passed_tasks}/{self.total_tasks} passed ({self.pass_rate:.1f}%) "
            f"| Mean Score: {self.mean_score:.2f} | Total Time: {self.total_duration_s:.1f}s"
        )
        print(summary.center(width))
        print("=" * width + "\n")


def _build_report(
    suite_name: str,
    total_tasks: int,
    task_reports: list[TaskReport],
    duration_s: float,
) -> BenchmarkReport:
    """Builds a BenchmarkReport with consistent denominators and zero-division guards."""
    passed = sum(1 for t in task_reports if t.passed)
    denominator = total_tasks if total_tasks > 0 else len(task_reports)
    pass_rate = (passed / denominator * 100.0) if denominator > 0 else 0.0
    mean_score = (sum(t.score for t in task_reports) / denominator) if denominator > 0 else 0.0
    return BenchmarkReport(
        suite_name=suite_name,
        total_tasks=total_tasks,
        passed_tasks=passed,
        pass_rate=pass_rate,
        mean_score=mean_score,
        total_duration_s=duration_s,
        tasks=list(task_reports),
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
) -> BenchmarkReport:
    """Executes a list of (Task, Evaluator) instances sequentially and aggregates results."""
    start_time = time.time()
    task_reports: list[TaskReport] = []

    if verbose:
        print(f"\n🏁 Launching {suite_name} Benchmark ({len(tasks)} tasks)...")

    for idx, (task, evaluator) in enumerate(tasks, start=1):
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
            )
        )

        # Incrementally persist benchmark report after each task
        if output_dir:
            cur_report = _build_report(
                suite_name=suite_name,
                total_tasks=len(tasks),
                task_reports=task_reports,
                duration_s=time.time() - start_time,
            )
            _save_report_json(cur_report, output_dir, suite_name)

    report = _build_report(
        suite_name=suite_name,
        total_tasks=len(tasks),
        task_reports=task_reports,
        duration_s=time.time() - start_time,
    )

    if verbose:
        report.print_scorecard()

    if output_dir:
        report_file = _save_report_json(report, output_dir, suite_name)
        if verbose:
            print(f"📄 Benchmark report saved to: {report_file}")

    return report
