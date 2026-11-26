"""Zero-dependency central CSV experiment tracking and historical backfill."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from core.defaults import load_defaults
from core.subagents import discover_subagents

if TYPE_CHECKING:
    from benchmarks.runner import BenchmarkReport, TaskReport

_DEFAULTS = load_defaults()

_NON_BENCHMARK_DIR_PREFIXES: tuple[str, ...] = (
    "interactive_session_",
    "tui_session_",
    "user_task_",
    "fix_calculator_bug_",
    "write_greeting_",
)
DEFAULT_RUNS_CSV: str = str(_DEFAULTS["experiments_runs_csv"])
DEFAULT_TASKS_CSV: str = str(_DEFAULTS["experiments_tasks_csv"])
DEFAULT_LIST_LIMIT: int = int(_DEFAULTS["experiments_list_limit"])

RUN_COLUMNS: tuple[str, ...] = (
    "run_id",
    "run_name",
    "timestamp_utc",
    "suite_name",
    "harness_version",
    "git_commit",
    "git_dirty",
    "provider",
    "model",
    "reasoning_effort",
    "max_steps",
    "max_context_tokens",
    "trigger_ratio",
    "bash_timeout",
    "docker",
    "task_filter",
    "subagents",
    "total_tasks",
    "passed_tasks",
    "pass_rate",
    "mean_score",
    "total_steps",
    "duration_s",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "cache_hit_rate_pct",
    "rate_limit_wait_s",
    "report_path",
)

TASK_COLUMNS: tuple[str, ...] = (
    "run_id",
    "run_name",
    "timestamp_utc",
    "suite_name",
    "harness_version",
    "git_commit",
    "provider",
    "model",
    "reasoning_effort",
    "task_id",
    "passed",
    "score",
    "total_steps",
    "duration_s",
    "finish_reason",
    "prompt_tokens",
    "cached_tokens",
    "completion_tokens",
    "cache_hit_rate_pct",
    "rate_limit_wait_s",
    "trace_path",
    "patch_path",
)


@runtime_checkable
class ExperimentTracker(Protocol):
    """Protocol for persisting aggregated benchmark runs and task-level outcomes."""

    def record_run(
        self,
        report: BenchmarkReport,
        *,
        output_dir: Path | None = None,
        run_id: str | None = None,
    ) -> Any: ...


def _utc_now_iso() -> str:
    """Returns the current UTC timestamp formatted in ISO-8601 with trailing 'Z'."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _get_package_version() -> str:
    """Returns the installed microharness package version."""
    from core import __version__

    return __version__


def collect_agent_metadata(repo_dir: Path | str | None = None) -> dict[str, Any]:
    """Collects package version and git commit/dirty state when inside a git repo."""
    cwd = Path(repo_dir).resolve() if repo_dir else Path.cwd().resolve()
    git_commit = ""
    git_dirty: bool | None = None

    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if rev.returncode == 0:
            git_commit = rev.stdout.strip()
            status = subprocess.run(
                ["git", "status", "--porcelain"],
                cwd=cwd,
                capture_output=True,
                text=True,
                timeout=2.0,
                check=False,
            )
            if status.returncode == 0:
                git_dirty = bool(status.stdout.strip())
    except (OSError, subprocess.SubprocessError):
        pass

    return {
        "harness_version": _get_package_version(),
        "git_commit": git_commit,
        "git_dirty": git_dirty,
    }


def collect_subagents_summary(
    global_dir: Path | str | None = None,
    target_dir: Path | str | None = None,
) -> dict[str, dict[str, Any]]:
    """Builds a compact sorted config snapshot of discovered and built-in subagents."""
    subagents = discover_subagents(target_dir, global_dir=global_dir)
    summary: dict[str, dict[str, Any]] = {}
    for name in sorted(subagents):
        spec = subagents[name]
        entry: dict[str, Any] = {"scope": spec.scope, "read_only": spec.read_only}
        if spec.model is not None:
            entry["model"] = spec.model
        if spec.effort is not None:
            entry["effort"] = spec.effort
        if spec.max_steps is not None:
            entry["max_steps"] = spec.max_steps
        if spec.tools is not None:
            entry["tools"] = list(spec.tools)
        summary[name] = entry
    return summary


def derive_run_id(
    suite_name: str,
    *,
    output_dir: Path | str | None = None,
    explicit_run_id: str | None = None,
    timestamp_utc: str | None = None,
) -> tuple[str, str]:
    """Derives a deterministic ``(run_id, run_name)`` pair for CSV upsert."""
    slug = suite_name.lower().replace("-", "_").replace(" ", "_")
    if output_dir is not None:
        resolved = Path(output_dir).resolve()
        report_key = str(resolved / f"{slug}_report.json")
        digest = hashlib.sha256(report_key.encode("utf-8")).hexdigest()[:8]
        run_name = resolved.name or slug
        run_id = explicit_run_id or f"{run_name}@{digest}"
        return run_id, run_name

    ts = (timestamp_utc or _utc_now_iso()).replace(":", "").replace("-", "")
    if explicit_run_id:
        return explicit_run_id, explicit_run_id
    run_id = f"{slug}_{ts}"
    return run_id, run_id


def _format_scalar(value: Any) -> str:
    """Formats scalar/JSON values consistently for CSV serialization."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (dict, list)):
        return json.dumps(value, sort_keys=True, separators=(",", ":"))
    return str(value)


def _read_csv_rows(csv_path: Path) -> tuple[list[str], list[dict[str, str]]]:
    """Reads an existing CSV into ``(header, rows)``."""
    if not csv_path.is_file():
        return [], []
    with csv_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        header = list(reader.fieldnames or [])
        rows = [dict(row) for row in reader]
    return header, rows


def _upsert_csv_rows(
    csv_path: Path,
    fieldnames: tuple[str, ...],
    new_rows: list[dict[str, str]],
    key_fields: tuple[str, ...],
) -> None:
    """Atomically upserts ``new_rows`` into ``csv_path`` keyed by ``key_fields``."""
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    existing_header, existing_rows = _read_csv_rows(csv_path)

    merged_fields: list[str] = list(fieldnames)
    for col in existing_header:
        if col not in merged_fields:
            merged_fields.append(col)

    index_by_key: dict[tuple[str, ...], int] = {}
    output_rows: list[dict[str, str]] = []
    for row in existing_rows:
        key = tuple(row.get(k, "") for k in key_fields)
        if all(key) and key not in index_by_key:
            index_by_key[key] = len(output_rows)
        output_rows.append({col: row.get(col, "") for col in merged_fields})

    for row in new_rows:
        normalized = {col: row.get(col, "") for col in merged_fields}
        key = tuple(normalized.get(k, "") for k in key_fields)
        if key in index_by_key:
            pos = index_by_key[key]
            # Preserve non-empty original timestamp_utc if caller passed the same timestamp fallback.
            if not normalized.get("timestamp_utc") and output_rows[pos].get("timestamp_utc"):
                normalized["timestamp_utc"] = output_rows[pos]["timestamp_utc"]
            output_rows[pos] = normalized
        else:
            if all(key):
                index_by_key[key] = len(output_rows)
            output_rows.append(normalized)

    tmp_path = csv_path.with_name(f".{csv_path.name}.{os.getpid()}.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=merged_fields)
        writer.writeheader()
        writer.writerows(output_rows)
    os.replace(tmp_path, csv_path)


class CsvExperimentTracker:
    """Persists benchmark run summaries and per-task outcomes to central CSV files."""

    def __init__(
        self,
        base_dir: Path | str,
        *,
        runs_filename: str = DEFAULT_RUNS_CSV,
        tasks_filename: str = DEFAULT_TASKS_CSV,
        auto_collect_agent_metadata: bool = True,
    ):
        self.base_dir = Path(base_dir).resolve()
        self.runs_csv_path = self.base_dir / runs_filename
        self.tasks_csv_path = self.base_dir / tasks_filename
        self.auto_collect_agent_metadata = auto_collect_agent_metadata

    def _existing_timestamp(self, run_id: str) -> str:
        _, rows = _read_csv_rows(self.runs_csv_path)
        for row in rows:
            if row.get("run_id") == run_id and row.get("timestamp_utc"):
                return row["timestamp_utc"]
        return ""

    def record_run(
        self,
        report: BenchmarkReport,
        *,
        output_dir: Path | None = None,
        run_id: str | None = None,
    ) -> tuple[Path, Path]:
        """Upserts one summary row and per-task rows for ``report``."""
        cfg = dict(report.config or {})
        meta: dict[str, Any] = {}
        if self.auto_collect_agent_metadata and any(
            k not in cfg for k in ("harness_version", "git_commit", "git_dirty")
        ):
            meta = collect_agent_metadata()

        slug = report.suite_name.lower().replace("-", "_").replace(" ", "_")
        report_path = ""
        if output_dir is not None:
            candidate = Path(output_dir).resolve() / f"{slug}_report.json"
            if candidate.is_file():
                report_path = str(candidate)

        resolved_run_id, run_name = derive_run_id(
            report.suite_name,
            output_dir=output_dir,
            explicit_run_id=run_id,
            timestamp_utc=_format_scalar(cfg.get("timestamp_utc")) or None,
        )

        timestamp_utc = (
            _format_scalar(cfg.get("timestamp_utc"))
            or self._existing_timestamp(resolved_run_id)
            or _utc_now_iso()
        )
        harness_version = _format_scalar(
            cfg.get("harness_version") if "harness_version" in cfg else meta.get("harness_version")
        )
        git_commit = _format_scalar(
            cfg.get("git_commit") if "git_commit" in cfg else meta.get("git_commit")
        )
        git_dirty = _format_scalar(
            cfg.get("git_dirty") if "git_dirty" in cfg else meta.get("git_dirty")
        )
        provider = _format_scalar(cfg.get("provider"))
        model = _format_scalar(cfg.get("model"))
        reasoning_effort = _format_scalar(cfg.get("reasoning_effort"))

        total_steps = sum(t.total_steps for t in report.tasks)

        run_row = {
            "run_id": resolved_run_id,
            "run_name": run_name,
            "timestamp_utc": timestamp_utc,
            "suite_name": report.suite_name,
            "harness_version": harness_version,
            "git_commit": git_commit,
            "git_dirty": git_dirty,
            "provider": provider,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "max_steps": _format_scalar(cfg.get("max_steps")),
            "max_context_tokens": _format_scalar(cfg.get("max_context_tokens")),
            "trigger_ratio": _format_scalar(cfg.get("trigger_ratio")),
            "bash_timeout": _format_scalar(cfg.get("bash_timeout")),
            "docker": _format_scalar(cfg.get("docker")),
            "task_filter": _format_scalar(cfg.get("task_filter")),
            "subagents": _format_scalar(cfg.get("subagents")),
            "total_tasks": str(report.total_tasks),
            "passed_tasks": str(report.passed_tasks),
            "pass_rate": f"{report.pass_rate:.2f}",
            "mean_score": f"{report.mean_score:.4f}",
            "total_steps": str(total_steps),
            "duration_s": f"{report.total_duration_s:.2f}",
            "prompt_tokens": str(report.total_prompt_tokens),
            "cached_tokens": str(report.total_cached_tokens),
            "completion_tokens": str(report.total_completion_tokens),
            "cache_hit_rate_pct": f"{report.cache_hit_rate_pct:.1f}",
            "rate_limit_wait_s": f"{report.total_rate_limit_wait_s:.2f}",
            "report_path": report_path,
        }

        task_rows: list[dict[str, str]] = []
        for t in report.tasks:
            task_rows.append(
                {
                    "run_id": resolved_run_id,
                    "run_name": run_name,
                    "timestamp_utc": timestamp_utc,
                    "suite_name": report.suite_name,
                    "harness_version": harness_version,
                    "git_commit": git_commit,
                    "provider": provider,
                    "model": model,
                    "reasoning_effort": reasoning_effort,
                    "task_id": t.task_id,
                    "passed": _format_scalar(t.passed),
                    "score": f"{t.score:.4f}",
                    "total_steps": str(t.total_steps),
                    "duration_s": f"{t.duration_s:.2f}",
                    "finish_reason": t.finish_reason,
                    "prompt_tokens": str(t.prompt_tokens),
                    "cached_tokens": str(t.cached_tokens),
                    "completion_tokens": str(t.completion_tokens),
                    "cache_hit_rate_pct": f"{t.cache_hit_rate_pct:.1f}",
                    "rate_limit_wait_s": f"{t.rate_limit_wait_s:.2f}",
                    "trace_path": t.trace_path,
                    "patch_path": t.patch_path or "",
                }
            )

        _upsert_csv_rows(self.runs_csv_path, RUN_COLUMNS, [run_row], ("run_id",))
        if task_rows:
            _upsert_csv_rows(
                self.tasks_csv_path,
                TASK_COLUMNS,
                task_rows,
                ("run_id", "task_id"),
            )
        return self.runs_csv_path, self.tasks_csv_path


def _extract_timestamp_from_report(report_file: Path, data: dict[str, Any]) -> str:
    """Resolves a best-effort historical UTC timestamp from config, traces, or file mtime."""
    cfg = data.get("config") or {}
    if isinstance(cfg, dict) and cfg.get("timestamp_utc"):
        return str(cfg["timestamp_utc"])

    timestamps: list[str] = []
    for t in data.get("tasks") or []:
        if not isinstance(t, dict):
            continue
        tp_raw = t.get("trace_path")
        if not tp_raw:
            continue
        tp = Path(str(tp_raw))
        if not tp.is_file():
            continue
        try:
            tr = json.loads(tp.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        started = tr.get("started_at")
        if isinstance(started, str) and started.strip():
            timestamps.append(started.strip())
    if timestamps:
        return min(timestamps)

    mtime = datetime.fromtimestamp(report_file.stat().st_mtime, tz=timezone.utc)
    return mtime.isoformat(timespec="seconds").replace("+00:00", "Z")


def _discover_report_files(runs_dir: Path) -> list[Path]:
    """Finds benchmark ``*_report.json`` files at depth 0 or 1 under ``runs_dir``."""
    if not runs_dir.is_dir():
        return []
    candidates: list[Path] = list(runs_dir.glob("*_report.json"))
    for child in runs_dir.iterdir():
        if child.is_dir():
            candidates.extend(child.glob("*_report.json"))
    valid = [p for p in candidates if p.is_file()]
    valid.sort(key=lambda p: p.stat().st_mtime)
    return valid


def _report_from_dict(data: dict[str, Any]) -> BenchmarkReport:
    """Constructs a ``BenchmarkReport`` from a parsed ``*_report.json`` payload."""
    from benchmarks.runner import BenchmarkReport, TaskReport

    task_reports: list[TaskReport] = []
    for t in data.get("tasks") or []:
        if not isinstance(t, dict) or not str(t.get("task_id", "")).strip():
            continue
        task_reports.append(
            TaskReport(
                task_id=str(t["task_id"]),
                passed=bool(t.get("passed", False)),
                score=float(t.get("score", 0.0)),
                total_steps=int(t.get("total_steps", 0)),
                duration_s=float(t.get("duration_s", 0.0)),
                finish_reason=str(t.get("finish_reason", "")),
                trace_path=str(t.get("trace_path", "")),
                workspace_path=str(t.get("workspace_path", "")),
                patch_path=t.get("patch_path"),
                feedback=str(t.get("feedback", "")),
                prompt_tokens=int(t.get("prompt_tokens", 0)),
                completion_tokens=int(t.get("completion_tokens", 0)),
                cached_tokens=int(t.get("cached_tokens", 0)),
                cache_hit_rate_pct=float(t.get("cache_hit_rate_pct", 0.0)),
                rate_limit_wait_s=float(t.get("rate_limit_wait_s", 0.0)),
            )
        )

    return BenchmarkReport(
        suite_name=str(data.get("suite_name", "Unknown")),
        total_tasks=int(data.get("total_tasks", len(task_reports))),
        passed_tasks=int(data.get("passed_tasks", 0)),
        pass_rate=float(data.get("pass_rate", 0.0)),
        mean_score=float(data.get("mean_score", 0.0)),
        total_duration_s=float(data.get("total_duration_s", 0.0)),
        tasks=task_reports,
        total_prompt_tokens=int(data.get("total_prompt_tokens", 0)),
        total_completion_tokens=int(data.get("total_completion_tokens", 0)),
        total_cached_tokens=int(data.get("total_cached_tokens", 0)),
        cache_hit_rate_pct=float(data.get("cache_hit_rate_pct", 0.0)),
        total_rate_limit_wait_s=float(data.get("total_rate_limit_wait_s", 0.0)),
        config=dict(data.get("config") or {}),
    )


_DIR_TIMESTAMP_RE = re.compile(r"_(\d{8})_(\d{6})(?:_|$)")


def _infer_timestamp_from_trace(trace_file: Path, tr: dict[str, Any]) -> str:
    """Extracts ISO-8601 UTC timestamp from trace, folder timestamp, or file mtime."""
    started = tr.get("started_at")
    if isinstance(started, str) and started.strip():
        return started.strip()

    match = _DIR_TIMESTAMP_RE.search(trace_file.parent.name)
    if match:
        ymd, hms = match.groups()
        try:
            dt = datetime.strptime(f"{ymd}{hms}", "%Y%m%d%H%M%S").astimezone()
            return dt.astimezone(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")
        except ValueError:
            pass

    mtime = datetime.fromtimestamp(trace_file.stat().st_mtime, tz=timezone.utc)
    return mtime.isoformat(timespec="seconds").replace("+00:00", "Z")


def _infer_duration_from_trace(tr: dict[str, Any]) -> float:
    """Computes run duration from timestamps or summed step durations."""
    started = tr.get("started_at")
    finished = tr.get("finished_at")
    if isinstance(started, str) and isinstance(finished, str):
        try:
            t0 = datetime.fromisoformat(started.replace("Z", "+00:00"))
            t1 = datetime.fromisoformat(finished.replace("Z", "+00:00"))
            delta = (t1 - t0).total_seconds()
            if delta >= 0:
                return delta
        except ValueError:
            pass
    return sum(
        float(s.get("duration_s", 0.0) or 0.0)
        for s in (tr.get("steps") or [])
        if isinstance(s, dict)
    )


def _report_from_standalone_trace(trace_file: Path, tr: dict[str, Any]) -> BenchmarkReport | None:
    """Builds a single-task ``BenchmarkReport`` from an older standalone ``trace.json``."""
    from benchmarks.runner import BenchmarkReport, TaskReport

    task_obj = tr.get("task")
    if not isinstance(task_obj, dict):
        return None
    task_id = str(task_obj.get("id") or "").strip()
    if not task_id:
        return None

    eval_obj = tr.get("eval_result") if isinstance(tr.get("eval_result"), dict) else {}
    assert isinstance(eval_obj, dict)
    passed = bool(eval_obj.get("passed", False))
    score = float(eval_obj.get("score", 1.0 if passed else 0.0) or 0.0)
    feedback = str(eval_obj.get("feedback") or "")

    if task_id.startswith("mini_swe_"):
        suite_name = "Mini-SWE"
    elif "Harbor/DeepSWE" in feedback or "-" in task_id:
        suite_name = "DeepSWE-mini"
    else:
        suite_name = "Standalone"

    telemetry = tr.get("telemetry") if isinstance(tr.get("telemetry"), dict) else {}
    assert isinstance(telemetry, dict)
    prompt_tokens = int(telemetry.get("total_prompt_tokens", 0) or 0)
    completion_tokens = int(telemetry.get("total_completion_tokens", 0) or 0)
    cached_tokens = int(telemetry.get("total_cached_tokens", 0) or 0)
    hit_rate = (
        float(telemetry["cache_hit_rate_pct"])
        if telemetry.get("cache_hit_rate_pct") is not None
        else (round(cached_tokens / prompt_tokens * 100.0, 1) if prompt_tokens > 0 else 0.0)
    )
    rl_wait = float(telemetry.get("total_rate_limit_wait_s", 0.0) or 0.0)

    duration_s = _infer_duration_from_trace(tr)
    steps = tr.get("total_steps")
    total_steps = int(steps) if steps is not None else len(tr.get("steps") or [])
    patch_file = trace_file.parent / "patch.diff"
    ws_dir = trace_file.parent / "workspace"

    cfg: dict[str, Any] = {
        "timestamp_utc": _infer_timestamp_from_trace(trace_file, tr),
        "harness_version": "",
        "git_commit": "",
    }
    if telemetry.get("cache_ttl"):
        cfg["cache_ttl"] = telemetry["cache_ttl"]
    if task_obj.get("max_steps") is not None:
        cfg["max_steps"] = task_obj["max_steps"]
    if task_obj.get("docker_image"):
        cfg["docker"] = True

    task_rep = TaskReport(
        task_id=task_id,
        passed=passed,
        score=score,
        total_steps=total_steps,
        duration_s=duration_s,
        finish_reason=str(tr.get("finish_reason") or ""),
        trace_path=str(trace_file),
        workspace_path=str(ws_dir) if ws_dir.exists() else "",
        patch_path=str(patch_file) if patch_file.is_file() else None,
        feedback=feedback,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cached_tokens=cached_tokens,
        cache_hit_rate_pct=hit_rate,
        rate_limit_wait_s=rl_wait,
    )
    return BenchmarkReport(
        suite_name=suite_name,
        total_tasks=1,
        passed_tasks=1 if passed else 0,
        pass_rate=100.0 if passed else 0.0,
        mean_score=score,
        total_duration_s=duration_s,
        tasks=[task_rep],
        total_prompt_tokens=prompt_tokens,
        total_completion_tokens=completion_tokens,
        total_cached_tokens=cached_tokens,
        cache_hit_rate_pct=hit_rate,
        total_rate_limit_wait_s=rl_wait,
        config=cfg,
    )


def _discover_standalone_traces(runs_dir: Path, reported_dirs: set[Path]) -> list[Path]:
    """Finds standalone benchmark ``trace.json`` directories lacking ``*_report.json``."""
    if not runs_dir.is_dir():
        return []
    traces: list[Path] = []
    for child in runs_dir.iterdir():
        if not child.is_dir():
            continue
        if child.resolve() in reported_dirs:
            continue
        if child.name.startswith(_NON_BENCHMARK_DIR_PREFIXES):
            continue
        trace_file = child / "trace.json"
        if trace_file.is_file():
            traces.append(trace_file)
    traces.sort(key=lambda p: p.stat().st_mtime)
    return traces


def backfill_runs(
    runs_dir: Path | str,
    tracker: CsvExperimentTracker,
    *,
    include_standalone_traces: bool = True,
) -> int:
    """Imports historical ``*_report.json`` runs and standalone traces under ``runs_dir``.

    Preserves missing historical version/commit fields as blank rather than stamping
    them with the current runtime version.
    """
    resolved_runs = Path(runs_dir).resolve()
    report_files = _discover_report_files(resolved_runs)
    reported_dirs: set[Path] = set()
    prev_auto = tracker.auto_collect_agent_metadata
    tracker.auto_collect_agent_metadata = False
    imported = 0
    try:
        for report_file in report_files:
            try:
                data = json.loads(report_file.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError):
                continue
            if not isinstance(data, dict) or "suite_name" not in data:
                continue
            reported_dirs.add(report_file.parent.resolve())
            report = _report_from_dict(data)
            if "timestamp_utc" not in report.config:
                report.config["timestamp_utc"] = _extract_timestamp_from_report(report_file, data)
            tracker.record_run(report, output_dir=report_file.parent)
            imported += 1

        if include_standalone_traces:
            for trace_file in _discover_standalone_traces(resolved_runs, reported_dirs):
                try:
                    tr = json.loads(trace_file.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    continue
                if not isinstance(tr, dict):
                    continue
                report = _report_from_standalone_trace(trace_file, tr)
                if report is None:
                    continue
                tracker.record_run(report, output_dir=trace_file.parent)
                imported += 1
    finally:
        tracker.auto_collect_agent_metadata = prev_auto
    return imported


def load_experiment_runs(
    csv_path: Path | str,
    *,
    limit: int | None = None,
) -> list[dict[str, str]]:
    """Loads rows from ``experiments.csv`` sorted newest-first by ``timestamp_utc``."""
    _, rows = _read_csv_rows(Path(csv_path))
    rows.sort(key=lambda r: r.get("timestamp_utc", ""), reverse=True)
    if limit is not None and limit >= 0:
        return rows[:limit]
    return rows
