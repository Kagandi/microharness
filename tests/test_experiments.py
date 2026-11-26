import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import MagicMock

import main
from benchmarks.experiments import (
    DEFAULT_RUNS_CSV,
    DEFAULT_TASKS_CSV,
    RUN_COLUMNS,
    TASK_COLUMNS,
    CsvExperimentTracker,
    ExperimentTracker,
    backfill_runs,
    collect_agent_metadata,
    collect_subagents_summary,
    derive_run_id,
    load_experiment_runs,
)
from benchmarks.mini_swe import create_rate_limiter_task
from benchmarks.runner import BenchmarkReport, TaskReport, run_benchmark_suite
from core import __version__
from core.defaults import load_defaults
from core.types import RunResult


class TestExperimentTracking(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path(tempfile.mkdtemp())

    def test_defaults_and_protocol(self):
        defaults = load_defaults()
        self.assertEqual(defaults["experiments_runs_csv"], DEFAULT_RUNS_CSV)
        self.assertEqual(defaults["experiments_tasks_csv"], DEFAULT_TASKS_CSV)
        tracker = CsvExperimentTracker(self.tmp_dir)
        self.assertIsInstance(tracker, ExperimentTracker)

    def test_collect_agent_metadata_outside_git_repo(self):
        non_git = self.tmp_dir / "not_a_repo"
        non_git.mkdir()
        meta = collect_agent_metadata(non_git)
        self.assertEqual(meta["harness_version"], __version__)
        self.assertEqual(meta["git_commit"], "")
        self.assertIsNone(meta["git_dirty"])

    def test_collect_subagents_summary_includes_project_overrides(self):
        global_dir = self.tmp_dir / "global"
        project_dir = self.tmp_dir / "project"
        (project_dir / ".microharness" / "agents").mkdir(parents=True)
        (project_dir / ".microharness" / "agents" / "reviewer.md").write_text(
            "---\n"
            "name: reviewer\n"
            "description: Custom reviewer\n"
            "model: deepseek-flash\n"
            "effort: high\n"
            "max_steps: 60\n"
            "tools: read_file, grep\n"
            "read_only: true\n"
            "---\nReview strictly.\n",
            encoding="utf-8",
        )
        summary = collect_subagents_summary(global_dir=global_dir, target_dir=project_dir)
        self.assertEqual(
            summary["reviewer"],
            {
                "scope": "project",
                "read_only": True,
                "model": "deepseek-flash",
                "effort": "high",
                "max_steps": 60,
                "tools": ["read_file", "grep"],
            },
        )
        self.assertEqual(summary["explore"]["scope"], "builtin")
        self.assertTrue(summary["explore"]["read_only"])

    def test_collect_subagents_summary_and_metadata_for_preset(self):
        global_dir = self.tmp_dir / "global"
        project_dir = self.tmp_dir / "project"
        preset_dir = global_dir / "presets" / "paper-researcher"
        (preset_dir / "subagents").mkdir(parents=True)
        (preset_dir / "agent.md").write_text(
            "---\nname: Paper Researcher\nisolated: true\n---\nResearch papers.\n",
            encoding="utf-8",
        )
        (preset_dir / "subagents" / "summarizer.md").write_text(
            "---\ndescription: Summarizes papers\ntools: [read_file]\nread_only: true\n---\nSummarize.\n",
            encoding="utf-8",
        )
        (project_dir / ".microharness" / "agents").mkdir(parents=True)
        (project_dir / ".microharness" / "agents" / "proj-helper.md").write_text(
            "---\ndescription: Project helper\n---\nHelp.\n",
            encoding="utf-8",
        )

        summary = collect_subagents_summary(
            global_dir=global_dir,
            target_dir=project_dir,
            preset="paper-researcher",
        )
        self.assertEqual(
            summary,
            {
                "summarizer": {
                    "scope": "preset",
                    "read_only": True,
                    "tools": ["read_file"],
                }
            },
        )

        from core.model import MockModel
        from core.presets import resolve_preset

        preset = resolve_preset("paper-researcher", workspace=project_dir, global_dir=global_dir)
        ctx = main.CliContext(
            model=MockModel(script=[]),
            global_dir=global_dir,
            runs_dir=self.tmp_dir / "runs",
            target_dir=project_dir,
            preset=preset,
        )
        meta = main._benchmark_metadata(ctx)
        self.assertEqual(meta["preset"], "paper-researcher")
        self.assertEqual(meta["subagents"], summary)

    def test_record_run_writes_and_upserts_csv_without_duplicates(self):
        tracker = CsvExperimentTracker(self.tmp_dir / "global")
        run_dir = self.tmp_dir / "runs" / "exp-alpha"

        report_step1 = BenchmarkReport(
            suite_name="DeepSWE-mini",
            total_tasks=2,
            passed_tasks=1,
            pass_rate=50.0,
            mean_score=0.5,
            total_duration_s=12.5,
            total_prompt_tokens=1000,
            total_completion_tokens=100,
            total_cached_tokens=800,
            cache_hit_rate_pct=80.0,
            total_rate_limit_wait_s=1.5,
            config={
                "timestamp_utc": "2026-11-26T10:00:00Z",
                "harness_version": "0.1.22",
                "git_commit": "abc1234",
                "git_dirty": False,
                "provider": "vercel",
                "model": "tencent/hy4-flash",
                "reasoning_effort": "high",
                "max_steps": 500,
                "max_context_tokens": 262144,
                "trigger_ratio": 0.8,
                "bash_timeout": 300.0,
                "docker": True,
                "task_filter": "koota,bandit",
                "preset": "developer-1",
                "subagents": {"reviewer": {"model": "deepseek-flash"}},
            },
            tasks=[
                TaskReport(
                    task_id="koota",
                    passed=True,
                    score=1.0,
                    total_steps=40,
                    duration_s=12.5,
                    finish_reason="completed",
                    trace_path="/tmp/koota/trace.json",
                    workspace_path="/tmp/koota/ws",
                    patch_path="/tmp/koota/patch.diff",
                    prompt_tokens=1000,
                    completion_tokens=100,
                    cached_tokens=800,
                    cache_hit_rate_pct=80.0,
                    rate_limit_wait_s=1.5,
                )
            ],
        )
        runs_csv, tasks_csv = tracker.record_run(report_step1, output_dir=run_dir)

        # Resume / second task completes in the same run_dir
        report_step2 = BenchmarkReport(
            suite_name="DeepSWE-mini",
            total_tasks=2,
            passed_tasks=2,
            pass_rate=100.0,
            mean_score=1.0,
            total_duration_s=30.0,
            total_prompt_tokens=2500,
            total_completion_tokens=250,
            total_cached_tokens=2000,
            cache_hit_rate_pct=80.0,
            total_rate_limit_wait_s=1.5,
            config=dict(report_step1.config),
            tasks=[
                report_step1.tasks[0],
                TaskReport(
                    task_id="bandit",
                    passed=True,
                    score=1.0,
                    total_steps=50,
                    duration_s=17.5,
                    finish_reason="completed",
                    trace_path="/tmp/bandit/trace.json",
                    workspace_path="/tmp/bandit/ws",
                    patch_path=None,
                    prompt_tokens=1500,
                    completion_tokens=150,
                    cached_tokens=1200,
                    cache_hit_rate_pct=80.0,
                    rate_limit_wait_s=0.0,
                ),
            ],
        )
        tracker.record_run(report_step2, output_dir=run_dir)

        with runs_csv.open("r", encoding="utf-8", newline="") as handle:
            run_rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(run_rows[0].keys()), RUN_COLUMNS)
        self.assertEqual(len(run_rows), 1)
        expected_run_id, expected_run_name = derive_run_id("DeepSWE-mini", output_dir=run_dir)
        self.assertEqual(run_rows[0]["run_id"], expected_run_id)
        self.assertEqual(run_rows[0]["run_name"], expected_run_name)
        self.assertEqual(run_rows[0]["timestamp_utc"], "2026-11-26T10:00:00Z")
        self.assertEqual(run_rows[0]["harness_version"], "0.1.22")
        self.assertEqual(run_rows[0]["git_commit"], "abc1234")
        self.assertEqual(run_rows[0]["git_dirty"], "false")
        self.assertEqual(run_rows[0]["docker"], "true")
        self.assertEqual(run_rows[0]["passed_tasks"], "2")
        self.assertEqual(run_rows[0]["total_steps"], "90")
        self.assertEqual(run_rows[0]["mean_score"], "1.0000")
        self.assertEqual(run_rows[0]["preset"], "developer-1")
        self.assertEqual(run_rows[0]["subagents"], '{"reviewer":{"model":"deepseek-flash"}}')

        with tasks_csv.open("r", encoding="utf-8", newline="") as handle:
            task_rows = list(csv.DictReader(handle))
            self.assertEqual(tuple(task_rows[0].keys()), TASK_COLUMNS)
        self.assertEqual(len(task_rows), 2)
        self.assertEqual([r["task_id"] for r in task_rows], ["koota", "bandit"])
        self.assertEqual(task_rows[1]["patch_path"], "")

    def test_distinct_directories_with_same_leaf_name_do_not_collide(self):
        tracker = CsvExperimentTracker(self.tmp_dir / "global")
        dir_a = self.tmp_dir / "a" / "runs"
        dir_b = self.tmp_dir / "b" / "runs"

        report = BenchmarkReport(
            suite_name="Mini-SWE",
            total_tasks=1,
            passed_tasks=1,
            pass_rate=100.0,
            mean_score=1.0,
            total_duration_s=1.0,
            tasks=[],
            config={"model": "m1"},
        )
        tracker.record_run(report, output_dir=dir_a)
        tracker.record_run(report, output_dir=dir_b)

        rows = load_experiment_runs(tracker.runs_csv_path)
        self.assertEqual(len(rows), 2)
        self.assertNotEqual(rows[0]["run_id"], rows[1]["run_id"])

    def test_run_benchmark_suite_records_to_tracker(self):
        task1, eval1 = create_rate_limiter_task()
        mock_harness = MagicMock()
        mock_harness.run.return_value = RunResult(
            task_id=task1.id,
            passed=True,
            score=1.0,
            total_steps=4,
            duration_s=3.2,
            finish_reason="completed",
            trace_path="trace.json",
            workspace_path="ws",
            prompt_tokens=900,
            completion_tokens=80,
            cached_tokens=600,
            cache_hit_rate_pct=66.7,
        )
        tracker = CsvExperimentTracker(self.tmp_dir / "global")
        run_out = self.tmp_dir / "runs" / "smoke"

        run_benchmark_suite(
            harness_factory=lambda _ev: mock_harness,
            tasks=[(task1, eval1)],
            suite_name="Mini-SWE",
            output_dir=run_out,
            verbose=False,
            config={"model": "test-model", "provider": "mock", "harness_version": "0.1.22"},
            tracker=tracker,
        )

        rows = load_experiment_runs(tracker.runs_csv_path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["run_name"], "smoke")
        self.assertEqual(rows[0]["passed_tasks"], "1")
        self.assertEqual(rows[0]["total_steps"], "4")
        self.assertEqual(rows[0]["model"], "test-model")

    def test_backfill_runs_imports_valid_reports_and_skips_corrupt_files(self):
        runs_dir = self.tmp_dir / "runs"
        exp1 = runs_dir / "legacy-exp"
        trace_dir = exp1 / "task_a_20261124"
        trace_dir.mkdir(parents=True)
        trace_file = trace_dir / "trace.json"
        trace_file.write_text(
            json.dumps({"started_at": "2026-11-24T12:00:00Z"}),
            encoding="utf-8",
        )
        (exp1 / "deepswe_mini_report.json").write_text(
            json.dumps(
                {
                    "suite_name": "DeepSWE-mini",
                    "total_tasks": 1,
                    "passed_tasks": 1,
                    "pass_rate": 100.0,
                    "mean_score": 1.0,
                    "total_duration_s": 50.0,
                    "total_prompt_tokens": 2000,
                    "total_completion_tokens": 150,
                    "total_cached_tokens": 1800,
                    "cache_hit_rate_pct": 90.0,
                    "total_rate_limit_wait_s": 0.0,
                    "config": {"model": "tencent/hy4-flash", "provider": "vercel"},
                    "tasks": [
                        {
                            "task_id": "task_a",
                            "passed": True,
                            "score": 1.0,
                            "total_steps": 12,
                            "duration_s": 50.0,
                            "finish_reason": "completed",
                            "trace_path": str(trace_file),
                            "workspace_path": str(trace_dir / "workspace"),
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        # Standalone Sept-style benchmark trace (no *_report.json) must also be backfilled
        old_run = runs_dir / "bandit-structured-nosec-directives_20260926_235402_eab89a"
        old_run.mkdir(parents=True)
        (old_run / "trace.json").write_text(
            json.dumps(
                {
                    "task": {"id": "bandit-structured-nosec-directives", "max_steps": 15},
                    "finish_reason": "completed",
                    "total_steps": 124,
                    "eval_result": {
                        "passed": True,
                        "score": 1.0,
                        "feedback": "Harbor/DeepSWE Verification PASSED ✅",
                    },
                    "telemetry": {
                        "total_prompt_tokens": 9300768,
                        "total_completion_tokens": 63840,
                        "total_cached_tokens": 9213952,
                        "cache_hit_rate_pct": 99.1,
                    },
                    "steps": [{"duration_s": 12.5}, {"duration_s": 7.5}],
                }
            ),
            encoding="utf-8",
        )

        # Interactive sessions must be ignored by backfill
        interactive_dir = runs_dir / "interactive_session_20260926_120000_abcd12"
        interactive_dir.mkdir(parents=True)
        (interactive_dir / "trace.json").write_text(
            json.dumps({"task": {"id": "interactive_session"}, "steps": []}),
            encoding="utf-8",
        )

        # Negative cases: corrupt JSON and unrelated JSON must be ignored gracefully.
        bad_dir = runs_dir / "corrupt-exp"
        bad_dir.mkdir(parents=True)
        (bad_dir / "deepswe_mini_report.json").write_text("{not json", encoding="utf-8")
        (bad_dir / "other_report.json").write_text(
            json.dumps({"not": "a report"}), encoding="utf-8"
        )

        tracker = CsvExperimentTracker(self.tmp_dir / "global")
        imported = backfill_runs(runs_dir, tracker)
        self.assertEqual(imported, 2)

        rows = load_experiment_runs(tracker.runs_csv_path)
        self.assertEqual(len(rows), 2)
        by_name = {r["run_name"]: r for r in rows}
        self.assertEqual(by_name["legacy-exp"]["timestamp_utc"], "2026-11-24T12:00:00Z")
        # Historical runs without saved version metadata must stay blank rather than fake current version.
        self.assertEqual(by_name["legacy-exp"]["harness_version"], "")
        self.assertEqual(by_name["legacy-exp"]["git_commit"], "")
        old_row = by_name["bandit-structured-nosec-directives_20260926_235402_eab89a"]
        self.assertEqual(old_row["suite_name"], "DeepSWE-mini")
        self.assertEqual(old_row["total_steps"], "124")
        self.assertEqual(old_row["duration_s"], "20.00")
        self.assertEqual(old_row["report_path"], "")

    def test_experiments_cli_backfill_and_list(self):
        global_dir = self.tmp_dir / "global"
        runs_dir = self.tmp_dir / "runs"
        exp_dir = runs_dir / "cli-exp"
        exp_dir.mkdir(parents=True)
        (exp_dir / "mini_swe_report.json").write_text(
            json.dumps(
                {
                    "suite_name": "Mini-SWE",
                    "total_tasks": 1,
                    "passed_tasks": 0,
                    "pass_rate": 0.0,
                    "mean_score": 0.5,
                    "total_duration_s": 9.0,
                    "config": {
                        "timestamp_utc": "2026-11-26T15:00:00Z",
                        "harness_version": "0.1.22",
                        "model": "deepseek-flash",
                    },
                    "tasks": [
                        {
                            "task_id": "t1",
                            "passed": False,
                            "score": 0.5,
                            "total_steps": 7,
                            "duration_s": 9.0,
                            "finish_reason": "completed",
                            "trace_path": "",
                            "workspace_path": "",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        buf = io.StringIO()
        with redirect_stdout(buf):
            main.main(
                [
                    "experiments",
                    "backfill",
                    "--global-dir",
                    str(global_dir),
                    "--runs-dir",
                    str(runs_dir),
                ]
            )
            main.main(["experiments", "list", "--global-dir", str(global_dir), "--limit", "5"])

        out = buf.getvalue()
        self.assertIn("Imported 1 benchmark run(s)", out)
        self.assertIn("cli-exp", out)
        self.assertIn("deepseek-flash", out)
        self.assertIn("0/1", out)


if __name__ == "__main__":
    unittest.main()
