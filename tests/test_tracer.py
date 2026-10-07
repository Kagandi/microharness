from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import patch

from core.harness import Task
from core.session import (
    find_recent_sessions,
    load_session_messages,
    reconstruct_messages_from_steps,
)
from core.tracer import (
    Tracer,
    _pair_calls_with_results,
)


class TestTracer(unittest.TestCase):
    def test_pair_calls_with_results_shuffled_order(self):
        """Invariant: Tool calls with IDs are paired with matching results regardless of order.

        Mutation G detection: If ID-based matching is disabled and positional matching is used,
        this test fails because call_1 gets result_2 instead of result_1.
        """
        tool_calls: list[dict[str, Any]] = [
            {"id": "call_1", "name": "read_file", "arguments": {"path": "a.txt"}},
            {"id": "call_2", "name": "bash", "arguments": {"command": "ls"}},
        ]
        # Shuffled results (call_2 comes first, call_1 second)
        tool_results: list[dict[str, Any]] = [
            {"call_id": "call_2", "output": "file_b.txt", "exit_code": 0},
            {"call_id": "call_1", "output": "file_a content", "exit_code": 0},
        ]

        pairs = _pair_calls_with_results(tool_calls, tool_results)

        self.assertEqual(len(pairs), 2)
        # First pair must strictly be call_1 with result_1
        self.assertEqual(pairs[0][0]["id"], "call_1")
        self.assertEqual(pairs[0][1]["call_id"], "call_1")
        self.assertEqual(pairs[0][1]["output"], "file_a content")

        # Second pair must strictly be call_2 with result_2
        self.assertEqual(pairs[1][0]["id"], "call_2")
        self.assertEqual(pairs[1][1]["call_id"], "call_2")
        self.assertEqual(pairs[1][1]["output"], "file_b.txt")

    def test_pair_calls_with_results_positional_fallback_no_ids(self):
        """Invariant: When calls lack IDs, results are matched in order without orphaning results."""
        tool_calls: list[dict[str, Any]] = [
            {"name": "tool_x", "arguments": {}},
            {"name": "tool_y", "arguments": {}},
        ]
        tool_results: list[dict[str, Any]] = [
            {"call_id": "x1", "output": "out_x"},
            {"call_id": "y1", "output": "out_y"},
        ]

        pairs = _pair_calls_with_results(tool_calls, tool_results)

        self.assertEqual(len(pairs), 2)
        self.assertEqual(pairs[0][0]["name"], "tool_x")
        self.assertEqual(pairs[0][1]["output"], "out_x")

        self.assertEqual(pairs[1][0]["name"], "tool_y")
        self.assertEqual(pairs[1][1]["output"], "out_y")

    def test_pair_calls_with_results_mixed_id_and_fallback(self):
        """Invariant: Calls with IDs match first, and remaining calls take remaining results."""
        tool_calls: list[dict[str, Any]] = [
            {"name": "positional_call"},
            {"id": "id_call", "name": "id_matched_call"},
        ]
        tool_results: list[dict[str, Any]] = [
            {"call_id": "id_call", "output": "id_output"},
            {"output": "positional_output"},
        ]

        pairs = _pair_calls_with_results(tool_calls, tool_results)

        self.assertEqual(len(pairs), 2)
        # Positional call takes the remaining un-id'd result
        self.assertEqual(pairs[0][0]["name"], "positional_call")
        self.assertEqual(pairs[0][1]["output"], "positional_output")

        # ID call takes the ID matched result even though it was first in tool_results
        self.assertEqual(pairs[1][0]["id"], "id_call")
        self.assertEqual(pairs[1][1]["output"], "id_output")

    def test_pair_calls_with_results_extra_results_and_calls(self):
        """Invariant: Unmatched calls get None, and leftover results become (None, result)."""
        # Case A: More calls than results
        calls_a: list[dict[str, Any]] = [
            {"id": "c1", "name": "call_1"},
            {"id": "c2", "name": "call_2"},
        ]
        results_a: list[dict[str, Any]] = [
            {"call_id": "c1", "output": "res_1"},
        ]
        pairs_a = _pair_calls_with_results(calls_a, results_a)
        self.assertEqual(len(pairs_a), 2)
        self.assertEqual(pairs_a[0], (calls_a[0], results_a[0]))
        self.assertEqual(pairs_a[1], (calls_a[1], None))

        # Case B: More results than calls (orphaned results)
        calls_b: list[dict[str, Any]] = [
            {"id": "c1", "name": "call_1"},
        ]
        results_b: list[dict[str, Any]] = [
            {"call_id": "c1", "output": "res_1"},
            {"call_id": "c_orphan", "output": "res_orphan"},
        ]
        pairs_b = _pair_calls_with_results(calls_b, results_b)
        self.assertEqual(len(pairs_b), 2)
        self.assertEqual(pairs_b[0], (calls_b[0], results_b[0]))
        self.assertEqual(pairs_b[1], (None, results_b[1]))

    def test_tracer_export_markdown_and_json(self):
        temp_dir = Path(tempfile.mkdtemp(prefix="test_tracer_"))
        task = Task(id="test_task", instruction="Test instruction")
        tracer = Tracer(task=task, output_dir=temp_dir, verbose=False)

        from core.tracer import StepTrace

        step = StepTrace(
            step_index=1,
            prompt_messages=[],
            model_response={
                "content": "Running ls",
                "tool_calls": [
                    {"id": "call_1", "name": "bash", "arguments": {"command": "ls -la"}}
                ],
            },
            tool_results=[
                {
                    "call_id": "call_1",
                    "output": "total 0",
                    "exit_code": 0,
                    "was_truncated": False,
                }
            ],
            duration_s=0.5,
        )
        tracer.record_step(step)

        json_path, md_path = tracer.finalize(finish_reason="completed")

        self.assertTrue(json_path.exists())
        self.assertTrue(md_path.exists())

        trace_data = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(trace_data["task"]["id"], "test_task")
        self.assertEqual(len(trace_data["steps"]), 1)
        self.assertEqual(trace_data["finish_reason"], "completed")

        md_content = md_path.read_text(encoding="utf-8")
        self.assertIn("Running ls", md_content)
        self.assertIn("ls -la", md_content)
        self.assertIn("total 0", md_content)

    def test_tracer_token_telemetry_in_logs_and_exports(self):
        """Invariant: Tracer prints token telemetry in step header and exports metrics to JSON and Markdown."""
        import io
        import sys

        from core.types import StepTrace

        temp_dir = Path(tempfile.mkdtemp(prefix="test_tracer_telemetry_"))
        task = Task(id="telemetry_task", instruction="Test telemetry")
        tracer = Tracer(task=task, output_dir=temp_dir, verbose=True)

        step = StepTrace(
            step_index=1,
            prompt_messages=[],
            model_response={"content": "Thinking deeply...", "tool_calls": []},
            tool_results=[],
            duration_s=1.25,
            usage={
                "prompt_tokens": 1200,
                "completion_tokens": 80,
                "cached_tokens": 600,
                "cache_hit_rate_pct": 50.0,
            },
        )

        captured_stdout = io.StringIO()
        orig_stdout = sys.stdout
        try:
            sys.stdout = captured_stdout
            tracer.record_step(step)
        finally:
            sys.stdout = orig_stdout

        output_log = captured_stdout.getvalue()
        # Verify step divider contains input, cached tokens, hit rate, and output tokens
        self.assertIn("Step 1", output_log)
        self.assertIn("1,200 in (600 cached - 50.0%)", output_log)
        self.assertIn("80 out", output_log)

        json_path, md_path = tracer.finalize(finish_reason="completed")
        trace_json = json.loads(json_path.read_text(encoding="utf-8"))

        self.assertIn("telemetry", trace_json)
        self.assertEqual(trace_json["telemetry"]["total_prompt_tokens"], 1200)
        self.assertEqual(trace_json["telemetry"]["total_completion_tokens"], 80)
        self.assertEqual(trace_json["telemetry"]["total_cached_tokens"], 600)
        self.assertEqual(trace_json["telemetry"]["cache_hit_rate_pct"], 50.0)
        self.assertEqual(trace_json["steps"][0]["usage"]["cached_tokens"], 600)

        md_text = md_path.read_text(encoding="utf-8")
        self.assertIn("- **Tokens**: 1,200 in (600 cached - 50.0% hit rate), 80 out", md_text)
        self.assertIn("### Step 1 (1.25s · 1,200 in [600 cached - 50.0%], 80 out)", md_text)

    def test_tracer_compact_mode_intermediate_and_final_steps(self):
        """Invariant: In compact mode, intermediate steps render as 1-liners and final step prints full response."""
        import io
        import sys

        from core.types import StepTrace

        temp_dir = Path(tempfile.mkdtemp(prefix="test_tracer_compact_"))
        task = Task(id="compact_task", instruction="Test compact mode")
        tracer = Tracer(task=task, output_dir=temp_dir, verbose=True, compact=True)

        # Step 1: Intermediate step with tool call and output
        step1 = StepTrace(
            step_index=1,
            prompt_messages=[],
            model_response={
                "content": "I will inspect the workspace files now.\nExtra details not needed.",
                "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "git status"}}],
            },
            tool_results=[{"call_id": "c1", "output": "On branch main\nnothing to commit", "exit_code": 0}],
            duration_s=0.3,
            usage={"prompt_tokens": 500, "completion_tokens": 40},
        )

        # Step 2: Final response with no tool calls
        step2 = StepTrace(
            step_index=2,
            prompt_messages=[],
            model_response={
                "content": "All changes are committed and the workspace is clean.",
                "tool_calls": [],
            },
            tool_results=[],
            duration_s=0.8,
            usage={"prompt_tokens": 800, "completion_tokens": 60},
        )

        captured_stdout = io.StringIO()
        orig_stdout = sys.stdout
        try:
            sys.stdout = captured_stdout
            tracer.record_step(step1)
            tracer.record_step(step2)
        finally:
            sys.stdout = orig_stdout

        output = captured_stdout.getvalue()

        # Invariant 1: Step 1 collapsed into 1-line thinking and 1-line action
        self.assertIn("● Thinking: I will inspect the workspace files now.", output)
        self.assertIn("$ git status", output)
        self.assertIn("(0.3s)", output)

        # Invariant 2: Step 1 does NOT dump multi-line divider or raw tool stdout
        self.assertNotIn("─── Step 1", output)
        self.assertNotIn("nothing to commit", output)

        # Invariant 3: Step 2 prints full final message
        self.assertIn("All changes are committed and the workspace is clean.", output)

    def test_tracer_scoped_instance_formatters(self):
        """Invariant: Tracer instance formatters do not leak between instances or mutate global defaults."""
        import io
        import sys

        from core.tracer import _ACTION_FORMATTERS, Tracer
        from core.types import StepTrace

        initial_globals = dict(_ACTION_FORMATTERS)

        # Instance 1 with custom formatter for bash
        tracer1 = Tracer(
            task=Task(id="t1", instruction=""),
            output_dir=Path(tempfile.mkdtemp()),
            verbose=True,
            formatters={"bash": lambda args: f"CUSTOM_BASH: {args.get('command')}"},
        )

        # Instance 2 with custom formatter for custom_tool
        tracer2 = Tracer(
            task=Task(id="t2", instruction=""),
            output_dir=Path(tempfile.mkdtemp()),
            verbose=True,
            formatters={"custom_tool": lambda args: f"CUSTOM_TOOL: {args.get('foo')}"},
        )

        step = StepTrace(
            step_index=1,
            prompt_messages=[],
            model_response={
                "content": "",
                "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "ls -la"}}],
            },
            tool_results=[{"call_id": "c1", "output": "ok", "exit_code": 0}],
            duration_s=0.1,
        )

        buf1 = io.StringIO()
        old_stdout = sys.stdout
        try:
            sys.stdout = buf1
            tracer1.record_step(step)
        finally:
            sys.stdout = old_stdout

        buf2 = io.StringIO()
        try:
            sys.stdout = buf2
            tracer2.record_step(step)
        finally:
            sys.stdout = old_stdout

        out1 = buf1.getvalue()
        out2 = buf2.getvalue()

        # Invariant 1: tracer1 used its custom bash formatter
        self.assertIn("CUSTOM_BASH: ls -la", out1)

        # Invariant 2: tracer2 did NOT use tracer1's custom formatter; used default bash formatter
        self.assertNotIn("CUSTOM_BASH", out2)
        self.assertIn("$ ls -la", out2)

        # Invariant 3: Global _ACTION_FORMATTERS was untouched
        self.assertEqual(_ACTION_FORMATTERS, initial_globals)

    def test_trajectory_jsonl_streaming_per_step(self):
        """Invariant: record_step immediately appends valid JSON to trajectory.jsonl on each step."""
        from core.types import StepTrace

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            tracer = Tracer(
                task=Task(id="task_stream", instruction="Test streaming"),
                output_dir=out_dir,
                verbose=False,
                workspace=Path("/some/repo"),
            )
            jsonl_path = out_dir / "trajectory.jsonl"
            self.assertFalse(jsonl_path.exists())

            # Step 1
            step1 = StepTrace(
                step_index=1,
                prompt_messages=[{"role": "user", "content": "hello"}],
                model_response={"content": "response 1", "tool_calls": []},
                tool_results=[],
                duration_s=0.5,
            )
            tracer.record_step(step1)

            self.assertTrue(jsonl_path.is_file())
            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 1)
            row1 = json.loads(lines[0])
            self.assertEqual(row1["step_index"], 1)
            self.assertEqual(row1["model_response"]["content"], "response 1")

            # Step 2
            step2 = StepTrace(
                step_index=2,
                prompt_messages=[
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "response 1"},
                ],
                model_response={"content": "response 2", "tool_calls": []},
                tool_results=[],
                duration_s=0.3,
            )
            tracer.record_step(step2)

            lines = jsonl_path.read_text(encoding="utf-8").strip().splitlines()
            self.assertEqual(len(lines), 2)
            row2 = json.loads(lines[1])
            self.assertEqual(row2["step_index"], 2)
            self.assertEqual(row2["model_response"]["content"], "response 2")

            # Finalize creates trace.json with messages and workspace
            tracer.finalize(finish_reason="goal_reached")
            trace_json = out_dir / "trace.json"
            self.assertTrue(trace_json.is_file())
            data = json.loads(trace_json.read_text(encoding="utf-8"))
            self.assertEqual(data["workspace"], "/some/repo")
            self.assertEqual(data["finish_reason"], "goal_reached")
            self.assertEqual(len(data["messages"]), 3)
            self.assertEqual(data["messages"][0]["role"], "user")
            self.assertEqual(data["messages"][1]["role"], "assistant")
            self.assertEqual(data["messages"][1]["content"], "response 1")
            self.assertEqual(data["messages"][2]["role"], "assistant")
            self.assertEqual(data["messages"][2]["content"], "response 2")

    def test_reconstruct_messages_from_steps(self):
        """Invariant: reconstruct_messages_from_steps builds complete history including assistant calls and tool results."""
        from core.session import reconstruct_messages_from_steps

        # Empty steps -> empty messages
        self.assertEqual(reconstruct_messages_from_steps([]), [])

        steps = [
            {
                "step_index": 1,
                "prompt_messages": [
                    {"role": "system", "content": "sys prompt"},
                    {"role": "user", "content": "run ls"},
                ],
                "model_response": {
                    "content": "calling bash",
                    "tool_calls": [
                        {
                            "id": "c_1",
                            "name": "bash",
                            "arguments": {"command": "ls"},
                            "extra_content": "extra_note",
                        }
                    ],
                },
                "tool_results": [
                    {"call_id": "c_1", "output": "file1.txt\nfile2.txt", "exit_code": 0}
                ],
            }
        ]
        reconstructed = reconstruct_messages_from_steps(steps)
        self.assertEqual(len(reconstructed), 4)
        self.assertEqual(reconstructed[0]["role"], "system")
        self.assertEqual(reconstructed[0]["content"], "sys prompt")
        self.assertEqual(reconstructed[1]["role"], "user")
        self.assertEqual(reconstructed[1]["content"], "run ls")
        self.assertEqual(reconstructed[2]["role"], "assistant")
        self.assertEqual(reconstructed[2]["content"], "calling bash")
        self.assertEqual(len(reconstructed[2]["tool_calls"]), 1)
        self.assertEqual(reconstructed[2]["tool_calls"][0]["id"], "c_1")
        self.assertEqual(reconstructed[2]["tool_calls"][0]["function"]["name"], "bash")
        self.assertEqual(
            json.loads(reconstructed[2]["tool_calls"][0]["function"]["arguments"]),
            {"command": "ls"},
        )
        self.assertEqual(reconstructed[2]["tool_calls"][0]["extra_content"], "extra_note")
        self.assertEqual(reconstructed[3]["role"], "tool")
        self.assertEqual(reconstructed[3]["content"], "file1.txt\nfile2.txt")
        self.assertEqual(reconstructed[3]["tool_call_id"], "c_1")

    def test_load_session_messages_variants(self):
        """Invariant: load_session_messages recovers from trace.json, legacy steps-only trace, and crashed trajectory.jsonl."""
        from core.session import load_session_messages

        with tempfile.TemporaryDirectory() as tmp_dir:
            run_dir = Path(tmp_dir)

            # 1. Non-existent returns empty messages and default metadata
            msgs, meta = load_session_messages(run_dir / "nonexistent")
            self.assertEqual(msgs, [])
            self.assertEqual(meta["total_steps"], 0)
            self.assertEqual(meta["finish_reason"], "unknown")

            # 2. Modern trace.json with explicit messages
            dir_modern = run_dir / "sess_modern"
            dir_modern.mkdir()
            trace_modern = {
                "workspace": "/repo/foo",
                "total_steps": 3,
                "finish_reason": "goal_reached",
                "messages": [
                    {"role": "user", "content": "hello modern"},
                    {"role": "assistant", "content": "hi"},
                ],
            }
            (dir_modern / "trace.json").write_text(json.dumps(trace_modern), encoding="utf-8")
            msgs, meta = load_session_messages(dir_modern)
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0]["content"], "hello modern")
            self.assertEqual(meta["total_steps"], 3)
            self.assertEqual(meta["workspace"], "/repo/foo")
            self.assertEqual(meta["finish_reason"], "goal_reached")

            # 3. Legacy trace.json without messages but with steps
            dir_legacy = run_dir / "sess_legacy"
            dir_legacy.mkdir()
            trace_legacy = {
                "workspace": "/repo/legacy",
                "total_steps": 1,
                "finish_reason": "max_steps",
                "steps": [
                    {
                        "step_index": 1,
                        "prompt_messages": [{"role": "user", "content": "hello legacy"}],
                        "model_response": {"content": "legacy response", "tool_calls": []},
                        "tool_results": [],
                    }
                ],
            }
            (dir_legacy / "trace.json").write_text(json.dumps(trace_legacy), encoding="utf-8")
            msgs, meta = load_session_messages(dir_legacy)
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0]["content"], "hello legacy")
            self.assertEqual(msgs[1]["content"], "legacy response")
            self.assertEqual(meta["workspace"], "/repo/legacy")

            # 4. Crashed session: NO trace.json, only trajectory.jsonl
            dir_crashed = run_dir / "sess_crashed"
            dir_crashed.mkdir()
            step_line = json.dumps({
                "step_index": 1,
                "prompt_messages": [{"role": "user", "content": "crashed turn"}],
                "model_response": {"content": "in progress", "tool_calls": []},
                "tool_results": [],
            })
            (dir_crashed / "trajectory.jsonl").write_text(step_line + "\n", encoding="utf-8")
            msgs, meta = load_session_messages(dir_crashed)
            self.assertEqual(len(msgs), 2)
            self.assertEqual(msgs[0]["content"], "crashed turn")
            self.assertEqual(msgs[1]["content"], "in progress")
            self.assertEqual(meta["total_steps"], 1)
            self.assertEqual(meta["finish_reason"], "unknown")

    def test_find_recent_sessions_ordering_and_affinity(self):
        """Invariant: find_recent_sessions sorts by workspace affinity and recency with limit enforcement."""
        import os

        from core.session import find_recent_sessions

        with tempfile.TemporaryDirectory() as tmp_dir:
            runs_dir = Path(tmp_dir)
            ws_target = Path(tmp_dir) / "target_repo"
            ws_other = Path(tmp_dir) / "other_repo"

            # Create session 1: older, other repo
            s1 = runs_dir / "run_old_other"
            s1.mkdir()
            (s1 / "trace.json").write_text(
                json.dumps({
                    "workspace": str(ws_other.resolve()),
                    "total_steps": 2,
                    "task": {"id": "t1", "instruction": "Old other instruction"},
                }),
                encoding="utf-8",
            )
            os.utime(s1, (1000, 1000))

            # Create session 2: newer, other repo
            s2 = runs_dir / "run_new_other"
            s2.mkdir()
            (s2 / "trace.json").write_text(
                json.dumps({
                    "workspace": str(ws_other.resolve()),
                    "total_steps": 5,
                    "task": {"id": "t2", "instruction": "New other instruction"},
                }),
                encoding="utf-8",
            )
            os.utime(s2, (2000, 2000))

            # Create session 3: target workspace match
            s3 = runs_dir / "run_target"
            s3.mkdir()
            (s3 / "trace.json").write_text(
                json.dumps({
                    "workspace": str(ws_target.resolve()),
                    "total_steps": 1,
                    "messages": [{"role": "user", "content": "Fix bug in target"}],
                }),
                encoding="utf-8",
            )
            os.utime(s3, (1500, 1500))

            # When querying with workspace=ws_target:
            # s3 must be ranked FIRST due to workspace match, even though s2 has newer mtime (2000 > 1500)
            res = find_recent_sessions(runs_dir, workspace=ws_target, limit=10)
            self.assertEqual(len(res), 3)
            self.assertEqual(res[0]["run_id"], "run_target")
            self.assertTrue(res[0]["matches_workspace"])
            self.assertEqual(res[0]["preview"], "Fix bug in target")

            # Between non-matching, s2 is newer than s1
            self.assertEqual(res[1]["run_id"], "run_new_other")
            self.assertFalse(res[1]["matches_workspace"])
            self.assertEqual(res[2]["run_id"], "run_old_other")

            # Limit parameter
            res_limited = find_recent_sessions(runs_dir, workspace=ws_target, limit=2)
            self.assertEqual(len(res_limited), 2)
            self.assertEqual([r["run_id"] for r in res_limited], ["run_target", "run_new_other"])

    def test_tracer_serialization_resilience(self):
        """Invariant: Tracer serializes non-primitive types without dropping steps or index holes."""
        from core.types import StepTrace

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            task = Task(id="test_task", instruction="Test resilience")
            tracer = Tracer(task=task, output_dir=out_dir, verbose=False)

            # Step 1: Contains Path, set, and custom object
            class CustomObj:
                def __str__(self):
                    return "CustomObjRepr"

            step1 = StepTrace(
                step_index=1,
                duration_s=0.5,
                prompt_messages=[{"role": "user", "content": "hello"}],
                model_response={"content": "response with object", "obj": CustomObj()},
                tool_results=[{"call_id": "c1", "output": Path("/tmp/some_file"), "tags": {"a", "b"}}],
            )
            tracer.record_step(step1)

            # Step 2: Normal step
            step2 = StepTrace(
                step_index=2,
                duration_s=0.3,
                prompt_messages=[{"role": "user", "content": "next"}],
                model_response={"content": "done"},
                tool_results=[],
            )
            tracer.record_step(step2)

            # Step 3: Malformed non-string dict key (e.g. tuple key) which causes json.dumps to raise TypeError
            step3 = StepTrace(
                step_index=3,
                duration_s=0.2,
                prompt_messages=[{"role": "user", "content": "error step"}],
                model_response={(1, 2): "tuple key raises TypeError"},  # type: ignore
                tool_results=[],
            )
            tracer.record_step(step3)

            # Verify trajectory.jsonl has exactly 3 lines with consecutive indices [1, 2, 3] (no index hole!)
            traj_lines = [
                json.loads(line)
                for line in tracer.jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(traj_lines), 3)
            self.assertEqual([line["step_index"] for line in traj_lines], [1, 2, 3])
            self.assertTrue(traj_lines[2].get("serialization_error"))

            # Steps 1 and 2 are serializable via default=str, so they must be written
            # in full rather than degraded to the serialization_error placeholder.
            self.assertIsNone(traj_lines[0].get("serialization_error"))
            self.assertIsNone(traj_lines[1].get("serialization_error"))
            self.assertEqual(traj_lines[0]["model_response"]["obj"], "CustomObjRepr")
            self.assertEqual(traj_lines[0]["tool_results"][0]["output"], "/tmp/some_file")
            self.assertIn("a", traj_lines[0]["tool_results"][0]["tags"])
            self.assertEqual(traj_lines[1]["prompt_messages"], [{"role": "user", "content": "next"}])

    def test_tracer_export_json_survives_non_primitive_values(self):
        """Invariant: a Decimal in usage must not crash finalization (default=str on export)."""
        from decimal import Decimal

        from core.types import StepTrace, Task

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            tracer = Tracer(task=Task(id="t", instruction="i"), output_dir=out_dir, verbose=False)
            tracer.record_step(
                StepTrace(
                    step_index=0,
                    duration_s=0.1,
                    compaction_applied=False,
                    usage={"input": Decimal("10.5"), "output": Decimal("2.25")},
                    prompt_messages=[{"role": "user", "content": "hi"}],
                    model_response={"content": "ok", "tool_calls": []},
                    tool_results=[],
                )
            )
            tracer.finish_reason = "completed"
            json_path, _ = tracer.finalize(finish_reason="completed")

            data = json.loads(json_path.read_text(encoding="utf-8"))
            self.assertEqual(data["steps"][0]["usage"], {"input": "10.5", "output": "2.25"})

    def test_resume_survives_trailing_serialization_placeholder(self):
        """Invariant: a placeholder final step must not discard the recoverable conversation."""
        from core.types import StepTrace

        with tempfile.TemporaryDirectory() as tmp_dir:
            out_dir = Path(tmp_dir)
            tracer = Tracer(task=Task(id="t", instruction="i"), output_dir=out_dir, verbose=False)

            history = [{"role": "system", "content": "SYS"}]
            for j in range(2):
                history += [
                    {"role": "user", "content": f"turn{j}"},
                    {"role": "assistant", "content": f"reply{j}"},
                ]
            tracer.record_step(
                StepTrace(
                    step_index=0,
                    duration_s=0.1,
                    prompt_messages=list(history),
                    model_response={"content": "reply0", "tool_calls": []},
                    tool_results=[],
                )
            )

            # Final step carries a circular reference: default=str cannot help, so the
            # placeholder record is written instead.
            circular: dict[str, Any] = {"input": 1}
            circular["self"] = circular
            tracer.record_step(
                StepTrace(
                    step_index=1,
                    duration_s=0.1,
                    prompt_messages=list(history),
                    usage=circular,  # type: ignore[arg-type]
                    model_response={"content": "reply1", "tool_calls": []},
                    tool_results=[],
                )
            )

            lines = [
                json.loads(line)
                for line in tracer.jsonl_path.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertTrue(lines[-1]["serialization_error"])

            # Reconstruction must fall back to the last step that recorded a prompt,
            # then append that step's own assistant reply.
            msgs = reconstruct_messages_from_steps(lines)
            self.assertEqual(
                [m["role"] for m in msgs],
                ["system", "user", "assistant", "user", "assistant", "assistant"],
            )
            self.assertEqual(msgs[-1]["content"], "reply0")

            loaded, meta = load_session_messages(out_dir)
            self.assertEqual(len(loaded), 6)
            self.assertEqual(meta["total_steps"], 2)

    def test_reconstruction_never_uses_placeholder_step(self):
        """Invariant: a step flagged serialization_error is skipped even if it carries prompt data."""
        good = {
            "step_index": 0,
            "prompt_messages": [{"role": "system", "content": "SYS"}],
            "model_response": {"content": "real", "tool_calls": []},
            "tool_results": [],
        }
        # A placeholder that (defensively) still carries prompt data must still be rejected,
        # otherwise a partially-written final record would become the reconstruction source.
        placeholder = {
            "step_index": 1,
            "serialization_error": True,
            "prompt_messages": [{"role": "user", "content": "ghost"}],
            "model_response": {"content": "ghost", "tool_calls": []},
            "tool_results": [],
        }

        self.assertEqual(reconstruct_messages_from_steps([good])[0]["content"], "SYS")
        msgs = reconstruct_messages_from_steps([good, placeholder])
        self.assertEqual([m["role"] for m in msgs], ["system", "assistant"])
        self.assertEqual(msgs[-1]["content"], "real")

        # With only placeholders available the last step is still the best we have:
        # its prompt snapshot plus its own assistant reply.
        self.assertEqual(
            [m["content"] for m in reconstruct_messages_from_steps([placeholder])],
            ["ghost", "ghost"],
        )

    def test_find_recent_sessions_reaches_old_workspace_match(self):
        """Invariant: a same-workspace session must stay reachable past the fast-path ceiling."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            runs = Path(tmp_dir) / "runs"
            runs.mkdir()

            def write(name: str, workspace: str, age_days: float) -> None:
                d = runs / name
                d.mkdir()
                (d / "trace.json").write_text(
                    json.dumps(
                        {
                            "task": {"id": name, "instruction": "work"},
                            "total_steps": 5,
                            "workspace": workspace,
                            "finish_reason": "completed",
                            "messages": [],
                            "steps": [],
                        }
                    ),
                    encoding="utf-8",
                )
                stamp = time.time() - age_days * 86400
                os.utime(d, (stamp, stamp))

            write("mine_old", "/repo", 100.0)
            for i in range(80):
                write(f"other_{i:03d}", "/elsewhere", 1.0 + i * 0.001)

            res = find_recent_sessions(runs, workspace="/repo", limit=10)
            self.assertEqual(res[0]["run_id"], "mine_old")
            self.assertTrue(res[0]["matches_workspace"])

            # With no workspace filter the ceiling must not change plain recency ordering
            # (ages increase with i, so other_000 is the newest).
            res_any = find_recent_sessions(runs, limit=10)
            self.assertEqual([r["run_id"] for r in res_any][0], "other_000")
            self.assertEqual(len(res_any), 10)

    def test_find_recent_sessions_short_circuits_on_recent_match(self):
        """Invariant: when the newest session matches this workspace, the scan must not read the rest."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            runs = Path(tmp_dir) / "runs"
            runs.mkdir()

            def write(name: str, workspace: str, age_days: float) -> None:
                d = runs / name
                d.mkdir()
                (d / "trace.json").write_text(
                    json.dumps(
                        {
                            "task": {"id": name, "instruction": "work"},
                            "total_steps": 1,
                            "workspace": workspace,
                            "finish_reason": "completed",
                            "messages": [],
                            "steps": [],
                        }
                    ),
                    encoding="utf-8",
                )
                stamp = time.time() - age_days * 86400
                os.utime(d, (stamp, stamp))

            write("mine_newest", "/repo", 0.0)
            for i in range(400):
                write(f"other_{i:03d}", "/elsewhere", 1.0 + i * 0.001)

            reads: list[str] = []
            real_read = Path.read_text

            def counting_read(self, *args, **kwargs):
                reads.append(self.name)
                return real_read(self, *args, **kwargs)

            with patch.object(Path, "read_text", counting_read):
                res = find_recent_sessions(runs, workspace="/repo", limit=10)

            self.assertEqual(res[0]["run_id"], "mine_newest")
            self.assertEqual(len(res), 10)
            # Only the pass-1 ceiling is consulted, far fewer than the 401 on disk.
            self.assertEqual(len(reads), 50)


if __name__ == "__main__":
    unittest.main()

