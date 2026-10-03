from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import Any

from core.harness import Task
from core.tracer import Tracer, _pair_calls_with_results


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


if __name__ == "__main__":
    unittest.main()
