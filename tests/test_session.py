import json
import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.model import MockModel, RateLimitError
from core.session import (
    ERROR,
    MAX_STEPS_EXCEEDED,
    TIMEOUT,
    find_recent_sessions,
    load_session,
    load_session_messages,
    rate_limit_wait,
    reconstruct_messages_from_steps,
    resolve_session_dir,
    run_steps,
)


class FakeTracer:
    """Minimal tracer double recording only what the step driver reports."""

    def __init__(self):
        self.steps: list[object] = []
        self.logs: list[str] = []

    def log(self, text: str) -> None:
        self.logs.append(text)

    def record_step(self, step_trace) -> None:
        self.steps.append(step_trace)


def _traced_loop(*results):
    """A loop double whose step() returns the given (finished, trace) pairs in order."""
    loop = MagicMock()
    loop.step.side_effect = results
    return loop


class _Trace:
    """Stand-in for a StepTrace, exposing only the attributes the driver reads."""

    def __init__(self, content="done"):
        self.model_response = {"content": content, "tool_calls": []}
        self.tool_results: list[dict] = []
        self.duration_s = 0.0


def _trace(content="done"):
    return _Trace(content)


class TestRateLimitWait(unittest.TestCase):
    def test_reads_the_counter_off_a_capable_model(self):
        """Invariant: a model's cumulative throttle time is reported as a float."""
        model = MockModel([])
        model.total_rate_limit_wait_s = 2.5
        self.assertEqual(rate_limit_wait(model), 2.5)

    def test_missing_counter_defaults_to_zero(self):
        """Invariant: a model without the capability reports no throttling at all."""
        self.assertEqual(rate_limit_wait(object()), 0.0)

    def test_unparsable_counter_defaults_to_zero(self):
        """Invariant: telemetry garbage never breaks a run."""
        model = MagicMock()
        model.total_rate_limit_wait_s = "not a number"
        self.assertEqual(rate_limit_wait(model), 0.0)


class TestRunSteps(unittest.TestCase):
    def test_runs_until_the_model_finishes(self):
        """Invariant: a finishing step ends the turn and reports 'completed'."""
        tracer = FakeTracer()
        loop = _traced_loop((False, _trace("a")), (True, _trace("b")))

        outcome = run_steps(MockModel([]), loop, tracer, start_step=1, max_steps=5)

        self.assertTrue(outcome.finished)
        self.assertEqual(outcome.reason, "completed")
        self.assertEqual(outcome.next_step, 3)
        self.assertEqual(len(tracer.steps), 2)

    def test_stops_at_max_steps(self):
        """Invariant: the step budget is a hard ceiling."""
        loop = _traced_loop(*[(False, _trace()) for _ in range(3)])
        outcome = run_steps(MockModel([]), loop, FakeTracer(), start_step=1, max_steps=3)
        self.assertFalse(outcome.finished)
        self.assertEqual(outcome.reason, MAX_STEPS_EXCEEDED)
        self.assertEqual(outcome.next_step, 4)

    def test_expired_deadline_stops_before_the_first_step(self):
        """Invariant: an already-exhausted budget prevents any model call."""
        loop = MagicMock()
        outcome = run_steps(
            MockModel([]), loop, FakeTracer(), start_step=1, max_steps=5, deadline=time.time() - 1
        )
        self.assertEqual(outcome.reason, TIMEOUT)
        loop.step.assert_not_called()

    def test_transient_errors_are_retried_with_backoff(self):
        """Invariant: a failing step is retried with growing delays before giving up."""
        loop = MagicMock()
        loop.step.side_effect = [RuntimeError("boom"), (True, _trace())]
        tracer = FakeTracer()

        with patch("time.sleep") as sleep:
            outcome = run_steps(
                MockModel([]), loop, tracer, start_step=1, max_steps=3, retry_delay=2.0
            )

        self.assertTrue(outcome.finished)
        sleep.assert_called_once_with(2.0)

    def test_retries_are_bounded_then_reported_as_error(self):
        """Invariant: exhausting retries ends the run with reason 'error'."""
        loop = MagicMock()
        loop.step.side_effect = RuntimeError("boom")
        tracer = FakeTracer()

        with patch("time.sleep"):
            outcome = run_steps(
                MockModel([]), loop, tracer, start_step=1, max_steps=9, max_retries=2
            )

        self.assertFalse(outcome.finished)
        self.assertEqual(outcome.reason, ERROR)
        self.assertEqual(loop.step.call_count, 3)
        self.assertTrue(any("failed after 2 retries" in line for line in tracer.logs))

    def test_rate_limit_error_stops_without_retrying(self):
        """Invariant: a 429 is terminal for the turn so the caller can resume later."""
        loop = MagicMock()
        loop.step.side_effect = RateLimitError("HTTP 429", retry_after=30)
        outcome = run_steps(MockModel([]), loop, FakeTracer(), start_step=1, max_steps=5)
        self.assertEqual(outcome.reason, "rate_limited")
        self.assertEqual(loop.step.call_count, 1)

    def test_timeout_error_is_not_retried(self):
        """Invariant: an expired task deadline must surface immediately."""
        loop = MagicMock()
        loop.step.side_effect = TimeoutError("Task timeout reached during model request")
        outcome = run_steps(MockModel([]), loop, FakeTracer(), start_step=1, max_steps=5)
        self.assertEqual(outcome.reason, TIMEOUT)
        self.assertEqual(loop.step.call_count, 1)

    def test_keyboard_interrupt_is_reported_not_raised(self):
        """Invariant: Ctrl+C ends the turn cleanly so the session can save its trace."""
        loop = MagicMock()
        loop.step.side_effect = KeyboardInterrupt
        outcome = run_steps(MockModel([]), loop, FakeTracer(), start_step=1, max_steps=5)
        self.assertEqual(outcome.reason, "interrupted")

    def test_on_finish_receives_the_final_text(self):
        """Invariant: the completion callback gets the model's last response text."""
        seen: list[str] = []
        loop = _traced_loop((True, _trace("all done")))
        run_steps(
            MockModel([]), loop, FakeTracer(), start_step=1, max_steps=1, on_finish=seen.append
        )
        self.assertEqual(seen, ["all done"])

    def test_rate_limit_wait_extends_the_effective_deadline(self):
        """Invariant: time spent honouring provider backoff is not charged to the task."""
        model = MockModel([])
        model.total_rate_limit_wait_s = 600.0
        loop = _traced_loop((True, _trace()))
        deadline = time.time() + 1.0

        outcome = run_steps(model, loop, FakeTracer(), start_step=1, max_steps=1, deadline=deadline)

        self.assertTrue(outcome.finished)


class TestSessionResolution(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.runs_dir = self.tmp / "runs"
        self.runs_dir.mkdir()
        self.sessions = [
            {"run_id": "alpha_1", "run_dir": str(self.runs_dir / "alpha_1"), "total_steps": 2},
            {"run_id": "beta_2", "run_dir": str(self.runs_dir / "beta_2"), "total_steps": 5},
        ]

    def test_exact_directory_path(self):
        """Invariant: passing a run directory loads exactly that run."""
        Path(self.sessions[0]["run_dir"]).mkdir()
        target, error = resolve_session_dir(self.runs_dir, self.sessions[0]["run_dir"], self.sessions)
        self.assertEqual(target, Path(self.sessions[0]["run_dir"]))
        self.assertEqual(error, "")

    def test_exact_run_id(self):
        """Invariant: passing a run id resolves inside the runs directory."""
        (self.runs_dir / "alpha_1").mkdir()
        target, error = resolve_session_dir(self.runs_dir, "alpha_1", self.sessions)
        self.assertEqual(target, self.runs_dir / "alpha_1")
        self.assertEqual(error, "")

    def test_one_based_index(self):
        """Invariant: indices are 1-based, matching the listing the user sees."""
        target, _ = resolve_session_dir(self.runs_dir, "2", self.sessions)
        self.assertEqual(target, Path(self.sessions[1]["run_dir"]))

    def test_substring_match(self):
        """Invariant: a partial run id still selects the right session."""
        target, _ = resolve_session_dir(self.runs_dir, "beta", self.sessions)
        self.assertEqual(target, Path(self.sessions[1]["run_dir"]))

    def test_no_selector_picks_the_most_recent(self):
        """Invariant: resuming with no argument continues the latest session."""
        target, _ = resolve_session_dir(self.runs_dir, None, self.sessions)
        self.assertEqual(target, Path(self.sessions[0]["run_dir"]))

    def test_out_of_range_index_reports_the_valid_range(self):
        """Invariant: an impossible index says how many sessions exist."""
        target, error = resolve_session_dir(self.runs_dir, "99", self.sessions)
        self.assertIsNone(target)
        self.assertIn("Invalid session index: 99", error)
        self.assertIn("found 2 sessions", error)

    def test_unknown_id_reports_not_found(self):
        """Invariant: an unrecognized selector is reported as a missing session."""
        target, error = resolve_session_dir(self.runs_dir, "nonexistent", self.sessions)
        self.assertIsNone(target)
        self.assertIn("not found", error)

    def test_no_sessions_and_no_selector(self):
        """Invariant: with nothing to resume, resolution defers the message to load_session."""
        target, error = resolve_session_dir(self.runs_dir, None, [])
        self.assertIsNone(target)
        self.assertEqual(error, "")
        self.assertEqual(load_session(self.runs_dir, None, None), ([], {"error": "Session not found: no resumable session available"}))


class TestLoadSession(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.runs_dir = self.tmp / "runs"
        run_dir = self.runs_dir / "session_a"
        run_dir.mkdir(parents=True)
        (run_dir / "trace.json").write_text(
            json.dumps({
                "task_id": "session_a",
                "messages": [
                    {"role": "user", "content": "hello"},
                    {"role": "assistant", "content": "hi"},
                ],
            }),
            encoding="utf-8",
        )

    def test_loads_messages_and_metadata(self):
        """Invariant: a resumed session restores real Message objects, not raw dicts."""
        messages, meta = load_session(self.runs_dir, None, "session_a")
        self.assertEqual([m.role for m in messages], ["user", "assistant"])
        self.assertEqual(messages[0].content, "hello")
        self.assertEqual(meta["run_id"], "session_a")

    def test_unknown_selector_returns_an_error_not_an_exception(self):
        """Invariant: /resume with a bad argument reports rather than crashes."""
        messages, meta = load_session(self.runs_dir, None, "nope")
        self.assertEqual(messages, [])
        self.assertIn("not found", meta["error"])


if __name__ == "__main__":
    unittest.main()

class TestSessionIndex(unittest.TestCase):
    """Scanning a runs directory into the /resume listing."""

    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.runs = self.tmp / "runs"
        self.runs.mkdir()

    def _write_session(self, name, workspace="/repo", age_days=0.0, steps=1, instruction="do work"):
        entry = self.runs / name
        entry.mkdir()
        (entry / "trace.json").write_text(
            json.dumps({
                "task": {"id": name, "instruction": instruction},
                "total_steps": steps,
                "workspace": workspace,
                "finish_reason": "completed",
                "messages": [],
                "steps": [],
            }),
            encoding="utf-8",
        )
        stamp = time.time() - age_days * 86400
        os.utime(entry, (stamp, stamp))
        return entry

    def test_missing_runs_dir_yields_nothing(self):
        """Invariant: a runs directory that does not exist is not an error."""
        self.assertEqual(find_recent_sessions(self.tmp / "absent"), [])

    def test_directory_without_a_trace_is_not_a_session(self):
        """Invariant: stray directories in runs/ are ignored rather than listed."""
        (self.runs / "not-a-run").mkdir()
        self._write_session("real")
        self.assertEqual([s["run_id"] for s in find_recent_sessions(self.runs)], ["real"])

    def test_newest_first_when_no_workspace_is_given(self):
        """Invariant: without a workspace filter, recency alone decides the order."""
        self._write_session("older", age_days=2.0)
        self._write_session("newer", age_days=1.0)
        self.assertEqual([s["run_id"] for s in find_recent_sessions(self.runs)],
                         ["newer", "older"])

    def test_matching_workspace_sorts_ahead_of_newer_sessions(self):
        """Invariant: sessions from this repository outrank newer ones from elsewhere."""
        self._write_session("mine", workspace="/repo", age_days=5.0)
        self._write_session("theirs", workspace="/elsewhere", age_days=1.0)
        sessions = find_recent_sessions(self.runs, workspace="/repo")
        self.assertEqual([s["run_id"] for s in sessions], ["mine", "theirs"])
        self.assertEqual([s["matches_workspace"] for s in sessions], [True, False])

    def test_session_fields_are_read_from_the_trace(self):
        """Invariant: the listing carries what /resume needs to show and restore."""
        self._write_session("abc", steps=7, instruction="Fix the bug in auth.py")
        record = find_recent_sessions(self.runs)[0]
        self.assertEqual(record["task_id"], "abc")
        self.assertEqual(record["total_steps"], 7)
        self.assertEqual(record["preview"], "Fix the bug in auth.py")
        self.assertEqual(record["finish_reason"], "completed")
        self.assertEqual(record["run_dir"], str(self.runs / "abc"))

    def test_preview_falls_back_to_the_first_user_message(self):
        """Invariant: a run with no task instruction still previews its first prompt."""
        entry = self.runs / "noinstr"
        entry.mkdir()
        (entry / "trace.json").write_text(
            json.dumps({
                "task": {"id": "noinstr", "instruction": "   "},
                "total_steps": 1,
                "messages": [
                    {"role": "system", "content": "sys"},
                    {"role": "user", "content": "the real ask"},
                ],
            }),
            encoding="utf-8",
        )
        self.assertEqual(find_recent_sessions(self.runs)[0]["preview"], "the real ask")

    def test_preview_is_clipped_to_one_line(self):
        """Invariant: an enormous instruction cannot flood the /resume listing."""
        self._write_session("long", instruction="x" * 400)
        preview = find_recent_sessions(self.runs)[0]["preview"]
        self.assertEqual(len(preview), 60)
        self.assertTrue(preview.endswith("..."))

    def test_missing_preview_is_reported_explicitly(self):
        """Invariant: a session with no user text shows a placeholder, not a blank row."""
        entry = self.runs / "silent"
        entry.mkdir()
        (entry / "trace.json").write_text(json.dumps({"total_steps": 0}), encoding="utf-8")
        self.assertEqual(find_recent_sessions(self.runs)[0]["preview"], "(no user prompt)")

    def test_corrupt_trace_does_not_abort_the_scan(self):
        """Invariant: one unreadable trace does not hide the sessions behind it.

        The corrupt run still appears (it has a trace file), but with empty
        summary fields rather than fabricated ones.
        """
        broken = self.runs / "broken"
        broken.mkdir()
        (broken / "trace.json").write_text("{ not json", encoding="utf-8")
        self._write_session("healthy")

        by_id = {s["run_id"]: s for s in find_recent_sessions(self.runs)}
        self.assertIn("healthy", by_id)
        self.assertEqual(by_id["broken"]["total_steps"], 0)
        self.assertEqual(by_id["broken"]["workspace"], None)
        self.assertEqual(by_id["broken"]["preview"], "(no user prompt)")

    def test_jsonl_only_run_is_discovered(self):
        """Invariant: a run that crashed before writing trace.json is still resumable."""
        entry = self.runs / "crashed"
        entry.mkdir()
        lines = [
            json.dumps({"prompt_messages": [{"role": "user", "content": "first ask"}]}),
            json.dumps({"prompt_messages": [{"role": "user", "content": "first ask"}]}),
        ]
        (entry / "trajectory.jsonl").write_text("\n".join(lines), encoding="utf-8")

        record = find_recent_sessions(self.runs)[0]
        self.assertEqual(record["run_id"], "crashed")
        self.assertEqual(record["total_steps"], 2)
        self.assertEqual(record["preview"], "first ask")

    def test_older_match_is_still_found_beyond_the_cheap_window(self):
        """Invariant: the two-pass scan reaches an older session for the same repository."""
        self._write_session("old_match", workspace="/repo", age_days=90.0)
        for index in range(80):
            self._write_session(f"noise_{index:03d}", workspace="/elsewhere", age_days=1.0 + index)

        sessions = find_recent_sessions(self.runs, workspace="/repo", limit=5)
        self.assertEqual(sessions[0]["run_id"], "old_match")
        self.assertTrue(sessions[0]["matches_workspace"])

    def test_limit_caps_the_result(self):
        """Invariant: /resume never returns more sessions than it asked for."""
        for index in range(6):
            self._write_session(f"s{index}", age_days=float(index))
        self.assertEqual(len(find_recent_sessions(self.runs, limit=3)), 3)


class TestReconstructMessages(unittest.TestCase):
    """Rebuilding a conversation from persisted step records."""

    def test_empty_input_yields_no_messages(self):
        """Invariant: a run with no steps reconstructs to nothing."""
        self.assertEqual(reconstruct_messages_from_steps([]), [])

    def test_prompt_plus_assistant_reply(self):
        """Invariant: the last step's prompt and its own response are both restored."""
        steps = [{
            "prompt_messages": [{"role": "user", "content": "hi"}],
            "model_response": {"content": "hello", "tool_calls": []},
            "tool_results": [],
        }]
        self.assertEqual(reconstruct_messages_from_steps(steps), [
            {"role": "user", "content": "hi"},
            {"role": "assistant", "content": "hello"},
        ])

    def test_tool_calls_and_results_are_restored_in_order(self):
        """Invariant: an interrupted tool exchange is replayed with its result attached."""
        steps = [{
            "prompt_messages": [{"role": "user", "content": "list files"}],
            "model_response": {
                "content": "",
                "tool_calls": [{"id": "c1", "name": "bash", "arguments": {"command": "ls"}}],
            },
            "tool_results": [{"call_id": "c1", "output": "a.py", "raw_log_path": None}],
        }]
        messages = reconstruct_messages_from_steps(steps)
        self.assertEqual(messages[1]["tool_calls"], [{
            "id": "c1",
            "type": "function",
            "function": {"name": "bash", "arguments": '{"command": "ls"}'},
        }])
        self.assertEqual(messages[2]["role"], "tool")
        self.assertEqual(messages[2]["tool_call_id"], "c1")

    def test_serialization_placeholder_is_skipped(self):
        """Invariant: a step that failed to serialize does not wipe the conversation."""
        steps = [
            {
                "prompt_messages": [{"role": "user", "content": "real history"}],
                "model_response": {"content": "kept", "tool_calls": []},
                "tool_results": [],
            },
            {
                "prompt_messages": [],
                "model_response": {"content": "lost"},
                "serialization_error": True,
            },
        ]
        messages = reconstruct_messages_from_steps(steps)
        self.assertEqual(messages[0]["content"], "real history")
        self.assertEqual(messages[1]["content"], "kept")

    def test_latest_step_with_state_wins(self):
        """Invariant: reconstruction uses the most recent usable step, not the first."""
        steps = [
            {"prompt_messages": [{"role": "user", "content": "one"}],
             "model_response": {"content": "a"}, "tool_results": []},
            {"prompt_messages": [{"role": "user", "content": "two"}],
             "model_response": {"content": "b"}, "tool_results": []},
        ]
        self.assertEqual(reconstruct_messages_from_steps(steps)[0]["content"], "two")

    def test_unusable_arguments_are_preserved_as_strings(self):
        """Invariant: non-dict arguments are still replayable after a round trip."""
        steps = [{
            "prompt_messages": [{"role": "user", "content": "x"}],
            "model_response": {"content": "", "tool_calls": [
                {"id": "1", "name": "bash", "arguments": "not json at all"}
            ]},
            "tool_results": [],
        }]
        call = reconstruct_messages_from_steps(steps)[1]["tool_calls"][0]
        self.assertEqual(call["function"]["arguments"], "not json at all")

    def test_provider_extra_content_is_preserved(self):
        """Invariant: provider-specific call metadata survives a resume."""
        steps = [{
            "prompt_messages": [{"role": "user", "content": "x"}],
            "model_response": {"content": "", "tool_calls": [
                {"id": "1", "function": {"name": "bash", "arguments": "{}"},
                 "extra_content": {"provider": "x"}}
            ]},
            "tool_results": [],
        }]
        call = reconstruct_messages_from_steps(steps)[1]["tool_calls"][0]
        self.assertEqual(call["extra_content"], {"provider": "x"})


class TestLoadSessionMessagesFallbacks(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def test_trace_messages_are_preferred(self):
        """Invariant: a complete trace replays its own recorded messages."""
        run = self.tmp / "run"
        run.mkdir()
        (run / "trace.json").write_text(
            json.dumps({"total_steps": 2, "messages": [{"role": "user", "content": "hi"}]}),
            encoding="utf-8",
        )
        messages, meta = load_session_messages(run)
        self.assertEqual(messages, [{"role": "user", "content": "hi"}])
        self.assertEqual(meta["total_steps"], 2)

    def test_falls_back_to_trace_steps_when_messages_are_absent(self):
        """Invariant: a trace without a messages block is rebuilt from its steps."""
        run = self.tmp / "steps_only"
        run.mkdir()
        (run / "trace.json").write_text(
            json.dumps({
                "total_steps": 1,
                "steps": [{"prompt_messages": [{"role": "user", "content": "q"}],
                           "model_response": {"content": "a"}, "tool_results": []}],
            }),
            encoding="utf-8",
        )
        messages, _ = load_session_messages(run)
        self.assertEqual([m["content"] for m in messages], ["q", "a"])

    def test_falls_back_to_jsonl_when_trace_is_corrupt(self):
        """Invariant: a truncated trace.json still leaves a resumable session."""
        run = self.tmp / "corrupt"
        run.mkdir()
        (run / "trace.json").write_text("{ truncated", encoding="utf-8")
        (run / "trajectory.jsonl").write_text(
            json.dumps({"prompt_messages": [{"role": "user", "content": "q"}],
                        "model_response": {"content": "a"}, "tool_results": []}),
            encoding="utf-8",
        )
        messages, meta = load_session_messages(run)
        self.assertEqual([m["content"] for m in messages], ["q", "a"])
        self.assertEqual(meta["total_steps"], 1)

    def test_unreadable_session_reports_metadata_without_messages(self):
        """Invariant: a directory with no trace at all yields empty messages, not an error."""
        run = self.tmp / "empty"
        run.mkdir()
        messages, meta = load_session_messages(run)
        self.assertEqual(messages, [])
        self.assertEqual(meta["finish_reason"], "unknown")
        self.assertEqual(meta["run_id"], "empty")
