import json
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
    load_session,
    rate_limit_wait,
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