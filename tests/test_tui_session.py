import unittest
from unittest.mock import MagicMock

from core.memory import Memory
from core.model import MockModel
from core.protocols import ModelCancelledError
from core.tui_session import TuiTurnRunner, replay_messages
from core.types import Message, ToolCall


class FakeState:
    """TUIState double recording every rendering call in order."""

    def __init__(self):
        self.turns: list[str] = []
        self.agents: list[str] = []
        self.actions: list[str] = []
        self.errors: list[str] = []
        self.systems: list[str] = []
        self.status_updates: list[tuple] = []
        self.turn_active = False

    def add_thought(self, text):
        self.turns.append(text)

    def add_agent(self, text):
        self.agents.append(text)

    def add_action(self, text):
        self.actions.append(text)

    def add_error(self, text):
        self.errors.append(text)

    def add_system(self, text):
        self.systems.append(text)

    def add_user(self, text):
        self.systems.append(f"> {text}")

    def update_status(self, stats, cache_stats=None):
        self.status_updates.append((stats, cache_stats))


class FakeTui:
    """CursesTUI double: tracks draws and can request a cancel a set number of times."""

    def __init__(self, cancel_after: int | None = None):
        self.stdscr = object()
        self.draws = 0
        self.cancel_after = cancel_after
        self.cancelled = 0

    def draw(self, _stdscr):
        self.draws += 1

    def check_cancel_requested(self):
        if self.cancel_after is None:
            return False
        self.cancelled += 1
        return self.cancelled > self.cancel_after


def _trace(content="", tool_calls=None, results=None, duration=1.0):
    trace = MagicMock()
    trace.model_response = {"content": content, "tool_calls": tool_calls or []}
    trace.tool_results = results or []
    trace.duration_s = duration
    return trace


class TuiSessionTestCase(unittest.TestCase):
    def setUp(self):
        self.memory = Memory(system_prompt="sys", user_prompt="")
        self.state = FakeState()
        self.tracer = MagicMock()
        self.model = MockModel([])
        self.tui = FakeTui()

    def _runner(self, step_results, tui=None, model=None):
        loop = MagicMock()
        loop.step.side_effect = step_results
        return TuiTurnRunner(
            model=model or self.model,
            loop=loop,
            memory=self.memory,
            tracer=self.tracer,
            state=self.state,
            tui=tui or self.tui,
            cache_stats=lambda: {"cache_ttl": "5m"},
        )


class TestTurnExecution(TuiSessionTestCase):
    def test_turn_stops_when_the_model_finishes(self):
        """Invariant: a finishing step ends the turn and the runner keeps the screen updated."""
        acting = _trace(
            content="thinking",
            tool_calls=[{"name": "bash", "arguments": {"command": "ls"}}],
            results=[{"exit_code": 0, "output": "a.py"}],
        )
        runner = self._runner([(False, acting), (True, _trace("final answer"))])

        runner.run_turn()

        self.assertEqual(self.state.agents, ["final answer"])
        self.assertEqual(self.state.turns, ["thinking"])
        self.assertEqual(runner.step_index, 3)
        self.assertFalse(self.state.turn_active)

    def test_turn_respects_the_step_budget(self):
        """Invariant: a runaway model cannot exceed the per-turn step ceiling."""
        runner = self._runner([(False, _trace()) for _ in range(10)])

        runner.run_turn(max_steps=4)

        self.assertEqual(runner.step_index, 5)
        self.assertEqual(runner.loop.step.call_count, 4)

    def test_tool_calls_render_as_actions_not_agent_text(self):
        """Invariant: a step with tool calls shows actions and hides the final answer."""
        trace = _trace(
            content="I will list the files",
            tool_calls=[{"name": "bash", "arguments": {"command": "ls"}}],
            results=[{"exit_code": 0, "output": "a.py"}],
            duration=0.5,
        )
        runner = self._runner([(False, trace)])

        runner.run_turn()

        self.assertEqual(self.state.turns, ["I will list the files"])
        self.assertEqual(self.state.actions, ["$ ls (0.5s)"])

    def test_failed_tool_call_surfaces_the_first_output_line(self):
        """Invariant: a non-zero exit is reported with a bounded error excerpt."""
        trace = _trace(
            tool_calls=[{"name": "bash", "arguments": {"command": "false"}}],
            results=[{"exit_code": 1, "output": "boom: it failed\nmore detail"}],
        )
        runner = self._runner([(False, trace), (True, _trace("recovered"))])

        runner.run_turn()

        self.assertEqual(self.state.errors, ["$ false (exit 1: boom: it failed)"])

    def test_long_reasoning_is_previewed_to_one_line(self):
        """Invariant: a long thought is truncated so it cannot flood the transcript."""
        trace = _trace(
            content="x" * 200,
            tool_calls=[{"name": "bash", "arguments": {"command": "ls"}}],
            results=[{"exit_code": 0, "output": ""}],
        )
        runner = self._runner([(False, trace)])

        runner.run_turn()

        self.assertEqual(len(self.state.turns[0]), 83)
        self.assertTrue(self.state.turns[0].endswith("..."))

    def test_steps_are_recorded_in_the_trace(self):
        """Invariant: every completed step is written to the tracer."""
        runner = self._runner([(True, _trace("done"))])
        runner.run_turn()
        self.tracer.record_step.assert_called_once()


class TestCancellation(TuiSessionTestCase):
    def test_esc_before_a_step_stops_the_turn(self):
        """Invariant: Esc at a step boundary ends the turn without running the model."""
        runner = self._runner([(False, _trace())], tui=FakeTui(cancel_after=0))

        runner.run_turn()

        runner.loop.step.assert_not_called()
        self.assertIn("cancelled by user (Esc)", self.state.systems[0])

    def test_esc_after_a_step_stops_the_next_one(self):
        """Invariant: Esc pressed mid-turn is honoured at the next boundary."""
        runner = self._runner([(False, _trace())] * 5, tui=FakeTui(cancel_after=2))

        runner.run_turn()

        self.assertEqual(runner.loop.step.call_count, 1)
        self.assertIn("Session is still active", self.state.systems[-1])

    def test_mid_request_cancellation_is_reported(self):
        """Invariant: ModelCancelledError aborts the turn rather than crashing the session."""
        runner = self._runner([ModelCancelledError("cancelled by user")])

        runner.run_turn()

        self.assertIn("cancelled by user (Esc)", self.state.systems[0])
        self.assertFalse(self.state.turn_active)

    def test_cancel_probe_reports_progress_once(self):
        """Invariant: a probe that fires both announces the abort and returns True."""
        probe_tui = FakeTui(cancel_after=0)
        runner = self._runner([], tui=probe_tui)

        self.assertTrue(runner.cancel_probe())
        self.assertIn("Cancelling", self.state.systems[0])
        self.assertIn("Ctrl+C cancels immediately", self.state.systems[0])

    def test_cancel_probe_is_false_while_no_cancel_is_pending(self):
        """Invariant: the probe stays quiet during a normal request."""
        runner = self._runner([])
        self.assertFalse(runner.cancel_probe())
        self.assertEqual(self.state.systems, [])

    def test_keyboard_interrupt_aborts_the_turn(self):
        """Invariant: Ctrl+C ends the turn and leaves the session usable."""
        runner = self._runner([KeyboardInterrupt])
        runner.run_turn()
        self.assertIn("interrupted by user (Ctrl+C)", self.state.systems[0])

    def test_unexpected_step_error_is_reported_without_aborting_the_session(self):
        """Invariant: a failing step is surfaced as an error and the turn ends cleanly."""
        runner = self._runner([RuntimeError("provider exploded")])
        runner.run_turn()
        self.assertEqual(self.state.errors, ["Step 1 error: provider exploded"])
        self.assertFalse(self.state.turn_active)

    def test_turn_active_flag_is_always_cleared(self):
        """Invariant: even a failing turn restores the UI's idle state."""
        runner = self._runner([RuntimeError("boom")])
        runner.run_turn()
        self.assertFalse(self.state.turn_active)


class TestAbortTurn(TuiSessionTestCase):
    def test_reports_the_reason_and_step(self):
        """Invariant: an abort names its cause and the step it happened at."""
        runner = self._runner([])
        runner.step_index = 4

        runner.abort_turn("cancelled by user (Esc)", at_step=4)

        self.assertIn("Turn cancelled by user (Esc) at step 4", self.state.systems[0])

    def test_reports_dropped_incomplete_messages(self):
        """Invariant: a half-finished tool exchange is reported so the user is not misled."""
        self.memory.add_assistant("calling", tool_calls=[
            ToolCall(id="1", name="bash", arguments={"command": "ls"})
        ])
        runner = self._runner([])

        runner.abort_turn("cancelled by user (Esc)", at_step=2)

        self.assertIn("Dropped 1 incomplete message(s)", self.state.systems[0])


class TestReplayMessages(TuiSessionTestCase):
    def test_replays_users_assistants_and_tool_calls(self):
        """Invariant: a resumed transcript is reconstructed in the same shapes as live turns."""
        messages = [
            Message(role="user", content="do it"),
            Message(role="assistant", content="on it", tool_calls=[
                ToolCall(id="1", name="bash", arguments={"command": "ls"})
            ]),
            Message(role="tool", content="output"),
        ]

        replay_messages(self.state, messages)

        self.assertIn("> do it", self.state.systems)
        self.assertEqual(self.state.agents, ["on it"])
        self.assertEqual(self.state.actions, ["$ ls"])

    def test_empty_messages_are_a_noop(self):
        """Invariant: resuming with no messages renders nothing."""
        replay_messages(self.state, [])
        self.assertEqual(self.state.systems, [])


if __name__ == "__main__":
    unittest.main()