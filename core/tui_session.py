"""Running an agent turn inside the full-screen TUI.

The TUI maintains its own turn loop because it must stay responsive, probe
for cooperative cancellation, and redraw between individual steps. This module
holds that TUI-specific runner: step rendering, cancellation probes, and
turn abort handling.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from core.loop import AgentLoop
from core.memory import Memory
from core.protocols import CancellableModel, Model, ModelCancelledError
from core.tracer import Tracer, _format_action_summary, _pair_calls_with_results, _preview
from core.types import Message, StepTrace

MAX_STEPS_PER_TURN = 30
THOUGHT_PREVIEW_CHARS = 80
ERROR_PREVIEW_CHARS = 60


def replay_messages(state: Any, messages: list[Message]) -> None:
    """Re-renders a restored conversation into the TUI transcript."""
    for message in messages:
        if message.role == "user" and message.content:
            state.add_user(message.content)
        elif message.role == "assistant":
            if message.content:
                state.add_agent(message.content)
            for call in message.tool_calls or []:
                state.add_action(_format_action_summary(call.name, call.arguments, color=False))


class TuiTurnRunner:
    """Executes agent turns for the TUI, redrawing the screen after every step."""

    def __init__(
        self,
        *,
        model: Model,
        loop: AgentLoop,
        memory: Memory,
        tracer: Tracer,
        state: Any,
        tui: Any,
        cache_stats: Callable[[], dict[str, Any]],
        step_index: int = 1,
    ):
        self.model = model
        self.loop = loop
        self.memory = memory
        self.tracer = tracer
        self.state = state
        self.tui = tui
        self.cache_stats = cache_stats
        self.step_index = step_index

    def refresh(self) -> None:
        """Updates the status area and requests a redraw from the main UI thread."""
        self.state.update_status(self.memory.stats(), self.cache_stats())
        if hasattr(self.tui, "request_redraw"):
            self.tui.request_redraw()
        elif getattr(self.tui, "stdscr", None):
            self.tui.draw(self.tui.stdscr)

    def run_turn(self, max_steps: int = MAX_STEPS_PER_TURN) -> None:
        """Runs one agent turn, honouring Esc cancellation and Ctrl+C interrupts."""
        turn_start = self.step_index
        self.state.turn_active = True
        self._install_cancel_probe()
        try:
            while (self.step_index - turn_start) < max_steps:
                if self.tui.check_cancel_requested():
                    self.abort_turn("cancelled by user (Esc)", at_step=self.step_index)
                    return
                if not self._step_once():
                    return
        finally:
            self.state.turn_active = False
            self._remove_cancel_probe()
            self.refresh()

    def _step_once(self) -> bool:
        """Runs a single step and renders it. Returns False when the turn should stop."""
        try:
            is_finished, step_trace = self.loop.step(step_index=self.step_index)
        except ModelCancelledError:
            self.abort_turn("cancelled by user (Esc)", at_step=self.step_index)
            return False
        except KeyboardInterrupt:
            self.abort_turn("interrupted by user (Ctrl+C)", at_step=self.step_index)
            return False
        except Exception as exc:
            self.state.add_error(f"Step {self.step_index} error: {exc}")
            return False

        self.tracer.record_step(step_trace)
        self.step_index += 1
        self.render_step(step_trace)
        self.refresh()

        if is_finished:
            return False
        if self.tui.check_cancel_requested():
            self.abort_turn("cancelled by user (Esc)", at_step=self.step_index)
            return False
        return True

    def render_step(self, step_trace: StepTrace) -> None:
        """Renders one completed step: thoughts and actions, or the final answer."""
        text = step_trace.model_response.get("content") or ""
        tool_calls = step_trace.model_response.get("tool_calls") or []

        if not tool_calls:
            if text.strip():
                self.state.add_agent(text.strip())
            return

        if text.strip():
            self.state.add_thought(_preview(text, THOUGHT_PREVIEW_CHARS))
        for call, result in _pair_calls_with_results(tool_calls, step_trace.tool_results):
            self._render_action(call, result, step_trace.duration_s)

    def _render_action(
        self, call: dict[str, Any] | None, result: dict[str, Any] | None, duration: float
    ) -> None:
        """Renders one tool call, surfacing the first output line on failure."""
        summary = _format_action_summary(
            (call or {}).get("name", ""), (call or {}).get("arguments", {}), color=False
        )
        exit_code = (result or {}).get("exit_code", 0)
        if exit_code == 0:
            self.state.add_action(f"{summary} ({duration:.1f}s)")
            return

        lines = (result or {}).get("output", "").strip().splitlines()
        detail = _preview("\n".join(lines[:1]), ERROR_PREVIEW_CHARS) or "Failed"
        self.state.add_error(f"{summary} (exit {exit_code}: {detail})")

    def abort_turn(self, reason: str, at_step: int) -> None:
        """Ends the turn cleanly, dropping any half-finished tool exchange.

        Shared by the boundary check and the in-flight cancel probe so both
        report identically and leave the session in the same state.
        """
        dropped = self.memory.drop_incomplete_tool_turn()
        suffix = f" Dropped {dropped} incomplete message(s)." if dropped else ""
        self.state.add_system(
            f"⚠️ Turn {reason} at step {at_step}. Session is still active.{suffix}"
        )
        self.refresh()

    def cancel_probe(self) -> bool:
        """Non-blocking Esc check invoked from the model's retry/backoff path.

        Surfaces progress the moment it fires, so a user who pressed Esc during
        a long rate-limit backoff sees the request is aborting rather than
        assuming the key did nothing. Every caller treats True as terminal, so
        this runs at most once per turn.
        """
        if not self.tui.check_cancel_requested():
            return False
        self.state.add_system(
            "⚠️ Cancelling… aborting the pending model request "
            "(Ctrl+C cancels immediately, even mid-request)."
        )
        self.refresh()
        return True

    def _install_cancel_probe(self) -> None:
        if isinstance(self.model, CancellableModel):
            self.model.set_cancel_probe(self.cancel_probe)

    def _remove_cancel_probe(self) -> None:
        if isinstance(self.model, CancellableModel):
            self.model.set_cancel_probe(None)
