from __future__ import annotations

import contextlib
import io
import unicodedata
import unittest
from unittest.mock import MagicMock, patch

from core.commands import Command
from core.interactive import COMMAND_SPECS
from core.skills import Skill
from core.tui import (
    SPINNER_FRAMES,
    SPINNER_INTERVAL,
    CursesTUI,
    TUILine,
    TUIState,
    _read_pending,
    decode_terminal_sequence,
    parse_inline_segments,
    render_markdown_table,
)


def display_width(text: str) -> int:
    """Terminal cells occupied by text; wide chars count 2, combining marks 0."""
    return sum(
        0
        if unicodedata.combining(ch)
        else (2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1)
        for ch in text
    )


def status_bar_text(mock_stdscr) -> str:
    """The most recently drawn reverse-video status bar text."""
    import curses

    texts = [
        call.args[2]
        for call in mock_stdscr.addstr.call_args_list
        if len(call.args) >= 4 and call.args[3] == curses.A_REVERSE
    ]
    if not texts:
        raise AssertionError("no status bar was drawn")
    return texts[-1]


class TestMarkdownTable(unittest.TestCase):
    def test_never_exceeds_max_width_across_widths_and_column_counts(self):
        """Invariant: no rendered line is wider than max_width, for any width/column count.

        Property-style rather than a fixed sample: the per-column width floor means an
        n-column table can never render narrower than 6n+1 cells, so the renderer must
        shed columns rather than let the box art overflow the pane.
        """
        longest_cell = "x" * 40
        for num_cols in range(1, 6):
            headers = [f"H{i}" for i in range(num_cols)]
            rows = [[longest_cell] * num_cols]
            for max_width in range(1, 90):
                lines = render_markdown_table(headers, rows, max_width=max_width)
                if not lines:
                    continue
                widest = max(display_width(line.text) for line in lines)
                self.assertLessEqual(
                    widest,
                    max_width,
                    f"{num_cols} cols at max_width={max_width} rendered {widest} cells",
                )

    def test_wide_characters_align_to_equal_display_width(self):
        """Invariant: every line of a CJK table has identical display width."""
        lines = render_markdown_table(
            ["名前", "Qty"], [["田中太郎", "1"]], max_width=40
        )
        widths = {display_width(line.text) for line in lines}
        self.assertEqual(len(widths), 1, f"ragged box, display widths were {widths}")

    def test_wide_characters_preserved_when_not_truncated(self):
        """Invariant: cells that fit keep their characters intact (no mojibake/clipping)."""
        lines = render_markdown_table(["名前"], [["東京都"]], max_width=40)
        body = [line.text for line in lines if "東京都" in line.text]
        self.assertEqual(len(body), 1)
        self.assertIn("東京都", body[0])

    def test_ragged_rows_are_padded_to_header_count(self):
        """Invariant: rows with too few cells render blanks for the missing ones."""
        lines = render_markdown_table(["A", "B", "C"], [["1", "2"], ["3"]], max_width=40)
        self.assertEqual(
            [line.text for line in lines],
            [
                "┌─────┬─────┬─────┐",
                "│ A   │ B   │ C   │",
                "├─────┼─────┼─────┤",
                "│ 1   │ 2   │     │",
                "│ 3   │     │     │",
                "└─────┴─────┴─────┘",
            ],
        )

    def test_extra_cells_are_truncated_to_header_count(self):
        """Invariant: rows with too many cells drop the surplus columns."""
        lines = render_markdown_table(["A", "B"], [["1", "2", "3", "4"]], max_width=40)
        self.assertEqual(
            [line.text for line in lines],
            [
                "┌─────┬─────┐",
                "│ A   │ B   │",
                "├─────┼─────┤",
                "│ 1   │ 2   │",
                "└─────┴─────┘",
            ],
        )

    def test_wide_characters_truncate_on_character_boundary(self):
        """Invariant: truncating a wide-char cell never splits a character or overflows."""
        # Column is forced narrow so the long CJK cell must be truncated.
        lines = render_markdown_table(["H"], [["東京都渋谷区神南"]], max_width=12)
        self.assertTrue(lines)
        for line in lines:
            self.assertLessEqual(display_width(line.text), 12)
        body = [line.text for line in lines if "…" in line.text]
        self.assertTrue(body, "expected an elision marker on the truncated cell")
        # The truncated cell must keep whole characters and an intact marker.
        cell = body[0].strip("│").strip()
        self.assertTrue(cell.endswith("…"), f"cell should end with an elision marker: {cell!r}")
        for ch in cell[:-1]:
            self.assertIn(ch, "東京都渋谷区神南", "truncation split a character")

    def test_ascii_truncation_adds_elision_marker(self):
        """Invariant: over-long ASCII cells are truncated with a visible marker."""
        lines = render_markdown_table(["H"], [["abcdefghijklmnop"]], max_width=12)
        body = [line.text for line in lines if "…" in line.text]
        self.assertEqual(len(body), 1)
        self.assertLessEqual(display_width(body[0]), 12)

    def test_no_columns_returns_no_lines(self):
        """Invariant: an empty header list renders nothing rather than a broken box."""
        self.assertEqual(render_markdown_table([], [["a"]], max_width=40), [])


class TestTUIInterrupt(unittest.TestCase):
    def test_ctrl_c_during_turn_is_handled_without_crashing(self):
        """Invariant: Ctrl+C mid-turn aborts the turn and keeps the session alive.

        Regression guard: the TUI reimplements the step loop instead of reusing
        _execute_steps, so it must handle KeyboardInterrupt itself. It is not an
        Exception subclass and would otherwise escape through curses.wrapper,
        leaving the terminal in an unknown state.
        """
        import tempfile
        from pathlib import Path

        from core.harness import Harness
        from core.model import MockModel

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            harness = Harness(
                model=MockModel(script=[("thinking", [])]),
                workspace_dir=tmp_path / "ws",
                global_dir=tmp_path / "global",
                runs_dir=tmp_path / "runs",
                sandbox=False,
                verbose=False,
            )

            steps = {"n": 0}

            def boom_step(self, step_index):
                steps["n"] += 1
                raise KeyboardInterrupt

            captured = {}

            def mock_tui_run(tui_instance, on_user_submit, initial_instruction=None):
                captured["state"] = tui_instance.state
                on_user_submit("do the thing")

            with patch("core.loop.AgentLoop.step", boom_step):
                with patch("core.tui.CursesTUI.run", side_effect=mock_tui_run, autospec=True):
                    try:
                        harness.run_tui()
                    except KeyboardInterrupt:
                        # An escaping Ctrl+C aborts the whole pytest session, so convert
                        # it into an ordinary failure to keep the rest of the run intact.
                        self.fail(
                            "KeyboardInterrupt escaped Harness.run_tui; the TUI turn loop "
                            "must catch it instead of unwinding through curses.wrapper"
                        )

            self.assertEqual(steps["n"], 1, "turn should abort on the first Ctrl+C")
            lines = [line.text for line in captured["state"].lines]
            self.assertTrue(
                any("interrupted by user" in text for text in lines),
                f"expected an interruption notice, got {lines}",
            )

    def test_ctrl_c_during_tool_execution_repairs_dangling_tool_calls(self):
        """Invariant: interrupting during tool dispatch leaves no unmatched tool_call.

        Reproduces the real hazard: loop.step records the assistant message (with
        tool_calls) before dispatching, so a Ctrl+C inside dispatch would otherwise
        leave a tool_call with no tool message, which chat APIs reject with a 400 --
        i.e. the user presses Ctrl+C to recover and the session stays broken.
        """
        import tempfile
        from pathlib import Path

        from core.harness import Harness
        from core.model import MockModel
        from core.types import ToolCall

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            harness = Harness(
                model=MockModel(
                    script=[
                        (
                            "running a command",
                            [ToolCall(id="c1", name="bash", arguments={"command": "ls"})],
                        )
                    ]
                ),
                workspace_dir=tmp_path / "ws",
                global_dir=tmp_path / "global",
                runs_dir=tmp_path / "runs",
                sandbox=False,
                verbose=False,
            )

            captured = {}

            def mock_tui_run(tui_instance, on_user_submit, initial_instruction=None):
                captured["state"] = tui_instance.state
                on_user_submit("run ls")

            def boom_dispatch(self, tool_call, step_index):
                raise KeyboardInterrupt

            with patch("core.tools.ToolRegistry.dispatch", boom_dispatch):
                with patch("core.tui.CursesTUI.run", side_effect=mock_tui_run, autospec=True):
                    try:
                        harness.run_tui()
                    except KeyboardInterrupt:
                        self.fail("KeyboardInterrupt escaped Harness.run_tui")

            lines = [line.text for line in captured["state"].lines]
            self.assertTrue(
                any("interrupted by user" in text for text in lines),
                f"expected an interruption notice, got {lines}",
            )
            self.assertTrue(
                any("Dropped 1 incomplete message" in text for text in lines),
                f"the dangling tool_call should have been trimmed, got {lines}",
            )

    def test_esc_during_turn_cancels_turn_without_crashing_or_exiting_session(self):
        """Invariant: Esc mid-turn cancels the turn cleanly, drops dangling messages, and keeps the session active."""
        import tempfile
        from pathlib import Path

        from core.harness import Harness
        from core.model import MockModel
        from core.types import ToolCall

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            harness = Harness(
                model=MockModel(
                    script=[
                        (
                            "step 1 output",
                            [ToolCall(id="c1", name="bash", arguments={"command": "echo step1"})],
                        ),
                        (
                            "step 2 output",
                            [ToolCall(id="c2", name="bash", arguments={"command": "echo step2"})],
                        ),
                    ]
                ),
                workspace_dir=tmp_path / "ws",
                global_dir=tmp_path / "global",
                runs_dir=tmp_path / "runs",
                sandbox=False,
                verbose=False,
            )

            captured = {}
            call_count = {"n": 0}

            def mock_check_cancel(tui_self):
                call_count["n"] += 1
                return call_count["n"] >= 2

            def mock_tui_run(tui_instance, on_user_submit, initial_instruction=None):
                captured["state"] = tui_instance.state
                captured["tui"] = tui_instance
                on_user_submit("run two steps")

            with patch("core.tui.CursesTUI.check_cancel_requested", mock_check_cancel):
                with patch("core.tui.CursesTUI.run", side_effect=mock_tui_run, autospec=True):
                    harness.run_tui()

            lines = [line.text for line in captured["state"].lines]
            self.assertTrue(
                any("Turn cancelled by user (Esc)" in text for text in lines),
                f"expected a cancellation notice, got {lines}",
            )
            self.assertFalse(captured["state"].turn_active)

    def test_read_pending_keeps_the_window_timeout(self):
        """Invariant: polling for input hands the read timeout back, so the screen keeps repainting."""
        import curses

        # 1. A window with a 50 ms timeout gets it back after the poll
        window = MagicMock()
        window.gettimeout.return_value = 50
        window.get_wch.side_effect = curses.error("no input")
        self.assertEqual(_read_pending(window), [])
        window.timeout.assert_called_once_with(50)
        window.nodelay.assert_called_once_with(True)

        # 2. A window with no timeout is left blocking again
        window = MagicMock()
        window.gettimeout.return_value = -1
        window.get_wch.side_effect = curses.error("no input")
        self.assertEqual(_read_pending(window), [])
        window.nodelay.assert_any_call(False)

        # 3. The characters that were pending are returned
        window = MagicMock()
        window.gettimeout.return_value = 50
        window.get_wch.side_effect = ["\x1b", "[", "A", curses.error("end")]
        self.assertEqual(_read_pending(window), ["\x1b", "[", "A"])
        window.timeout.assert_called_once_with(50)

    def test_check_cancel_requested_variants(self):
        """Invariant: the cancel flag is consumed once and the curses window is never read off-thread."""
        import curses

        state = TUIState()
        tui = CursesTUI(state)

        # 1. No cancel requested -> False
        self.assertFalse(tui.check_cancel_requested())

        # 2. A pending cancel is reported once and then consumed
        tui._cancel_requested = True
        self.assertTrue(tui.check_cancel_requested())
        self.assertFalse(tui.check_cancel_requested())

        # 3. The window is never read: the main thread owns the keyboard
        mock_scr = MagicMock()
        mock_scr.get_wch.side_effect = curses.error("no input")
        tui.stdscr = mock_scr
        self.assertFalse(tui.check_cancel_requested())
        mock_scr.get_wch.assert_not_called()

        # 4. Esc during a turn sets the flag the worker thread polls
        state.turn_active = True
        self.assertTrue(tui.handle_key(27, lambda _t: None))
        self.assertTrue(tui.check_cancel_requested())
        self.assertFalse(tui._cancel_requested)

    def test_status_bar_displays_running_indicator_when_turn_active(self):
        """Invariant: status bar states Esc's real timing and points at Ctrl+C for immediate abort."""
        state = TUIState()
        state.turn_active = True
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 120)
        tui = CursesTUI(state)
        tui.draw(mock_stdscr)

        calls = mock_stdscr.addstr.call_args_list
        rendered_texts = [c.args[2] for c in calls if len(c.args) >= 3 and isinstance(c.args[2], str)]
        joined = " ".join(rendered_texts)
        self.assertIn("running", joined)
        # Esc is deferred to the next step boundary, so the hint must not claim otherwise.
        self.assertIn("Esc cancels after current step", joined)
        self.assertIn("Ctrl+C cancels now", joined)
        self.assertNotIn("Esc to cancel]", joined)

    def test_status_bar_starts_with_spinner_frame_and_running_when_turn_active(self):
        """Invariant: the first status field is the spinner frame followed by the word running."""
        state = TUIState()
        state.turn_active = True
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 120)
        clock = [0.0]
        with patch("core.tui.time.monotonic", side_effect=lambda: clock[0]):
            tui = CursesTUI(state)
            tui.draw(mock_stdscr)

        status = status_bar_text(mock_stdscr)
        # 1. The spinner leads the bar so a narrow terminal cannot truncate it away.
        self.assertTrue(status.startswith(f" {SPINNER_FRAMES[0]} running "))
        # 2. The cancel hint keeps both timings and drops the duplicated word.
        self.assertIn("[Esc cancels after current step · Ctrl+C cancels now]", status)
        self.assertNotIn("[running ·", status)

    def test_status_bar_has_no_spinner_when_no_turn_active(self):
        """Invariant: an idle status bar shows no spinner frame and is unchanged otherwise."""
        state = TUIState()
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 120)
        tui = CursesTUI(state)
        tui.draw(mock_stdscr)

        status = status_bar_text(mock_stdscr)
        # 1. No spinner glyph leaks into the idle bar.
        for frame in SPINNER_FRAMES:
            self.assertNotIn(frame, status)
        # 2. The idle bar is exactly what it was before the spinner existed.
        self.assertEqual(status, " context: 0 / 0 (0.0%) ")

    def test_advance_spinner_steps_frame_per_interval_and_wraps(self):
        """Invariant: one frame per whole interval, remainder kept, last frame wraps to first."""
        state = TUIState()
        # 1. Less than a full interval leaves the frame alone.
        state.advance_spinner(SPINNER_INTERVAL / 2)
        self.assertEqual(state.spinner_frame, 0)
        self.assertAlmostEqual(state.spinner_elapsed, SPINNER_INTERVAL / 2)
        # 2. The saved remainder completes an interval and advances exactly one frame.
        state.advance_spinner(SPINNER_INTERVAL / 2)
        self.assertEqual(state.spinner_frame, 1)
        self.assertAlmostEqual(state.spinner_elapsed, 0.0)
        # 3. An interval and a half advances one frame and keeps the extra half.
        state.advance_spinner(SPINNER_INTERVAL * 1.5)
        self.assertEqual(state.spinner_frame, 2)
        self.assertAlmostEqual(state.spinner_elapsed, SPINNER_INTERVAL / 2)
        # 4. The frame wraps back to the first frame after the last one.
        state.spinner_frame = len(SPINNER_FRAMES) - 1
        state.advance_spinner(SPINNER_INTERVAL)
        self.assertEqual(state.spinner_frame, 0)

    def test_spinner_does_not_advance_while_no_turn_is_active(self):
        """Invariant: idle draws never move the spinner frame, however long the idle gap."""
        state = TUIState()
        state.turn_active = False
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 120)
        clock = [0.0]
        with patch("core.tui.time.monotonic", side_effect=lambda: clock[0]):
            tui = CursesTUI(state)
            # 1. A long idle gap must not accumulate animation time.
            clock[0] += 5.0
            tui.draw(mock_stdscr)
            tui.draw(mock_stdscr)
        self.assertEqual(state.spinner_frame, 0)
        self.assertEqual(state.spinner_elapsed, 0.0)

    def test_spinner_advances_between_draws_while_turn_is_active(self):
        """Invariant: consecutive draws during a turn advance the frame and repaint the new glyph."""
        state = TUIState()
        state.turn_active = True
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 120)
        clock = [0.0]
        with patch("core.tui.time.monotonic", side_effect=lambda: clock[0]):
            tui = CursesTUI(state)
            tui.draw(mock_stdscr)
            first_frame = state.spinner_frame
            clock[0] += SPINNER_INTERVAL
            tui.draw(mock_stdscr)
        # 1. The frame index moved forward.
        self.assertNotEqual(state.spinner_frame, first_frame)
        # 2. The repainted bar shows the new frame glyph.
        self.assertIn(SPINNER_FRAMES[state.spinner_frame], status_bar_text(mock_stdscr))

    def test_escape_key_during_turn_active_keeps_session_alive(self):
        """Invariant: Esc during turn_active does not terminate the session."""
        state = TUIState()
        state.turn_active = True
        tui = CursesTUI(state)
        self.assertTrue(tui.handle_key(27, lambda _t: None))

    def test_initial_instruction_is_submitted_after_first_draw(self):
        """Invariant: run() draws a frame before submitting initial_instruction.

        Regression guard: the initial instruction used to be submitted before
        tui.run(), while stdscr was still None, so the entire first turn ran with
        no UI at all -- indistinguishable from a hang.
        """
        import curses as _curses

        state = TUIState(title="t")
        tui = CursesTUI(state)
        events = []

        class FakeScr:
            def erase(self):
                pass

            def getmaxyx(self):
                return (24, 80)

            def keypad(self, _v):
                pass

            def nodelay(self, _v):
                pass

            def get_wch(self):
                raise EOFError

        def draw_spy(scr):
            events.append("draw")

        def submit(text):
            events.append(f"submit:{text}")

        with patch.object(tui, "draw", draw_spy):
            with (
                patch.object(_curses, "wrapper", lambda fn: fn(FakeScr())),
                patch.object(_curses, "cbreak", lambda: None),
                patch.object(_curses, "curs_set", lambda _v: None),
                patch.object(tui, "_init_colors", lambda: None),
                patch("sys.stdout"),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                tui.run(on_user_submit=submit, initial_instruction="hello there")

        self.assertIn("submit:hello there", events)
        self.assertLess(
            events.index("draw"),
            events.index("submit:hello there"),
            "initial instruction must be submitted only after a frame is drawn",
        )


class TestTUI(unittest.TestCase):
    def test_tui_state_initialization_and_status(self):
        """Invariant: TUIState initializes with defaults and updates token metrics."""
        state = TUIState(title="test_harness")
        self.assertEqual(state.title, "test_harness")
        self.assertEqual(len(state.lines), 0)
        self.assertEqual(state.scroll_offset, 0)

        stats = {
            "estimated_tokens": 1250,
            "max_estimated_tokens": 8000,
            "usage_pct": 15.6,
            "total_messages": 4,
        }
        cache_stats = {"cache_hit_rate_pct": 82.5}
        state.update_status(stats, cache_stats)

        self.assertEqual(state.status_tokens, 1250)
        self.assertEqual(state.max_tokens, 8000)
        self.assertEqual(state.usage_pct, 15.6)
        self.assertEqual(state.cache_hit_pct, 82.5)
        self.assertEqual(state.messages_count, 4)

    def test_tui_state_add_messages_and_tags(self):
        """Invariant: TUIState appends properly tagged lines and resets scroll to bottom."""
        state = TUIState()
        state.scroll_up(10)
        self.assertEqual(state.scroll_offset, 10)

        state.add_user("What is microharness?")
        # Adding a message auto-scrolls to bottom (scroll_offset = 0)
        self.assertEqual(state.scroll_offset, 0)
        self.assertEqual(state.lines[-1], TUILine(text="> What is microharness?", tag="user"))

        state.add_thought("Analyzing codebase")
        self.assertEqual(state.lines[-1], TUILine(text="Thinking: Analyzing codebase", tag="thought"))

        state.add_action("bash: git status (0.2s)")
        self.assertEqual(state.lines[-1], TUILine(text="bash: git status (0.2s)", tag="action"))

        state.add_error("bash failed (exit 1)")
        self.assertEqual(state.lines[-1], TUILine(text="error: bash failed (exit 1)", tag="error"))

        state.add_agent("Microharness is an AI agent harness.\nMinimal and pluggable.")
        # Agent response splits lines with dim spacers
        agent_lines = [line.text for line in state.lines if line.tag == "agent"]
        self.assertEqual(agent_lines, ["Microharness is an AI agent harness.", "Minimal and pluggable."])

    def test_tui_state_wrapping_and_visible_slice(self):
        """Invariant: Lines wrap to width, and visible slice calculates correct window rows."""
        state = TUIState()
        state.add_user("Short message")
        # Line longer than 30 characters
        state.add_thought("This is a very long thinking line that definitely exceeds thirty columns in terminal")

        wrapped = state.get_wrapped_lines(width=30)
        # Verify wrapping occurred (at least 3 wrapped rows for the long line)
        self.assertGreaterEqual(len(wrapped), 3)

        # Slice when total lines < height
        slice_small = state.get_visible_slice(height=20, width=30)
        self.assertEqual(len(slice_small), len(wrapped))

        # Slice when total lines > height (window height = 2)
        slice_fixed = state.get_visible_slice(height=2, width=30)
        self.assertEqual(len(slice_fixed), 2)
        # Should be the bottom-most 2 lines by default
        self.assertEqual(slice_fixed, wrapped[-2:])

        # Scroll up by 1
        state.scroll_up(1)
        slice_scrolled = state.get_visible_slice(height=2, width=30)
        self.assertEqual(slice_scrolled, wrapped[-3:-1])

        # Scroll down back to bottom
        state.scroll_down(5)
        self.assertEqual(state.scroll_offset, 0)
        self.assertEqual(state.get_visible_slice(height=2, width=30), wrapped[-2:])

    def test_tui_input_buffer_and_backspace(self):
        """Invariant: Input buffer appends chars, deletes backspace, and clears on submit."""
        state = TUIState()
        state.handle_input_char("h")
        state.handle_input_char("i")
        self.assertEqual(state.input_buffer, "hi")

        state.handle_backspace()
        self.assertEqual(state.input_buffer, "h")

        state.handle_input_char("ello world")
        self.assertEqual(state.input_buffer, "hello world")

        cleared = state.clear_input()
        self.assertEqual(cleared, "hello world")
        self.assertEqual(state.input_buffer, "")

    def test_curses_tui_draw_with_mock_screen(self):
        """Invariant: CursesTUI.draw renders header, transcript, status, and input without error."""
        state = TUIState(title="microharness")
        state.provider_name = "openrouter"
        state.model_name = "deepseek"
        state.cache_ttl = "5M"
        state.reasoning = "MEDIUM"
        state.add_user("ping")
        state.add_action("pong")
        state.handle_input_char("test input")

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)

        tui = CursesTUI(state)
        # Should draw without throwing exceptions
        tui.draw(mock_stdscr)

        mock_stdscr.erase.assert_called_once()
        mock_stdscr.refresh.assert_called_once()
        # Verify prompt line added
        prompt_calls = [c for c in mock_stdscr.addstr.call_args_list if "> " in str(c)]
        self.assertGreaterEqual(len(prompt_calls), 1)

        # Verify bottom status bar includes model, cache, and reasoning
        status_calls = [c for c in mock_stdscr.addstr.call_args_list if "cache: 5M" in str(c)]
        self.assertGreaterEqual(len(status_calls), 1)
        status_str = str(status_calls[0])
        self.assertIn("openrouter/deepseek", status_str)
        self.assertIn("reasoning: MEDIUM", status_str)

    def test_cli_defaults_to_interactive(self):
        """Invariant: default CLI arguments parse without requiring flags."""
        from main import _build_arg_parser

        parser = _build_arg_parser()
        args = parser.parse_args([])
        self.assertFalse(args.batch)
        self.assertIsNone(args.instruction)

    def test_harness_run_tui_headless_flow(self):
        """Invariant: Harness.run_tui initializes TUIState, runs user callbacks, and teardown succeeds."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from core.harness import Harness
        from core.model import MockModel

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            ws_dir = tmp_path / "workspace"
            skill_dir = ws_dir / ".microharness" / "skills" / "deploy-tool"
            skill_dir.mkdir(parents=True)
            (skill_dir / "SKILL.md").write_text("# Deploy Instructions\nRun deployment.\n", encoding="utf-8")

            mock_model = MockModel(script=[("Deploy in progress", []), ("Hello from agent", [])])
            harness = Harness(
                model=mock_model,
                workspace_dir=ws_dir,
                global_dir=tmp_path / "global",
                runs_dir=tmp_path / "runs",
                sandbox=False,
            )

            captured_state = []

            def mock_tui_run(tui_instance, on_user_submit, initial_instruction=None):
                captured_state.append(tui_instance.state)
                if initial_instruction:
                    on_user_submit(initial_instruction)
                on_user_submit("/status")
                on_user_submit("/help")
                on_user_submit("/skill")
                on_user_submit("/skill deploy-tool please deploy")
                on_user_submit("Hi agent")

            with patch("core.tui.CursesTUI.run", side_effect=mock_tui_run, autospec=True):
                harness.run_tui()

            # Verify runs directory has run record and logs
            runs = list((tmp_path / "runs").iterdir())
            self.assertEqual(len(runs), 1)

            # Verify captured state lines
            self.assertEqual(len(captured_state), 1)
            lines = captured_state[0].lines
            # Confirm presence of system /status output, /help output, /skill list, /skill activation, user input, and agent response
            has_status = any("Context Tokens:" in line.text for line in lines)
            has_help = any("Available commands:" in line.text for line in lines)
            has_skill_list = any("deploy-tool" in line.text for line in lines)
            has_skill_activation = any("Activated skill: deploy-tool" in line.text for line in lines)
            has_user = any("> Hi agent" in line.text for line in lines)
            has_agent = any("Hello from agent" in line.text for line in lines)
            self.assertTrue(has_status)
            self.assertTrue(has_help)
            self.assertTrue(has_skill_list)
            self.assertTrue(has_skill_activation)
            self.assertTrue(has_user)
            self.assertTrue(has_agent)

    def test_compact_path_and_get_git_branch(self):
        """Invariant: compact_path contracts home dir and get_git_branch detects HEAD ref."""
        import tempfile
        from pathlib import Path

        from core.tui import compact_path, get_git_branch

        home = Path.home()
        self.assertEqual(compact_path(str(home)), "~")
        self.assertEqual(compact_path(str(home / "test" / "dir")), "~/test/dir")
        self.assertEqual(compact_path("/var/log"), "/var/log")
        self.assertEqual(compact_path(""), "")

        with tempfile.TemporaryDirectory() as tmp_dir:
            p = Path(tmp_dir)
            self.assertEqual(get_git_branch(p), "")

            git_dir = p / ".git"
            git_dir.mkdir()
            head_file = git_dir / "HEAD"
            head_file.write_text("ref: refs/heads/feature-tui\n", encoding="utf-8")
            self.assertEqual(get_git_branch(p), "feature-tui")

            head_file.write_text("a1b2c3d4e5f6\n", encoding="utf-8")
            self.assertEqual(get_git_branch(p), "a1b2c3d")

    def test_markdown_table_and_formatting(self):
        """Invariant: parse_markdown_to_tui_lines renders tables with box drawing, code fences, and headings."""
        from core.tui import parse_markdown_to_tui_lines

        md_text = (
            "# Agent Execution Report\n\n"
            "Summary table below:\n\n"
            "| Tool | Status | Duration |\n"
            "| :--- | :--- | :--- |\n"
            "| `bash` | passed | 0.2s |\n"
            "| `read_file` | passed | 0.1s |\n\n"
            "```python\n"
            "def hello():\n"
            "    return True\n"
            "```\n\n"
            "> Note: Workspace is clean\n"
            "- Item 1\n"
            "1. Step one\n"
            "---\n"
        )

        lines = parse_markdown_to_tui_lines(md_text, max_width=80)
        tags = [line.tag for line in lines]

        # Invariant 1: Heading is uppercase and tagged
        self.assertIn("AGENT EXECUTION REPORT", [entry.text for entry in lines if entry.tag == "heading"])

        # Invariant 2: Table has border, header, and data rows
        self.assertIn("table_border", tags)
        self.assertIn("table_header", tags)
        self.assertIn("table_row", tags)

        table_header = next(entry.text for entry in lines if entry.tag == "table_header")
        self.assertIn("Tool", table_header)
        self.assertIn("Status", table_header)
        self.assertIn("Duration", table_header)

        table_rows = [entry.text for entry in lines if entry.tag == "table_row"]
        self.assertEqual(len(table_rows), 2)
        self.assertIn("bash", table_rows[0])
        self.assertIn("read_file", table_rows[1])

        # Invariant 3: Code fence has borders and code tag
        self.assertIn("code_border", tags)
        self.assertIn("code", tags)
        code_lines = [entry.text for entry in lines if entry.tag == "code"]
        self.assertTrue(any("def hello():" in cl for cl in code_lines))

        # Invariant 4: Blockquote and bullets
        self.assertIn("blockquote", tags)
        self.assertTrue(any("• Item 1" in entry.text for entry in lines))
        self.assertTrue(any("Step one" in entry.text for entry in lines))

    def test_floating_command_menu_logic(self):
        """Invariant: TUIState filters matching commands, handles navigation, and completes command."""
        state = TUIState()
        # Input not starting with / -> no matches
        state.input_buffer = "hello"
        self.assertEqual(state.get_matching_commands(), [])

        # Input "/" -> every command the router supports, in catalog order
        state.input_buffer = "/"
        matches = state.get_matching_commands()
        self.assertEqual(
            [m[0] for m in matches],
            [spec.canonical for spec in COMMAND_SPECS],
        )
        self.assertEqual(len(matches), len(COMMAND_SPECS))

        # Input "/comp" -> matches /compact only ("/cache" also starts with /c)
        state.input_buffer = "/comp"
        c_matches = state.get_matching_commands()
        self.assertEqual(len(c_matches), 1)
        self.assertEqual(c_matches[0][0], "/compact")

        # Command completion
        completed = state.complete_selected_command()
        self.assertEqual(completed, "/compact")
        self.assertEqual(state.input_buffer, "/compact")

        # Custom workspace commands
        state.set_custom_commands(
            {"review": Command("review", "Review PR", "template", "project")}
        )
        state.input_buffer = "/rev"
        r_matches = state.get_matching_commands()
        self.assertEqual(len(r_matches), 1)
        self.assertEqual(r_matches[0][0], "/review")

        # Navigation wrap-around
        state.input_buffer = "/"
        all_matches = state.get_matching_commands()
        state.selected_command_idx = 0
        state.select_prev_command()
        self.assertEqual(state.selected_command_idx, len(all_matches) - 1)
        state.select_next_command()
        self.assertEqual(state.selected_command_idx, 0)

    def test_curses_draw_cockpit_and_floating_menu(self):
        """Invariant: CursesTUI renders path cockpit row and floating commands menu when matching."""
        state = TUIState(title="microharness")
        state.cwd = "~/Projects/microharness"
        state.git_branch = "feature-ui"
        state.input_buffer = "/co"

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)

        tui = CursesTUI(state)
        tui.draw(mock_stdscr)

        # 1. Path & Git branch in cockpit
        path_calls = [c for c in mock_stdscr.addstr.call_args_list if "~/Projects/microharness" in str(c)]
        self.assertGreaterEqual(len(path_calls), 1)
        self.assertIn("(feature-ui)", str(path_calls[0]))

        # 2. Floating menu border and suggestion
        menu_border_calls = [c for c in mock_stdscr.addstr.call_args_list if "Commands" in str(c)]
        self.assertGreaterEqual(len(menu_border_calls), 1)
        # Verify starts at col 0 (full screen width)
        self.assertEqual(menu_border_calls[0].args[1], 0)

        menu_item_calls = [c for c in mock_stdscr.addstr.call_args_list if "/compact" in str(c)]
        self.assertGreaterEqual(len(menu_item_calls), 1)

        # 3. Tip row in floating menu
        tip_calls = [c for c in mock_stdscr.addstr.call_args_list if "Navigate" in str(c)]
        self.assertGreaterEqual(len(tip_calls), 1)

    def test_curses_tui_handle_key_tab_and_navigation(self):
        """Invariant: handle_key supports str/int Tab autocompletion, cycling, Shift-Tab, and Right-Arrow."""
        import curses

        state = TUIState()
        tui = CursesTUI(state)
        submitted: list[str] = []

        def dummy_submit(text: str) -> None:
            submitted.append(text)

        # 1. Single-match Tab autocompletion (str "\t")
        state.input_buffer = "/comp"
        state.selected_command_idx = 0
        ret = tui.handle_key("\t", dummy_submit)
        self.assertTrue(ret)
        self.assertEqual(state.input_buffer, "/compact")

        # 2. Single-match Tab autocompletion (int 9)
        state.input_buffer = "/st"
        state.selected_command_idx = 0
        ret = tui.handle_key(9, dummy_submit)
        self.assertTrue(ret)
        self.assertEqual(state.input_buffer, "/status")

        # 3. Multi-match Tab cycling forward
        state.input_buffer = "/"
        state.selected_command_idx = 0
        tui.handle_key("\t", dummy_submit)
        self.assertEqual(state.selected_command_idx, 1)
        tui.handle_key(9, dummy_submit)
        self.assertEqual(state.selected_command_idx, 2)

        # 4. Shift-Tab and Up arrow cycling backward
        tui.handle_key(curses.KEY_BTAB, dummy_submit)
        self.assertEqual(state.selected_command_idx, 1)
        tui.handle_key(353, dummy_submit)  # Shift-Tab integer
        self.assertEqual(state.selected_command_idx, 0)
        tui.handle_key("\x1b[Z", dummy_submit)  # Shift-Tab ANSI escape sequence
        self.assertEqual(state.selected_command_idx, len(COMMAND_SPECS) - 1)
        tui.handle_key(curses.KEY_UP, dummy_submit)
        self.assertEqual(state.selected_command_idx, len(COMMAND_SPECS) - 2)

        # 5. Down arrow cycling forward, then wrapping past the last entry
        tui.handle_key(curses.KEY_DOWN, dummy_submit)
        self.assertEqual(state.selected_command_idx, len(COMMAND_SPECS) - 1)
        tui.handle_key(curses.KEY_DOWN, dummy_submit)
        self.assertEqual(state.selected_command_idx, 0)

        # 6. Right Arrow autocompletion into input buffer
        state.selected_command_idx = len(COMMAND_SPECS) - 1  # /help
        tui.handle_key(curses.KEY_RIGHT, dummy_submit)
        self.assertEqual(state.input_buffer, "/help")

        # 7. Enter key executes command and clears input
        ret = tui.handle_key("\n", dummy_submit)
        self.assertTrue(ret)
        self.assertEqual(submitted, ["/help"])
        self.assertEqual(state.input_buffer, "")

        # 8. Escape dismisses slash input
        state.input_buffer = "/st"
        tui.handle_key(27, dummy_submit)
        self.assertEqual(state.input_buffer, "")

        # 9. Negative test: Tab with no matches does not mutate input
        state.input_buffer = "normal message"
        tui.handle_key("\t", dummy_submit)
        self.assertEqual(state.input_buffer, "normal message")

        # 10. Exit command returns False
        state.input_buffer = "/exit"
        ret = tui.handle_key("\n", dummy_submit)
        self.assertFalse(ret)

    def test_cursor_movement_and_editing_at_cursor(self):
        """Invariant: Left/Right move the caret, and edits land at the caret instead of the buffer end."""
        import curses

        state = TUIState()
        tui = CursesTUI(state)
        submitted: list[str] = []

        def dummy_submit(text: str) -> None:
            submitted.append(text)

        # 1. Left and Right move the caret one character at a time
        state.input_buffer = "abc"
        state.cursor_pos = 3
        tui.handle_key(curses.KEY_LEFT, dummy_submit)
        self.assertEqual(state.cursor_pos, 2)
        tui.handle_key(curses.KEY_RIGHT, dummy_submit)
        self.assertEqual(state.cursor_pos, 3)

        # 2. Left clamps at the start of the buffer
        state.cursor_pos = 0
        tui.handle_key(curses.KEY_LEFT, dummy_submit)
        self.assertEqual(state.cursor_pos, 0)

        # 3. Right clamps at the end of the buffer
        state.cursor_pos = len(state.input_buffer)
        tui.handle_key(curses.KEY_RIGHT, dummy_submit)
        self.assertEqual(state.cursor_pos, 3)

        # 4. Typed text lands at the moved caret, not at the end of the buffer
        state.input_buffer = "abc"
        state.cursor_pos = 3
        tui.handle_key(curses.KEY_LEFT, dummy_submit)
        tui.handle_key("X", dummy_submit)
        self.assertEqual(state.input_buffer, "abXc")
        self.assertEqual(state.cursor_pos, 3)

        # 5. Backspace deletes the character before the moved caret
        state.input_buffer = "abc"
        state.cursor_pos = 3
        tui.handle_key(curses.KEY_LEFT, dummy_submit)
        tui.handle_key(curses.KEY_LEFT, dummy_submit)
        tui.handle_key(curses.KEY_BACKSPACE, dummy_submit)
        self.assertEqual(state.input_buffer, "bc")
        self.assertEqual(state.cursor_pos, 0)

        # 6. Home and End move within the caret's own line
        state.input_buffer = "abc\ndef"
        state.cursor_pos = 5
        tui.handle_key(curses.KEY_HOME, dummy_submit)
        self.assertEqual(state.cursor_pos, 4)
        tui.handle_key(curses.KEY_END, dummy_submit)
        self.assertEqual(state.cursor_pos, 7)

        # 7. Ctrl+A and Ctrl+E do the same as Home and End
        state.input_buffer = "abc\ndef"
        state.cursor_pos = 5
        tui.handle_key(1, dummy_submit)
        self.assertEqual(state.cursor_pos, 4)
        tui.handle_key(5, dummy_submit)
        self.assertEqual(state.cursor_pos, 7)

        # 8. Delete removes the character under the caret and leaves it in place
        state.input_buffer = "abc"
        state.cursor_pos = 1
        tui.handle_key(curses.KEY_DC, dummy_submit)
        self.assertEqual(state.input_buffer, "ac")
        self.assertEqual(state.cursor_pos, 1)

        # 9. Negative: Delete at the end of the buffer changes nothing
        state.input_buffer = "abc"
        state.cursor_pos = 3
        tui.handle_key(curses.KEY_DC, dummy_submit)
        self.assertEqual(state.input_buffer, "abc")
        self.assertEqual(state.cursor_pos, 3)

        # 10. Negative: Backspace at the start of the buffer changes nothing
        state.input_buffer = "abc"
        state.cursor_pos = 0
        tui.handle_key(curses.KEY_BACKSPACE, dummy_submit)
        self.assertEqual(state.input_buffer, "abc")
        self.assertEqual(state.cursor_pos, 0)

        # 11. Moving the caret never submits the input
        self.assertEqual(submitted, [])

    def test_cursor_keys_with_open_menu_and_raw_escape_sequences(self):
        """Invariant: Right completes an open menu, Left moves the caret, and raw arrows act like their key codes."""
        import curses

        state = TUIState()
        state.set_custom_commands({"alpha": Command("alpha", "does alpha", "t", "project")})
        tui = CursesTUI(state)

        def dummy_submit(text: str) -> None:
            return None

        # 1. Right arrow with an open command menu completes the selected entry
        state.input_buffer = "/al"
        state.cursor_pos = 3
        self.assertEqual(len(state.get_matching_commands()), 1)
        tui.handle_key(curses.KEY_RIGHT, dummy_submit)
        self.assertEqual(state.input_buffer, "/alpha")

        # 2. Left arrow with an open menu moves the caret instead of completing
        state.input_buffer = "/al"
        state.cursor_pos = 3
        tui.handle_key(curses.KEY_LEFT, dummy_submit)
        self.assertEqual(state.cursor_pos, 2)
        self.assertEqual(state.input_buffer, "/al")

        # 3. Raw left and right sequences move the caret like the key codes do
        state.input_buffer = "abc"
        state.cursor_pos = 3
        tui.handle_key("\x1b[D", dummy_submit)
        self.assertEqual(state.cursor_pos, 2)
        tui.handle_key("\x1b[C", dummy_submit)
        self.assertEqual(state.cursor_pos, 3)

        # 4. A raw right sequence with an open menu completes the entry, like KEY_RIGHT
        state.input_buffer = "/al"
        state.cursor_pos = 3
        tui.handle_key("\x1b[C", dummy_submit)
        self.assertEqual(state.input_buffer, "/alpha")

        # 5. Raw Home and End sequences move the caret to the ends of its line
        state.input_buffer = "abc\ndef"
        state.cursor_pos = 5
        tui.handle_key("\x1b[7~", dummy_submit)
        self.assertEqual(state.cursor_pos, 4)
        tui.handle_key("\x1b[8~", dummy_submit)
        self.assertEqual(state.cursor_pos, 7)

        # 6. Negative: an unknown escape sequence moves nothing and inserts nothing
        state.input_buffer = "abc"
        state.cursor_pos = 3
        tui.handle_key("\x1b[Q", dummy_submit)
        self.assertEqual(state.cursor_pos, 3)
        self.assertEqual(state.input_buffer, "abc")

        # 7. Negative: a bare Escape is not read as an arrow and still clears the input
        state.input_buffer = "abc"
        state.cursor_pos = 3
        tui.handle_key("\x1b", dummy_submit)
        self.assertEqual(state.input_buffer, "")

        # 8. Negative: the Shift+Enter sequences still insert a newline
        state.input_buffer = "ab"
        state.cursor_pos = 2
        tui.handle_key("\x1b[13;2u", dummy_submit)
        self.assertEqual(state.input_buffer, "ab\n")
        self.assertEqual(state.cursor_pos, 3)

    def test_caret_stays_visible_after_moving_left(self):
        """Invariant: the character under the caret stays on screen once the caret moves left."""
        state = TUIState()
        tui = CursesTUI(state)
        buffer = "0123456789" * 10
        state.input_buffer = buffer
        state.cursor_pos = 40

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 40)
        tui.draw(mock_stdscr)

        # 1. The input row is drawn from the window offset that reveals the caret
        drawn = [c.args[2] for c in mock_stdscr.addstr.call_args_list if len(c.args) >= 3 and c.args[1] == 2]
        self.assertEqual(len(drawn), 1)
        self.assertEqual(drawn[0], buffer[5:41])
        self.assertEqual(drawn[0][35], buffer[40])

        # 2. The cursor is placed on the column where the caret's character is drawn
        mock_stdscr.move.assert_called_once_with(23, 37)

        # 3. With the caret at the end of the line the tail is still what gets drawn
        state.cursor_pos = len(buffer)
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 40)
        tui.draw(mock_stdscr)

        drawn = [c.args[2] for c in mock_stdscr.addstr.call_args_list if len(c.args) >= 3 and c.args[1] == 2]
        self.assertEqual(len(drawn), 1)
        self.assertEqual(drawn[0], buffer[-36:])
        mock_stdscr.move.assert_called_once_with(23, 38)

    def test_vertical_cursor_movement_between_lines(self):
        """Invariant: Up and Down move the caret between the lines of a multiline input."""
        state = TUIState()
        state.input_buffer = "abc\ndef\nghi"
        state.cursor_pos = 0

        # 1. Down walks down the lines, keeping the column
        state.move_cursor_down()
        self.assertEqual(state.cursor_pos, 4)
        state.move_cursor_down()
        self.assertEqual(state.cursor_pos, 8)

        # 2. Up walks back up the lines, keeping the column
        state.move_cursor_up()
        self.assertEqual(state.cursor_pos, 4)
        state.move_cursor_up()
        self.assertEqual(state.cursor_pos, 0)

        # 3. Negative: Up on the first line leaves the caret where it is
        state.move_cursor_up()
        self.assertEqual(state.cursor_pos, 0)

        # 4. Negative: Down on the last line leaves the caret where it is
        state.cursor_pos = len(state.input_buffer)
        state.move_cursor_down()
        self.assertEqual(state.cursor_pos, len(state.input_buffer))

        # 5. Negative: a single-line input has no line to move to
        state.input_buffer = "abc"
        state.cursor_pos = 1
        state.move_cursor_up()
        self.assertEqual(state.cursor_pos, 1)
        state.move_cursor_down()
        self.assertEqual(state.cursor_pos, 1)

    def test_vertical_cursor_movement_keeps_goal_column(self):
        """Invariant: vertical moves keep aiming at the column the caret left."""
        state = TUIState()
        state.input_buffer = "0123456789\nabcde\n0123456789"
        state.cursor_pos = 10

        # 1. Down from column 10 lands on the end of the 5-character line
        state.move_cursor_down()
        self.assertEqual(state.cursor_pos, 16)

        # 2. Up returns to column 10 of the first line
        state.move_cursor_up()
        self.assertEqual(state.cursor_pos, 10)

    def test_horizontal_move_resets_goal_column(self):
        """Invariant: a horizontal move makes the next vertical move aim at the new column."""
        state = TUIState()
        state.input_buffer = "0123456789\n0123456789\n0123456789"
        state.cursor_pos = 10

        # 1. Down remembers column 10 as the goal column
        state.move_cursor_down()
        self.assertEqual(state.cursor_pos, 21)

        # 2. Left moves the caret to column 9 and forgets the goal column
        state.move_cursor_left()
        self.assertEqual(state.cursor_pos, 20)

        # 3. The next Down aims at column 9, not at the remembered column 10
        state.move_cursor_down()
        self.assertEqual(state.cursor_pos, 31)

    def test_typing_resets_goal_column(self):
        """Invariant: typing at the caret makes the next vertical move aim at the new column."""
        state = TUIState()
        state.input_buffer = "0123456789\n0123456789\n0123456789"
        state.cursor_pos = 9

        # 1. Down remembers column 9 as the goal column
        state.move_cursor_down()
        self.assertEqual(state.cursor_pos, 20)

        # 2. Typing moves the caret to column 10 and forgets the goal column
        state.handle_input_char("X")
        self.assertEqual(state.cursor_pos, 21)
        self.assertEqual(state.input_buffer, "0123456789\n012345678X9\n0123456789")

        # 3. The next Down aims at column 10, not at the remembered column 9
        state.move_cursor_down()
        self.assertEqual(state.cursor_pos, 33)

    def test_truncating_the_input_resets_the_goal_column(self):
        """Invariant: cutting the input buffer forgets the column vertical moves aim at."""
        state = TUIState()
        state.input_buffer = "abc\n0123456789"
        state.cursor_pos = 14

        # 1. Up remembers column 10 as the goal column
        state.move_cursor_up()
        self.assertEqual(state.cursor_pos, 3)

        # 2. Truncating the buffer forgets the goal column
        state.truncate_input(3)
        self.assertEqual(state.input_buffer, "abc")
        self.assertEqual(state.cursor_pos, 3)

        # 3. The next vertical move aims at the caret's own column, not at column 10
        state.handle_newline()
        state.move_cursor_up()
        self.assertEqual(state.cursor_pos, 0)

    def test_raw_key_sequence_map_covers_both_cursor_key_encodings(self):
        """Invariant: raw arrow sequences map to the right key in both cursor key encodings."""
        import curses

        from core.tui import RAW_KEY_SEQUENCES

        # 1. Normal mode sequences
        self.assertEqual(RAW_KEY_SEQUENCES["\x1b[A"], curses.KEY_UP)
        self.assertEqual(RAW_KEY_SEQUENCES["\x1b[B"], curses.KEY_DOWN)
        self.assertEqual(RAW_KEY_SEQUENCES["\x1b[C"], curses.KEY_RIGHT)
        self.assertEqual(RAW_KEY_SEQUENCES["\x1b[D"], curses.KEY_LEFT)

        # 2. Application cursor mode sequences
        self.assertEqual(RAW_KEY_SEQUENCES["\x1bOA"], curses.KEY_UP)
        self.assertEqual(RAW_KEY_SEQUENCES["\x1bOB"], curses.KEY_DOWN)
        self.assertEqual(RAW_KEY_SEQUENCES["\x1bOC"], curses.KEY_RIGHT)
        self.assertEqual(RAW_KEY_SEQUENCES["\x1bOD"], curses.KEY_LEFT)

        # 3. Home, End, Delete and the shifted arrows
        self.assertEqual(RAW_KEY_SEQUENCES["\x1bOH"], curses.KEY_HOME)
        self.assertEqual(RAW_KEY_SEQUENCES["\x1bOF"], curses.KEY_END)
        self.assertEqual(RAW_KEY_SEQUENCES["\x1b[3~"], curses.KEY_DC)
        self.assertEqual(RAW_KEY_SEQUENCES["\x1b[1;2A"], curses.KEY_SR)
        self.assertEqual(RAW_KEY_SEQUENCES["\x1b[1;2B"], curses.KEY_SF)

    def test_raw_up_and_down_sequences_move_the_caret(self):
        """Invariant: raw up and down sequences move the caret between lines."""
        state = TUIState()
        tui = CursesTUI(state)
        state.input_buffer = "abc\ndef"
        state.cursor_pos = 7

        # 1. A raw up sequence moves the caret to the line above
        tui.handle_key("\x1b[A", lambda _t: None)
        self.assertEqual(state.cursor_pos, 3)

        # 2. A raw down sequence moves it back to the line below
        tui.handle_key("\x1b[B", lambda _t: None)
        self.assertEqual(state.cursor_pos, 7)

        # 3. The application cursor mode encodings do the same
        state.cursor_pos = 7
        tui.handle_key("\x1bOA", lambda _t: None)
        self.assertEqual(state.cursor_pos, 3)
        tui.handle_key("\x1bOB", lambda _t: None)
        self.assertEqual(state.cursor_pos, 7)

    def test_up_and_down_do_not_scroll_the_transcript(self):
        """Invariant: with no menu open, Up and Down move the caret and never scroll the transcript."""
        import curses

        state = TUIState()
        state.input_buffer = "abc\ndef"
        state.cursor_pos = 0
        tui = CursesTUI(state)

        # 1. Up on the first line moves nothing and scrolls nothing
        tui.handle_key(curses.KEY_UP, lambda _t: None)
        self.assertEqual(state.cursor_pos, 0)
        self.assertEqual(state.scroll_offset, 0)

        # 2. Down moves the caret and leaves the scroll offset alone
        tui.handle_key(curses.KEY_DOWN, lambda _t: None)
        self.assertEqual(state.cursor_pos, 4)
        self.assertEqual(state.scroll_offset, 0)

        # 3. Shift-Tab with no menu open scrolls nothing
        tui.handle_key(curses.KEY_BTAB, lambda _t: None)
        self.assertEqual(state.cursor_pos, 4)
        self.assertEqual(state.scroll_offset, 0)

    def test_shift_up_and_down_scroll_the_transcript_by_one_line(self):
        """Invariant: Shift+Up and Shift+Down scroll the transcript by exactly one line."""
        import curses

        state = TUIState()
        tui = CursesTUI(state)

        # 1. Shift+Up scrolls up one line
        tui.handle_key(curses.KEY_SR, lambda _t: None)
        self.assertEqual(state.scroll_offset, 1)

        # 2. Shift+Down scrolls back down one line
        tui.handle_key(curses.KEY_SF, lambda _t: None)
        self.assertEqual(state.scroll_offset, 0)

        # 3. The raw shift sequences scroll the same way
        tui.handle_key("\x1b[1;2A", lambda _t: None)
        self.assertEqual(state.scroll_offset, 1)
        tui.handle_key("\x1b[1;2B", lambda _t: None)
        self.assertEqual(state.scroll_offset, 0)

        # 4. Shift+Up scrolls even while a menu is open
        state.input_buffer = "/"
        tui.handle_key(curses.KEY_SR, lambda _t: None)
        self.assertEqual(state.scroll_offset, 1)
        self.assertEqual(state.selected_command_idx, 0)

    def test_up_and_down_navigate_an_open_menu_without_moving_the_caret(self):
        """Invariant: with a menu open, Up and Down select entries instead of moving the caret."""
        import curses

        state = TUIState()
        state.input_buffer = "/"
        state.cursor_pos = 1
        tui = CursesTUI(state)
        self.assertGreater(len(state.get_matching_commands()), 1)

        # 1. Up selects the previous entry and leaves the caret alone
        tui.handle_key(curses.KEY_UP, lambda _t: None)
        self.assertEqual(state.selected_command_idx, len(state.get_matching_commands()) - 1)
        self.assertEqual(state.cursor_pos, 1)

        # 2. Down selects the next entry and leaves the caret alone
        tui.handle_key(curses.KEY_DOWN, lambda _t: None)
        self.assertEqual(state.selected_command_idx, 0)
        self.assertEqual(state.cursor_pos, 1)

    def test_input_box_follows_the_caret_vertically(self):
        """Invariant: the input box scrolls so the caret's line stays visible."""
        import curses

        state = TUIState()
        state.input_buffer = "\n".join(f"line {i}" for i in range(1, 11))
        state.cursor_pos = len(state.input_buffer)
        tui = CursesTUI(state)

        def drawn_rows(mock_stdscr: MagicMock) -> list[tuple[int, str]]:
            return [
                (c.args[0], c.args[2])
                for c in mock_stdscr.addstr.call_args_list
                if len(c.args) >= 3 and c.args[1] == 2 and c.args[0] >= 18
            ]

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        tui.draw(mock_stdscr)

        # 1. With the caret on the last line the box shows the last six lines
        self.assertEqual(
            drawn_rows(mock_stdscr),
            [
                (18, "line 5"),
                (19, "line 6"),
                (20, "line 7"),
                (21, "line 8"),
                (22, "line 9"),
                (23, "line 10"),
            ],
        )
        mock_stdscr.move.assert_called_once_with(23, 9)

        # 2. Moving the caret up scrolls the box so its line stays visible
        for _ in range(5):
            tui.handle_key(curses.KEY_UP, lambda _t: None)

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        tui.draw(mock_stdscr)

        self.assertEqual(
            drawn_rows(mock_stdscr),
            [
                (18, "line 1"),
                (19, "line 2"),
                (20, "line 3"),
                (21, "line 4"),
                (22, "line 5"),
                (23, "line 6"),
            ],
        )
        mock_stdscr.move.assert_called_once_with(22, 8)

    def test_escape_never_exits_on_empty_prompt(self):
        """Invariant: Esc never terminates the session on an empty prompt; it clears input or dismisses menus."""
        state = TUIState(title="t")
        tui = CursesTUI(state)
        self.assertTrue(
            tui.handle_key(27, lambda _t: None),
            "Esc on an empty prompt must keep the session alive",
        )

        state = TUIState(title="t")
        state.input_buffer = "half typed"
        tui = CursesTUI(state)
        self.assertTrue(
            tui.handle_key(27, lambda _t: None), "Esc with text must not terminate"
        )
        self.assertEqual(state.input_buffer, "", "Esc with text should clear it instead")

    def test_escape_dismisses_menu_without_exiting(self):
        """Invariant: Esc closes an open command menu and keeps the session alive."""
        state = TUIState(title="t")
        state.set_custom_commands({"alpha": Command("alpha", "does alpha", "t", "project")})
        state.input_buffer = "/al"
        tui = CursesTUI(state)
        self.assertEqual(len(state.get_matching_commands()), 1)
        self.assertTrue(tui.handle_key(27, lambda _t: None))
        self.assertEqual(state.get_matching_commands(), [])

    def test_skill_subcommand_autocomplete_and_draw(self):
        """Invariant: /skill matches available skills under /skill <name> with navigation and drawing."""
        state = TUIState()
        skills = {
            name: Skill(name, summary, path, scope, instructions)
            for name, summary, path, scope, instructions in (
                ("deploy-tool", "Deploy app to production", "skills/deploy/SKILL.md",
                 "project", "deploy instructions"),
                ("my-linter", "Lint code with ruff", "skills/lint/SKILL.md",
                 "global", "lint instructions"),
            )
        }
        state.set_skills(skills)

        # 1. Matching /sk matches /skill command
        state.input_buffer = "/sk"
        m = state.get_matching_commands()
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0][0], "/skill")

        # 2. Matching /skill  with space matches all skills
        state.input_buffer = "/skill "
        m = state.get_matching_commands()
        self.assertEqual(len(m), 2)
        self.assertEqual([cmd for cmd, _ in m], ["/skill deploy-tool", "/skill my-linter"])
        self.assertEqual(m[0][1], "[project] Deploy app to production")
        self.assertEqual(m[1][1], "[global] Lint code with ruff")

        # 3. Matching with partial skill name
        state.input_buffer = "/skill my"
        m = state.get_matching_commands()
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0][0], "/skill my-linter")

        # 4. Tab autocompletion of single matching skill
        completed = state.complete_selected_command()
        self.assertEqual(completed, "/skill my-linter")
        self.assertEqual(state.input_buffer, "/skill my-linter")

        # 5. Curses drawing renders 'Available Skills' header
        state.input_buffer = "/skill "
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        tui = CursesTUI(state)
        tui.draw(mock_stdscr)

        skill_border_calls = [c for c in mock_stdscr.addstr.call_args_list if "Available Skills" in str(c)]
        self.assertGreaterEqual(len(skill_border_calls), 1)
        skill_item_calls = [c for c in mock_stdscr.addstr.call_args_list if "/skill deploy-tool" in str(c)]
        self.assertGreaterEqual(len(skill_item_calls), 1)

    def test_direct_skill_slash_command_autocomplete(self):
        """Invariant: Discovered skills are registered as direct /{skill} commands in the catalog and autocompleted."""
        state = TUIState()
        skills = {
            name: Skill(name, summary, path, scope, instructions)
            for name, summary, path, scope, instructions in (
                ("deploy-tool", "Deploy app to production", "skills/deploy/SKILL.md",
                 "project", "deploy instructions"),
                ("my-linter", "Lint code with ruff", "skills/lint/SKILL.md",
                 "global", "lint instructions"),
            )
        }
        state.set_skills(skills)

        # 1. Matching /dep matches direct skill /deploy-tool
        state.input_buffer = "/dep"
        m = state.get_matching_commands()
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0][0], "/deploy-tool")
        self.assertEqual(m[0][1], "Skill [project] Deploy app to production")

        # 2. Matching /my matches direct skill /my-linter
        state.input_buffer = "/my"
        m = state.get_matching_commands()
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0][0], "/my-linter")
        self.assertEqual(m[0][1], "Skill [global] Lint code with ruff")

        # 3. Tab completion completes direct skill name
        completed = state.complete_selected_command()
        self.assertEqual(completed, "/my-linter")
        self.assertEqual(state.input_buffer, "/my-linter")

        # 4. Custom command overrides skill if name collides
        state.set_custom_commands(
            {"deploy-tool": Command("deploy-tool", "Override deploy", "echo override", "workspace")}
        )
        state.input_buffer = "/dep"
        m = state.get_matching_commands()
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0][0], "/deploy-tool")
        self.assertEqual(m[0][1], "Custom command [workspace]")

    def test_reasoning_subcommand_autocomplete_and_draw(self):
        """Invariant: /reasoning offers levels popup menu and draws title."""
        state = TUIState()
        state.input_buffer = "/reasoning "
        matches = state.get_matching_commands()
        self.assertGreaterEqual(len(matches), 5)
        names = [cmd for cmd, _ in matches]
        self.assertIn("/reasoning off", names)
        self.assertIn("/reasoning low", names)
        self.assertIn("/reasoning medium", names)
        self.assertIn("/reasoning high", names)
        self.assertIn("/reasoning xhigh", names)
        self.assertIn("/reasoning max", names)

        # Curses drawing renders 'Reasoning / Thinking Effort' header
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        tui = CursesTUI(state)
        tui.draw(mock_stdscr)
        header_calls = [c for c in mock_stdscr.addstr.call_args_list if "Reasoning / Thinking Effort" in str(c)]
        self.assertGreaterEqual(len(header_calls), 1)

    def test_inline_markdown_parsing_and_styling(self):
        """Invariant: parse_inline_segments correctly classifies bold, italic, code, and preserves snake_case."""
        # 1. Empty or plain text
        self.assertEqual(parse_inline_segments(""), [("", "normal")])
        self.assertEqual(parse_inline_segments("plain text"), [("plain text", "normal")])

        # 2. Asterisks bold and underscores bold
        self.assertEqual(parse_inline_segments("**bold text**"), [("bold text", "bold")])
        self.assertEqual(parse_inline_segments("__bold text__"), [("bold text", "bold")])

        # 3. Asterisks italic and underscores italic
        self.assertEqual(parse_inline_segments("*italic text*"), [("italic text", "italic")])
        self.assertEqual(parse_inline_segments("_italic text_"), [("italic text", "italic")])

        # 4. Bold italic combinations
        self.assertEqual(parse_inline_segments("***bold italic***"), [("bold italic", "bold_italic")])
        self.assertEqual(parse_inline_segments("___bold italic___"), [("bold italic", "bold_italic")])

        # 5. Inline code
        self.assertEqual(parse_inline_segments("`code snippet`"), [("code snippet", "code")])

        # 6. Mixed text in a single line
        mixed = "Check **important** and *crucial* with `git status`!"
        segments = parse_inline_segments(mixed)
        expected = [
            ("Check ", "normal"),
            ("important", "bold"),
            (" and ", "normal"),
            ("crucial", "italic"),
            (" with ", "normal"),
            ("git status", "code"),
            ("!", "normal"),
        ]
        self.assertEqual(segments, expected)

        # 7. Preserve snake_case identifiers without treating underscores as italic
        ident = "use micro_harness_v2 and test_case_runner now"
        self.assertEqual(parse_inline_segments(ident), [(ident, "normal")])

    def test_tui_draw_renders_inline_markdown_with_curses_attributes(self):
        """Invariant: CursesTUI.draw applies curses bold and italic attributes and strips markdown markers."""
        import curses

        state = TUIState()
        state.add_agent("Result: **strong result** and *careful note*")

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        tui = CursesTUI(state)
        tui.draw(mock_stdscr)

        # Check call arguments of addstr
        calls = mock_stdscr.addstr.call_args_list

        # Verify raw ** and * markers are NOT in any text rendered
        rendered_texts = [c.args[2] for c in calls if len(c.args) >= 3 and isinstance(c.args[2], str)]
        for t in rendered_texts:
            self.assertNotIn("**", t)

        # Verify that bold text was rendered with curses.A_BOLD
        bold_calls = [
            c for c in calls
            if len(c.args) >= 4 and (c.args[3] & curses.A_BOLD) and c.args[2] == "strong result"
        ]
        self.assertGreaterEqual(len(bold_calls), 1)

        # Verify that italic text was rendered with italic attribute (A_ITALIC or fallback A_UNDERLINE)
        italic_attr = getattr(curses, "A_ITALIC", 0) or curses.A_UNDERLINE
        italic_calls = [
            c for c in calls
            if len(c.args) >= 4 and (c.args[3] & italic_attr) and c.args[2] == "careful note"
        ]
        self.assertGreaterEqual(len(italic_calls), 1)

    def test_file_tag_query_detection_and_matching(self):
        """Invariant: TUIState detects @file tag queries and matches workspace files with ranking."""
        state = TUIState()
        state.set_workspace_files([
            "core/harness.py",
            "core/tui.py",
            "tests/test_tui.py",
            "README.md",
        ])

        # 1. Detection at beginning
        state.input_buffer = "@"
        state.cursor_pos = 1
        self.assertEqual(state.get_current_file_tag_query(), (0, ""))

        state.input_buffer = "@co"
        state.cursor_pos = 3
        self.assertEqual(state.get_current_file_tag_query(), (0, "co"))

        # 2. Detection in sentence
        state.input_buffer = "Review @tui"
        state.cursor_pos = len(state.input_buffer)
        self.assertEqual(state.get_current_file_tag_query(), (7, "tui"))

        # 3. Negative detection: email address or plain text
        state.input_buffer = "email@test.com"
        state.cursor_pos = len(state.input_buffer)
        self.assertIsNone(state.get_current_file_tag_query())

        state.input_buffer = "plain text"
        state.cursor_pos = len(state.input_buffer)
        self.assertIsNone(state.get_current_file_tag_query())

        # 4. Matching file tags
        state.input_buffer = "Review @tui"
        state.cursor_pos = len(state.input_buffer)
        matches = state.get_matching_file_tags()
        self.assertEqual(len(matches), 2)
        # Matches core/tui.py and tests/test_tui.py
        matched_tags = [tag for tag, _ in matches]
        self.assertIn("@core/tui.py", matched_tags)
        self.assertIn("@tests/test_tui.py", matched_tags)

        # 5. Autocompleting chosen file tag in-place with trailing space
        state.selected_command_idx = 0
        completed = state.complete_selected_file_tag()
        self.assertEqual(completed, matches[0][0])
        self.assertEqual(state.input_buffer, f"Review {matches[0][0]} ")
        # After completion (trailing space), query is inactive
        self.assertIsNone(state.get_current_file_tag_query())

    def test_shift_enter_multiline_editing_and_submit(self):
        """Invariant: Shift+Enter and Alt+Enter insert newlines, while Enter submits full multiline text."""
        state = TUIState()
        tui = CursesTUI(state)
        submitted: list[str] = []

        def dummy_submit(text: str) -> None:
            submitted.append(text)

        # 1. Type first line
        state.handle_input_char("First line")
        self.assertEqual(state.input_buffer, "First line")

        # 2. Shift+Enter via CSI-u sequence \x1b[13;2u
        ret = tui.handle_key("\x1b[13;2u", dummy_submit)
        self.assertTrue(ret)
        self.assertEqual(submitted, [])
        self.assertEqual(state.input_buffer, "First line\n")

        # 3. Type second line
        state.handle_input_char("Second line")
        self.assertEqual(state.input_buffer, "First line\nSecond line")

        # 4. Alt/Option+Enter via \x1b\r
        ret = tui.handle_key("\x1b\r", dummy_submit)
        self.assertTrue(ret)
        self.assertEqual(submitted, [])
        self.assertEqual(state.input_buffer, "First line\nSecond line\n")

        # 5. Type third line
        state.handle_input_char("Third line")
        self.assertEqual(state.input_buffer, "First line\nSecond line\nThird line")

        # 6. Backspace removes across characters
        state.handle_backspace()
        self.assertEqual(state.input_buffer, "First line\nSecond line\nThird lin")

        # 7. Standard Enter submits complete multiline text and clears buffer
        ret = tui.handle_key("\n", dummy_submit)
        self.assertTrue(ret)
        self.assertEqual(submitted, ["First line\nSecond line\nThird lin"])
        self.assertEqual(state.input_buffer, "")

    def test_multiline_input_rendering_and_file_autocomplete_menu(self):
        """Invariant: CursesTUI.draw renders multiline input prompts and Workspace Files menu."""
        state = TUIState()
        state.set_workspace_files(["core/harness.py", "core/tui.py"])
        state.input_buffer = "Line 1\nLine 2 @core"
        state.cursor_pos = len(state.input_buffer)

        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (24, 80)
        tui = CursesTUI(state)
        tui.draw(mock_stdscr)

        calls = mock_stdscr.addstr.call_args_list

        # Verify multiline prompt prefix (> and continuation space)
        rendered_texts = [c.args[2] for c in calls if len(c.args) >= 3 and isinstance(c.args[2], str)]
        self.assertIn("> ", rendered_texts)
        self.assertIn("Line 1", rendered_texts)
        self.assertIn("Line 2 @core", rendered_texts)

        # Verify Workspace Files popup menu is rendered
        ws_menu_calls = [c for c in calls if "Workspace Files" in str(c)]
        self.assertGreaterEqual(len(ws_menu_calls), 1)

        # Verify file item @core/harness.py in menu
        item_calls = [c for c in calls if "@core/harness.py" in str(c)]
        self.assertGreaterEqual(len(item_calls), 1)

        # Escape key dismisses the file tag query
        tui.handle_key(27, lambda text: None)
        self.assertEqual(state.input_buffer, "Line 1\nLine 2 ")

    def test_shift_2_and_terminal_sequence_decoding_for_at_symbol(self):
        """Invariant: Shift+2 produces '@' via normal character or decoded escape sequences."""
        # 1. Direct decoder assertions
        self.assertEqual(decode_terminal_sequence("\x1b[27;2;50~"), "@")  # modifyOtherKeys Shift+2
        self.assertEqual(decode_terminal_sequence("\x1b[50;2u"), "@")     # CSI-u Shift+2
        self.assertEqual(decode_terminal_sequence("\x1b[64;2u"), "@")     # CSI-u codepoint 64 with shift
        self.assertEqual(decode_terminal_sequence("\x1b[64u"), "@")       # CSI-u codepoint 64
        self.assertEqual(decode_terminal_sequence("\x1b[27;2;49~"), "!")  # Shift+1
        self.assertEqual(decode_terminal_sequence("\x1b[27;2;51~"), "#")  # Shift+3
        self.assertIsNone(decode_terminal_sequence("\x1b[13;2u"))         # Enter is not character
        self.assertIsNone(decode_terminal_sequence("plain"))

        state = TUIState()
        state.set_workspace_files(["core/tui.py"])
        tui = CursesTUI(state)

        # 2. Standard character input "@"
        tui.handle_key("@", lambda text: None)
        self.assertEqual(state.input_buffer, "@")
        self.assertEqual(len(state.get_matching_file_tags()), 1)

        # 3. Encoded modifyOtherKeys Shift+2 sequence "\x1b[27;2;50~"
        state.clear_input()
        tui.handle_key("\x1b[27;2;50~", lambda text: None)
        self.assertEqual(state.input_buffer, "@")
        self.assertEqual(len(state.get_matching_file_tags()), 1)

        # 4. Encoded CSI-u Shift+2 sequence "\x1b[50;2u"
        state.clear_input()
        tui.handle_key("\x1b[50;2u", lambda text: None)
        self.assertEqual(state.input_buffer, "@")
        self.assertEqual(len(state.get_matching_file_tags()), 1)

    def test_model_and_resume_subcommand_autocomplete(self):
        """Invariant: /model and /resume provide subcommand autocomplete for models and previous runs."""
        state = TUIState()
        models = [
            {"name": "fast", "model": "gemini-2.5-flash", "provider": "google", "description": "Fast flash model"},
            {"name": "smart", "model": "claude-3-5-sonnet", "provider": "anthropic", "description": "Smart sonnet"},
        ]
        state.set_available_models(models)

        sessions = [
            {"run_id": "run-2026-10-03-alpha", "total_steps": 5, "preview": "Fix caching bug in harness"},
            {"run_id": "run-2026-10-02-beta", "total_steps": 12, "preview": "Refactor discovery"},
        ]
        state.set_available_sessions(sessions)

        # 1. /model with space matches all available models
        state.input_buffer = "/model "
        m = state.get_matching_commands()
        self.assertEqual(len(m), 2)
        self.assertEqual(m[0][0], "/model fast")
        self.assertEqual(m[0][1], "[google] Fast flash model")
        self.assertEqual(m[1][0], "/model smart")

        # 2. /model fa matches fast
        state.input_buffer = "/model fa"
        m_fa = state.get_matching_commands()
        self.assertEqual(len(m_fa), 1)
        self.assertEqual(m_fa[0][0], "/model fast")

        # 3. Tab completion on /model fa completes buffer to /model fast
        completed = state.complete_selected_command()
        self.assertEqual(completed, "/model fast")
        self.assertEqual(state.input_buffer, "/model fast")

        # 4. /resume with space matches all available sessions
        state.input_buffer = "/resume "
        s = state.get_matching_commands()
        self.assertEqual(len(s), 2)
        self.assertEqual(s[0][0], "/resume run-2026-10-03-alpha")
        self.assertEqual(s[0][1], "(5 steps) Fix caching bug in harness")

        # 5. /resume run-2026-10-02 matches beta
        state.input_buffer = "/resume run-2026-10-02"
        s_beta = state.get_matching_commands()
        self.assertEqual(len(s_beta), 1)
        self.assertEqual(s_beta[0][0], "/resume run-2026-10-02-beta")

        # 6. Tab completion completes session id
        completed_sess = state.complete_selected_command()
        self.assertEqual(completed_sess, "/resume run-2026-10-02-beta")
        self.assertEqual(state.input_buffer, "/resume run-2026-10-02-beta")

        # 7. /model without space also reveals models list
        state.input_buffer = "/model"
        m_bare = state.get_matching_commands()
        self.assertEqual(len(m_bare), 3)  # /model root + 2 models
        self.assertEqual(m_bare[0][0], "/model")
        self.assertEqual(m_bare[1][0], "/model fast")
        self.assertEqual(m_bare[2][0], "/model smart")

        # 8. /resume without space also reveals recent sessions
        state.input_buffer = "/resume"
        s_bare = state.get_matching_commands()
        self.assertEqual(len(s_bare), 3)  # /resume root + 2 sessions
        self.assertEqual(s_bare[0][0], "/resume")
        self.assertEqual(s_bare[1][0], "/resume run-2026-10-03-alpha")

        # 9. Tab on /m completes with trailing space directly into /model [space]
        tui = CursesTUI(state)
        submitted: list[str] = []
        state.input_buffer = "/m"
        state.selected_command_idx = 0
        tui.handle_key("\t", lambda text: submitted.append(text))
        self.assertEqual(state.input_buffer, "/model ")

        # 10. Enter on /model [space] selects and submits the highlighted model
        state.selected_command_idx = 0  # fast
        tui.handle_key("\n", lambda text: submitted.append(text))
        self.assertEqual(submitted, ["/model fast"])

        # 11. CursesTUI draw renders correct titles
        mock_stdscr = MagicMock()
        mock_stdscr.getmaxyx.return_value = (30, 100)
        state.input_buffer = "/model "
        tui.draw(mock_stdscr)
        model_border_calls = [c for c in mock_stdscr.addstr.call_args_list if "Available Models" in str(c)]
        self.assertGreaterEqual(len(model_border_calls), 1)

        state.input_buffer = "/resume "
        tui.draw(mock_stdscr)
        resume_border_calls = [c for c in mock_stdscr.addstr.call_args_list if "Previous Sessions" in str(c)]
        self.assertGreaterEqual(len(resume_border_calls), 1)

    def test_harness_run_tui_resume_continues_step_numbering(self):
        """Invariant: run_tui(resume=...) must continue step numbering past the resumed session."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from core.harness import Harness
        from core.model import MockModel
        from core.types import ToolCall

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            runs_dir = tmp_path / "runs"

            def build_harness() -> Harness:
                return Harness(
                    model=MockModel(script=[("working", []), ("done", [])]),
                    workspace_dir=tmp_path / "workspace",
                    global_dir=tmp_path / "global",
                    runs_dir=runs_dir,
                    sandbox=False,
                    verbose=False,
                )

            # Produce a prior session with exactly 2 recorded steps.
            seed = build_harness()
            _, _, _, tracer, loop, prior_run = seed._setup_run(
                task_id="prior",
                system_prompt="SYS",
                initial_instruction="prior work",
                initial_files={},
            )
            script = [("step one", [ToolCall(id="c1", name="write_file", arguments={"path": "a.txt", "content": "x"})]), ("step two", [])]
            seed.model = MockModel(script=script)
            loop.model = seed.model
            for idx in range(2):
                _, trace = loop.step(step_index=idx)
                tracer.record_step(trace)
            tracer.finalize(finish_reason="completed")
            self.assertEqual(len(tracer.steps), 2)

            # Resume it in TUI mode; the first step of the new turn must be step 3.
            harness = build_harness()
            observed_step_indices: list[int] = []
            real_step = None

            def mock_tui_run(tui_instance, on_user_submit, initial_instruction=None):
                on_user_submit("continue please")

            from core.loop import AgentLoop

            real_step = AgentLoop.step

            def recording_step(self, step_index=None, **kwargs):
                observed_step_indices.append(step_index)
                return real_step(self, step_index=step_index, **kwargs)

            with patch("core.tui.CursesTUI.run", side_effect=mock_tui_run, autospec=True), patch.object(
                AgentLoop, "step", recording_step
            ):
                harness.run_tui(resume=prior_run.name)

            self.assertTrue(observed_step_indices, "expected at least one step to run after resume")
            self.assertEqual(observed_step_indices[0], 3)
            self.assertEqual(observed_step_indices, sorted(set(observed_step_indices)))

    def test_harness_run_tui_announces_cancelling_and_survives(self):
        """Invariant: a cancel seen during a request announces 'Cancelling...' once and keeps the session."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from core.harness import Harness
        from core.model import CancellableModel, ModelCancelledError

        class CancellingModel:
            """Stands in for HttpChatModel: probes, reports a cancel, then raises."""

            model = "cancelling"
            provider = "test"

            def __init__(self) -> None:
                self.probe = None
                self.probe_calls = 0

            def set_cancel_probe(self, probe):
                self.probe = probe

            def generate(self, messages, tools):
                self.probe_calls += 1
                if self.probe is not None and self.probe():
                    raise ModelCancelledError("cancelled mid-request")
                return "done", []

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            model = CancellingModel()
            harness = Harness(
                model=model,
                workspace_dir=tmp / "ws",
                global_dir=tmp / "g",
                runs_dir=tmp / "r",
                sandbox=False,
                verbose=False,
            )
            self.assertIsInstance(model, CancellableModel)

            captured = []

            def mock_tui_run(tui_instance, on_user_submit, initial_instruction=None):
                captured.append(tui_instance.state)
                # Esc is "pressed" from the third poll onward: the probe must see it.
                on_user_submit("do the thing")
                on_user_submit("/status")

            polls = {"n": 0}

            def fake_cancel(self):
                # Per iteration the harness polls once at the loop boundary (odd
                # polls) and once from inside the request via the probe (even
                # polls). Report a cancel only on probe polls so the
                # mid-request path is what gets exercised.
                polls["n"] += 1
                return polls["n"] % 2 == 0

            with patch("core.tui.CursesTUI.run", side_effect=mock_tui_run, autospec=True), patch(
                "core.tui.CursesTUI.check_cancel_requested", fake_cancel
            ):
                harness.run_tui()

            self.assertTrue(captured)
            lines = [ln.text for ln in captured[0].lines]

            announcing = [t for t in lines if "Cancelling" in t]
            self.assertEqual(len(announcing), 1, f"expected exactly one announcement, got {announcing}")
            self.assertIn("Ctrl+C cancels immediately", announcing[0])

            # The turn was cancelled, not aborted: the session still responds afterwards.
            self.assertTrue(
                any("cancelled by user" in t for t in lines),
                f"expected a cancellation notice, got {lines[-6:]}",
            )
            self.assertTrue(any("Context Tokens:" in t for t in lines), "session should still be usable")
            self.assertFalse(captured[0].turn_active)
            # The probe must be detached once the turn ends.
            self.assertIsNone(model.probe)


if __name__ == "__main__":
    unittest.main()


class TestCommandCatalog(unittest.TestCase):
    """The autocomplete catalog and the router must not drift apart."""

    def test_catalog_covers_every_routable_command(self):
        """Invariant: the TUI offers exactly the commands the router can execute."""
        state = TUIState()
        state.input_buffer = "/"
        self.assertEqual(
            [name for name, _ in state.get_matching_commands()],
            [spec.canonical for spec in COMMAND_SPECS],
        )

    def test_catalog_survives_custom_command_and_skill_rebuilds(self):
        """Invariant: adding resources keeps the built-in entries and appends the new ones."""
        from core.commands import Command
        from core.skills import Skill

        state = TUIState()
        state.set_custom_commands(
            {"review": Command("review", "Review PR", "t", "project")}
        )
        state.set_skills(
            {"deploy": Skill("deploy", "Deploy app", "d/SKILL.md", "project", "body")}
        )

        names = [name for name, _ in state.command_catalog]
        self.assertEqual(names[: len(COMMAND_SPECS)], [s.canonical for s in COMMAND_SPECS])
        self.assertIn("/review", names)
        self.assertIn("/deploy", names)

    def test_skill_menu_shows_summary_and_scope_from_the_dataclass(self):
        """Invariant: the /skill menu reads Skill fields, so summaries are not lost."""
        from core.skills import Skill

        state = TUIState()
        state.set_skills(
            {
                "deploy-tool": Skill(
                    "deploy-tool", "Deploy app to production", "d/SKILL.md", "global", "body"
                )
            }
        )
        state.input_buffer = "/skill "
        matches = dict(state.get_matching_commands())

        self.assertEqual(
            matches["/skill deploy-tool"], "[global] Deploy app to production"
        )

    def test_model_menu_reuses_the_shared_description_formatter(self):
        """Invariant: the /model menu describes a model exactly like the /model listing."""
        from core.interactive import format_model_desc

        state = TUIState()
        state.set_available_models(
            [{"name": "fast", "model": "gemini", "provider": "google", "description": "Fast"}]
        )
        state.input_buffer = "/model fast"
        command, desc = state.get_matching_commands()[0]
        self.assertEqual(command, "/model fast")
        self.assertEqual(desc, format_model_desc(state.available_models[0]))

    def test_session_menu_truncates_long_previews(self):
        """Invariant: the /resume menu keeps previews inside its narrow column."""
        state = TUIState()
        state.set_available_sessions(
            [{"run_id": "abc", "total_steps": 3, "preview": "x" * 80}]
        )
        state.input_buffer = "/resume abc"
        _, desc = state.get_matching_commands()[0]
        self.assertEqual(desc, "(3 steps) " + "x" * 35)
