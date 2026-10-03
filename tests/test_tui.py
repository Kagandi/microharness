from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from core.tui import CursesTUI, TUILine, TUIState


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

    def test_cli_tui_flag_parsing(self):
        """Invariant: --tui flag is parsed properly in main CLI argument parser."""
        from main import _build_arg_parser

        parser = _build_arg_parser()
        args_with_tui = parser.parse_args(["--tui"])
        self.assertTrue(args_with_tui.tui)

        args_default = parser.parse_args([])
        self.assertFalse(args_default.tui)

    def test_harness_run_tui_headless_flow(self):
        """Invariant: Harness.run_tui initializes TUIState, runs user callbacks, and teardown succeeds."""
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        from core.harness import Harness
        from core.model import MockModel

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            mock_model = MockModel(script=[("Hello from agent", [])])
            harness = Harness(
                model=mock_model,
                workspace_dir=tmp_path / "workspace",
                global_dir=tmp_path / "global",
                runs_dir=tmp_path / "runs",
                sandbox=True,
            )

            captured_state = []

            def mock_tui_run(tui_instance, on_user_submit):
                captured_state.append(tui_instance.state)
                on_user_submit("/status")
                on_user_submit("/help")
                on_user_submit("Hi agent")

            with patch("core.tui.CursesTUI.run", side_effect=mock_tui_run, autospec=True):
                harness.run_tui()

            # Verify runs directory has run record and logs
            runs = list((tmp_path / "runs").iterdir())
            self.assertEqual(len(runs), 1)

            # Verify captured state lines
            self.assertEqual(len(captured_state), 1)
            lines = captured_state[0].lines
            # Confirm presence of system /status output, /help output, user input, and agent response
            has_status = any("Context Tokens:" in line.text for line in lines)
            has_help = any("Available commands:" in line.text for line in lines)
            has_user = any("> Hi agent" in line.text for line in lines)
            has_agent = any("Hello from agent" in line.text for line in lines)
            self.assertTrue(has_status)
            self.assertTrue(has_help)
            self.assertTrue(has_user)
            self.assertTrue(has_agent)


if __name__ == "__main__":
    unittest.main()
