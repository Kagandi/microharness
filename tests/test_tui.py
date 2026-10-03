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

        # Input "/" -> all built-in commands
        state.input_buffer = "/"
        matches = state.get_matching_commands()
        self.assertEqual(len(matches), 5)
        self.assertEqual([m[0] for m in matches], ["/compact", "/status", "/new", "/help", "/exit"])

        # Input "/c" -> matches /compact
        state.input_buffer = "/c"
        c_matches = state.get_matching_commands()
        self.assertEqual(len(c_matches), 1)
        self.assertEqual(c_matches[0][0], "/compact")

        # Command completion
        completed = state.complete_selected_command()
        self.assertEqual(completed, "/compact")
        self.assertEqual(state.input_buffer, "/compact")

        # Custom workspace commands
        state.set_custom_commands({"review": ("template", "prompt", "project")})
        state.input_buffer = "/r"
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


if __name__ == "__main__":
    unittest.main()
