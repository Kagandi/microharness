from __future__ import annotations

import unittest
from unittest.mock import MagicMock

from core.tui import (
    CursesTUI,
    TUILine,
    TUIState,
    decode_terminal_sequence,
    parse_inline_segments,
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

            def mock_tui_run(tui_instance, on_user_submit):
                captured_state.append(tui_instance.state)
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

        # Input "/" -> all built-in commands
        state.input_buffer = "/"
        matches = state.get_matching_commands()
        self.assertEqual(len(matches), 6)
        self.assertEqual([m[0] for m in matches], ["/compact", "/status", "/new", "/skill", "/help", "/exit"])

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

    def test_curses_tui_handle_key_tab_and_navigation(self):
        """Invariant: handle_key supports str/int Tab autocompletion, cycling, Shift-Tab, and Right-Arrow."""
        import curses

        state = TUIState()
        tui = CursesTUI(state)
        submitted: list[str] = []

        def dummy_submit(text: str) -> None:
            submitted.append(text)

        # 1. Single-match Tab autocompletion (str "\t")
        state.input_buffer = "/c"
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
        self.assertEqual(state.selected_command_idx, 5)
        tui.handle_key(curses.KEY_UP, dummy_submit)
        self.assertEqual(state.selected_command_idx, 4)

        # 5. Down arrow cycling forward
        tui.handle_key(curses.KEY_DOWN, dummy_submit)
        self.assertEqual(state.selected_command_idx, 5)
        tui.handle_key(curses.KEY_DOWN, dummy_submit)
        self.assertEqual(state.selected_command_idx, 0)

        # 6. Right Arrow autocompletion into input buffer
        state.selected_command_idx = 4  # /help
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

    def test_skill_subcommand_autocomplete_and_draw(self):
        """Invariant: /skill matches available skills under /skill <name> with navigation and drawing."""
        state = TUIState()
        skills = {
            "deploy-tool": ("Deploy app to production", "skills/deploy/SKILL.md", "project", "deploy instructions"),
            "my-linter": ("Lint code with ruff", "skills/lint/SKILL.md", "global", "lint instructions"),
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
            "deploy-tool": ("Deploy app to production", "skills/deploy/SKILL.md", "project", "deploy instructions"),
            "my-linter": ("Lint code with ruff", "skills/lint/SKILL.md", "global", "lint instructions"),
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
        state.set_custom_commands({"deploy-tool": ("Override deploy", "echo override", "workspace")})
        state.input_buffer = "/dep"
        m = state.get_matching_commands()
        self.assertEqual(len(m), 1)
        self.assertEqual(m[0][0], "/deploy-tool")
        self.assertEqual(m[0][1], "Custom command [workspace]")

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


if __name__ == "__main__":
    unittest.main()
