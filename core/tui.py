from __future__ import annotations

import curses
import re
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

DEFAULT_COMMANDS: list[tuple[str, str]] = [
    ("/compact", "Free context tokens by compacting older turn observations"),
    ("/status", "Show current token usage, cache rate, and context breakdown"),
    ("/new", "Start fresh session while preserving workspace changes"),
    ("/skill", "Show available skills or activate with /skill <name>"),
    ("/help", "List all available built-in and workspace custom commands"),
    ("/exit", "Exit the interactive agent session"),
]


def compact_path(path_str: str) -> str:
    """Compacts an absolute path with ~ if within the user's home directory."""
    if not path_str:
        return ""
    try:
        home = str(Path.home())
        p_str = str(path_str)
        if p_str == home or p_str == str(Path.home().resolve()):
            return "~"
        if p_str.startswith(home + "/"):
            return "~" + p_str[len(home) :]
        resolved_home = str(Path.home().resolve())
        if p_str.startswith(resolved_home + "/"):
            return "~" + p_str[len(resolved_home) :]
        return p_str
    except Exception:
        return path_str


INLINE_MD_PATTERN = re.compile(
    r"(?:\*\*\*(?P<bi>[^*]+)\*\*\*)"
    r"|(?:___(?P<bi2>[^_]+)___)"
    r"|(?:\*\*(?P<b>[^*]+)\*\*)"
    r"|(?:__(?P<b2>[^_]+)__)"
    r"|(?:\*(?P<i>[^*]+)\*)"
    r"|(?:(?<!\w)_(?P<i2>[^_]+)_(?!\w))"
    r"|(?:`(?P<code>[^`]+)`)"
)


def _clean_markdown_inline(text: str) -> str:
    """Strips inline markdown formatting markers (`code`, **bold**, *italic*) while preserving identifiers."""
    t = re.sub(r"`([^`]+)`", r"\1", text)
    t = re.sub(r"\*\*\*([^*]+)\*\*\*", r"\1", t)
    t = re.sub(r"___([^_]+)___", r"\1", t)
    t = re.sub(r"\*\*([^*]+)\*\*", r"\1", t)
    t = re.sub(r"__([^_]+)__", r"\1", t)
    t = re.sub(r"\*([^*]+)\*", r"\1", t)
    t = re.sub(r"(?<!\w)_([^_]+)_(?!\w)", r"\1", t)
    return t.strip()


def parse_inline_segments(text: str) -> list[tuple[str, str]]:
    """Splits a line into styled segments (normal, bold, italic, bold_italic, code).

    Markdown delimiter characters are stripped, allowing native terminal formatting.
    """
    if not text:
        return [("", "normal")]

    tokens: list[tuple[str, str]] = []
    last_idx = 0
    for m in INLINE_MD_PATTERN.finditer(text):
        if m.start() > last_idx:
            tokens.append((text[last_idx : m.start()], "normal"))
        if m.group("bi") or m.group("bi2"):
            val = m.group("bi") or m.group("bi2")
            tokens.append((val, "bold_italic"))
        elif m.group("b") or m.group("b2"):
            val = m.group("b") or m.group("b2")
            tokens.append((val, "bold"))
        elif m.group("i") or m.group("i2"):
            val = m.group("i") or m.group("i2")
            tokens.append((val, "italic"))
        elif m.group("code"):
            tokens.append((m.group("code"), "code"))
        last_idx = m.end()

    if last_idx < len(text):
        tokens.append((text[last_idx:], "normal"))

    cleaned: list[tuple[str, str]] = []
    for t_str, style in tokens:
        if style == "normal":
            cleaned_str = re.sub(r"^\*\*|\*\*$|^_|_$|^\*|\*$", "", t_str)
            if cleaned_str:
                cleaned.append((cleaned_str, style))
        else:
            cleaned.append((t_str, style))

    return cleaned or [("", "normal")]


def get_git_branch(workspace: Path | str) -> str:
    """Detects active git branch name from .git/HEAD using stdlib only."""
    try:
        ws_path = Path(workspace)
        git_dir = ws_path / ".git"
        if git_dir.is_file():
            text = git_dir.read_text(encoding="utf-8").strip()
            if text.startswith("gitdir:"):
                target = text[7:].strip()
                git_dir = (ws_path / target).resolve() if not Path(target).is_absolute() else Path(target)
        if not git_dir.is_dir():
            return ""
        head_file = git_dir / "HEAD"
        if head_file.is_file():
            content = head_file.read_text(encoding="utf-8").strip()
            if content.startswith("ref: refs/heads/"):
                return content[len("ref: refs/heads/") :]
            return content[:7]
    except Exception:
        pass
    return ""


def _is_table_separator(line: str) -> bool:
    """Checks if a line is a markdown table separator like | --- | :---: |."""
    stripped = line.strip()
    if not (stripped.startswith("|") and stripped.endswith("|")):
        return False
    cells = [c.strip() for c in stripped[1:-1].split("|")]
    return len(cells) > 0 and all(bool(re.match(r"^:?-+:?$", cell)) for cell in cells)


def _parse_table_row(line: str) -> list[str]:
    """Extracts cell texts from a pipe-delimited table row."""
    stripped = line.strip()
    if stripped.startswith("|"):
        stripped = stripped[1:]
    if stripped.endswith("|"):
        stripped = stripped[:-1]
    return [c.strip() for c in stripped.split("|")]


def render_markdown_table(headers: list[str], rows: list[list[str]], max_width: int = 80) -> list[TUILine]:
    """Renders a parsed markdown table into boxed TUILines fitting within max_width."""
    num_cols = len(headers)
    if num_cols == 0:
        return []

    clean_headers = [_clean_markdown_inline(h) for h in headers]
    padded_rows: list[list[str]] = []
    for r in rows:
        row = [_clean_markdown_inline(c) for c in r]
        if len(row) < num_cols:
            row.extend([""] * (num_cols - len(row)))
        elif len(row) > num_cols:
            row = row[:num_cols]
        padded_rows.append(row)

    col_widths = [len(h) for h in clean_headers]
    for r in padded_rows:
        for i in range(num_cols):
            col_widths[i] = max(col_widths[i], len(r[i]))

    col_widths = [max(3, w) for w in col_widths]
    total_w = sum(col_widths) + 3 * num_cols + 1

    if total_w > max_width and max_width >= (4 * num_cols + 1):
        avail = max_width - (3 * num_cols + 1)
        sum_content = sum(col_widths)
        if sum_content > 0:
            scale = avail / sum_content
            col_widths = [max(3, int(w * scale)) for w in col_widths]
            diff = avail - sum(col_widths)
            if diff > 0:
                col_widths[0] += diff

    def fit(text: str, w: int) -> str:
        if len(text) > w:
            return (text[: max(0, w - 1)] + "…") if w > 1 else text[:w]
        return text.ljust(w)

    top_border = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    mid_border = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    bot_border = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"

    header_cells = " │ ".join(fit(clean_headers[i], col_widths[i]) for i in range(num_cols))
    header_line = f"│ {header_cells} │"

    tui_lines: list[TUILine] = [
        TUILine(text=top_border, tag="table_border"),
        TUILine(text=header_line, tag="table_header"),
        TUILine(text=mid_border, tag="table_border"),
    ]

    for r in padded_rows:
        row_cells = " │ ".join(fit(r[i], col_widths[i]) for i in range(num_cols))
        tui_lines.append(TUILine(text=f"│ {row_cells} │", tag="table_row"))

    tui_lines.append(TUILine(text=bot_border, tag="table_border"))
    return tui_lines


def parse_markdown_to_tui_lines(text: str, max_width: int = 80) -> list[TUILine]:
    """Parses markdown text into styled TUILines with tables, code blocks, headings, lists."""
    raw_lines = text.splitlines()
    result: list[TUILine] = []
    i = 0
    n = len(raw_lines)

    in_code_block = False

    while i < n:
        line = raw_lines[i]
        stripped = line.strip()

        # 1. Code Fence (```)
        if stripped.startswith("```"):
            if not in_code_block:
                in_code_block = True
                code_lang = stripped[3:].strip()
                label = f" {code_lang} " if code_lang else " "
                border_w = max(10, min(max_width - 2, 76))
                dash_len = max(0, border_w - 2 - len(label))
                top = "┌─" + label + ("─" * dash_len) + "┐"
                result.append(TUILine(text=top[:border_w], tag="code_border"))
            else:
                in_code_block = False
                border_w = max(10, min(max_width - 2, 76))
                bot = "└" + ("─" * (border_w - 2)) + "┘"
                result.append(TUILine(text=bot[:border_w], tag="code_border"))
            i += 1
            continue

        if in_code_block:
            result.append(TUILine(text=f"│ {line}", tag="code"))
            i += 1
            continue

        # 2. Markdown Table Detection
        if (
            stripped.startswith("|")
            and stripped.endswith("|")
            and (i + 1 < n)
            and _is_table_separator(raw_lines[i + 1])
        ):
            headers = _parse_table_row(line)
            i += 2
            data_rows: list[list[str]] = []
            while i < n and raw_lines[i].strip().startswith("|") and raw_lines[i].strip().endswith("|"):
                data_rows.append(_parse_table_row(raw_lines[i]))
                i += 1
            table_lines = render_markdown_table(headers, data_rows, max_width=max_width)
            result.extend(table_lines)
            continue

        # 3. Headings
        if stripped.startswith("#"):
            match = re.match(r"^(#{1,6})\s+(.*)$", stripped)
            if match:
                level = len(match.group(1))
                h_text = match.group(2).strip()
                clean_h = _clean_markdown_inline(h_text)
                if level == 1:
                    result.append(TUILine(text=clean_h.upper(), tag="heading"))
                else:
                    result.append(TUILine(text=clean_h, tag="heading"))
                i += 1
                continue

        # 4. Horizontal Rule
        if stripped in ("---", "***", "___") or bool(re.match(r"^[-*_]{3,}$", stripped)):
            hr_w = max(10, min(max_width - 2, 76))
            result.append(TUILine(text="─" * hr_w, tag="dim"))
            i += 1
            continue

        # 5. Blockquote
        if stripped.startswith(">"):
            quote_text = stripped[1:].strip()
            result.append(TUILine(text=f"  │ {quote_text}", tag="blockquote"))
            i += 1
            continue

        # 6. Unordered List (- item or * item)
        if re.match(r"^[-*+]\s+", stripped):
            item_text = re.sub(r"^[-*+]\s+", "", stripped)
            result.append(TUILine(text=f"  • {item_text}", tag="agent"))
            i += 1
            continue

        # 7. Ordered List (1. item)
        list_match = re.match(r"^(\d+)\.\s+(.*)$", stripped)
        if list_match:
            num = list_match.group(1)
            item_text = list_match.group(2)
            result.append(TUILine(text=f"  {num}. {item_text}", tag="agent"))
            i += 1
            continue

        # 8. Standard Prose
        result.append(TUILine(text=line, tag="agent"))
        i += 1

    return result


@dataclass
class TUILine:
    text: str
    tag: str  # "user", "thought", "action", "error", "agent", "system", "dim", "heading", "table_border", "table_header", "table_row", "code_border", "code", "blockquote"


class TUIState:
    """Headless state manager for TUI transcript, scrolling, and token metrics.

    Separating state from curses rendering allows comprehensive unit testing
    without requiring an interactive terminal session during pytest.
    """

    def __init__(self, title: str = "microharness"):
        self.title = title
        self.lines: list[TUILine] = []
        self.scroll_offset: int = 0
        self.input_buffer: str = ""
        self.cursor_pos: int = 0
        self.status_tokens: int = 0
        self.max_tokens: int = 0
        self.usage_pct: float = 0.0
        self.cache_hit_pct: float = 0.0
        self.messages_count: int = 0
        self.model_name: str = ""
        self.provider_name: str = ""
        self.cache_ttl: str = ""
        self.reasoning: str = ""
        self.cwd: str = ""
        self.git_branch: str = ""
        self.command_catalog: list[tuple[str, str]] = list(DEFAULT_COMMANDS)
        self.custom_commands: dict[str, Any] = {}
        self.skills: dict[str, Any] = {}
        self.selected_command_idx: int = 0

    def _rebuild_command_catalog(self) -> None:
        """Rebuilds the catalog with built-in commands, custom commands, and direct skills."""
        catalog = list(DEFAULT_COMMANDS)
        for cmd_name, cmd_info in sorted(self.custom_commands.items()):
            desc = f"Custom command: /{cmd_name}"
            if isinstance(cmd_info, (tuple, list)) and len(cmd_info) >= 3:
                scope = cmd_info[2]
                desc = f"Custom command [{scope}]"
            catalog.append((f"/{cmd_name}", desc))

        existing_cmds = {cmd.lower() for cmd, _ in catalog}
        for s_name, s_info in sorted(self.skills.items()):
            slash_skill = f"/{s_name}"
            if slash_skill.lower() not in existing_cmds:
                summary = (
                    s_info[0]
                    if isinstance(s_info, (tuple, list)) and len(s_info) > 0
                    else s_name
                )
                scope = (
                    s_info[2]
                    if isinstance(s_info, (tuple, list)) and len(s_info) > 2
                    else "project"
                )
                catalog.append((slash_skill, f"Skill [{scope}] {summary}"))
                existing_cmds.add(slash_skill.lower())

        self.command_catalog = catalog

    def set_custom_commands(self, custom_commands: dict[str, Any]) -> None:
        """Loads custom workspace commands into the auto-completion catalog."""
        self.custom_commands = dict(custom_commands)
        self._rebuild_command_catalog()

    def set_skills(self, skills: dict[str, Any]) -> None:
        """Loads discovered skills into the auto-completion catalog."""
        self.skills = dict(skills)
        self._rebuild_command_catalog()

    def get_matching_commands(self) -> list[tuple[str, str]]:
        """Filters command catalog matching current slash-command prefix in input buffer."""
        buf = self.input_buffer.lstrip().lower()
        if not buf.startswith("/"):
            return []

        # If user is typing /skill <subcommand>, match against available skills
        if buf.startswith(("/skill ", "/skills ")):
            prefix = buf.split(" ", 1)[0]
            sub = buf.split(" ", 1)[1].lstrip()
            res: list[tuple[str, str]] = []
            for s_name, s_info in sorted(self.skills.items()):
                if s_name.lower().startswith(sub):
                    summary = (
                        s_info[0]
                        if isinstance(s_info, (tuple, list)) and len(s_info) > 0
                        else s_name
                    )
                    scope = (
                        s_info[2]
                        if isinstance(s_info, (tuple, list)) and len(s_info) > 2
                        else "project"
                    )
                    res.append((f"{prefix} {s_name}", f"[{scope}] {summary}"))
            return res

        if " " in buf:
            return []

        return [
            (cmd, desc)
            for cmd, desc in self.command_catalog
            if cmd.lower().startswith(buf)
        ]

    def select_next_command(self) -> None:
        matches = self.get_matching_commands()
        if matches:
            self.selected_command_idx = (self.selected_command_idx + 1) % len(matches)

    def select_prev_command(self) -> None:
        matches = self.get_matching_commands()
        if matches:
            self.selected_command_idx = (self.selected_command_idx - 1) % len(matches)

    def complete_selected_command(self) -> str | None:
        matches = self.get_matching_commands()
        if not matches:
            return None
        idx = max(0, min(self.selected_command_idx, len(matches) - 1))
        chosen_cmd, _ = matches[idx]
        self.input_buffer = chosen_cmd
        self.cursor_pos = len(self.input_buffer)
        return chosen_cmd

    def add_user(self, text: str) -> None:
        if self.lines and self.lines[-1].text != "":
            self.lines.append(TUILine(text="", tag="dim"))
        lines = text.splitlines() or [""]
        for i, line in enumerate(lines):
            prefix = "> " if i == 0 else "  "
            self.lines.append(TUILine(text=f"{prefix}{line}", tag="user"))
        self.scroll_offset = 0

    def add_thought(self, text: str) -> None:
        self.lines.append(TUILine(text=f"Thinking: {text}", tag="thought"))
        self.scroll_offset = 0

    def add_action(self, text: str) -> None:
        self.lines.append(TUILine(text=text, tag="action"))
        self.scroll_offset = 0

    def add_error(self, text: str) -> None:
        self.lines.append(TUILine(text=f"error: {text}", tag="error"))
        self.scroll_offset = 0

    def add_agent(self, text: str, max_width: int = 80) -> None:
        self.lines.append(TUILine(text="", tag="dim"))
        parsed = parse_markdown_to_tui_lines(text, max_width=max_width)
        self.lines.extend(parsed)
        self.lines.append(TUILine(text="", tag="dim"))
        self.scroll_offset = 0

    def add_system(self, text: str) -> None:
        for line in text.splitlines():
            self.lines.append(TUILine(text=line, tag="system"))
        self.scroll_offset = 0

    def update_status(self, stats: dict[str, Any], cache_stats: dict[str, Any] | None = None) -> None:
        self.status_tokens = int(stats.get("estimated_tokens") or 0)
        self.max_tokens = int(stats.get("max_estimated_tokens") or 0)
        self.usage_pct = float(stats.get("usage_pct") or 0.0)
        self.messages_count = int(stats.get("total_messages") or 0)
        if cache_stats:
            self.cache_hit_pct = float(cache_stats.get("cache_hit_rate_pct") or 0.0)

    def scroll_up(self, n: int = 3) -> None:
        self.scroll_offset += n

    def scroll_down(self, n: int = 3) -> None:
        self.scroll_offset = max(0, self.scroll_offset - n)

    def get_wrapped_lines(self, width: int) -> list[tuple[str, str]]:
        """Wraps transcript lines to fit the specified column width."""
        wrapped: list[tuple[str, str]] = []
        eff_width = max(10, width - 2)

        for line_item in self.lines:
            if not line_item.text:
                wrapped.append(("", line_item.tag))
            elif line_item.tag in ("table_border", "table_header", "table_row", "code_border", "code"):
                # Preserve formatted table and code lines without re-wrapping
                wrapped.append((line_item.text[:eff_width], line_item.tag))
            else:
                indent = "  " if line_item.tag == "user" and line_item.text.startswith("> ") else ""
                chunks = textwrap.wrap(
                    line_item.text,
                    width=eff_width,
                    replace_whitespace=False,
                    drop_whitespace=False,
                    subsequent_indent=indent,
                )
                if not chunks:
                    wrapped.append(("", line_item.tag))
                else:
                    for chunk in chunks:
                        wrapped.append((chunk, line_item.tag))

        return wrapped

    def get_visible_slice(self, height: int, width: int) -> list[tuple[str, str]]:
        """Calculates visible lines based on height and current scroll offset."""
        wrapped = self.get_wrapped_lines(width)
        total = len(wrapped)
        if total <= height:
            return wrapped

        max_scroll = max(0, total - height)
        actual_scroll = min(self.scroll_offset, max_scroll)
        end_idx = total - actual_scroll
        start_idx = max(0, end_idx - height)
        return wrapped[start_idx:end_idx]

    def handle_input_char(self, char_str: str) -> None:
        """Appends character to input buffer."""
        self.input_buffer += char_str
        self.cursor_pos += len(char_str)

    def handle_backspace(self) -> None:
        """Removes the last character from input buffer."""
        if self.input_buffer:
            self.input_buffer = self.input_buffer[:-1]
            self.cursor_pos = max(0, self.cursor_pos - 1)

    def clear_input(self) -> str:
        """Clears and returns the current input buffer."""
        val = self.input_buffer.strip()
        self.input_buffer = ""
        self.cursor_pos = 0
        return val


class CursesTUI:
    """Curses terminal user interface with bottom-anchored chat and scrollable activity."""

    def __init__(self, state: TUIState):
        self.state = state
        self.stdscr: Any = None
        self.has_colors = False

    def _init_colors(self) -> None:
        try:
            if curses.has_colors():
                curses.start_color()
                curses.use_default_colors()
                curses.init_pair(1, curses.COLOR_CYAN, -1)     # User / Heading
                curses.init_pair(2, curses.COLOR_GREEN, -1)    # Action / Code
                curses.init_pair(3, curses.COLOR_YELLOW, -1)   # Tool / Thought
                curses.init_pair(4, curses.COLOR_RED, -1)      # Error
                curses.init_pair(5, curses.COLOR_BLUE, -1)     # Header / Brand
                curses.init_pair(6, curses.COLOR_WHITE, -1)    # Agent text
                self.has_colors = True
        except curses.error:
            self.has_colors = False

    def draw(self, stdscr: Any) -> None:
        """Draws the complete TUI screen: Header, Transcript, Status, and Bottom Input."""
        stdscr.erase()
        h, w = stdscr.getmaxyx()
        if h < 8 or w < 20:
            stdscr.addstr(0, 0, "Terminal window too small! Please resize.")
            stdscr.refresh()
            return

        # 1. Header Bar (Row 0)
        header_text = f" {self.state.title} "
        header_text = header_text[: w - 1]
        try:
            stdscr.attron(curses.A_BOLD)
            stdscr.addstr(0, 0, header_text)
            stdscr.attroff(curses.A_BOLD)
        except curses.error:
            pass

        # 2. Header separator line (Row 1)
        try:
            stdscr.addstr(1, 0, "─" * (w - 1), curses.A_DIM)
        except curses.error:
            pass

        # 3. Main Transcript Area (Rows 2 to H - 5)
        sep_row = h - 4
        path_row = h - 3
        telemetry_row = h - 2
        input_row = h - 1

        transcript_height = max(1, h - 6)
        visible_lines = self.state.get_visible_slice(height=transcript_height, width=w)
        for i, (text, tag) in enumerate(visible_lines):
            row = 2 + i
            if row >= sep_row:
                break
            attr = curses.A_NORMAL
            if tag in ("user", "heading"):
                attr = curses.color_pair(1) | curses.A_BOLD if self.has_colors else curses.A_BOLD
            elif tag == "table_header":
                attr = curses.A_BOLD
            elif tag == "table_row":
                attr = curses.A_NORMAL
            elif tag in ("table_border", "code_border", "blockquote", "dim", "system"):
                attr = curses.A_DIM
            elif tag == "code":
                attr = curses.color_pair(2) if self.has_colors else curses.A_NORMAL
            elif tag == "thought":
                attr = curses.color_pair(3) | curses.A_DIM if self.has_colors else curses.A_DIM
            elif tag == "action":
                attr = curses.color_pair(2) if self.has_colors else curses.A_NORMAL
            elif tag == "error":
                attr = curses.color_pair(4) | curses.A_BOLD if self.has_colors else curses.A_BOLD
            elif tag == "agent":
                attr = curses.color_pair(6) if self.has_colors else curses.A_NORMAL

            if tag in ("agent", "user", "blockquote", "table_row"):
                segments = parse_inline_segments(text)
                col = 1
                italic_attr = getattr(curses, "A_ITALIC", 0) or curses.A_UNDERLINE
                for seg_text, style in segments:
                    if col >= w - 1:
                        break
                    avail = w - 1 - col
                    chunk = seg_text[:avail]
                    seg_attr = attr
                    if style == "bold":
                        seg_attr |= curses.A_BOLD
                    elif style == "italic":
                        seg_attr |= italic_attr
                    elif style == "bold_italic":
                        seg_attr |= (curses.A_BOLD | italic_attr)
                    elif style == "code":
                        seg_attr = (curses.color_pair(2) | curses.A_BOLD) if self.has_colors else curses.A_BOLD

                    try:
                        stdscr.addstr(row, col, chunk, seg_attr)
                    except curses.error:
                        pass
                    col += len(chunk)
            else:
                try:
                    stdscr.addstr(row, 1, text[: w - 2], attr)
                except curses.error:
                    pass

        # 4. Status Cockpit: Row H - 4 (Sep), Row H - 3 (Path), Row H - 2 (Telemetry)
        path_label = self.state.cwd or ""
        if self.state.git_branch:
            path_label = f"{path_label} ({self.state.git_branch})" if path_label else f"({self.state.git_branch})"
        path_text = f" {path_label} " if path_label else ""

        parts: list[str] = []
        model_label = ""
        if self.state.provider_name and self.state.model_name:
            if self.state.model_name.lower().startswith(self.state.provider_name.lower() + "/"):
                model_label = self.state.model_name
            else:
                model_label = f"{self.state.provider_name}/{self.state.model_name}"
        elif self.state.model_name:
            model_label = self.state.model_name
        elif self.state.provider_name:
            model_label = self.state.provider_name

        if model_label:
            parts.append(model_label)

        if self.state.cache_ttl:
            parts.append(f"cache: {self.state.cache_ttl}")
        if self.state.reasoning:
            parts.append(f"reasoning: {self.state.reasoning}")

        parts.append(
            f"context: {self.state.status_tokens:,} / {self.state.max_tokens:,} tokens ({self.state.usage_pct:.1f}%)"
        )
        if self.state.cache_hit_pct > 0:
            parts.append(f"cache hit: {self.state.cache_hit_pct:.1f}%")
        parts.append(f"msgs: {self.state.messages_count}")

        if self.state.scroll_offset > 0:
            parts.append(f"[scrolled +{self.state.scroll_offset}]")

        status_text = " " + " · ".join(parts) + " "

        try:
            stdscr.addstr(sep_row, 0, "─" * (w - 1), curses.A_DIM)
            if path_text:
                stdscr.addstr(path_row, 0, path_text[: w - 1], curses.A_DIM)
            stdscr.addstr(telemetry_row, 0, status_text[: w - 1], curses.A_REVERSE)
        except curses.error:
            pass

        # 5. Floating Slash-Command Menu (drawn above status cockpit, full screen width)
        matches = self.state.get_matching_commands()
        if matches:
            idx = max(0, min(self.state.selected_command_idx, len(matches) - 1))
            max_items = max(6, min(len(matches), max(6, sep_row - 6)))
            if len(matches) <= max_items:
                visible_matches = matches
                rel_idx = idx
            else:
                start_m = max(0, min(idx - max_items // 2, len(matches) - max_items))
                visible_matches = matches[start_m : start_m + max_items]
                rel_idx = idx - start_m

            box_h = len(visible_matches) + 4
            box_w = max(10, w - 1)
            start_y = max(1, sep_row - box_h)
            start_x = 0

            is_skill_menu = self.state.input_buffer.lstrip().lower().startswith(("/skill ", "/skills "))
            title_part = " Available Skills " if is_skill_menu else " Available Commands "
            dash_count = max(0, box_w - 2 - len(title_part) - 2)
            border_top = "┌─" + title_part + ("─" * dash_count) + "┐"
            mid_sep = "├" + "─" * (box_w - 2) + "┤"
            border_bot = "└" + "─" * (box_w - 2) + "┘"
            tip_text = "   Tab / ↑ / ↓: Navigate  ·  Enter / →: Select  ·  Esc: Dismiss"

            try:
                stdscr.addstr(start_y, start_x, border_top[:box_w], curses.A_DIM)
                cmd_col_w = 22 if is_skill_menu else 12
                for row_i, (cmd, desc) in enumerate(visible_matches):
                    y = start_y + 1 + row_i
                    is_selected = (row_i == rel_idx)
                    prefix = "  > " if is_selected else "    "
                    line_content = f"{prefix}{cmd:<{cmd_col_w}} {desc}"
                    line_content = line_content[: box_w - 2].ljust(box_w - 2)
                    attr = (
                        (curses.color_pair(1) | curses.A_BOLD | curses.A_REVERSE)
                        if (is_selected and self.has_colors)
                        else (curses.A_REVERSE if is_selected else curses.A_NORMAL)
                    )
                    stdscr.addstr(y, start_x, "│", curses.A_DIM)
                    stdscr.addstr(y, start_x + 1, line_content, attr)
                    stdscr.addstr(y, start_x + box_w - 1, "│", curses.A_DIM)

                y_sep = start_y + 1 + len(visible_matches)
                stdscr.addstr(y_sep, start_x, mid_sep[:box_w], curses.A_DIM)

                y_tip = y_sep + 1
                tip_content = tip_text[: box_w - 2].ljust(box_w - 2)
                stdscr.addstr(y_tip, start_x, "│", curses.A_DIM)
                stdscr.addstr(y_tip, start_x + 1, tip_content, curses.A_DIM)
                stdscr.addstr(y_tip, start_x + box_w - 1, "│", curses.A_DIM)

                stdscr.addstr(start_y + box_h - 1, start_x, border_bot[:box_w], curses.A_DIM)
            except curses.error:
                pass

        # 6. Bottom Anchored Input Box (Row H - 1)
        prompt_label = "> "
        avail_input_w = max(5, w - len(prompt_label) - 2)
        display_input = self.state.input_buffer[-avail_input_w:]

        try:
            stdscr.addstr(input_row, 0, prompt_label, curses.A_BOLD)
            stdscr.addstr(input_row, len(prompt_label), display_input)
            cursor_col = min(w - 1, len(prompt_label) + len(display_input))
            stdscr.move(input_row, cursor_col)
        except curses.error:
            pass

        stdscr.refresh()

    def handle_key(self, ch: str | int, on_user_submit: Callable[[str], None]) -> bool:
        """Processes a single keystroke event.

        Returns False if the interactive session should terminate (e.g. /exit), else True.
        """
        matches = self.state.get_matching_commands()

        if ch in ("\n", "\r", curses.KEY_ENTER, 10, 13):
            if matches and not self.state.input_buffer.endswith(" "):
                self.state.complete_selected_command()
            user_text = self.state.clear_input()
            if not user_text:
                return True
            if user_text.lower() in ("/exit", "/quit", "exit", "quit"):
                return False
            on_user_submit(user_text)
            return True

        if ch in ("\t", 9):
            if matches:
                if len(matches) == 1:
                    self.state.complete_selected_command()
                else:
                    self.state.select_next_command()
            return True

        if ch in (curses.KEY_UP, curses.KEY_BTAB, 353, "\x1b[Z"):
            if matches:
                self.state.select_prev_command()
                return True
            self.state.scroll_up(5)
            return True

        if ch in (curses.KEY_DOWN,):
            if matches:
                self.state.select_next_command()
                return True
            self.state.scroll_down(5)
            return True

        if ch in (curses.KEY_RIGHT,):
            if matches:
                self.state.complete_selected_command()
            return True

        if ch in (curses.KEY_PPAGE,):
            self.state.scroll_up(5)
            return True

        if ch in (curses.KEY_NPAGE,):
            self.state.scroll_down(5)
            return True

        if ch in (27, "\x1b"):
            if matches:
                self.state.clear_input()
            return True

        if ch in (curses.KEY_BACKSPACE, "\x7f", "\x08", 127):
            self.state.handle_backspace()
            return True

        if isinstance(ch, str) and ch.isprintable():
            self.state.handle_input_char(ch)
            return True

        return True

    def run(self, on_user_submit: Callable[[str], None]) -> None:
        """Launches the interactive curses event loop."""

        def _main(stdscr: Any) -> None:
            self.stdscr = stdscr
            self._init_colors()
            curses.cbreak()
            stdscr.keypad(True)
            curses.curs_set(1)

            self.draw(stdscr)

            while True:
                try:
                    ch = stdscr.get_wch()
                except (KeyboardInterrupt, EOFError):
                    break
                except curses.error:
                    continue

                if not self.handle_key(ch, on_user_submit):
                    break
                self.draw(stdscr)

        curses.wrapper(_main)
