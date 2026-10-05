from __future__ import annotations

import curses
import os
import re
import sys
import textwrap
import unicodedata
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.memory import format_compact_num
from core.interactive import command_catalog, format_model_desc, format_session_desc

# Commands that open a value menu instead of the command catalog: their
# spellings, the menu title, and the hint shown before a value is typed.
SUBCOMMAND_MENUS: dict[str, tuple[tuple[str, ...], str, str]] = {
    "skill": (("/skill", "/skills"), " Available Skills ", "Load or inspect a custom skill prompt"),
    "model": (("/model", "/models"), " Available Models ", "Switch active model or show models from config"),
    "resume": (("/resume", "/resumes"), " Previous Sessions ", "Show previous sessions or resume with /resume <id>"),
}


# (min, max) width of the command column per value menu. A fixed pair means a
# fixed width; otherwise the column shrinks with the terminal. File menus use
# their own bounds because their values are paths.
MENU_COLUMN_BOUNDS: dict[str, tuple[int, int]] = {
    "skill": (22, 22),
    "model": (20, 36),
    "resume": (26, 38),
}
FILE_MENU_COLUMN_BOUNDS = (24, 42)


def menu_column_width(kind: str | None, box_w: int, is_file_menu: bool) -> int:
    """Width of the command column for the menu being drawn."""
    if is_file_menu:
        low, high = FILE_MENU_COLUMN_BOUNDS
        return max(low, min(high, box_w - 12))
    if kind is None:
        return 12
    low, high = MENU_COLUMN_BOUNDS[kind]
    return low if low == high else max(low, min(high, box_w - 20))


def active_subcommand(buf: str) -> tuple[str, str | None] | None:
    """Returns ``(kind, sub)`` when the input has opened a value menu.

    ``sub`` is the text typed after the command word; ``None`` means the command
    word itself was just typed, which is when the menu still shows its hint. The
    outer ``None`` means no value menu is open at all.
    """
    command, space, sub = buf.partition(" ")
    for kind, (spellings, _title, _hint) in SUBCOMMAND_MENUS.items():
        if command in spellings:
            return kind, sub.lstrip() if space else None
    return None

SHIFT_ENTER_SEQUENCES: set[str | int] = {
    "\x1b[13;2u",
    "\x1b[27;2;13~",
    "\x1b[13;2~",
    "\x1bOM",
    "\x1b[M",
    "\x1b\r",
    "\x1b\n",
    545,
    546,
}

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
    except (OSError, ValueError):
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
    except OSError:
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


MIN_COL_WIDTH = 3


def _char_width(ch: str) -> int:
    """Terminal cells occupied by a single character."""
    if unicodedata.combining(ch):
        return 0
    return 2 if unicodedata.east_asian_width(ch) in ("W", "F") else 1


def _dwidth(text: str) -> int:
    """Display width in terminal cells, accounting for wide and combining characters."""
    return sum(_char_width(ch) for ch in text)


def _truncate_display(text: str, width: int, ellipsis: str = "…") -> str:
    """Truncates text so its display width fits within `width`, marking elision."""
    if _dwidth(text) <= width:
        return text
    if width <= 0:
        return ""
    budget = width - _dwidth(ellipsis)
    if budget <= 0:
        return ellipsis[:width]
    kept: list[str] = []
    used = 0
    for ch in text:
        cw = _char_width(ch)
        if used + cw > budget:
            break
        kept.append(ch)
        used += cw
    return "".join(kept) + ellipsis


def _pad_display(text: str, width: int) -> str:
    """Right-pads text with spaces to reach the given display width."""
    return text + " " * max(0, width - _dwidth(text))


def _fitted_widths(desired: list[int], max_width: int, num_cols: int) -> list[int]:
    """Scales column widths down so the rendered table fits within max_width.

    Widths never drop below MIN_COL_WIDTH, so a caller that still overflows must
    shed columns (see render_markdown_table) rather than expect a fit here.
    """
    overhead = 3 * num_cols + 1
    avail = max_width - overhead
    if avail <= 0:
        return [MIN_COL_WIDTH] * num_cols
    total = sum(desired)
    if total <= avail:
        return desired
    scaled = [max(MIN_COL_WIDTH, int(w * avail / total)) for w in desired]
    leftover = avail - sum(scaled)
    if leftover > 0 and scaled:
        scaled[scaled.index(max(scaled))] += leftover
    return scaled


def render_markdown_table(headers: list[str], rows: list[list[str]], max_width: int = 80) -> list[TUILine]:
    """Renders a parsed markdown table into boxed TUILines that fit within max_width.

    Columns are sized by display width (wide CJK characters count as two cells) and,
    when the table cannot fit the requested width, trailing columns are dropped rather
    than allowing the box art to overflow the available pane.
    """
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

    desired: list[int] = []
    for i in range(num_cols):
        widest = _dwidth(clean_headers[i])
        for r in padded_rows:
            widest = max(widest, _dwidth(r[i]))
        desired.append(max(MIN_COL_WIDTH, widest))

    # Drop trailing columns until the table fits; the per-column floor of
    # MIN_COL_WIDTH means a table of n columns can never render narrower than
    # 6n+1 cells, so width-fitting alone cannot guarantee the max_width contract.
    visible = num_cols
    while visible > 0:
        col_widths = _fitted_widths(desired[:visible], max_width, visible)
        if sum(col_widths) + 3 * visible + 1 <= max_width:
            break
        visible -= 1
    if visible == 0:
        return []

    if visible < num_cols:
        clean_headers = clean_headers[:visible]
        padded_rows = [r[:visible] for r in padded_rows]

    def fit(text: str, w: int) -> str:
        return _pad_display(_truncate_display(text, w), w)

    top_border = "┌" + "┬".join("─" * (w + 2) for w in col_widths) + "┐"
    mid_border = "├" + "┼".join("─" * (w + 2) for w in col_widths) + "┤"
    bot_border = "└" + "┴".join("─" * (w + 2) for w in col_widths) + "┘"

    header_cells = " │ ".join(fit(clean_headers[i], col_widths[i]) for i in range(visible))
    header_line = f"│ {header_cells} │"

    tui_lines: list[TUILine] = [
        TUILine(text=top_border, tag="table_border"),
        TUILine(text=header_line, tag="table_header"),
        TUILine(text=mid_border, tag="table_border"),
    ]

    for r in padded_rows:
        row_cells = " │ ".join(fit(r[i], col_widths[i]) for i in range(visible))
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
        self.command_catalog: list[tuple[str, str]] = command_catalog()
        self.custom_commands: dict[str, Any] = {}
        self.skills: dict[str, Any] = {}
        self.available_models: list[dict[str, Any]] = []
        self.available_sessions: list[dict[str, Any]] = []
        self.workspace_files: list[str] = []
        self.selected_command_idx: int = 0
        self.turn_active: bool = False

    def set_available_models(self, models: list[dict[str, Any]]) -> None:
        """Loads available models for /model autocomplete."""
        self.available_models = list(models)

    def set_available_sessions(self, sessions: list[dict[str, Any]]) -> None:
        """Loads available previous sessions for /resume autocomplete."""
        self.available_sessions = list(sessions)

    def set_workspace_files(self, files: list[str]) -> None:
        """Loads workspace relative file paths for @ file tagging and autocomplete."""
        self.workspace_files = list(files)

    def _rebuild_command_catalog(self) -> None:
        """Rebuilds the catalog with built-in commands, custom commands, and direct skills."""
        catalog = command_catalog()
        for cmd_name, cmd_info in sorted(self.custom_commands.items()):
            scope = getattr(cmd_info, "scope", "")
            desc = f"Custom command [{scope}]" if scope else f"Custom command: /{cmd_name}"
            catalog.append((f"/{cmd_name}", desc))

        existing_cmds = {cmd.lower() for cmd, _ in catalog}
        for s_name, s_info in sorted(self.skills.items()):
            slash_skill = f"/{s_name}"
            if slash_skill.lower() not in existing_cmds:
                summary = getattr(s_info, "summary", s_name)
                scope = getattr(s_info, "scope", "project")
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

    def get_current_file_tag_query(self) -> tuple[int, str] | None:
        """Detects if cursor is currently inside or right after an @file tag token.

        Returns (tag_start_idx, query) where tag_start_idx is the index of '@'.
        Returns None if not currently typing an @ tag.
        """
        pos = min(self.cursor_pos, len(self.input_buffer)) if self.cursor_pos >= 0 else len(self.input_buffer)
        sub = self.input_buffer[:pos]
        m = re.search(r"(?:^|\s)@([^\s]*)$", sub)
        if not m:
            return None
        full_match = m.group(0)
        at_offset = full_match.find("@")
        at_idx = m.start() + at_offset
        query = m.group(1)
        return (at_idx, query)

    def get_matching_file_tags(self) -> list[tuple[str, str]]:
        """Filters workspace files matching current @file tag query."""
        tag_info = self.get_current_file_tag_query()
        if tag_info is None:
            return []
        _, query = tag_info
        q = query.lower()

        prefix_matches: list[str] = []
        substring_matches: list[str] = []
        for f in self.workspace_files:
            f_low = f.lower()
            fname = os.path.basename(f).lower()
            if not q:
                prefix_matches.append(f)
            elif f_low.startswith(q) or fname.startswith(q):
                prefix_matches.append(f)
            elif q in f_low:
                substring_matches.append(f)

        ranked = prefix_matches + substring_matches
        return [(f"@{f}", "[file]") for f in ranked[:30]]

    def complete_selected_file_tag(self) -> str | None:
        """Completes the currently selected @file tag into input_buffer with a trailing space."""
        tag_info = self.get_current_file_tag_query()
        if tag_info is None:
            return None
        matches = self.get_matching_file_tags()
        if not matches:
            return None
        idx = max(0, min(self.selected_command_idx, len(matches) - 1))
        chosen_tag, _ = matches[idx]  # e.g. "@core/tui.py"
        at_idx, query = tag_info
        before = self.input_buffer[:at_idx]
        after = self.input_buffer[at_idx + 1 + len(query):]
        self.input_buffer = f"{before}{chosen_tag} {after}"
        self.cursor_pos = len(before) + len(chosen_tag) + 1
        return chosen_tag

    def get_matching_commands(self) -> list[tuple[str, str]]:
        """Filters command catalog matching current slash-command prefix in input buffer."""
        buf = self.input_buffer.lstrip().lower()
        if not buf.startswith("/"):
            return []

        # An argument-taking command lists its own values instead of the catalog.
        subcommand = active_subcommand(buf)
        if subcommand is None:
            if " " in buf:
                return []
            return [
                (cmd, desc)
                for cmd, desc in self.command_catalog
                if cmd.lower().startswith(buf)
            ]

        kind, sub = subcommand
        _spellings, _title, hint = SUBCOMMAND_MENUS[kind]
        prefix = buf.split(" ", 1)[0]
        rows: list[tuple[str, str]] = []
        if sub is None:
            rows.append((prefix, hint))
        rows += [(f"{prefix} {name}", desc) for name, desc in self._subcommand_values(kind, sub)]
        return rows

    def _subcommand_values(self, kind: str, sub: str) -> list[tuple[str, str]]:
        """The ``(name, description)`` values a value menu can offer."""
        if kind == "skill":
            return [
                (name, f"[{getattr(info, 'scope', 'project')}] "
                       f"{getattr(info, 'summary', name)}")
                for name, info in sorted(self.skills.items())
                if not sub or name.lower().startswith(sub)
            ]
        if kind == "model":
            return [
                (model.get("name", ""), format_model_desc(model))
                for model in self.available_models
                if not sub
                or model.get("name", "").lower().startswith(sub)
                or (model.get("model") or "").lower().startswith(sub)
            ]
        return [
            (session.get("run_id", ""), format_session_desc(session, preview_limit=35))
            for session in self.available_sessions
            if not sub or session.get("run_id", "").lower().startswith(sub)
        ]

    def select_next_command(self) -> None:
        file_matches = self.get_matching_file_tags()
        if file_matches:
            self.selected_command_idx = (self.selected_command_idx + 1) % len(file_matches)
            return
        matches = self.get_matching_commands()
        if matches:
            self.selected_command_idx = (self.selected_command_idx + 1) % len(matches)

    def select_prev_command(self) -> None:
        file_matches = self.get_matching_file_tags()
        if file_matches:
            self.selected_command_idx = (self.selected_command_idx - 1) % len(file_matches)
            return
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
        """Inserts character into input buffer at cursor position."""
        pos = min(self.cursor_pos, len(self.input_buffer))
        self.input_buffer = self.input_buffer[:pos] + char_str + self.input_buffer[pos:]
        self.cursor_pos = pos + len(char_str)

    def handle_newline(self) -> None:
        """Inserts a newline character at cursor position for multiline editing."""
        pos = min(self.cursor_pos, len(self.input_buffer))
        self.input_buffer = self.input_buffer[:pos] + "\n" + self.input_buffer[pos:]
        self.cursor_pos = pos + 1

    def handle_backspace(self) -> None:
        """Removes the character before cursor position from input buffer."""
        if self.input_buffer and self.cursor_pos > 0:
            pos = min(self.cursor_pos, len(self.input_buffer))
            self.input_buffer = self.input_buffer[: pos - 1] + self.input_buffer[pos:]
            self.cursor_pos = max(0, pos - 1)

    def clear_input(self) -> str:
        """Clears and returns the current input buffer."""
        val = self.input_buffer.strip()
        self.input_buffer = ""
        self.cursor_pos = 0
        return val


# Terminals that report shifted number-row keys with a modifier need the
# character the unshifted key would not produce.
SHIFTED_NUMBER_KEYS: dict[int, str] = {
    49: "!", 50: "@", 51: "#", 52: "$", 53: "%",
    54: "^", 55: "&", 56: "*", 57: "(", 48: ")",
}

# The two encodings a terminal may use for a modified key, each with the capture
# group holding the codepoint and the group holding the modifier.
#   CSI-u:          \x1b[<code>;<modifiers>u
#   modifyOtherKeys: \x1b[27;<modifiers>;<code>~
KEY_SEQUENCE_PATTERNS: tuple[tuple[re.Pattern[str], int, int], ...] = (
    (re.compile(r"^\x1b\[(\d+)(?:;(\d+))?u$"), 0, 1),
    (re.compile(r"^\x1b\[27;(\d+);(\d+)~$"), 1, 0),
)

SHIFTED_MODIFIERS = (2, 3)


def _parse_key_sequence(seq: str) -> tuple[int, int] | None:
    """Extracts ``(codepoint, modifier)`` from a recognized key sequence."""
    for pattern, code_group, modifier_group in KEY_SEQUENCE_PATTERNS:
        if match := pattern.match(seq):
            groups = match.groups()
            return int(groups[code_group]), int(groups[modifier_group] or 1)
    return None


def _character_for(code: int, modifier: int) -> str | None:
    """Maps a decoded key to a character, or ``None`` when it produces none."""
    if code == 13:
        return None  # Enter / Shift-Enter are handled as navigation, not text
    if code == 64:
        return "@"  # terminals disagree on whether Shift+2 arrives as 50 or 64
    if modifier in SHIFTED_MODIFIERS and code in SHIFTED_NUMBER_KEYS:
        return SHIFTED_NUMBER_KEYS[code]
    if 32 <= code <= 126:
        return chr(code)
    return None


# Transcript tags whose lines are drawn segment by segment for inline styling.
INLINE_MARKUP_TAGS = ("agent", "user", "blockquote", "table_row")


def _safe(write: Any, *args: Any) -> None:
    """Writes to a curses window, ignoring writes that land outside it.

    Drawing right up to the screen edge is normal, and curses reports it as an
    error rather than clipping, so the failure is expected and not worth
    interrupting a repaint for.
    """
    try:
        write(*args)
    except curses.error:
        pass


@dataclass(frozen=True)
class ScreenLayout:
    """Row positions for one frame, derived from the terminal height and input size."""

    input_lines: int
    input_box_height: int
    input_start_row: int
    telemetry_row: int
    path_row: int
    separator_row: int


def _set_nodelay(window: Any, enabled: bool) -> None:
    """Switches a curses window between blocking reads and polling reads."""
    try:
        window.nodelay(enabled)
    except curses.error:
        pass


def _drain_escape_sequence(window: Any) -> str:
    """Reads the remainder of an escape sequence without blocking.

    Returns the sequence text including its leading ESC. A bare Escape leaves
    nothing behind, which is how a lone keypress is told apart from CSI-u and
    modifyOtherKeys sequences that decode to ordinary characters.
    """
    seq = "\x1b"
    while True:
        try:
            nxt = window.get_wch()
        except (curses.error, EOFError, StopIteration):
            break
        seq += chr(nxt) if isinstance(nxt, int) and 0 <= nxt <= 0x10FFFF else str(nxt)
    return seq


def _reset_key_reporting() -> None:
    """Restores normal key reporting so typing "Shift+2" reaches us as "@".

    Terminals left in modifyOtherKeys mode deliver shifted keys as escape
    sequences, which would turn ordinary typing into navigation.
    """
    try:
        sys.stdout.write("\x1b[>4;0m")
        sys.stdout.flush()
    except OSError:
        pass


def decode_terminal_sequence(seq: str) -> str | None:
    """Decodes modifyOtherKeys or CSI-u key sequences into characters if applicable.

    For example, Shift+2 can be encoded as ``\x1b[27;2;50~`` or ``\x1b[50;2u``, which
    maps to ``@``.
    """
    if not isinstance(seq, str) or not seq.startswith("\x1b"):
        return None
    if (parsed := _parse_key_sequence(seq)) is None:
        return None
    return _character_for(*parsed)


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

        self._draw_header(stdscr, w)
        layout = self._screen_layout(h)
        self._draw_transcript(stdscr, w, layout)
        self._draw_status(stdscr, w, layout)
        self._draw_autocomplete(stdscr, w, layout)
        self._draw_input(stdscr, w, layout)
        stdscr.refresh()

    def _draw_header(self, stdscr: Any, w: int) -> None:
        """Draws the title bar and the rule beneath it."""
        _safe(stdscr.addstr, 0, 0, f" {self.state.title} "[: w - 1], curses.A_BOLD)
        _safe(stdscr.addstr, 1, 0, "─" * (w - 1), curses.A_DIM)

    def _screen_layout(self, height: int) -> "ScreenLayout":
        """Resolves the row each region occupies for the current input height."""
        input_lines = len(self.state.input_buffer.split("\n"))
        box_height = max(1, min(6, input_lines))
        input_start = height - box_height
        telemetry_row = input_start - 1
        path_row = telemetry_row - 1
        return ScreenLayout(
            input_lines=input_lines,
            input_box_height=box_height,
            input_start_row=input_start,
            telemetry_row=telemetry_row,
            path_row=path_row,
            separator_row=path_row - 1,
        )

    def _draw_transcript(self, stdscr: Any, w: int, layout: "ScreenLayout") -> None:
        """Draws the scrollable conversation area above the status cockpit."""
        height = max(1, layout.separator_row - 2)
        italic = getattr(curses, "A_ITALIC", 0) or curses.A_UNDERLINE
        for index, (text, tag) in enumerate(
            self.state.get_visible_slice(height=height, width=w)
        ):
            row = 2 + index
            if row >= layout.separator_row:
                break
            attr = self._tag_attribute(tag)
            if tag in INLINE_MARKUP_TAGS:
                self._draw_markup_row(stdscr, row, w, text, attr, italic)
            else:
                _safe(stdscr.addstr, row, 1, text[: w - 2], attr)

    def _tag_attribute(self, tag: str) -> int:
        """Curses attribute for a transcript line tag, falling back to normal."""
        if tag in ("user", "heading"):
            return self._colored(1, curses.A_BOLD)
        if tag == "table_header":
            return curses.A_BOLD
        if tag in ("table_border", "code_border", "blockquote", "dim", "system"):
            return curses.A_DIM
        if tag == "action":
            return self._colored(2, curses.A_NORMAL)
        if tag == "thought":
            return self._colored(3, curses.A_DIM)
        if tag == "error":
            return self._colored(4, curses.A_BOLD)
        if tag == "code":
            return self._colored(2, curses.A_NORMAL)
        if tag == "agent":
            return self._colored(6, curses.A_NORMAL)
        return curses.A_NORMAL

    def _colored(self, pair: int, fallback: int) -> int:
        """A color pair when colors are available, otherwise a plain attribute."""
        return curses.color_pair(pair) | fallback if self.has_colors else fallback

    def _draw_markup_row(
        self, stdscr: Any, row: int, w: int, text: str, attr: int, italic: int
    ) -> None:
        """Draws one line segment by segment so inline styles survive."""
        col = 1
        for segment, style in parse_inline_segments(text):
            if col >= w - 1:
                break
            chunk = segment[: w - 1 - col]
            _safe(stdscr.addstr, row, col, chunk, self._styled(attr, style, italic))
            col += len(chunk)

    def _styled(self, base: int, style: str, italic: int) -> int:
        """Applies an inline markdown style on top of a line's base attribute."""
        if style == "bold":
            return base | curses.A_BOLD
        if style == "italic":
            return base | italic
        if style == "bold_italic":
            return base | curses.A_BOLD | italic
        if style == "code":
            return self._colored(2, curses.A_BOLD)
        return base

    def _draw_status(self, stdscr: Any, w: int, layout: "ScreenLayout") -> None:
        """Draws the path line and the reverse-video telemetry bar."""
        _safe(stdscr.addstr, layout.separator_row, 0, "─" * (w - 1), curses.A_DIM)
        if path := self._path_label():
            _safe(stdscr.addstr, layout.path_row, 0, f" {path} "[: w - 1], curses.A_DIM)
        status = " " + " · ".join(self._status_parts()) + " "
        _safe(stdscr.addstr, layout.telemetry_row, 0, status[: w - 1], curses.A_REVERSE)

    def _path_label(self) -> str:
        """Workspace path with the git branch appended, when one is known."""
        if not self.state.cwd:
            return f"({self.state.git_branch})" if self.state.git_branch else ""
        if self.state.git_branch:
            return f"{self.state.cwd} ({self.state.git_branch})"
        return self.state.cwd

    def _status_parts(self) -> list[str]:
        """The telemetry fields shown in the status bar, in display order."""
        state = self.state
        parts: list[str] = [self._model_label()] if self._model_label() else []
        if state.cache_ttl:
            parts.append(f"cache: {state.cache_ttl}")
        if state.reasoning:
            parts.append(f"reasoning: {state.reasoning}")
        used_tok = format_compact_num(self.state.status_tokens)
        max_tok = format_compact_num(self.state.max_tokens)
        parts.append(f"context: {used_tok} / {max_tok} ({self.state.usage_pct:.1f}%)")
        if state.cache_hit_pct > 0:
            parts.append(f"cache hit: {state.cache_hit_pct:.1f}%")
        if state.scroll_offset > 0:
            parts.append(f"[scrolled +{state.scroll_offset}]")
        if state.turn_active:
            parts.append("[running · Esc cancels after current step · Ctrl+C cancels now]")
        return parts

    def _model_label(self) -> str:
        """Provider-qualified model name, without repeating a provider already in the id."""
        state = self.state
        if state.provider_name and state.model_name:
            if state.model_name.lower().startswith(state.provider_name.lower() + "/"):
                return state.model_name
            return f"{state.provider_name}/{state.model_name}"
        return state.model_name or state.provider_name

    def _draw_autocomplete(self, stdscr: Any, w: int, layout: "ScreenLayout") -> None:
        """Draws the floating file/command menu above the input box."""
        file_matches = self.state.get_matching_file_tags()
        matches = file_matches or self.state.get_matching_commands()
        if not matches:
            return
        is_file_menu = bool(file_matches)
        subcommand = active_subcommand(self.state.input_buffer.lstrip().lower())
        menu_kind = None if is_file_menu else (subcommand[0] if subcommand else None)

        visible, selected = self._visible_menu_slice(matches, layout.separator_row)
        box_height = len(visible) + 4
        box_width = max(10, w - 1)
        top = max(1, layout.separator_row - box_height)
        try:
            self._draw_menu_frame(stdscr, top, box_width, box_height, menu_kind, is_file_menu)
            column_width = menu_column_width(menu_kind, box_width, is_file_menu)
            for row, (command, description) in enumerate(visible, start=1):
                self._draw_menu_row(
                    stdscr, top + row, box_width, command, description, column_width,
                    row - 1 == selected,
                )
            self._draw_menu_footer(stdscr, top, box_width, box_height, len(visible))
        except curses.error:
            pass

    def _visible_menu_slice(
        self, matches: list[tuple[str, str]], separator_row: int
    ) -> tuple[list[tuple[str, str]], int]:
        """Scrolls the match list around the selection to fit above the input box."""
        limit = max(6, min(len(matches), max(6, separator_row - 6)))
        index = max(0, min(self.state.selected_command_idx, len(matches) - 1))
        if len(matches) <= limit:
            return matches, index
        start = max(0, min(index - limit // 2, len(matches) - limit))
        return matches[start : start + limit], index - start

    def _menu_title(self, menu_kind: str | None, is_file_menu: bool) -> str:
        """Heading shown in the menu's top border."""
        if menu_kind is not None:
            return SUBCOMMAND_MENUS[menu_kind][1]
        return " Workspace Files " if is_file_menu else " Available Commands "

    def _draw_menu_frame(
        self, stdscr: Any, top: int, width: int, height: int, menu_kind: str | None, is_file: bool
    ) -> None:
        """Draws the menu's top border with its title."""
        title = self._menu_title(menu_kind, is_file)
        dash_count = max(0, width - 2 - len(title) - 2)
        stdscr.addstr(top, 0, ("┌─" + title + "─" * dash_count + "┐")[:width], curses.A_DIM)

    def _draw_menu_row(
        self, stdscr: Any, y: int, width: int, command: str, description: str,
        column_width: int, selected: bool,
    ) -> None:
        """Draws one menu entry between the box's vertical rules."""
        prefix = "  > " if selected else "    "
        content = f"{prefix}{command:<{column_width}} {description}"
        content = content[: width - 2].ljust(width - 2)
        attr = (
            (curses.color_pair(1) | curses.A_BOLD | curses.A_REVERSE)
            if (selected and self.has_colors)
            else (curses.A_REVERSE if selected else curses.A_NORMAL)
        )
        stdscr.addstr(y, 0, "│", curses.A_DIM)
        stdscr.addstr(y, 1, content, attr)
        stdscr.addstr(y, width - 1, "│", curses.A_DIM)

    def _draw_menu_footer(
        self, stdscr: Any, top: int, width: int, height: int, rows: int
    ) -> None:
        """Draws the menu's separator, keyboard hint, and bottom border."""
        tip = "   Tab / ↑ / ↓: Navigate  ·  Enter / →: Select  ·  Esc: Dismiss"
        separator_row = top + 1 + rows
        stdscr.addstr(separator_row, 0, ("├" + "─" * (width - 2) + "┤")[:width], curses.A_DIM)
        stdscr.addstr(separator_row + 1, 0, "│", curses.A_DIM)
        stdscr.addstr(separator_row + 1, 1, tip[: width - 2].ljust(width - 2), curses.A_DIM)
        stdscr.addstr(separator_row + 1, width - 1, "│", curses.A_DIM)
        stdscr.addstr(top + height - 1, 0, ("└" + "─" * (width - 2) + "┘")[:width], curses.A_DIM)

    def _draw_input(self, stdscr: Any, w: int, layout: "ScreenLayout") -> None:
        """Draws the bottom-anchored prompt and places the cursor."""
        prompt, continuation = "> ", "  "
        available = max(5, w - len(prompt) - 2)
        lines = self.state.input_buffer.split("\n")
        first_line = max(0, layout.input_lines - layout.input_box_height)

        try:
            for row in range(layout.input_box_height):
                index = first_line + row
                text = lines[index]
                label = prompt if index == 0 else continuation
                y = layout.input_start_row + row
                stdscr.addstr(y, 0, label, curses.A_BOLD if index == 0 else curses.A_DIM)
                stdscr.addstr(y, len(label), text[-available:])
            if (cursor := self._cursor_position(w, layout, available, len(prompt))):
                stdscr.move(*cursor)
        except curses.error:
            pass

    def _cursor_position(
        self, w: int, layout: "ScreenLayout", available: int, prompt_width: int
    ) -> tuple[int, int] | None:
        """Where the cursor sits, accounting for the tail of a scrolled input line."""
        first_line = max(0, layout.input_lines - layout.input_box_height)
        before = self.state.input_buffer[: self.state.cursor_pos]
        line_index = before.count("\n")
        if line_index < first_line:
            return None  # the cursor sits above the visible input box
        line_text = self.state.input_buffer.split("\n")[line_index]
        column = len(before.split("\n")[-1])
        offset = max(0, len(line_text) - available)
        return layout.input_start_row + (line_index - first_line), min(
            w - 1, prompt_width + max(0, column - offset)
        )

    def check_cancel_requested(self) -> bool:
        """Non-blocking check if Escape key was pressed by user during execution."""
        if not self.stdscr:
            return False
        try:
            _set_nodelay(self.stdscr, True)
            while True:
                try:
                    ch = self.stdscr.get_wch()
                except (curses.error, EOFError, StopIteration):
                    break
                except KeyboardInterrupt:
                    raise
                if ch in (27, "\x1b") and len(_drain_escape_sequence(self.stdscr)) == 1:
                    return True
            return False
        finally:
            _set_nodelay(self.stdscr, False)

    def _complete_command_keeping_value_menu(self) -> None:
        """Completes the selected command, adding a space when it takes an argument.

        Leaving "/skill" alone would close the value menu the user just opened.
        """
        completed = self.state.complete_selected_command()
        if completed is None:
            return
        takes_argument = any(
            completed in spellings for spellings, _t, _h in SUBCOMMAND_MENUS.values()
        )
        if takes_argument:
            self.state.input_buffer = f"{completed} "
            self.state.cursor_pos = len(self.state.input_buffer)

    def handle_key(self, ch: str | int, on_user_submit: Callable[[str], None]) -> bool:
        """Processes a single keystroke event.

        Returns False if the interactive session should terminate (e.g. /exit), else True.
        """
        # Decode any terminal escape sequences that represent characters (e.g. Shift+2 -> @)
        if isinstance(ch, str) and ch.startswith("\x1b"):
            decoded_char = decode_terminal_sequence(ch)
            if decoded_char:
                ch = decoded_char

        file_matches = self.state.get_matching_file_tags()
        cmd_matches = self.state.get_matching_commands()

        # 1. Shift+Enter / Alt+Enter: Insert newline for multiline input
        if ch in SHIFT_ENTER_SEQUENCES:
            self.state.handle_newline()
            return True

        # 2. Enter: Complete open autocomplete selection or submit
        if ch in ("\n", "\r", curses.KEY_ENTER, 10, 13):
            if file_matches:
                self.state.complete_selected_file_tag()
                return True
            if cmd_matches:
                self.state.complete_selected_command()
            user_text = self.state.clear_input()
            if not user_text:
                return True
            if user_text.lower() in ("/exit", "/quit", "exit", "quit"):
                return False
            on_user_submit(user_text)
            return True

        # 3. Tab: Complete selection or cycle forward
        if ch in ("\t", 9):
            if file_matches:
                if len(file_matches) == 1:
                    self.state.complete_selected_file_tag()
                else:
                    self.state.select_next_command()
                return True
            if cmd_matches:
                if len(cmd_matches) == 1:
                    self._complete_command_keeping_value_menu()
                else:
                    self.state.select_next_command()
                return True
            return True

        # 4. Up / Shift-Tab: Cycle backward or scroll up
        if ch in (curses.KEY_UP, curses.KEY_BTAB, 353, "\x1b[Z"):
            if file_matches or cmd_matches:
                self.state.select_prev_command()
                return True
            self.state.scroll_up(5)
            return True

        # 5. Down: Cycle forward or scroll down
        if ch in (curses.KEY_DOWN,):
            if file_matches or cmd_matches:
                self.state.select_next_command()
                return True
            self.state.scroll_down(5)
            return True

        # 6. Right Arrow: Complete selection
        if ch in (curses.KEY_RIGHT,):
            if file_matches:
                self.state.complete_selected_file_tag()
                return True
            if cmd_matches:
                self._complete_command_keeping_value_menu()
                return True
            return True

        if ch in (curses.KEY_PPAGE,):
            self.state.scroll_up(5)
            return True

        if ch in (curses.KEY_NPAGE,):
            self.state.scroll_down(5)
            return True

        # 7. Escape: cancel the running turn, else dismiss menu / clear tag query,
        #    else (with an empty prompt) exit the session.
        if ch in (27, "\x1b"):
            if self.state.turn_active:
                return True
            if file_matches:
                tag_info = self.state.get_current_file_tag_query()
                if tag_info:
                    at_idx, _ = tag_info
                    self.state.input_buffer = self.state.input_buffer[:at_idx]
                    self.state.cursor_pos = at_idx
                return True
            if cmd_matches:
                self.state.clear_input()
                return True
            if self.state.input_buffer:
                self.state.clear_input()
                return True
            return False

        if ch in (curses.KEY_BACKSPACE, "\x7f", "\x08", 127):
            self.state.handle_backspace()
            return True

        if isinstance(ch, str) and ch.isprintable():
            self.state.handle_input_char(ch)
            return True

        return True

    def run(
        self,
        on_user_submit: Callable[[str], None],
        initial_instruction: str | None = None,
    ) -> None:
        """Launches the interactive curses event loop.

        If `initial_instruction` is given it is submitted *after* the first frame is
        drawn, so a turn started from the command line is visible from its first step
        instead of running against an uninitialised screen.
        """

        def _main(stdscr: Any) -> None:
            self.stdscr = stdscr
            self._init_colors()
            curses.cbreak()
            stdscr.keypad(True)
            curses.curs_set(1)

            _reset_key_reporting()

            self.draw(stdscr)

            if initial_instruction and initial_instruction.strip():
                on_user_submit(initial_instruction.strip())
                self.draw(stdscr)

            while True:
                try:
                    ch = stdscr.get_wch()
                except (KeyboardInterrupt, EOFError):
                    break
                except curses.error:
                    continue

                if ch in (27, "\x1b"):
                    _set_nodelay(stdscr, True)
                    seq = _drain_escape_sequence(stdscr)
                    _set_nodelay(stdscr, False)
                    if len(seq) > 1:
                        ch = seq

                if not self.handle_key(ch, on_user_submit):
                    break
                self.draw(stdscr)

            _reset_key_reporting()

        curses.wrapper(_main)
