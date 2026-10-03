from __future__ import annotations

import curses
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class TUILine:
    text: str
    tag: str  # "user", "thought", "action", "error", "agent", "system", "dim"


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

    def add_user(self, text: str) -> None:
        self.lines.append(TUILine(text=f"👤 You: {text}", tag="user"))
        self.scroll_offset = 0

    def add_thought(self, text: str) -> None:
        self.lines.append(TUILine(text=f"● Thinking: {text}", tag="thought"))
        self.scroll_offset = 0

    def add_action(self, text: str) -> None:
        self.lines.append(TUILine(text=f"● {text}", tag="action"))
        self.scroll_offset = 0

    def add_error(self, text: str) -> None:
        self.lines.append(TUILine(text=f"❌ {text}", tag="error"))
        self.scroll_offset = 0

    def add_agent(self, text: str) -> None:
        self.lines.append(TUILine(text="", tag="dim"))
        for line in text.splitlines():
            self.lines.append(TUILine(text=line, tag="agent"))
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
            else:
                chunks = textwrap.wrap(
                    line_item.text,
                    width=eff_width,
                    replace_whitespace=False,
                    drop_whitespace=False,
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
                curses.init_pair(1, curses.COLOR_CYAN, -1)     # User
                curses.init_pair(2, curses.COLOR_GREEN, -1)    # Action / Success
                curses.init_pair(3, curses.COLOR_YELLOW, -1)   # Tool / Warning
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
        brand = f" ⚡ {self.state.title} "
        info_parts: list[str] = []
        if self.state.provider_name or self.state.model_name:
            info_parts.append(f"[{self.state.provider_name}] {self.state.model_name}".strip())
        if self.state.cache_ttl:
            info_parts.append(f"cache: {self.state.cache_ttl}")
        if self.state.reasoning:
            info_parts.append(f"reasoning: {self.state.reasoning}")

        info_str = " · ".join(info_parts)
        header_text = f"{brand}· {info_str}" if info_str else brand
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

        # 3. Main Transcript Area (Rows 2 to H - 4)
        transcript_height = max(1, h - 5)
        visible_lines = self.state.get_visible_slice(height=transcript_height, width=w)
        for i, (text, tag) in enumerate(visible_lines):
            row = 2 + i
            if row >= h - 3:
                break
            attr = curses.A_NORMAL
            if tag == "user":
                attr = curses.color_pair(1) | curses.A_BOLD if self.has_colors else curses.A_BOLD
            elif tag in ("action", "thought"):
                attr = curses.color_pair(2) if self.has_colors else curses.A_NORMAL
            elif tag == "error":
                attr = curses.color_pair(4) | curses.A_BOLD if self.has_colors else curses.A_BOLD
            elif tag in ("dim", "system"):
                attr = curses.A_DIM

            try:
                stdscr.addstr(row, 1, text[: w - 2], attr)
            except curses.error:
                pass

        # 4. Status Bar (Row H - 3)
        status_row = h - 3
        status_text = (
            f" 📊 Context: {self.state.status_tokens:,} / {self.state.max_tokens:,} tokens "
            f"({self.state.usage_pct:.1f}%) · Cache: {self.state.cache_hit_pct:.1f}% · "
            f"Messages: {self.state.messages_count} "
        )
        if self.state.scroll_offset > 0:
            status_text += f" [▲ Scrolled +{self.state.scroll_offset}]"

        try:
            stdscr.addstr(status_row, 0, "─" * (w - 1), curses.A_DIM)
            stdscr.addstr(status_row + 1, 0, status_text[: w - 1], curses.A_REVERSE)
        except curses.error:
            pass

        # 5. Bottom Anchored Input Box (Row H - 1)
        input_row = h - 1
        prompt_label = "❯ "
        avail_input_w = max(5, w - len(prompt_label) - 2)
        display_input = self.state.input_buffer[-avail_input_w:]

        try:
            stdscr.addstr(input_row, 0, prompt_label, curses.A_BOLD)
            stdscr.addstr(input_row, len(prompt_label), display_input)
            # Position hardware cursor
            cursor_col = min(w - 1, len(prompt_label) + len(display_input))
            stdscr.move(input_row, cursor_col)
        except curses.error:
            pass

        stdscr.refresh()

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

                if ch in ("\n", "\r", curses.KEY_ENTER, 10, 13):
                    user_text = self.state.clear_input()
                    if not user_text:
                        continue
                    if user_text.lower() in ("/exit", "/quit", "exit", "quit"):
                        break
                    on_user_submit(user_text)
                    self.draw(stdscr)
                    continue

                if ch in (curses.KEY_BACKSPACE, "\x7f", "\x08", 127):
                    self.state.handle_backspace()
                    self.draw(stdscr)
                    continue

                if ch in (curses.KEY_UP, curses.KEY_PPAGE):
                    self.state.scroll_up(5)
                    self.draw(stdscr)
                    continue

                if ch in (curses.KEY_DOWN, curses.KEY_NPAGE):
                    self.state.scroll_down(5)
                    self.draw(stdscr)
                    continue

                if isinstance(ch, str) and ch.isprintable():
                    self.state.handle_input_char(ch)
                    self.draw(stdscr)

        curses.wrapper(_main)
