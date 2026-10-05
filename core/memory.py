from __future__ import annotations

import re
from typing import Any

from core.types import Message, ToolCall, ToolResult


class Memory:
    """Manages conversation history and implements context compaction policies."""

    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        max_context_chars: int = 15000,
        trigger_ratio: float = 0.80,
        keep_recent_turns: int = 4,
    ):
        self.system_prompt = system_prompt
        self.max_context_chars = max_context_chars
        self.trigger_ratio = trigger_ratio
        self.keep_recent_turns = keep_recent_turns
        self.messages: list[Message] = []
        self.modified_files: set[str] = set()
        self._init(system_prompt, user_prompt)

    def _init(self, system_prompt: str, user_prompt: str) -> None:
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))
        if user_prompt:
            self.messages.append(Message(role="user", content=user_prompt))

    def restore_messages(
        self, messages: list[Message], refresh_system_prompt: str | None = None
    ) -> None:
        """Restores memory messages from a previous session, optionally updating system prompt."""
        if not messages:
            return
        restored = list(messages)
        if refresh_system_prompt is not None:
            self.system_prompt = refresh_system_prompt
            if restored and restored[0].role == "system":
                restored[0] = Message(role="system", content=refresh_system_prompt)
            else:
                restored.insert(0, Message(role="system", content=refresh_system_prompt))
        self.messages = restored

    def add_user(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content or ""))

    def add_assistant(self, content: str, tool_calls: list[ToolCall] | None = None) -> None:
        self.messages.append(
            Message(
                role="assistant",
                content=content or "",
                tool_calls=tool_calls if tool_calls else None,
            )
        )

    def add_tool_result(self, result: ToolResult) -> None:
        self.messages.append(
            Message(
                role="tool",
                content=result.output,
                tool_call_id=result.call_id,
                raw_log_path=result.raw_log_path,
            )
        )

    def record_modified_file(self, path: str) -> None:
        """Records a file path modified during this session."""
        self.modified_files.add(path)

    def total_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)

    def stats(self) -> dict[str, Any]:
        """Returns diagnostic metrics on memory and context budget."""
        chars = self.total_chars()
        pct = round((chars / self.max_context_chars) * 100, 1) if self.max_context_chars else 0.0
        trigger_threshold = int(self.max_context_chars * self.trigger_ratio)
        return {
            "total_messages": len(self.messages),
            "total_chars": chars,
            "estimated_tokens": chars // 4,
            "max_context_chars": self.max_context_chars,
            "max_estimated_tokens": self.max_context_chars // 4,
            "trigger_ratio": self.trigger_ratio,
            "trigger_threshold_chars": trigger_threshold,
            "usage_pct": pct,
            "modified_files": sorted(self.modified_files),
            "by_role": {
                "system": sum(1 for m in self.messages if m.role == "system"),
                "user": sum(1 for m in self.messages if m.role == "user"),
                "assistant": sum(1 for m in self.messages if m.role == "assistant"),
                "tool": sum(1 for m in self.messages if m.role == "tool"),
            },
        }

    def _prune_reasoning_blocks(self, msg: Message) -> bool:
        """Strips historical reasoning blocks from older assistant turns."""
        if msg.role != "assistant" or (
            "<think>" not in msg.content and "<thinking>" not in msg.content
        ):
            return False
        cleaned = re.sub(r"<(think|thinking)>.*?</\1>\s*", "", msg.content, flags=re.DOTALL).strip()
        if cleaned != msg.content.strip():
            msg.content = (
                cleaned if cleaned else "[Reasoning stream collapsed to conserve context window]"
            )
            return True
        return False

    def _collapse_tool_output(self, msg: Message, min_chars: int = 120) -> bool:
        """Replaces large historical tool observations with a tombstone pointing to disk logs."""
        if (
            msg.role != "tool"
            or len(msg.content) <= min_chars
            or msg.content.startswith("[Tool output collapsed")
        ):
            return False
        original_len = len(msg.content)
        if msg.raw_log_path:
            msg.content = (
                f"[Tool output collapsed ({original_len} chars). "
                f"Full log: `{msg.raw_log_path}`. Use `read_file` to inspect if needed.]"
            )
        else:
            msg.content = (
                f"[Tool output collapsed ({original_len} chars) to conserve context window]"
            )
        return True

    def _sync_files_ledger(
        self, safe_prefix_count: int, ledger_index: int | None, pruned_any: bool
    ) -> bool:
        """Maintains an active files-modified ledger in context when compaction occurs."""
        if not self.modified_files or (not pruned_any and ledger_index is None):
            return False
        files_list = ", ".join(sorted(self.modified_files))
        ledger_content = f"[Session Files Modified: {files_list}]"
        if ledger_index is not None:
            if self.messages[ledger_index].content != ledger_content:
                self.messages[ledger_index].content = ledger_content
                return True
        else:
            self.messages.insert(
                safe_prefix_count,
                Message(role="system", content=ledger_content),
            )
        return False

    def compact_if_needed(self, force: bool = False) -> bool:
        """Prunes older observations & thinking streams if context exceeds budget or force=True.

        Preserves:
        1. System prompt (if present).
        2. Initial task prompt (first user message).
        3. The most recent `keep_recent_turns` messages.
        4. Replaces reasoning blocks (<think>...</think>) with placeholders.
        5. Replaces large tool observations with enriched tombstones containing disk log pointers.
        6. Injects or updates an active session ledger of modified files.

        Returns:
            True if any messages were compacted, False otherwise.
        """
        threshold = int(self.max_context_chars * self.trigger_ratio)
        if not force and self.total_chars() <= threshold:
            return False

        safe_prefix_count = sum(1 for m in self.messages[:2] if m.role in ("system", "user"))
        ledger_index: int | None = None
        if len(self.messages) > safe_prefix_count and self.messages[
            safe_prefix_count
        ].content.startswith("[Session Files Modified:"):
            ledger_index = safe_prefix_count
            safe_prefix_count += 1

        if len(self.messages) <= safe_prefix_count + self.keep_recent_turns:
            return False

        pruned_any = False
        middle_end_index = len(self.messages) - self.keep_recent_turns
        for i in range(safe_prefix_count, middle_end_index):
            msg = self.messages[i]
            if self._prune_reasoning_blocks(msg) or self._collapse_tool_output(msg):
                pruned_any = True

        if self._sync_files_ledger(safe_prefix_count, ledger_index, pruned_any):
            pruned_any = True

        return pruned_any

    def reset(self, new_instruction: str = "") -> None:
        """Resets the conversation history while preserving the system instructions."""
        self.messages.clear()
        self.modified_files.clear()
        self._init(self.system_prompt, new_instruction)

    def drop_incomplete_tool_turn(self) -> int:
        """Removes a trailing assistant turn whose tool results were never recorded.

        An interruption between recording the assistant message and appending its
        tool results leaves a tool_call with no matching tool message, which
        chat-completions APIs reject. Returns the number of messages removed.
        """
        last_assistant = None
        for i in range(len(self.messages) - 1, -1, -1):
            if self.messages[i].role == "assistant" and self.messages[i].tool_calls:
                last_assistant = i
                break
        if last_assistant is None:
            return 0

        answered = {
            m.tool_call_id for m in self.messages[last_assistant + 1 :] if m.role == "tool"
        }
        expected = {tc.id for tc in (self.messages[last_assistant].tool_calls or [])}
        if expected <= answered:
            return 0

        removed = len(self.messages) - last_assistant
        del self.messages[last_assistant:]
        return removed

    def get_messages(self) -> list[Message]:
        return list(self.messages)


def format_compact_num(n: int | float | None) -> str:
    """Formats an integer or float count concisely using 'k' and 'M' suffixes.

    Examples:
        0 -> '0'
        500 -> '500'
        1000 -> '1k'
        1420 -> '1.4k'
        8000 -> '8k'
        128000 -> '128k'
        1000000 -> '1M'
        1500000 -> '1.5M'
    """
    if n is None:
        return "0"
    try:
        val = float(n)
    except (TypeError, ValueError):
        return str(n)
    if abs(val) >= 999_950:
        scaled = round(val / 1_000_000, 1)
        return f"{scaled:g}M"
    if abs(val) >= 999.5:
        scaled = round(val / 1_000, 1)
        return f"{scaled:g}k"
    return str(int(val)) if isinstance(n, int) or val.is_integer() else f"{val:.1f}"

