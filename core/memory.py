from __future__ import annotations

from typing import Any

from core.types import Message, ToolCall, ToolResult


class Memory:
    """Manages conversation history and implements context compaction policies."""

    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        max_context_chars: int = 15000,
        keep_recent_turns: int = 4,
    ):
        self.system_prompt = system_prompt
        self.max_context_chars = max_context_chars
        self.keep_recent_turns = keep_recent_turns
        self.messages: list[Message] = []
        self._init(system_prompt, user_prompt)

    def _init(self, system_prompt: str, user_prompt: str) -> None:
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))
        if user_prompt:
            self.messages.append(Message(role="user", content=user_prompt))

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
            )
        )

    def total_chars(self) -> int:
        return sum(len(m.content) for m in self.messages)

    def stats(self) -> dict[str, Any]:
        """Returns diagnostic metrics on memory and context budget."""
        chars = self.total_chars()
        pct = round((chars / self.max_context_chars) * 100, 1) if self.max_context_chars else 0.0
        return {
            "total_messages": len(self.messages),
            "total_chars": chars,
            "estimated_tokens": chars // 4,
            "max_context_chars": self.max_context_chars,
            "max_estimated_tokens": self.max_context_chars // 4,
            "usage_pct": pct,
            "by_role": {
                "system": sum(1 for m in self.messages if m.role == "system"),
                "user": sum(1 for m in self.messages if m.role == "user"),
                "assistant": sum(1 for m in self.messages if m.role == "assistant"),
                "tool": sum(1 for m in self.messages if m.role == "tool"),
            },
        }

    def compact_if_needed(self, force: bool = False) -> bool:
        """Prunes older tool observations if total context exceeds budget or if force=True.

        Preserves:
        1. System prompt (if present).
        2. Initial task prompt (first user message).
        3. The most recent `keep_recent_turns` messages.

        Returns:
            True if any messages were compacted, False otherwise.
        """
        if not force and self.total_chars() <= self.max_context_chars:
            return False

        safe_prefix_count = sum(1 for m in self.messages[:2] if m.role in ("system", "user"))
        if len(self.messages) <= safe_prefix_count + self.keep_recent_turns:
            return False

        pruned_any = False
        middle_end_index = len(self.messages) - self.keep_recent_turns
        for i in range(safe_prefix_count, middle_end_index):
            msg = self.messages[i]
            if (
                msg.role == "tool"
                and len(msg.content) > 120
                and not msg.content.startswith("[Tool output collapsed")
            ):
                original_len = len(msg.content)
                msg.content = (
                    f"[Tool output collapsed ({original_len} chars) to conserve context window]"
                )
                pruned_any = True

        return pruned_any

    def reset(self, new_instruction: str = "") -> None:
        """Resets the conversation history while preserving the system instructions."""
        self.messages.clear()
        self._init(self.system_prompt, new_instruction)

    def get_messages(self) -> list[Message]:
        return list(self.messages)
