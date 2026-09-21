from __future__ import annotations

from typing import Optional

from core.types import Message, ToolCall, ToolResult


class Memory:
    """Manages the conversation history and implements context compaction policies."""

    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        max_context_chars: int = 15000,
        keep_recent_turns: int = 4,
    ):
        self.max_context_chars = max_context_chars
        self.keep_recent_turns = keep_recent_turns
        self.messages: list[Message] = []
        self._init(system_prompt, user_prompt)

    def _init(self, system_prompt: str, user_prompt: str) -> None:
        if system_prompt:
            self.messages.append(Message(role="system", content=system_prompt))
        if user_prompt:
            self.messages.append(Message(role="user", content=user_prompt))

    def add_assistant(self, content: str, tool_calls: Optional[list[ToolCall]] = None) -> None:
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

    def compact_if_needed(self) -> bool:
        """Prunes older tool observations if total message content exceeds max_context_chars.

        Preserves:
        1. System prompt (first message).
        2. Task prompt (second message).
        3. The most recent `keep_recent_turns` messages.

        Returns:
            True if compaction occurred, False otherwise.
        """
        if self.total_chars() <= self.max_context_chars:
            return False

        # If history is too short to safely prune, skip
        safe_prefix_count = 2  # System + Task
        if len(self.messages) <= safe_prefix_count + self.keep_recent_turns:
            return False

        pruned_any = False
        # Target middle messages (between initial prompts and recent turns)
        middle_end_index = len(self.messages) - self.keep_recent_turns
        for i in range(safe_prefix_count, middle_end_index):
            msg = self.messages[i]
            # Prune older tool observations that are still substantial
            if msg.role == "tool" and len(msg.content) > 120:
                original_len = len(msg.content)
                collapsed_notice = (
                    f"[Tool output collapsed ({original_len} chars) to conserve context window]"
                )
                msg.content = collapsed_notice
                pruned_any = True

        return pruned_any

    def get_messages(self) -> list[Message]:
        return list(self.messages)
