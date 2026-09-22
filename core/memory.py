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

            # 1. Prune historical reasoning / thinking blocks from older assistant turns
            if msg.role == "assistant" and (
                "<think>" in msg.content or "<thinking>" in msg.content
            ):
                cleaned = re.sub(
                    r"<(think|thinking)>.*?</\1>\s*", "", msg.content, flags=re.DOTALL
                ).strip()
                if cleaned != msg.content.strip():
                    msg.content = (
                        cleaned
                        if cleaned
                        else "[Reasoning stream collapsed to conserve context window]"
                    )
                    pruned_any = True

            # 2. Prune large tool outputs and inject raw log file pointer for recovery
            elif (
                msg.role == "tool"
                and len(msg.content) > 120
                and not msg.content.startswith("[Tool output collapsed")
            ):
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
                pruned_any = True

        # 3. If files were modified, maintain an active files ledger at the safe prefix
        if self.modified_files and (pruned_any or ledger_index is not None):
            files_list = ", ".join(sorted(self.modified_files))
            ledger_content = f"[Session Files Modified: {files_list}]"
            if ledger_index is not None:
                if self.messages[ledger_index].content != ledger_content:
                    self.messages[ledger_index].content = ledger_content
                    pruned_any = True
            else:
                self.messages.insert(
                    safe_prefix_count,
                    Message(role="system", content=ledger_content),
                )

        return pruned_any

    def reset(self, new_instruction: str = "") -> None:
        """Resets the conversation history while preserving the system instructions."""
        self.messages.clear()
        self.modified_files.clear()
        self._init(self.system_prompt, new_instruction)

    def get_messages(self) -> list[Message]:
        return list(self.messages)
