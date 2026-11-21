from __future__ import annotations

import json
import re
from typing import Any

from core.types import ASSISTANT_EXTRA_KEYS, Message, ToolCall, ToolResult

COMPACTION_SUMMARY_HEADER = "[Context Compaction Summary]"
_REASONING_TAG_RE = re.compile(r"<(think|thinking)>.*?</\1>\s*", flags=re.DOTALL)
_ERROR_LINE_RE = re.compile(
    r"(error|exception|traceback|failed|assertionerror|permissionerror|filenotfounderror|exit code\s+[1-9])",
    re.IGNORECASE,
)


def _serialize_extra_value(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, default=str)
    except Exception:
        return str(value)


def _estimate_message_chars(msg: Message) -> int:
    """Estimates character weight of a message including tool_calls and extra_fields."""
    total = len(msg.content or "")
    if msg.tool_calls:
        for tc in msg.tool_calls:
            total += len(tc.id or "") + len(tc.name or "")
            if isinstance(tc.arguments, dict):
                total += len(json.dumps(tc.arguments, default=str))
            elif tc.arguments is not None:
                total += len(str(tc.arguments))
            if tc.extra_content:
                total += len(_serialize_extra_value(tc.extra_content))
    if msg.extra_fields:
        for key, value in msg.extra_fields.items():
            if value is not None:
                total += len(str(key)) + len(_serialize_extra_value(value))
    return total


def _strip_thinking_tags(text: str) -> str:
    if not text or ("<think>" not in text and "<thinking>" not in text):
        return text.strip() if text else ""
    return _REASONING_TAG_RE.sub("", text).strip()


def _truncate_text(text: str, limit: int) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: max(0, limit - 3)].rstrip() + "..."


def _is_compaction_summary(msg: Message) -> bool:
    return (msg.content or "").startswith(COMPACTION_SUMMARY_HEADER)


def _format_tool_call_brief(tc: ToolCall) -> str:
    args = tc.arguments if isinstance(tc.arguments, dict) else {}
    if tc.name in ("read_file", "write_file", "edit_file") and "path" in args:
        arg_str = str(args["path"])
    elif tc.name == "bash" and "command" in args:
        arg_str = _truncate_text(str(args["command"]), 72)
    elif args:
        arg_str = _truncate_text(
            ", ".join(f"{k}={v!r}" for k, v in list(args.items())[:3]),
            72,
        )
    else:
        arg_str = ""
    return f"{tc.name}({arg_str})" if arg_str else f"{tc.name}()"


def _summarize_tool_output(content: str, raw_log_path: str | None) -> tuple[str, str | None]:
    """Produces a concise digest of a tool output and extracts any prominent error line."""
    lines = [line.strip() for line in (content or "").splitlines() if line.strip()]
    error_line: str | None = None
    for line in lines:
        if _ERROR_LINE_RE.search(line):
            error_line = _truncate_text(line, 120)
            break

    if raw_log_path:
        outcome = f"see log `{raw_log_path}` ({len(content)} chars)"
        if error_line:
            outcome = f"{outcome}; {error_line}"
        return outcome, error_line

    if not lines:
        return "(empty output)", None
    if len(lines) == 1:
        return _truncate_text(lines[0], 110), error_line
    first = _truncate_text(lines[0], 90)
    return f"{len(lines)} lines: {first}", error_line


def _parse_prior_summary(content: str) -> dict[str, list[str]]:
    """Extracts bullet lists from a prior compaction summary message."""
    sections: dict[str, list[str]] = {
        "followups": [],
        "actions": [],
        "findings": [],
        "errors": [],
    }
    current: str | None = None
    header_map = {
        "User follow-ups:": "followups",
        "Completed actions:": "actions",
        "Key findings:": "findings",
        "Errors / warnings:": "errors",
    }
    for raw_line in content.splitlines():
        line = raw_line.strip()
        if line in header_map:
            current = header_map[line]
            continue
        if line.startswith("- ") and current is not None:
            sections[current].append(line[2:].strip())
        elif line and not line.startswith("[") and not line.startswith("Modified files:"):
            current = None
    return sections


class Memory:
    """Manages conversation history and implements KV-cache-friendly context compaction."""

    def __init__(
        self,
        system_prompt: str,
        user_prompt: str,
        max_context_tokens: int = 64000,
        trigger_ratio: float = 0.80,
        keep_recent_turns: int = 4,
    ):
        self.system_prompt = system_prompt
        self.max_context_tokens = max_context_tokens
        self.trigger_ratio = trigger_ratio
        self.keep_recent_turns = keep_recent_turns
        self.messages: list[Message] = []
        self.modified_files: set[str] = set()
        self._calibrated_prompt_tokens: int | None = None
        self._calibrated_raw_tokens: int | None = None
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
        self._calibrated_prompt_tokens = None
        self._calibrated_raw_tokens = None

    def add_user(self, content: str) -> None:
        self.messages.append(Message(role="user", content=content or ""))

    def add_assistant(
        self,
        content: str,
        tool_calls: list[ToolCall] | None = None,
        extra_fields: dict[str, Any] | None = None,
    ) -> None:
        self.messages.append(
            Message(
                role="assistant",
                content=content or "",
                tool_calls=tool_calls if tool_calls else None,
                extra_fields=dict(extra_fields) if extra_fields else None,
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
        """Records a file path modified during this session without mutating history."""
        self.modified_files.add(path)

    def raw_estimated_tokens(self) -> int:
        """Estimates tokens purely from in-memory message characters (~4 chars per token)."""
        return sum(_estimate_message_chars(m) for m in self.messages) // 4

    def calibrate_tokens(
        self, actual_prompt_tokens: int, baseline_raw_tokens: int | None = None
    ) -> None:
        """Calibrates token estimates against actual prompt_tokens reported by a provider."""
        if actual_prompt_tokens <= 0:
            return
        baseline = (
            baseline_raw_tokens if baseline_raw_tokens is not None else self.raw_estimated_tokens()
        )
        self._calibrated_prompt_tokens = int(actual_prompt_tokens)
        self._calibrated_raw_tokens = max(0, int(baseline))

    def estimated_tokens(self) -> int:
        """Returns calibrated token estimate when provider telemetry is available, else raw estimate."""
        raw_now = self.raw_estimated_tokens()
        if self._calibrated_prompt_tokens is None or self._calibrated_raw_tokens is None:
            return raw_now
        if self._calibrated_raw_tokens <= 0:
            return max(raw_now, self._calibrated_prompt_tokens)
        delta = raw_now - self._calibrated_raw_tokens
        if delta >= 0:
            return self._calibrated_prompt_tokens + delta
        return max(
            0,
            round(self._calibrated_prompt_tokens * (raw_now / self._calibrated_raw_tokens)),
        )

    def stats(self) -> dict[str, Any]:
        """Returns diagnostic metrics on memory and context budget."""
        tokens = self.estimated_tokens()
        pct = round((tokens / self.max_context_tokens) * 100, 1) if self.max_context_tokens else 0.0
        trigger_threshold = int(self.max_context_tokens * self.trigger_ratio)
        return {
            "total_messages": len(self.messages),
            "estimated_tokens": tokens,
            "calibrated_prompt_tokens": self._calibrated_prompt_tokens,
            "max_context_tokens": self.max_context_tokens,
            "max_estimated_tokens": self.max_context_tokens,
            "trigger_ratio": self.trigger_ratio,
            "trigger_threshold_tokens": trigger_threshold,
            "usage_pct": pct,
            "modified_files": sorted(self.modified_files),
            "by_role": {
                "system": sum(1 for m in self.messages if m.role == "system"),
                "user": sum(1 for m in self.messages if m.role == "user"),
                "assistant": sum(1 for m in self.messages if m.role == "assistant"),
                "tool": sum(1 for m in self.messages if m.role == "tool"),
            },
        }

    def _immutable_prefix_length(self) -> int:
        """Returns the length of the immutable leading [system?, initial_user?] prefix."""
        if not self.messages:
            return 0
        if self.messages[0].role == "system":
            if len(self.messages) > 1 and self.messages[1].role == "user":
                return 2
            return 1
        if self.messages[0].role == "user":
            return 1
        return 0

    def _prune_message_reasoning(self, msg: Message) -> bool:
        """Strips thinking tags and provider reasoning extra_fields from an assistant message."""
        if msg.role != "assistant":
            return False
        changed = False
        if "<think>" in msg.content or "<thinking>" in msg.content:
            cleaned = _strip_thinking_tags(msg.content)
            if cleaned != msg.content.strip():
                if cleaned:
                    msg.content = cleaned
                elif msg.tool_calls:
                    msg.content = ""
                else:
                    msg.content = "[Reasoning stream collapsed to conserve context window]"
                changed = True
        if msg.extra_fields:
            for key in ASSISTANT_EXTRA_KEYS:
                if key in msg.extra_fields:
                    msg.extra_fields.pop(key)
                    changed = True
            if not msg.extra_fields:
                msg.extra_fields = None
        return changed

    def _prune_stale_reasoning(self, messages: list[Message]) -> bool:
        """Prunes reasoning payloads from older retained assistant turns."""
        last_assistant_idx: int | None = None
        last_user_idx: int | None = None
        for idx, msg in enumerate(messages):
            if msg.role == "assistant":
                last_assistant_idx = idx
            elif msg.role == "user":
                last_user_idx = idx

        if last_assistant_idx is None:
            return False

        changed = False
        for idx, msg in enumerate(messages):
            if msg.role != "assistant":
                continue
            is_stale = idx < last_assistant_idx or (
                last_user_idx is not None and idx < last_user_idx
            )
            if is_stale and self._prune_message_reasoning(msg):
                changed = True
        return changed

    def _build_compaction_summary(self, compacted_slice: list[Message]) -> Message:
        """Collapses compacted middle turns into one structured, bounded summary message."""
        followups: list[str] = []
        actions: list[str] = []
        findings: list[str] = []
        errors: list[str] = []

        tool_results_by_id: dict[str, Message] = {
            m.tool_call_id: m for m in compacted_slice if m.role == "tool" and m.tool_call_id
        }
        matched_tool_ids: set[str] = set()

        for msg in compacted_slice:
            if _is_compaction_summary(msg):
                prior = _parse_prior_summary(msg.content)
                followups.extend(prior["followups"])
                actions.extend(prior["actions"])
                findings.extend(prior["findings"])
                errors.extend(prior["errors"])
            elif msg.role == "user":
                text = _truncate_text(msg.content, 160)
                if text:
                    followups.append(text)
            elif msg.role == "assistant":
                cleaned = _strip_thinking_tags(msg.content)
                if cleaned and cleaned != "[Reasoning stream collapsed to conserve context window]":
                    findings.append(_truncate_text(cleaned, 160))
                for tc in msg.tool_calls or []:
                    brief = _format_tool_call_brief(tc)
                    tmsg = tool_results_by_id.get(tc.id)
                    if tmsg is not None:
                        matched_tool_ids.add(tc.id)
                        outcome, err = _summarize_tool_output(tmsg.content, tmsg.raw_log_path)
                        actions.append(f"{brief} -> {outcome}")
                        if err:
                            errors.append(f"{brief}: {err}")
                    else:
                        actions.append(brief)
            elif msg.role == "tool":
                if msg.tool_call_id and msg.tool_call_id in matched_tool_ids:
                    continue
                outcome, err = _summarize_tool_output(msg.content, msg.raw_log_path)
                actions.append(f"tool -> {outcome}")
                if err:
                    errors.append(err)

        followups = followups[-6:]
        actions = actions[-12:]
        findings = findings[-8:]
        errors = errors[-8:]

        max_summary_chars = max(400, min(3200, self.max_context_tokens * 2))

        def _render() -> str:
            parts = [COMPACTION_SUMMARY_HEADER]
            if self.modified_files:
                parts.append(f"Modified files: {', '.join(sorted(self.modified_files))}")
            if followups:
                parts.append("User follow-ups:\n" + "\n".join(f"- {item}" for item in followups))
            if actions:
                parts.append("Completed actions:\n" + "\n".join(f"- {item}" for item in actions))
            if findings:
                parts.append("Key findings:\n" + "\n".join(f"- {item}" for item in findings))
            if errors:
                parts.append("Errors / warnings:\n" + "\n".join(f"- {item}" for item in errors))
            return "\n".join(parts)

        rendered = _render()
        while len(rendered) > max_summary_chars and (actions or findings or followups or errors):
            if len(actions) > 2:
                actions.pop(0)
            elif len(findings) > 1:
                findings.pop(0)
            elif len(followups) > 1:
                followups.pop(0)
            elif len(errors) > 1:
                errors.pop(0)
            elif actions:
                actions.pop(0)
            elif findings:
                findings.pop(0)
            else:
                break
            rendered = _render()

        if len(rendered) > max_summary_chars:
            rendered = rendered[: max_summary_chars - 3].rstrip() + "..."

        return Message(role="user", content=rendered)

    def compact_if_needed(self, force: bool = False) -> bool:
        """Compacts older turns at a clean boundary if context exceeds budget or force=True.

        Preserves:
        1. Immutable leading prefix (system prompt and initial user task prompt).
        2. The most recent `keep_recent_turns` messages without splitting tool-call pairs.
        3. Replaces compacted middle turns with one concise structured summary (including
           modified files, follow-ups, tool actions/log paths, findings, and errors).
        4. Prunes stale reasoning tags and `extra_fields` from older retained assistant turns.

        Returns:
            True if memory was compacted or stale reasoning was pruned, False otherwise.
        """
        threshold_tokens = int(self.max_context_tokens * self.trigger_ratio)
        if not force and self.estimated_tokens() <= threshold_tokens:
            return False

        prefix_len = self._immutable_prefix_length()
        keep = max(0, self.keep_recent_turns)
        split_idx = len(self.messages) - keep
        while (
            split_idx > prefix_len
            and split_idx < len(self.messages)
            and self.messages[split_idx].role == "tool"
        ):
            split_idx -= 1

        if split_idx <= prefix_len:
            return self._prune_stale_reasoning(self.messages[prefix_len:])

        prefix = list(self.messages[:prefix_len])
        compacted_slice = self.messages[prefix_len:split_idx]
        recent_slice = list(self.messages[split_idx:])

        has_compaction_work = any(not _is_compaction_summary(m) for m in compacted_slice)
        if not has_compaction_work:
            return self._prune_stale_reasoning(recent_slice)

        summary_msg = self._build_compaction_summary(compacted_slice)
        self._prune_stale_reasoning(recent_slice)
        self.messages = [*prefix, summary_msg, *recent_slice]
        return True

    def reset(self, new_instruction: str = "") -> None:
        """Resets the conversation history while preserving the system instructions."""
        self.messages.clear()
        self.modified_files.clear()
        self._calibrated_prompt_tokens = None
        self._calibrated_raw_tokens = None
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

        answered = {m.tool_call_id for m in self.messages[last_assistant + 1 :] if m.role == "tool"}
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
