"""Run lifecycle: the step driver, session discovery, and resumption.

The step driver owns the retry-and-give-up policy for a single turn. The session
half owns "which previous run do I mean, what does it contain, and how do I load
it" — including rebuilding a conversation from a persisted trace. Both are pure
bookkeeping around :class:`~core.loop.AgentLoop`, so batch runs, the CLI
session, and the TUI all share one policy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable

from core.loop import AgentLoop
from core.protocols import DeadlineAwareModel, Model, RateLimitAwareModel
from core.ratelimit import RateLimitError
from core.types import Message

if TYPE_CHECKING:
    from core.tracer import Tracer

# finish_reason values, matching RunResult.finish_reason
COMPLETED = "completed"
MAX_STEPS_EXCEEDED = "max_steps_exceeded"
TIMEOUT = "timeout"
INTERRUPTED = "interrupted"
RATE_LIMITED = "rate_limited"
ERROR = "error"

NOT_FOUND = "Session not found: no resumable session available"

# Enough history for numeric / substring session selectors to resolve.
SEARCH_LIMIT = 50


@dataclass(frozen=True)
class StepOutcome:
    """Where a turn stopped, and which step index to continue from."""

    finished: bool
    reason: str
    next_step: int


def rate_limit_wait(model: Model) -> float:
    """Cumulative seconds ``model`` has spent honouring provider rate limits."""
    raw = (
        model.total_rate_limit_wait_s
        if isinstance(model, RateLimitAwareModel)
        else getattr(model, "total_rate_limit_wait_s", 0.0)
    )
    try:
        return float(raw or 0.0)
    except (TypeError, ValueError):
        return 0.0


def run_steps(
    model: Model,
    loop: AgentLoop,
    tracer: Tracer,
    start_step: int,
    max_steps: int,
    *,
    deadline: float | None = None,
    max_retries: int = 3,
    retry_delay: float = 5.0,
    on_finish: Callable[[str], None] | None = None,
) -> StepOutcome:
    """Drives the agent loop for up to ``max_steps`` steps, retrying transient errors.

    Time spent waiting out provider rate limits is added back to the deadline, so
    a throttled run is not charged for the provider's own backoff.
    """
    step_idx = start_step
    step_attempts = 0

    while (step_idx - start_step) < max_steps:
        effective_deadline = (deadline + rate_limit_wait(model)) if deadline else None
        if effective_deadline and time.time() > effective_deadline:
            tracer.log("\n⚠️ Task reached maximum allowed timeout.")
            return StepOutcome(False, TIMEOUT, step_idx)

        if effective_deadline and isinstance(model, DeadlineAwareModel):
            model.set_deadline(effective_deadline)

        try:
            is_finished, step_trace = loop.step(step_index=step_idx)
        except KeyboardInterrupt:
            tracer.log("\n⚠️ Interrupted by user (Ctrl+C).")
            return StepOutcome(False, INTERRUPTED, step_idx)
        except TimeoutError as exc:
            tracer.log(f"\n⚠️ Task reached maximum allowed timeout: {exc}")
            return StepOutcome(False, TIMEOUT, step_idx)
        except RateLimitError as exc:
            tracer.log(
                f"\n🛑 Step {step_idx} rate limited (HTTP 429): {exc}\n"
                "Progress has been saved. You can resume this benchmark with --resume once rate limits reset."
            )
            return StepOutcome(False, RATE_LIMITED, step_idx)
        except Exception as exc:
            if _retry_step(exc, step_idx, step_attempts + 1, max_retries, retry_delay, tracer):
                step_attempts += 1
                continue
            return StepOutcome(False, ERROR, step_idx)

        tracer.record_step(step_trace)
        step_attempts = 0
        step_idx += 1
        if is_finished:
            if on_finish:
                on_finish(step_trace.model_response.get("content", ""))
            return StepOutcome(True, COMPLETED, step_idx)

    return StepOutcome(False, MAX_STEPS_EXCEEDED, step_idx)


def _retry_step(
    exc: Exception,
    step_idx: int,
    attempt: int,
    max_retries: int,
    retry_delay: float,
    tracer: Tracer,
) -> bool:
    """Logs and sleeps before a step retry. Returns False once retries run out."""
    if attempt > max_retries:
        tracer.log(f"\n❌ Step {step_idx} failed after {max_retries} retries: {exc}")
        return False
    sleep_s = retry_delay * attempt
    tracer.log(
        f"\n⚠️ Step {step_idx} encountered an error: {exc}\n"
        f"⏳ Sleeping {sleep_s:.1f}s and retrying step ({attempt}/{max_retries})..."
    )
    time.sleep(sleep_s)
    return True


def resolve_session_dir(
    runs_dir: Path,
    selector: str | Path | None,
    sessions: list[dict],
) -> tuple[Path | None, str]:
    """Resolves a session selector to a run directory.

    Accepts a run directory path, a run id, a 1-based index into ``sessions``,
    or a substring of a run id. ``None`` selects the most recent session.

    Returns:
        ``(run_dir, error)``; exactly one of the two is set.
    """
    if not selector:
        return (Path(sessions[0]["run_dir"]) if sessions else None), ""

    candidate = Path(selector)
    if candidate.is_dir():
        return candidate, ""
    if (runs_dir / str(selector)).is_dir():
        return runs_dir / str(selector), ""

    text = str(selector).strip()
    if text.isdigit():
        index = int(text) - 1
        if not 0 <= index < len(sessions):
            return None, f"Invalid session index: {selector} (found {len(sessions)} sessions)"
        return Path(sessions[index]["run_dir"]), ""

    for session in sessions:
        if text in session["run_id"]:
            return Path(session["run_dir"]), ""
    return None, NOT_FOUND


def load_session(
    runs_dir: Path,
    workspace: Path | None,
    selector: str | Path | None,
) -> tuple[list[Message], dict]:
    """Loads a previous session's messages and metadata.

    Returns ``([], {"error": ...})`` when the selector cannot be resolved.
    """
    sessions = find_recent_sessions(runs_dir, workspace, limit=SEARCH_LIMIT)
    target, error = resolve_session_dir(runs_dir, selector, sessions)
    if target is None or not target.is_dir():
        return [], {"error": error or NOT_FOUND}

    raw_messages, meta = load_session_messages(target)
    return [Message.from_dict(m) for m in raw_messages], meta


# --------------------------------------------------------------------------
# Reading persisted sessions
# --------------------------------------------------------------------------

PREVIEW_MAX_CHARS = 60
_ELLIPSIS = "..."


def _read_json(path: Path) -> dict[str, Any] | None:
    """Reads a JSON object, returning ``None`` for missing, unreadable, or malformed files."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    return data if isinstance(data, dict) else None


def _read_jsonl_steps(path: Path) -> list[dict[str, Any]]:
    """Reads an append-only trajectory into step records, skipping unreadable lines."""
    steps: list[dict[str, Any]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return steps
    for line in lines:
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except Exception:
            continue
        if isinstance(record, dict):
            steps.append(record)
    return steps


def _first_user_text(messages: Any) -> str:
    """Returns the first user message content, or ``""`` when there is none."""
    for message in messages or []:
        if isinstance(message, dict) and message.get("role") == "user" and message.get("content"):
            return str(message["content"]).strip()
    return ""


def _preview_from_step(step: dict[str, Any] | None) -> str:
    """Returns the user prompt that opened a step, which is the session's first user turn."""
    return _first_user_text((step or {}).get("prompt_messages"))


def _clip_preview(text: str) -> str:
    """Bounds a preview to one display line."""
    if len(text) <= PREVIEW_MAX_CHARS:
        return text
    return text[: PREVIEW_MAX_CHARS - len(_ELLIPSIS)] + _ELLIPSIS


def _describe_session(entry: Path, mtime: float, target_ws: str | None) -> dict[str, Any] | None:
    """Summarizes one run directory, or returns ``None`` when it is not a session.

    Prefers ``trace.json`` (complete) and falls back to ``trajectory.jsonl`` for a
    run that crashed before it could write its trace.
    """
    trace_file = entry / "trace.json"
    jsonl_file = entry / "trajectory.jsonl"
    if not (trace_file.is_file() or jsonl_file.is_file()):
        return None

    total_steps = 0
    preview = ""
    workspace: str | None = None
    finish_reason = ""
    task_id = entry.name

    if trace_file.is_file():
        data = _read_json(trace_file) or {}
        total_steps = int(data.get("total_steps") or 0)
        workspace = data.get("workspace")
        finish_reason = data.get("finish_reason", "")
        task = data.get("task") or {}
        task_id = task.get("id") or entry.name
        preview = (task.get("instruction") or "").strip()
        if not preview and data.get("messages"):
            preview = _first_user_text(data["messages"])
        elif not preview and data.get("steps"):
            preview = _preview_from_step(data["steps"][-1])

    if not preview and jsonl_file.is_file():
        steps = _read_jsonl_steps(jsonl_file)
        total_steps = max(total_steps, len(steps))
        if steps:
            preview = _preview_from_step(steps[-1])

    return {
        "run_id": entry.name,
        "run_dir": str(entry),
        "task_id": task_id,
        "total_steps": total_steps,
        "preview": _clip_preview(preview) or "(no user prompt)",
        "workspace": workspace,
        "mtime": mtime,
        "finish_reason": finish_reason,
        "matches_workspace": bool(target_ws and workspace and target_ws == workspace),
    }


def _pass1_ceiling(limit: int, target_ws: str | None) -> int:
    """How many recent runs the cheap first pass may inspect.

    Without a workspace to match there is nothing to search for, so the limit is
    the ceiling and the expensive second pass never runs.
    """
    return max(limit * 5, 50) if target_ws else limit


def _scan_for_matches(
    candidates: list[tuple[float, Path]],
    target_ws: str | None,
    limit: int,
    *,
    scan_all: bool,
    cheap_ceiling: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """One pass over candidates, splitting them into workspace matches and the rest."""
    matching: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []

    for index, (mtime, entry) in enumerate(candidates):
        if scan_all:
            if index < cheap_ceiling:
                continue  # the cheap pass already inspected this one
        elif index >= cheap_ceiling:
            break  # the cheap pass stops at its ceiling
        record = _describe_session(entry, mtime, target_ws)
        if record is None:
            continue
        if record["matches_workspace"]:
            if len(matching) >= limit:
                break
            matching.append(record)
        elif not scan_all:
            others.append(record)

    return matching, others


def find_recent_sessions(
    runs_dir: Path | str,
    workspace: Path | str | None = None,
    limit: int = 10,
) -> list[dict[str, Any]]:
    """Discovers previous sessions from ``runs_dir``, newest first, workspace matches first.

    Scans a cheap window of recent runs first and only walks the full history when
    that window contains no match, so an older session for the same repository
    stays reachable instead of being hidden by the scan ceiling.
    """
    base = Path(runs_dir)
    if not base.is_dir():
        return []

    target_ws = str(Path(workspace).resolve()) if workspace else None

    candidates: list[tuple[float, Path]] = []
    for entry in base.iterdir():
        if not entry.is_dir():
            continue
        try:
            mtime = entry.stat().st_mtime
        except Exception:
            mtime = 0.0
        candidates.append((mtime, entry))
    candidates.sort(key=lambda c: c[0], reverse=True)

    ceiling = _pass1_ceiling(limit, target_ws)
    matching: list[dict[str, Any]] = []
    others: list[dict[str, Any]] = []
    for scan_all in (False, True):
        matching, found_others = _scan_for_matches(
            candidates, target_ws, limit, scan_all=scan_all, cheap_ceiling=ceiling
        )
        if not scan_all:
            others = found_others
        if matching or not target_ws:
            break

    combined = matching + others
    combined.sort(key=lambda s: (1 if s["matches_workspace"] else 0, s["mtime"]), reverse=True)
    return combined[:limit]


def _select_reconstruction_step(steps: list[dict[str, Any]]) -> dict[str, Any]:
    """Returns the most recent step that actually recorded conversation state.

    When a step cannot be serialized, :meth:`core.tracer.Tracer.record_step`
    writes a placeholder carrying ``serialization_error`` and an empty
    ``prompt_messages``. Because reconstruction is driven by a single step's
    prompt snapshot, picking such a placeholder would silently discard the entire
    conversation on resume. We therefore walk backwards to the last step that
    recorded a real prompt.
    """
    for step in reversed(steps):
        if step.get("serialization_error"):
            continue
        if step.get("prompt_messages"):
            return step
    return steps[-1]


def _rebuild_tool_calls(tool_calls: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Restores OpenAI-shaped tool calls, keeping provider ``extra_content`` intact."""
    rebuilt = []
    for call in tool_calls:
        if not isinstance(call, dict):
            continue
        function = call.get("function") or {}
        args = function.get("arguments") or call.get("arguments") or {}
        if not isinstance(args, str):
            args = json.dumps(args, default=str) if isinstance(args, dict) else str(args)
        entry: dict[str, Any] = {
            "id": call.get("id", ""),
            "type": "function",
            "function": {"name": function.get("name") or call.get("name", ""), "arguments": args},
        }
        if call.get("extra_content"):
            entry["extra_content"] = call["extra_content"]
        rebuilt.append(entry)
    return rebuilt


def _rebuild_assistant_message(response: dict[str, Any]) -> dict[str, Any]:
    """Rebuilds the assistant turn a step recorded, if it recorded anything."""
    content = response.get("content") or ""
    tool_calls = response.get("tool_calls") or []
    if not content and not tool_calls:
        return {}

    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls and (rebuilt := _rebuild_tool_calls(tool_calls)):
        message["tool_calls"] = rebuilt
    return message


def reconstruct_messages_from_steps(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Reconstructs the full conversation from a list of persisted step records."""
    if not steps:
        return []
    last_step = _select_reconstruction_step(steps)

    messages: list[dict[str, Any]] = [dict(m) for m in (last_step.get("prompt_messages") or [])]
    if assistant := _rebuild_assistant_message(last_step.get("model_response") or {}):
        messages.append(assistant)
    for result in last_step.get("tool_results") or []:
        if isinstance(result, dict) and "call_id" in result:
            messages.append(
                {
                    "role": "tool",
                    "content": result.get("output", ""),
                    "tool_call_id": result.get("call_id"),
                    "raw_log_path": result.get("raw_log_path"),
                }
            )
    return messages


def load_session_messages(run_dir: Path | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Loads reconstructed messages and metadata from a session directory.

    Prefers ``trace.json`` and falls back to ``trajectory.jsonl``, which is all a
    run that crashed mid-turn left behind.
    """
    dir_path = Path(run_dir)
    metadata: dict[str, Any] = {
        "run_id": dir_path.name,
        "run_dir": str(dir_path),
        "total_steps": 0,
        "workspace": None,
        "finish_reason": "unknown",
    }

    data = _read_json(dir_path / "trace.json")
    if data is not None:
        metadata["total_steps"] = int(data.get("total_steps") or 0)
        metadata["workspace"] = data.get("workspace")
        metadata["finish_reason"] = data.get("finish_reason", "unknown")
        metadata["task"] = data.get("task", {})
        if isinstance(data.get("messages"), list) and data["messages"]:
            return data["messages"], metadata
        if data.get("steps"):
            return reconstruct_messages_from_steps(data["steps"]), metadata

    steps = _read_jsonl_steps(dir_path / "trajectory.jsonl")
    if steps:
        metadata["total_steps"] = len(steps)
        return reconstruct_messages_from_steps(steps), metadata

    return [], metadata
