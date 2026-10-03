"""Run lifecycle: the step driver and session discovery / resumption.

The step driver owns the retry-and-give-up policy for a single turn; session
helpers own "which previous run do I mean and how do I load it". Both are pure
bookkeeping around :class:`~core.loop.AgentLoop`, so batch runs, the CLI
session, and the TUI all share one policy.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.loop import AgentLoop
from core.protocols import DeadlineAwareModel, Model, RateLimitAwareModel
from core.ratelimit import RateLimitError
from core.tracer import Tracer, find_recent_sessions, load_session_messages
from core.types import Message

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


def find_sessions(
    runs_dir: Path,
    workspace: Path | None,
    limit: int = 10,
) -> list[dict]:
    """Finds recent runs, preferring ones from the same workspace."""
    return find_recent_sessions(runs_dir=runs_dir, workspace=workspace, limit=limit)


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
    sessions = find_sessions(runs_dir, workspace, limit=SEARCH_LIMIT)
    target, error = resolve_session_dir(runs_dir, selector, sessions)
    if target is None or not target.is_dir():
        return [], {"error": error or NOT_FOUND}

    raw_messages, meta = load_session_messages(target)
    return [Message.from_dict(m) for m in raw_messages], meta
