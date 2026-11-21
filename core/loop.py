from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from core.hooks import POST_STEP, PRE_STEP, HookContext, HookRegistry
from core.memory import Memory
from core.model import AssistantMetadataModel, Model, UsageReportingModel
from core.tools import ToolRegistry
from core.types import (
    DEFAULT_DOOM_LOOP_MAX_NUDGES as DEFAULT_DOOM_LOOP_MAX_NUDGES,
)
from core.types import (
    DEFAULT_DOOM_LOOP_THRESHOLD as DEFAULT_DOOM_LOOP_THRESHOLD,
)
from core.types import (
    DOOM_LOOP as DOOM_LOOP,
)
from core.types import StepTrace, ToolResult
from core.usage import normalize_usage

_FILE_MODIFYING_TOOLS: frozenset[str] = frozenset({"write_file", "edit_file"})


@dataclass(frozen=True)
class DoomLoopCheck:
    """Outcome of inspecting one completed step for repeated failure / thrashing loops."""

    triggered: bool
    action: str  # "none" | "nudge" | "stop"
    reason: str = ""
    nudge_message: str = ""


def _call_id_name_args(call: Any) -> tuple[str, str, Any]:
    if isinstance(call, dict):
        fn = call.get("function") or {}
        name = str(fn.get("name") or call.get("name") or "")
        args = fn.get("arguments") if "arguments" in fn else call.get("arguments")
        if isinstance(args, str):
            try:
                args = json.loads(args)
            except Exception:
                pass
        return str(call.get("id") or ""), name, args
    return (
        str(getattr(call, "id", "") or ""),
        str(getattr(call, "name", "") or ""),
        getattr(call, "arguments", None),
    )


def _result_id_exit(result: Any) -> tuple[str, int]:
    if isinstance(result, dict):
        raw_code = result.get("exit_code", 0)
        return str(result.get("call_id") or ""), int(raw_code if raw_code is not None else 0)
    raw_code = getattr(result, "exit_code", 0)
    return str(getattr(result, "call_id", "") or ""), int(raw_code if raw_code is not None else 0)


def _canonical_args(arguments: Any) -> str:
    if isinstance(arguments, dict):
        return json.dumps(arguments, sort_keys=True, default=str)
    return str(arguments)


def _pair_step_calls(
    tool_calls: Sequence[Any], tool_results: Sequence[Any]
) -> list[tuple[Any, Any | None]]:
    results_by_id: dict[str, Any] = {}
    for res in tool_results:
        cid, _ = _result_id_exit(res)
        if cid:
            results_by_id[cid] = res

    remaining = list(tool_results)
    paired: list[Any | None] = [None] * len(tool_calls)
    for idx, call in enumerate(tool_calls):
        cid, _, _ = _call_id_name_args(call)
        if cid and cid in results_by_id:
            matched = results_by_id.pop(cid)
            remaining.remove(matched)
            paired[idx] = matched

    for idx in range(len(tool_calls)):
        if paired[idx] is None and remaining:
            paired[idx] = remaining.pop(0)

    return [(tool_calls[idx], paired[idx]) for idx in range(len(tool_calls))]


def _is_oscillating_states(history: Sequence[str], threshold: int) -> bool:
    window = max(3, threshold)
    if len(history) < window:
        return False
    tail = history[-window:]
    return tail[0] != tail[1] and all(tail[i] == tail[i % 2] for i in range(window))


def _is_repeating_states(history: Sequence[str], threshold: int) -> bool:
    if threshold < 2 or len(history) < threshold:
        return False
    return len(set(history[-threshold:])) == 1


def _is_oscillating_ops(ops: Sequence[tuple[str, str]], threshold: int) -> bool:
    window = max(3, threshold)
    if len(ops) < window:
        return False
    tail = ops[-window:]
    first = tail[0]
    inverse = (first[1], first[0])
    return first[0] != first[1] and all(
        tail[i] == (first if i % 2 == 0 else inverse) for i in range(window)
    )


def _build_nudge_message(reason: str) -> str:
    return (
        f"[Loop guard] {reason} Step back and reconsider your approach: re-read the relevant "
        "file(s) or error output carefully, verify your assumptions, and try a different fix "
        "instead of repeating the same edits or failing calls."
    )


class DoomLoopDetector:
    """Detects repeated identical failing calls and A->B->A edit oscillations across steps."""

    def __init__(
        self,
        threshold: int = DEFAULT_DOOM_LOOP_THRESHOLD,
        max_nudges: int = DEFAULT_DOOM_LOOP_MAX_NUDGES,
        env: Any = None,
    ):
        self.threshold = int(threshold)
        self.max_nudges = max(0, int(max_nudges))
        self.env = env
        self._last_failure_sig: tuple[tuple[str, str, int], ...] | None = None
        self._failure_streak: int = 0
        self._edit_history: dict[str, list[str]] = {}
        self._edit_op_history: dict[str, list[tuple[str, str]]] = {}
        self._nudges_sent: int = 0

    def _inspect_file_edit(
        self, name: str, arguments: Any
    ) -> tuple[str, str | None, tuple[str, str] | None] | None:
        if not isinstance(arguments, dict) or not arguments.get("path"):
            return None
        raw_path = str(arguments["path"])
        norm_path = os.path.normpath(raw_path)
        if self.env is not None:
            try:
                resolved = self.env.resolve_path(raw_path)
                norm_path = str(resolved)
                if isinstance(resolved, Path) and resolved.is_file():
                    digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
                    return norm_path, digest, None
            except Exception:
                pass
        if name == "write_file" and "content" in arguments:
            digest = hashlib.sha256(
                str(arguments.get("content", "")).encode("utf-8", errors="replace")
            ).hexdigest()
            return norm_path, digest, None
        if name == "edit_file":
            return (
                norm_path,
                None,
                (str(arguments.get("old_str", "")), str(arguments.get("new_str", ""))),
            )
        return None

    def observe_step(
        self,
        tool_calls: Sequence[Any],
        tool_results: Sequence[Any],
        *,
        modifies_files_fn: Callable[[str], bool] | None = None,
    ) -> DoomLoopCheck:
        """Records one step's tool calls/outcomes and flags repeated failures or edit thrashing."""
        if self.threshold <= 0 or not tool_calls:
            return DoomLoopCheck(False, "none")

        failing_sigs: list[tuple[str, str, int]] = []
        any_file_modified = False
        edit_reason = ""

        for call, result in _pair_step_calls(tool_calls, tool_results):
            _, name, arguments = _call_id_name_args(call)
            _, exit_code = _result_id_exit(result) if result is not None else ("", 0)

            is_modifier = (
                bool(modifies_files_fn(name))
                if modifies_files_fn is not None
                else name in _FILE_MODIFYING_TOOLS
            )
            if exit_code == 0 and is_modifier:
                any_file_modified = True
                inspected = self._inspect_file_edit(name, arguments)
                if inspected is not None:
                    norm_path, state_hash, op_pair = inspected
                    display_path = (
                        str(arguments.get("path")) if isinstance(arguments, dict) else norm_path
                    )
                    if state_hash is not None:
                        hist = self._edit_history.setdefault(norm_path, [])
                        hist.append(state_hash)
                        if _is_oscillating_states(hist, self.threshold):
                            edit_reason = (
                                f"Edit oscillation detected on '{display_path}' "
                                "(repeatedly reverting between prior file states)."
                            )
                        elif _is_repeating_states(hist, self.threshold):
                            edit_reason = (
                                f"Repeated identical edit state on '{display_path}' "
                                f"({self.threshold} times)."
                            )
                    elif op_pair is not None:
                        ops = self._edit_op_history.setdefault(norm_path, [])
                        ops.append(op_pair)
                        if _is_oscillating_ops(ops, self.threshold):
                            edit_reason = (
                                f"Edit oscillation detected on '{display_path}' "
                                "(repeatedly reverting between prior file states)."
                            )
                        elif (
                            len(ops) >= max(2, self.threshold)
                            and len(set(ops[-max(2, self.threshold) :])) == 1
                        ):
                            edit_reason = (
                                f"Repeated identical edit on '{display_path}' "
                                f"({self.threshold} times)."
                            )

            if exit_code != 0:
                failing_sigs.append((name, _canonical_args(arguments), exit_code))

        if any_file_modified or not failing_sigs:
            self._last_failure_sig = None
            self._failure_streak = 0
        else:
            step_failure_sig = tuple(failing_sigs)
            if step_failure_sig == self._last_failure_sig:
                self._failure_streak += 1
            else:
                self._last_failure_sig = step_failure_sig
                self._failure_streak = 1

        reason = ""
        if self._failure_streak >= self.threshold:
            failed_names = ", ".join(sorted({sig[0] for sig in self._last_failure_sig or ()}))
            reason = (
                f"Repeated identical failing tool call ({failed_names}) "
                f"{self._failure_streak} consecutive times."
            )
        elif edit_reason:
            reason = edit_reason

        if not reason:
            return DoomLoopCheck(False, "none")

        if self._nudges_sent < self.max_nudges:
            self._nudges_sent += 1
            return DoomLoopCheck(
                True,
                "nudge",
                reason=reason,
                nudge_message=_build_nudge_message(reason),
            )
        return DoomLoopCheck(True, "stop", reason=reason)


class AgentLoop:
    """The native tool-calling agent loop.

    Agnostic to sandboxing, benchmarks, and evaluation.
    Drives the structured LLM generation -> tool dispatch -> observation cycle.
    """

    def __init__(
        self,
        model: Model,
        memory: Memory,
        tools: ToolRegistry,
        *,
        hooks: HookRegistry | None = None,
        doom_loop_threshold: int = DEFAULT_DOOM_LOOP_THRESHOLD,
        doom_loop_max_nudges: int = DEFAULT_DOOM_LOOP_MAX_NUDGES,
    ):
        self.model = model
        self.memory = memory
        self.tools = tools
        resolved_hooks = hooks if hooks is not None else getattr(tools, "hooks", None)
        self.hooks: HookRegistry | None = (
            resolved_hooks if isinstance(resolved_hooks, HookRegistry) else None
        )
        self.doom_loop_threshold = doom_loop_threshold
        self.doom_loop_max_nudges = doom_loop_max_nudges
        self.doom_loop_detector = DoomLoopDetector(
            threshold=doom_loop_threshold,
            max_nudges=doom_loop_max_nudges,
            env=getattr(tools, "env", None),
        )
        self.last_doom_loop_check: DoomLoopCheck | None = None

    def step(self, step_index: int) -> tuple[bool, StepTrace]:
        """Executes a single tool-calling turn.

        Returns:
            (is_finished, step_trace)
        """
        start_time = time.time()
        self.last_doom_loop_check = None
        env = getattr(self.tools, "env", None)

        if self.hooks is not None:
            pre_step = self.hooks.run(
                PRE_STEP,
                HookContext(event=PRE_STEP, env=env, step_index=step_index),
            )
            if pre_step.blocked:
                trace = StepTrace(
                    step_index=step_index,
                    prompt_messages=[msg.to_dict() for msg in self.memory.get_messages()],
                    model_response={"content": pre_step.combined_message, "tool_calls": []},
                    tool_results=[],
                    compaction_applied=False,
                    duration_s=time.time() - start_time,
                    stop_reason="hook_blocked",
                    stop_detail=pre_step.combined_message or "Step blocked by pre_step hook.",
                )
                return True, trace

        # 1. Compaction check before sending context to LLM
        compaction_applied = self.memory.compact_if_needed()

        # 2. Snapshot current prompt state & raw token estimate for calibration
        prompt_messages = self.memory.get_messages()
        prompt_raw_tokens = self.memory.raw_estimated_tokens()
        prompt_snapshot = [msg.to_dict() for msg in prompt_messages]

        # 3. Model generation
        content, tool_calls = self.model.generate(
            messages=prompt_messages,
            tools=self.tools.to_openai_schemas(),
        )
        assistant_fields = (
            self.model.get_last_assistant_fields()
            if isinstance(self.model, AssistantMetadataModel)
            else {}
        )

        # 4. Record assistant thought / intent into memory
        self.memory.add_assistant(
            content=content,
            tool_calls=tool_calls,
            extra_fields=assistant_fields or None,
        )

        step_usage: dict[str, Any] = {}
        if isinstance(self.model, UsageReportingModel):
            step_usage = self.model.get_last_step_usage()
        elif hasattr(self.model, "last_usage"):
            step_usage = dict(getattr(self.model, "last_usage", {}) or {})

        actual_prompt_tokens = normalize_usage(step_usage).prompt_tokens
        if actual_prompt_tokens > 0:
            self.memory.calibrate_tokens(
                actual_prompt_tokens,
                baseline_raw_tokens=prompt_raw_tokens,
            )

        # 5. Stop condition: If no tools called, the agent has finished reasoning
        if not tool_calls:
            duration = time.time() - start_time
            trace = StepTrace(
                step_index=step_index,
                prompt_messages=prompt_snapshot,
                model_response={"content": content, "tool_calls": [], **assistant_fields},
                tool_results=[],
                compaction_applied=compaction_applied,
                duration_s=duration,
                usage=step_usage,
            )
            return True, trace

        # 6. Execute actions and record observations
        tool_results: list[ToolResult] = []
        terminated = False
        stop_reason: str | None = None
        stop_detail: str | None = None

        if len(tool_calls) > 1 and all(self.tools.is_parallel_safe(c.name) for c in tool_calls):
            with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as pool:
                dispatched = list(
                    pool.map(lambda c: self.tools.dispatch(c, step_index=step_index), tool_calls)
                )
        else:
            dispatched = [self.tools.dispatch(call, step_index=step_index) for call in tool_calls]

        for call, result in zip(tool_calls, dispatched):
            if result.exit_code == 0 and self.tools.modifies_files(call.name):
                target_path = call.arguments.get("path")
                if target_path:
                    self.memory.record_modified_file(str(target_path))
            if result.exit_code == 0 and self.tools.terminates_loop(call.name):
                terminated = True
            self.memory.add_tool_result(result)
            tool_results.append(result)

        if not terminated:
            check = self.doom_loop_detector.observe_step(
                tool_calls,
                tool_results,
                modifies_files_fn=self.tools.modifies_files,
            )
            self.last_doom_loop_check = check
            if check.action == "nudge":
                self.memory.add_user(check.nudge_message)
            elif check.action == "stop":
                terminated = True
                stop_reason = DOOM_LOOP
                stop_detail = check.reason

        duration = time.time() - start_time
        trace = StepTrace(
            step_index=step_index,
            prompt_messages=prompt_snapshot,
            model_response={
                "content": content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
                ],
                **assistant_fields,
            },
            tool_results=[tr.to_dict() for tr in tool_results],
            compaction_applied=compaction_applied,
            duration_s=duration,
            usage=step_usage,
            stop_reason=stop_reason,
            stop_detail=stop_detail,
        )

        if self.hooks is not None:
            self.hooks.run(
                POST_STEP,
                HookContext(
                    event=POST_STEP,
                    env=env,
                    step_index=step_index,
                    metadata={"terminated": terminated, "stop_reason": stop_reason},
                ),
            )

        return terminated, trace
