"""Generic lifecycle hook registry and built-in default hooks.

Lifecycle events: ``pre_turn``, ``post_turn``, ``pre_step``, ``post_step``,
``pre_tool``, ``post_tool``.

Built-in default hooks (``python_syntax_diagnostics`` and
``doom_loop_detection``) are registered automatically and can be removed or
overridden by name via :meth:`HookRegistry.unregister` /
:meth:`HookRegistry.disable`, ``disabled_hooks``, or ``hooks`` configuration.
"""

from __future__ import annotations

import fnmatch
import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from core.defaults import load_defaults
from core.editing import check_python_syntax
from core.types import DOOM_LOOP, ToolCall, ToolResult

_DEFAULTS = load_defaults()

if TYPE_CHECKING:
    from core.environment import Environment
    from core.memory import Memory

PRE_TURN = "pre_turn"
POST_TURN = "post_turn"
PRE_STEP = "pre_step"
POST_STEP = "post_step"
PRE_TOOL = "pre_tool"
POST_TOOL = "post_tool"

HOOK_EVENTS: frozenset[str] = frozenset(
    {PRE_TURN, POST_TURN, PRE_STEP, POST_STEP, PRE_TOOL, POST_TOOL}
)

PYTHON_SYNTAX_HOOK_NAME = "python_syntax_diagnostics"
DOOM_LOOP_HOOK_NAME = "doom_loop_detection"
DEFAULT_DOOM_LOOP_THRESHOLD: int = int(_DEFAULTS["doom_loop_threshold"])
DEFAULT_DOOM_LOOP_MAX_NUDGES: int = int(_DEFAULTS["doom_loop_max_nudges"])
DEFAULT_HOOK_COMMAND_TIMEOUT: float = float(_DEFAULTS["hook_command_timeout"])
DEFAULT_MAX_HOOK_OUTPUT_CHARS: int = int(_DEFAULTS["max_hook_output_chars"])


@dataclass
class HookContext:
    """Event payload passed to matching hooks."""

    event: str
    env: Environment | None = None
    memory: Memory | None = None
    tool_name: str = ""
    tool_call: ToolCall | None = None
    tool_args: dict[str, Any] | None = None
    exit_code: int | None = None
    output: str = ""
    step_index: int | None = None
    modifies_files: bool = False
    path: str | None = None
    tool_calls: Sequence[ToolCall] = ()
    tool_results: Sequence[ToolResult] = ()
    modifies_files_fn: Callable[[str], bool] | None = None
    state: dict[str, Any] = field(default_factory=dict)
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HookDecision:
    """Structured result returned by a hook to surface feedback or stop execution."""

    message: str = ""
    block: bool = False
    stop_reason: str | None = None


@dataclass(frozen=True)
class HookOutcome:
    """Aggregated outcome across all hooks that ran for one event."""

    blocked: bool = False
    stop_reason: str | None = None
    messages: tuple[str, ...] = ()

    @property
    def combined_message(self) -> str:
        return "\n\n".join(m for m in self.messages if m)


@runtime_checkable
class HookProtocol(Protocol):
    """Protocol for pluggable lifecycle hooks."""

    name: str
    event: str

    def matches(self, ctx: HookContext) -> bool: ...

    def run(self, ctx: HookContext) -> HookDecision | str | None: ...


def _matches_tool_pattern(matcher: str | None, tool_name: str) -> bool:
    if not matcher or matcher == "*":
        return True
    if not tool_name:
        return False
    try:
        return re.fullmatch(matcher, tool_name) is not None
    except re.error:
        return fnmatch.fnmatch(tool_name, matcher)


def _matches_path_glob(path_glob: str | None, path: str | None) -> bool:
    if not path_glob:
        return True
    if not path:
        return False
    return fnmatch.fnmatch(path, path_glob) or fnmatch.fnmatch(Path(path).name, path_glob)


def _run_command_hook(
    command: str, name: str, timeout: float, ctx: HookContext
) -> HookDecision | str | None:
    stripped = command.strip()
    if not stripped:
        return None
    replacements = {
        "{path}": shlex.quote(ctx.path or ""),
        "{tool}": shlex.quote(ctx.tool_name),
        "{event}": shlex.quote(ctx.event),
    }
    rendered = stripped
    for token, value in replacements.items():
        if token in rendered:
            rendered = rendered.replace(token, value)

    if ctx.env is not None:
        exit_code, output = ctx.env.execute_bash(rendered, timeout=timeout)
    else:
        try:
            proc = subprocess.run(
                rendered,
                shell=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                timeout=timeout,
                encoding="utf-8",
                errors="replace",
            )
            exit_code, output = proc.returncode, proc.stdout
        except subprocess.TimeoutExpired:
            exit_code, output = 124, f"[Hook '{name}' timed out after {timeout}s]"

    if exit_code == 0:
        return None
    cleaned = (output or "").strip()
    if not cleaned:
        cleaned = f"Hook '{name}' command exited with code {exit_code}."
    elif len(cleaned) > DEFAULT_MAX_HOOK_OUTPUT_CHARS:
        cleaned = f"{cleaned[:DEFAULT_MAX_HOOK_OUTPUT_CHARS].rstrip()}\n... (hook output truncated)"
    return HookDecision(message=cleaned, block=True) if ctx.event.startswith("pre_") else cleaned


class Hook:
    """Lifecycle hook backed by a Python callable or shell command."""

    def __init__(
        self,
        name: str,
        event: str,
        *,
        fn: Callable[[HookContext], HookDecision | str | None] | None = None,
        command: str | None = None,
        matcher: str | None = None,
        path_glob: str | None = None,
        only_on_success: bool = False,
        timeout: float = DEFAULT_HOOK_COMMAND_TIMEOUT,
    ):
        if event not in HOOK_EVENTS:
            raise ValueError(f"Unsupported hook event: '{event}'.")
        if fn is None and (not isinstance(command, str) or not command.strip()):
            raise ValueError(f"Hook '{name}' must define either 'fn' or a non-empty 'command'.")
        self.name = name
        self.event = event
        self.fn = fn
        self.command = command
        self.matcher = matcher
        self.path_glob = path_glob
        self.only_on_success = only_on_success
        self.timeout = float(timeout)

    def matches(self, ctx: HookContext) -> bool:
        if ctx.event != self.event:
            return False
        if self.only_on_success and ctx.exit_code is not None and ctx.exit_code != 0:
            return False
        if not _matches_tool_pattern(self.matcher, ctx.tool_name):
            return False
        return _matches_path_glob(self.path_glob, ctx.path)

    def run(self, ctx: HookContext) -> HookDecision | str | None:
        if self.fn is not None:
            return self.fn(ctx)
        assert self.command is not None
        return _run_command_hook(self.command, self.name, self.timeout, ctx)


def _python_syntax_hook(ctx: HookContext) -> str | None:
    """Built-in post-edit syntax check for ``.py`` files using stdlib ``ast.parse``."""
    if not ctx.path:
        return None
    source: str | None = None
    if (
        ctx.tool_name == "write_file"
        and isinstance(ctx.tool_args, dict)
        and "content" in ctx.tool_args
    ):
        source = str(ctx.tool_args["content"])
    elif ctx.env is not None:
        try:
            resolved = ctx.env.resolve_path(ctx.path)
            if isinstance(resolved, Path) and resolved.is_file():
                source = resolved.read_text(encoding="utf-8", errors="replace")
        except Exception:
            source = None
    if source is None:
        return None
    err = check_python_syntax(source, path_label=ctx.path)
    return f"[Post-edit diagnostics]\n{err}" if err else None


def _is_oscillating(history: Sequence[str], threshold: int) -> bool:
    window = max(3, threshold)
    if len(history) < window:
        return False
    tail = history[-window:]
    return tail[0] != tail[1] and all(tail[i] == tail[i % 2] for i in range(window))


def make_doom_loop_hook(
    threshold: int = DEFAULT_DOOM_LOOP_THRESHOLD,
    max_nudges: int = DEFAULT_DOOM_LOOP_MAX_NUDGES,
    *,
    name: str = DOOM_LOOP_HOOK_NAME,
) -> Hook:
    """Creates a stateful-per-loop ``post_step`` hook that detects repeated failures and edit oscillation."""
    if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
        raise ValueError(f"Unsupported doom-loop threshold: {threshold!r}. Must be >= 0.")
    if isinstance(max_nudges, bool) or not isinstance(max_nudges, int) or max_nudges < 0:
        raise ValueError(f"Unsupported doom-loop max_nudges: {max_nudges!r}. Must be >= 0.")

    def _run(ctx: HookContext) -> HookDecision | None:
        if threshold <= 0 or not ctx.tool_calls:
            return None

        state = ctx.state.setdefault(
            name,
            {
                "last_failure_sig": None,
                "failure_streak": 0,
                "edit_history": {},
                "nudges_sent": 0,
            },
        )
        results_by_id = {r.call_id: r for r in ctx.tool_results if r.call_id}
        failing_sigs: list[tuple[str, str, int]] = []
        any_file_modified = False
        edit_reason = ""

        for idx, call in enumerate(ctx.tool_calls):
            result = results_by_id.get(call.id) or (
                ctx.tool_results[idx] if idx < len(ctx.tool_results) else None
            )
            exit_code = result.exit_code if result is not None else 0
            is_modifier = (
                ctx.modifies_files_fn(call.name)
                if ctx.modifies_files_fn is not None
                else call.name in {"write_file", "edit_file"}
            )
            if (
                exit_code == 0
                and is_modifier
                and isinstance(call.arguments, dict)
                and call.arguments.get("path")
            ):
                any_file_modified = True
                raw_path = str(call.arguments["path"])
                norm_path = os.path.normpath(raw_path)
                digest: str | None = None
                if ctx.env is not None:
                    try:
                        resolved = ctx.env.resolve_path(raw_path)
                        norm_path = str(resolved)
                        if isinstance(resolved, Path) and resolved.is_file():
                            digest = hashlib.sha256(resolved.read_bytes()).hexdigest()
                    except Exception:
                        digest = None
                if digest is None and call.name == "write_file" and "content" in call.arguments:
                    digest = hashlib.sha256(
                        str(call.arguments["content"]).encode("utf-8", errors="replace")
                    ).hexdigest()
                if digest is not None:
                    hist: list[str] = state["edit_history"].setdefault(norm_path, [])
                    hist.append(digest)
                    if _is_oscillating(hist, threshold):
                        edit_reason = (
                            f"Edit oscillation detected on '{raw_path}' "
                            "(repeatedly reverting between prior file states)."
                        )
                    elif (
                        threshold >= 2
                        and len(hist) >= threshold
                        and len(set(hist[-threshold:])) == 1
                    ):
                        edit_reason = (
                            f"Repeated identical edit state on '{raw_path}' ({threshold} times)."
                        )
            elif exit_code != 0:
                canon_args = (
                    json.dumps(call.arguments, sort_keys=True, default=str)
                    if isinstance(call.arguments, dict)
                    else str(call.arguments)
                )
                failing_sigs.append((call.name, canon_args, exit_code))

        if any_file_modified or not failing_sigs:
            state["last_failure_sig"] = None
            state["failure_streak"] = 0
        else:
            sig = tuple(failing_sigs)
            if sig == state["last_failure_sig"]:
                state["failure_streak"] += 1
            else:
                state["last_failure_sig"] = sig
                state["failure_streak"] = 1

        reason = ""
        if state["failure_streak"] >= threshold:
            names = ", ".join(sorted({s[0] for s in state["last_failure_sig"] or ()}))
            reason = (
                f"Repeated identical failing tool call ({names}) "
                f"{state['failure_streak']} consecutive times."
            )
        elif edit_reason:
            reason = edit_reason

        if not reason:
            return None

        if state["nudges_sent"] < max_nudges:
            state["nudges_sent"] += 1
            if ctx.memory is not None:
                ctx.memory.add_user(
                    f"[Loop guard] {reason} Step back and reconsider your approach: "
                    "re-read the relevant file(s) or error output carefully, verify your "
                    "assumptions, and try a different fix instead of repeating the same "
                    "edits or failing calls."
                )
            return HookDecision(message=f"Loop guard: {reason}", block=False)

        return HookDecision(message=reason, block=True, stop_reason=DOOM_LOOP)

    return Hook(name=name, event=POST_STEP, fn=_run)


def build_default_hooks() -> list[Hook]:
    """Returns built-in default hooks registered on new harnesses and registries."""
    return [
        Hook(
            name=PYTHON_SYNTAX_HOOK_NAME,
            event=POST_TOOL,
            fn=_python_syntax_hook,
            matcher="write_file|edit_file",
            path_glob="*.py",
            only_on_success=True,
        ),
        make_doom_loop_hook(),
    ]


class HookRegistry:
    """Ordered registry of lifecycle hooks with name-based removal and disabling."""

    def __init__(self, hooks: Iterable[HookProtocol] = ()):
        self._hooks: list[HookProtocol] = []
        self._disabled: set[str] = set()
        for hook in hooks:
            self.register(hook)

    @classmethod
    def empty(cls) -> HookRegistry:
        return cls()

    @classmethod
    def with_defaults(cls) -> HookRegistry:
        return cls(build_default_hooks())

    def register(self, hook: HookProtocol) -> None:
        if hook.event not in HOOK_EVENTS:
            raise ValueError(f"Unsupported hook event: '{hook.event}'.")
        if not hook.name:
            raise ValueError("Hook name cannot be empty.")
        self._disabled.discard(hook.name)
        self._hooks = [h for h in self._hooks if h.name != hook.name]
        self._hooks.append(hook)

    def unregister(self, name: str) -> bool:
        before = len(self._hooks)
        self._hooks = [h for h in self._hooks if h.name != name]
        return len(self._hooks) < before

    def disable(self, name: str) -> bool:
        self._disabled.add(name)
        return self.unregister(name)

    def get(self, name: str) -> HookProtocol | None:
        return next((h for h in self._hooks if h.name == name), None)

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def list_hooks(self, event: str | None = None) -> list[HookProtocol]:
        return [h for h in self._hooks if event is None or h.event == event]

    def run(self, event: str, ctx: HookContext) -> HookOutcome:
        if not self._hooks:
            return HookOutcome()

        messages: list[str] = []
        for hook in list(self._hooks):
            if hook.event != event:
                continue
            try:
                if not hook.matches(ctx):
                    continue
                result = hook.run(ctx)
            except Exception as exc:
                msg = f"Hook '{hook.name}' error: {exc}"
                if msg not in messages:
                    messages.append(msg)
                continue

            if result is None:
                continue
            if isinstance(result, HookDecision):
                cleaned = result.message.strip()
                if cleaned and cleaned not in messages:
                    messages.append(cleaned)
                if result.block:
                    return HookOutcome(
                        blocked=True,
                        stop_reason=result.stop_reason,
                        messages=tuple(messages),
                    )
            else:
                cleaned = str(result).strip()
                if cleaned and cleaned not in messages:
                    messages.append(cleaned)

        return HookOutcome(blocked=False, messages=tuple(messages))


def parse_hook_specs(raw: Any, *, scope: str = "explicit") -> list[dict[str, Any]]:
    """Normalizes ``hooks`` config structures (event-mapped dict or list) into flat dicts."""
    if not raw:
        return []
    specs: list[dict[str, Any]] = []
    if isinstance(raw, Mapping):
        for event, entries in raw.items():
            if event not in HOOK_EVENTS:
                raise ValueError(
                    f"Unsupported hook event '{event}' in {scope} hooks config. "
                    f"Valid events: {', '.join(sorted(HOOK_EVENTS))}"
                )
            for item in entries if isinstance(entries, list) else [entries]:
                if not isinstance(item, Mapping):
                    raise ValueError(f"Invalid hook entry under '{event}': expected object.")
                specs.append({"event": event, **item, "_scope": scope})
        return specs

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if not isinstance(item, Mapping) or item.get("event") not in HOOK_EVENTS:
                raise ValueError(f"Invalid or missing hook event in {scope} hooks list.")
            specs.append({**item, "_scope": scope})
        return specs

    raise ValueError("Invalid hooks config: expected an event mapping or list of hooks.")


def build_hook_registry(
    *,
    include_defaults: bool = True,
    hook_specs: Sequence[Mapping[str, Any]] = (),
    extra_hooks: Iterable[HookProtocol | Mapping[str, Any]] = (),
    disabled_hooks: Iterable[str] = (),
    trust_project_hooks: bool = False,
) -> HookRegistry:
    """Builds a :class:`HookRegistry` from defaults, config specs, and explicit hooks."""
    registry = HookRegistry.with_defaults() if include_defaults else HookRegistry.empty()
    disabled: set[str] = {str(n).strip() for n in disabled_hooks if str(n).strip()}
    warned_untrusted = False

    def _apply_spec(spec: Mapping[str, Any], idx: int) -> None:
        nonlocal warned_untrusted
        event = str(spec.get("event") or "")
        name = str(spec.get("name") or "").strip() or f"{event}_hook_{idx}"
        if spec.get("enabled") is False:
            disabled.add(name)
            return
        if spec.get("_scope") == "project" and not trust_project_hooks and "command" in spec:
            if not warned_untrusted:
                print(
                    "⚠️ Warning: Skipping untrusted project hook(s) in .microharness/config.json. "
                    "Pass --trust-project-tools to enable.",
                    file=sys.stderr,
                )
                warned_untrusted = True
            return

        if name == DOOM_LOOP_HOOK_NAME and "command" not in spec and "fn" not in spec:
            threshold = spec.get("threshold", DEFAULT_DOOM_LOOP_THRESHOLD)
            max_nudges = spec.get("max_nudges", DEFAULT_DOOM_LOOP_MAX_NUDGES)
            registry.register(make_doom_loop_hook(threshold=threshold, max_nudges=max_nudges))
            return

        raw_timeout = spec.get("timeout", DEFAULT_HOOK_COMMAND_TIMEOUT)
        if (
            isinstance(raw_timeout, bool)
            or not isinstance(raw_timeout, (int, float))
            or raw_timeout <= 0
        ):
            raise ValueError(
                f"Hook '{name}' timeout must be a positive number, got {raw_timeout!r}."
            )

        registry.register(
            Hook(
                name=name,
                event=event,
                fn=spec.get("fn"),
                command=spec.get("command"),
                matcher=str(spec["matcher"]).strip() if spec.get("matcher") else None,
                path_glob=str(spec["path_glob"]).strip() if spec.get("path_glob") else None,
                only_on_success=bool(spec.get("only_on_success", False)),
                timeout=float(raw_timeout),
            )
        )

    for idx, spec in enumerate(hook_specs, start=1):
        _apply_spec(spec, idx)

    for idx, item in enumerate(extra_hooks, start=len(hook_specs) + 1):
        if isinstance(item, Mapping):
            _apply_spec(item, idx)
        else:
            registry.register(item)

    for name in disabled:
        registry.disable(name)

    return registry
