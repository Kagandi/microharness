"""Generic lifecycle hook registry, built-in default hooks, and config loader.

Hooks run at well-defined lifecycle events (``pre_turn``, ``post_turn``,
``pre_step``, ``post_step``, ``pre_tool``, ``post_tool``) without adding
feature-specific wiring to individual tools or loops.

Built-in default hooks (such as Python syntax diagnostics on ``write_file`` /
``edit_file``) are registered automatically and can be removed programmatically
via :meth:`HookRegistry.unregister` / :meth:`HookRegistry.disable` or via
``disabled_hooks`` / ``{"enabled": false}`` in configuration files.
"""

from __future__ import annotations

import fnmatch
import re
import shlex
import subprocess
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from core.editing import check_python_syntax
from core.types import ToolCall

if TYPE_CHECKING:
    from core.environment import Environment

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
DEFAULT_HOOK_COMMAND_TIMEOUT = 10.0
DEFAULT_MAX_HOOK_OUTPUT_CHARS = 2000


@dataclass
class HookContext:
    """Event payload passed to matching hooks."""

    event: str
    env: Environment | None = None
    tool_name: str = ""
    tool_call: ToolCall | None = None
    tool_args: dict[str, Any] | None = None
    exit_code: int | None = None
    output: str = ""
    step_index: int | None = None
    modifies_files: bool = False
    path: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class HookDecision:
    """Structured result returned by a hook when it needs to block or message."""

    message: str = ""
    block: bool = False


@dataclass(frozen=True)
class HookOutcome:
    """Aggregated outcome across all hooks that ran for one event."""

    blocked: bool = False
    messages: tuple[str, ...] = ()

    @property
    def combined_message(self) -> str:
        return "\n\n".join(m for m in self.messages if m)


@runtime_checkable
class Hook(Protocol):
    """Protocol for pluggable lifecycle hooks."""

    name: str
    event: str

    def matches(self, ctx: HookContext) -> bool:
        """Returns True if this hook should run for ``ctx``."""
        ...

    def run(self, ctx: HookContext) -> HookDecision | str | None:
        """Executes the hook for ``ctx`` and optionally returns a message or block decision."""
        ...


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


def _hook_matches(
    ctx: HookContext,
    *,
    event: str,
    matcher: str | None,
    path_glob: str | None,
    only_on_success: bool,
) -> bool:
    if ctx.event != event:
        return False
    if only_on_success and ctx.exit_code is not None and ctx.exit_code != 0:
        return False
    if not _matches_tool_pattern(matcher, ctx.tool_name):
        return False
    if not _matches_path_glob(path_glob, ctx.path):
        return False
    return True


def _render_command_template(command: str, ctx: HookContext) -> str:
    rendered = command.strip()
    replacements = {
        "{path}": shlex.quote(ctx.path or ""),
        "{tool}": shlex.quote(ctx.tool_name),
        "{event}": shlex.quote(ctx.event),
    }
    for token, value in replacements.items():
        if token in rendered:
            rendered = rendered.replace(token, value)
    return rendered


class FunctionHook:
    """Callable-backed lifecycle hook with optional tool/path filtering."""

    def __init__(
        self,
        name: str,
        event: str,
        fn: Callable[[HookContext], HookDecision | str | None],
        *,
        matcher: str | None = None,
        path_glob: str | None = None,
        only_on_success: bool = False,
    ):
        if event not in HOOK_EVENTS:
            raise ValueError(f"Unsupported hook event: '{event}'.")
        self.name = name
        self.event = event
        self.fn = fn
        self.matcher = matcher
        self.path_glob = path_glob
        self.only_on_success = only_on_success

    def matches(self, ctx: HookContext) -> bool:
        return _hook_matches(
            ctx,
            event=self.event,
            matcher=self.matcher,
            path_glob=self.path_glob,
            only_on_success=self.only_on_success,
        )

    def run(self, ctx: HookContext) -> HookDecision | str | None:
        return self.fn(ctx)


class CommandHook:
    """Shell-command lifecycle hook configured in config or Python.

    On ``pre_*`` events, a non-zero exit code blocks the action.
    On ``post_*`` events, non-zero exit output is surfaced concisely as feedback.
    """

    def __init__(
        self,
        name: str,
        event: str,
        command: str,
        *,
        matcher: str | None = None,
        path_glob: str | None = None,
        only_on_success: bool = False,
        timeout: float = DEFAULT_HOOK_COMMAND_TIMEOUT,
    ):
        if event not in HOOK_EVENTS:
            raise ValueError(f"Unsupported hook event: '{event}'.")
        self.name = name
        self.event = event
        self.command = command
        self.matcher = matcher
        self.path_glob = path_glob
        self.only_on_success = only_on_success
        self.timeout = float(timeout)

    def matches(self, ctx: HookContext) -> bool:
        return _hook_matches(
            ctx,
            event=self.event,
            matcher=self.matcher,
            path_glob=self.path_glob,
            only_on_success=self.only_on_success,
        )

    def run(self, ctx: HookContext) -> HookDecision | str | None:
        stripped = self.command.strip()
        if not stripped:
            return None
        rendered = _render_command_template(stripped, ctx)

        if ctx.env is not None:
            exit_code, output = ctx.env.execute_bash(rendered, timeout=self.timeout)
        else:
            try:
                proc = subprocess.run(
                    rendered,
                    shell=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    timeout=self.timeout,
                    encoding="utf-8",
                    errors="replace",
                )
                exit_code, output = proc.returncode, proc.stdout
            except subprocess.TimeoutExpired:
                exit_code, output = 124, f"[Hook '{self.name}' timed out after {self.timeout}s]"

        if exit_code == 0:
            return None

        cleaned = (output or "").strip()
        if not cleaned:
            cleaned = f"Hook '{self.name}' command exited with code {exit_code}."
        elif len(cleaned) > DEFAULT_MAX_HOOK_OUTPUT_CHARS:
            cleaned = (
                f"{cleaned[:DEFAULT_MAX_HOOK_OUTPUT_CHARS].rstrip()}\n... (hook output truncated)"
            )

        if self.event.startswith("pre_"):
            return HookDecision(message=cleaned, block=True)
        return cleaned


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


def build_default_hooks() -> list[Hook]:
    """Returns the built-in default hooks registered on new harnesses/registries."""
    return [
        FunctionHook(
            name=PYTHON_SYNTAX_HOOK_NAME,
            event=POST_TOOL,
            fn=_python_syntax_hook,
            matcher="write_file|edit_file",
            path_glob="*.py",
            only_on_success=True,
        )
    ]


class HookRegistry:
    """Ordered registry of lifecycle hooks with name-based removal and disabling."""

    def __init__(self, hooks: Iterable[Hook] = ()):
        self._hooks: list[Hook] = []
        self._disabled: set[str] = set()
        for hook in hooks:
            self.register(hook)

    @classmethod
    def empty(cls) -> HookRegistry:
        return cls()

    @classmethod
    def with_defaults(cls) -> HookRegistry:
        return cls(build_default_hooks())

    def register(self, hook: Hook) -> None:
        """Registers (or replaces by name) a hook."""
        if hook.event not in HOOK_EVENTS:
            raise ValueError(f"Unsupported hook event: '{hook.event}'.")
        if not hook.name:
            raise ValueError("Hook name cannot be empty.")
        self._disabled.discard(hook.name)
        self._hooks = [h for h in self._hooks if h.name != hook.name]
        self._hooks.append(hook)

    def unregister(self, name: str) -> bool:
        """Removes a registered hook by name. Returns True if one was removed."""
        before = len(self._hooks)
        self._hooks = [h for h in self._hooks if h.name != name]
        return len(self._hooks) < before

    def remove(self, name: str) -> bool:
        """Alias for :meth:`unregister`."""
        return self.unregister(name)

    def disable(self, name: str) -> bool:
        """Removes ``name`` and remembers it as disabled."""
        self._disabled.add(name)
        return self.unregister(name)

    def is_disabled(self, name: str) -> bool:
        return name in self._disabled

    def get(self, name: str) -> Hook | None:
        for hook in self._hooks:
            if hook.name == name:
                return hook
        return None

    def __contains__(self, name: str) -> bool:
        return self.get(name) is not None

    def list_hooks(self, event: str | None = None) -> list[Hook]:
        if event is None:
            return list(self._hooks)
        return [h for h in self._hooks if h.event == event]

    def run(self, event: str, ctx: HookContext) -> HookOutcome:
        """Executes all matching hooks for ``event`` and returns their aggregated outcome."""
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
                    return HookOutcome(blocked=True, messages=tuple(messages))
            else:
                cleaned = str(result).strip()
                if cleaned and cleaned not in messages:
                    messages.append(cleaned)

        return HookOutcome(blocked=False, messages=tuple(messages))


def parse_hook_specs(raw: Any, *, scope: str = "explicit") -> list[dict[str, Any]]:
    """Normalizes ``hooks`` config structures (event-mapped dict or list of specs) into flat dicts."""
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
            items = entries if isinstance(entries, list) else [entries]
            for item in items:
                if isinstance(item, Mapping):
                    spec = dict(item)
                    spec.setdefault("event", event)
                    spec["_scope"] = scope
                    specs.append(spec)
                else:
                    raise ValueError(f"Invalid hook entry under '{event}': expected object.")
        return specs

    if isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
        for item in raw:
            if not isinstance(item, Mapping):
                raise ValueError("Invalid hook entry in hooks list: expected object.")
            spec = dict(item)
            event = spec.get("event")
            if event not in HOOK_EVENTS:
                raise ValueError(
                    f"Unsupported or missing hook event {event!r} in {scope} hooks config."
                )
            spec["_scope"] = scope
            specs.append(spec)
        return specs

    raise ValueError(
        "Invalid hooks config: expected a mapping of event -> list or a list of hooks."
    )


def build_hook_from_spec(spec: Mapping[str, Any], index: int = 0) -> Hook:
    """Constructs a :class:`CommandHook` (or callable :class:`FunctionHook`) from a normalized dict."""
    event = str(spec.get("event") or "")
    if event not in HOOK_EVENTS:
        raise ValueError(f"Unsupported hook event: {event!r}.")

    name = str(spec.get("name") or "").strip() or f"{event}_hook_{index}"
    matcher = str(spec["matcher"]).strip() if spec.get("matcher") else None
    path_glob = str(spec["path_glob"]).strip() if spec.get("path_glob") else None
    only_on_success = bool(spec.get("only_on_success", False))

    fn = spec.get("fn")
    if callable(fn):
        return FunctionHook(
            name=name,
            event=event,
            fn=fn,
            matcher=matcher,
            path_glob=path_glob,
            only_on_success=only_on_success,
        )

    command = spec.get("command")
    if not isinstance(command, str) or not command.strip():
        raise ValueError(f"Hook '{name}' must specify a non-empty 'command' or callable 'fn'.")

    raw_timeout = spec.get("timeout", DEFAULT_HOOK_COMMAND_TIMEOUT)
    if (
        isinstance(raw_timeout, bool)
        or not isinstance(raw_timeout, (int, float))
        or raw_timeout <= 0
    ):
        raise ValueError(f"Hook '{name}' timeout must be a positive number, got {raw_timeout!r}.")

    return CommandHook(
        name=name,
        event=event,
        command=command.strip(),
        matcher=matcher,
        path_glob=path_glob,
        only_on_success=only_on_success,
        timeout=float(raw_timeout),
    )


def build_hook_registry(
    *,
    include_defaults: bool = True,
    hook_specs: Sequence[Mapping[str, Any]] = (),
    extra_hooks: Iterable[Hook | Mapping[str, Any]] = (),
    disabled_hooks: Iterable[str] = (),
    trust_project_hooks: bool = False,
) -> HookRegistry:
    """Assembles a :class:`HookRegistry` from defaults, config specs, and explicit hooks.

    Project-scoped command hooks are skipped unless ``trust_project_hooks=True``
    to preserve the workspace execution trust boundary.
    """
    registry = HookRegistry.with_defaults() if include_defaults else HookRegistry.empty()
    disabled: set[str] = {str(name).strip() for name in disabled_hooks if str(name).strip()}
    warned_untrusted = False

    def _apply_spec(spec: Mapping[str, Any], idx: int) -> None:
        nonlocal warned_untrusted
        name = str(spec.get("name") or "").strip()
        if spec.get("enabled") is False:
            if name:
                disabled.add(name)
            return
        scope = str(spec.get("_scope") or "explicit")
        if scope == "project" and not trust_project_hooks and "command" in spec:
            if not warned_untrusted:
                print(
                    "⚠️ Warning: Skipping untrusted project hook(s) in .microharness/config.json. "
                    "Pass --trust-project-tools to enable.",
                    file=sys.stderr,
                )
                warned_untrusted = True
            return
        hook = build_hook_from_spec(spec, index=idx)
        registry.register(hook)

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
