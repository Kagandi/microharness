"""Subagents: Markdown/frontmatter agent definitions, discovery, and isolated delegation tool.

A subagent is defined as a Markdown document with optional YAML frontmatter
under ``~/.microharness/agents/*.md`` (global) or ``.microharness/agents/*.md``
(project, shadowing global and built-in presets). Executing a subagent runs an
isolated child :class:`~core.loop.AgentLoop` with a fresh :class:`~core.memory.Memory`
and scoped :class:`~core.tools.ToolRegistry` sharing the workspace environment.
"""

from __future__ import annotations

import contextlib
import copy
import re
import sys
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from core.config import (
    DEFAULT_SUBAGENT_MAX_STEPS as DEFAULT_SUBAGENT_MAX_STEPS,
)
from core.config import (
    DISABLED_VALUES,
    normalize_reasoning_effort,
)
from core.discovery import discover_resources
from core.environment import Environment, ReadOnlyEnvironment
from core.loop import DEFAULT_DOOM_LOOP_MAX_NUDGES, DEFAULT_DOOM_LOOP_THRESHOLD, AgentLoop
from core.memory import Memory
from core.prompts import DEFAULT_SYSTEM_PROMPT, build_system_prompt
from core.protocols import (
    CacheAwareModel,
    CancellableModel,
    DeadlineAwareModel,
    EndpointAwareModel,
    Model,
    ModelCancelledError,
    ReasoningAwareModel,
)
from core.session import INTERRUPTED, _final_response_text, run_steps
from core.skills import _description_from_frontmatter, _first_heading, _split_frontmatter
from core.tools import (
    BashTool,
    EditFileTool,
    ReadFileTool,
    Tool,
    ToolRegistry,
    WriteFileTool,
)
from core.tracer import Tracer
from core.types import StepTrace, Task
from core.usage import TokenUsage, summarize_run_usage

TOOL_NAME_ALIASES: dict[str, str] = {
    "read": "read_file",
    "write": "write_file",
    "edit": "edit_file",
}

BLOCKED_CHILD_TOOLS: frozenset[str] = frozenset({"subagent", "finish"})

INHERIT_VALUES: frozenset[str] = frozenset({"", "inherit", "default", "parent"})

BLOCK_SCALAR_MARKERS: frozenset[str] = frozenset({">", "|", ">-", "|-"})


@dataclass(frozen=True)
class Subagent:
    """A discovered or built-in subagent profile."""

    name: str
    description: str
    system_prompt: str
    scope: str  # 'builtin', 'global', or 'project'
    tools: tuple[str, ...] | None = None
    model: str | None = None
    effort: str | None = None
    max_steps: int | None = None
    path: str = ""
    read_only: bool = False

    @property
    def badge(self) -> str:
        return f"[{self.scope}] "


EXPLORE_SYSTEM_PROMPT = """You are a fast, read-only codebase exploration specialist starting with a clean context.

Guidelines:
- Search and read relevant files to answer the question thoroughly and accurately.
- Use `read_file` and read-only `bash` commands (e.g. `rg`, `grep`, `find`, `git status`, `git log`).
- Do NOT modify any files in the workspace.
- Cite exact file paths, symbol names, and line numbers you observed, and return a concise structured report."""

REVIEWER_SYSTEM_PROMPT = """You are an independent, unbiased code reviewer starting with a clean context.

Guidelines:
- Inspect the relevant files, `git diff`, and tests without assuming prior claims are correct.
- Do NOT modify any files in the workspace.
- Check correctness, edge cases, error handling, security boundaries, and test coverage.
- Return concise, prioritized, actionable findings backed by observed file paths and line numbers."""

BUILTIN_SUBAGENTS: dict[str, Subagent] = {
    "explore": Subagent(
        name="explore",
        description="Fast read-only codebase exploration and search specialist",
        system_prompt=EXPLORE_SYSTEM_PROMPT,
        scope="builtin",
        tools=("read_file", "bash"),
        path="builtin:explore",
        read_only=True,
    ),
    "reviewer": Subagent(
        name="reviewer",
        description="Independent, unbiased code reviewer with a clean context",
        system_prompt=REVIEWER_SYSTEM_PROMPT,
        scope="builtin",
        tools=("read_file", "bash"),
        path="builtin:reviewer",
        read_only=True,
    ),
    "worker": Subagent(
        name="worker",
        description="General-purpose software engineering subagent with full file and bash tools",
        system_prompt=DEFAULT_SYSTEM_PROMPT,
        scope="builtin",
        tools=None,
        path="builtin:worker",
        read_only=False,
    ),
}


def _unquote_yaml_scalar(raw: str) -> str:
    """Strips quotes and trailing inline YAML comments from a scalar value."""
    s = raw.strip()
    if not s or s.startswith("#"):
        return ""
    if s[0] in ('"', "'"):
        quote = s[0]
        end = s.find(quote, 1)
        if end != -1:
            return s[1:end].strip()
        return s.strip(quote).strip()
    return re.sub(r"\s+#.*$", "", s).strip()


def _normalize_tool_token(raw: str) -> str:
    cleaned = _unquote_yaml_scalar(raw).strip()
    if not cleaned:
        return ""
    alias = TOOL_NAME_ALIASES.get(cleaned.lower())
    return alias if alias is not None else cleaned


def _normalize_tools_list(raw_items: list[str] | None) -> tuple[str, ...] | None:
    """Normalizes a parsed tool allow-list; None means inherit all non-subagent tools."""
    if raw_items is None:
        return None
    normalized: list[str] = []
    seen: set[str] = set()
    for item in raw_items:
        token = _normalize_tool_token(item)
        if not token:
            continue
        if token in ("*", "all"):
            return None
        if token.lower() in BLOCKED_CHILD_TOOLS:
            continue
        if token not in seen:
            seen.add(token)
            normalized.append(token)
    return tuple(normalized)


def _extract_frontmatter_fields(frontmatter: list[str]) -> dict[str, Any]:
    """Extracts scalar fields and YAML lists (such as ``tools``) from frontmatter lines."""
    fields: dict[str, Any] = {}
    i = 0
    n = len(frontmatter)

    while i < n:
        line = frontmatter[i]
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or line.startswith((" ", "\t")):
            i += 1
            continue

        key, sep, raw_val = stripped.partition(":")
        if not sep:
            i += 1
            continue

        key = key.strip().lower()
        val = raw_val.strip()

        if key == "tools":
            clean_val = _unquote_yaml_scalar(val)
            if clean_val.startswith("[") and clean_val.endswith("]"):
                inner = clean_val[1:-1].strip()
                fields["tools"] = (
                    [part.strip() for part in inner.split(",") if part.strip()] if inner else []
                )
                i += 1
                continue
            if clean_val:
                fields["tools"] = [part.strip() for part in clean_val.split(",") if part.strip()]
                i += 1
                continue

            # Empty value after `tools:`: check subsequent `- item` lines
            items: list[str] = []
            j = i + 1
            while j < n:
                sub = frontmatter[j]
                sub_stripped = sub.strip()
                if not sub_stripped or sub_stripped.startswith("#"):
                    j += 1
                    continue
                if sub_stripped == "-" or sub_stripped.startswith("- "):
                    item = sub_stripped[1:].strip()
                    if item:
                        items.append(item)
                    j += 1
                    continue
                if sub.startswith((" ", "\t")) and ":" not in sub_stripped:
                    j += 1
                    continue
                break
            if items:
                fields["tools"] = items
            i = j
            continue

        unquoted = _unquote_yaml_scalar(val)
        if unquoted in BLOCK_SCALAR_MARKERS:
            parts: list[str] = []
            j = i + 1
            while j < n:
                sub = frontmatter[j]
                if sub.startswith((" ", "\t")):
                    if sub.strip():
                        parts.append(sub.strip())
                    j += 1
                    continue
                if not sub.strip():
                    j += 1
                    continue
                break
            fields[key] = " ".join(parts)
            i = j
            continue

        fields[key] = unquoted
        i += 1

    return fields


def _parse_effort(fields: dict[str, Any], source: str) -> str | None:
    """Resolves ``effort`` / ``reasoning_effort`` / ``thinking`` from frontmatter fields."""
    raw: str | None = None
    for key in ("effort", "reasoning_effort", "thinking"):
        candidate = fields.get(key)
        if isinstance(candidate, str) and candidate.strip():
            raw = candidate.strip()
            break
    if raw is None:
        return None
    lowered = raw.lower()
    if lowered in INHERIT_VALUES:
        return None
    if lowered in DISABLED_VALUES:
        return "off"
    try:
        return normalize_reasoning_effort(raw)
    except ValueError as exc:
        print(
            f"⚠️ Warning: Ignoring invalid reasoning effort '{raw}' in subagent '{source}': {exc}",
            file=sys.stderr,
        )
        return None


def _parse_max_steps(fields: dict[str, Any], source: str) -> int | None:
    """Resolves optional positive ``max_steps`` / ``max_turns`` from frontmatter fields."""
    raw: str | None = None
    for key in ("max_steps", "max_turns"):
        candidate = fields.get(key)
        if isinstance(candidate, str) and candidate.strip():
            raw = candidate.strip()
            break
    if raw is None:
        return None
    if raw.lower() in {"", "none", "null", "default", "inherit"}:
        return None
    try:
        value = int(raw)
    except ValueError:
        print(
            f"⚠️ Warning: Ignoring non-integer max_steps '{raw}' in subagent '{source}'.",
            file=sys.stderr,
        )
        return None
    if value <= 0:
        print(
            f"⚠️ Warning: Ignoring non-positive max_steps '{raw}' in subagent '{source}'.",
            file=sys.stderr,
        )
        return None
    return value


def _parse_model(fields: dict[str, Any]) -> str | None:
    raw = fields.get("model")
    if not isinstance(raw, str):
        return None
    clean = raw.strip()
    if clean.lower() in {"", "inherit", "default", "parent", "none", "null"}:
        return None
    return clean


def _parse_read_only(fields: dict[str, Any], source: str, *, default: bool = False) -> bool:
    """Resolves optional ``read_only`` / ``readonly`` boolean from frontmatter fields."""
    raw: str | None = None
    for key in ("read_only", "readonly"):
        candidate = fields.get(key)
        if isinstance(candidate, str) and candidate.strip():
            raw = candidate.strip()
            break
    if raw is None:
        return default
    lowered = raw.lower()
    if lowered in {"true", "yes", "1", "on", "enable", "enabled"}:
        return True
    if lowered in {"false", "no", "0", "off", "disable", "disabled"}:
        return False
    print(
        f"⚠️ Warning: Ignoring invalid read_only '{raw}' in subagent '{source}'.",
        file=sys.stderr,
    )
    return default


def parse_subagent_markdown(
    content: str,
    default_name: str,
    scope: str = "project",
    path: str = "",
) -> Subagent:
    """Parses a subagent Markdown document with optional YAML frontmatter."""
    has_frontmatter, frontmatter, body = _split_frontmatter(content.splitlines())
    fields = _extract_frontmatter_fields(frontmatter) if has_frontmatter else {}

    raw_name = fields.get("name")
    name = (
        raw_name.strip().lower()
        if isinstance(raw_name, str) and raw_name.strip()
        else default_name.strip().lower()
    )
    described = _description_from_frontmatter(frontmatter) if has_frontmatter else None
    description = described or _first_heading(body) or name

    raw_tools = fields.get("tools")
    tools = _normalize_tools_list(raw_tools if isinstance(raw_tools, list) else None)
    model = _parse_model(fields)
    effort = _parse_effort(fields, source=path or name)
    max_steps = _parse_max_steps(fields, source=path or name)
    builtin_default = BUILTIN_SUBAGENTS.get(name)
    read_only = _parse_read_only(
        fields,
        source=path or name,
        default=bool(builtin_default and builtin_default.read_only),
    )

    body_text = "\n".join(body).strip()
    system_prompt = body_text or DEFAULT_SYSTEM_PROMPT

    return Subagent(
        name=name,
        description=description,
        system_prompt=system_prompt,
        scope=scope,
        tools=tools,
        model=model,
        effort=effort,
        max_steps=max_steps,
        path=path,
        read_only=read_only,
    )


def _load_subagent(entry: Path, scope: str, workspace: Path | None) -> tuple[str, Subagent] | None:
    """Loads one subagent ``.md`` file or directory containing ``AGENT.md``."""
    doc = (
        entry / "AGENT.md" if entry.is_dir() else (entry if entry.suffix.lower() == ".md" else None)
    )
    if not doc or not doc.is_file() or entry.name.lower() == "readme.md":
        return None

    content = doc.read_text(encoding="utf-8", errors="replace")
    resolved = doc.resolve()
    path_str = (
        str(resolved.relative_to(workspace))
        if workspace and resolved.is_relative_to(workspace)
        else str(doc)
    )
    agent = parse_subagent_markdown(
        content=content,
        default_name=entry.stem.lower(),
        scope=scope,
        path=path_str,
    )
    return agent.name, agent


def discover_subagents(
    workspace: Path | str | None,
    global_dir: Path | str | None = None,
    *,
    include_builtins: bool = True,
) -> dict[str, Subagent]:
    """Discovers subagents from built-ins, global dir, and project dir.

    Project subagents shadow global ones, and both shadow built-ins sharing
    the same name.
    """
    ws = Path(workspace).resolve() if workspace else None

    def loader(entry: Path, scope: str) -> tuple[str, Subagent] | None:
        return _load_subagent(entry, scope, ws)

    discovered = discover_resources(
        resource_name="agents",
        loader=loader,
        workspace_dir=workspace,
        global_dir=global_dir,
    )
    result: dict[str, Subagent] = dict(BUILTIN_SUBAGENTS) if include_builtins else {}
    for res in discovered.values():
        result[res.id] = res.item
    return result


def format_subagent_desc(agent: Subagent, *, include_scope: bool = True) -> str:
    """Formats a concise one-line summary of a subagent for listings and menus."""
    badge = agent.badge if include_scope else ""
    meta: list[str] = []
    if agent.model:
        meta.append(f"model:{agent.model}")
    if agent.effort:
        meta.append(f"effort:{agent.effort}")
    if agent.tools is not None:
        tools_label = ",".join(agent.tools) if agent.tools else "none"
        meta.append(f"tools:{tools_label}")
    if agent.max_steps is not None:
        meta.append(f"max_steps:{agent.max_steps}")
    suffix = f" ({', '.join(meta)})" if meta else ""
    return f"{badge}{agent.description}{suffix}"


def render_subagents_index(subagents: dict[str, Subagent]) -> str:
    """Renders the available subagents section injected into the parent system prompt."""
    if not subagents:
        return ""
    lines = ["--- Available Subagents ---"]
    for name, agent in sorted(subagents.items()):
        tools_label = (
            "all" if agent.tools is None else (", ".join(agent.tools) if agent.tools else "none")
        )
        extras = [f"tools: {tools_label}"]
        if agent.model:
            extras.append(f"model: {agent.model}")
        if agent.effort:
            extras.append(f"effort: {agent.effort}")
        if agent.max_steps is not None:
            extras.append(f"max_steps: {agent.max_steps}")
        lines.append(f"- {agent.badge}{name} ({', '.join(extras)}): {agent.description}")
    lines.append(
        "Tip: Use the `subagent` tool to delegate isolated tasks with a clean context "
        "(pass `agent` or `tasks` for parallel read-only execution)."
    )
    lines.append("---------------------------")
    return "\n".join(lines)


@dataclass(frozen=True)
class SubagentCommandResult:
    """Outcome of a ``/agent`` command: either info/error to display or a turn prompt."""

    action: str  # 'skip' or 'turn'
    message: str
    is_error: bool = False
    prompt: str | None = None


def compose_subagent_prompt(subagent: Subagent, instruction: str) -> str:
    """Builds a parent turn prompt directing delegation to the selected subagent."""
    return (
        f"Delegate the following task to the `{subagent.name}` subagent using the `subagent` "
        f'tool (`agent="{subagent.name}"`), then report the result concisely:\n\n'
        f"{instruction.strip()}"
    )


def _list_subagents(subagents: dict[str, Subagent]) -> str:
    names = sorted(subagents)
    lines = ["Available Subagents:"]
    for index, name in enumerate(names, start=1):
        lines.append(f"  {index}. {name:<14} - {format_subagent_desc(subagents[name])}")
    lines.append("\nTip: Type /agent <name|number> <task> to delegate a task to a subagent.")
    lines.append("     Type /agent <name|number> to inspect its configuration.")
    return "\n".join(lines)


def _describe_subagent(agent: Subagent) -> str:
    tools_str = "all (inherit)" if agent.tools is None else (", ".join(agent.tools) or "none")
    lines = [
        f"Subagent: {agent.name} {agent.badge.rstrip()}",
        f"  • Description: {agent.description}",
        f"  • Tools: {tools_str}",
        f"  • Model: {agent.model or 'inherit'}",
        f"  • Effort: {agent.effort or 'inherit'}",
        f"  • Max Steps: {agent.max_steps if agent.max_steps is not None else 'unlimited'}",
    ]
    if agent.path:
        lines.append(f"  • Source: {agent.path}")
    lines.append(f"  • Usage: /agent {agent.name} <task>")
    return "\n".join(lines)


def _select_subagent(
    target: str, subagents: dict[str, Subagent]
) -> tuple[Subagent | None, str | None]:
    """Resolves a subagent by 1-based index or case-insensitive name."""
    names = sorted(subagents)
    if target.isdigit():
        index = int(target) - 1
        if not 0 <= index < len(names):
            return (
                None,
                f"Invalid subagent number {target}. Type /agent to view available subagents.",
            )
        return subagents[names[index]], None

    lowered = target.lower()
    for name in names:
        if name.lower() == lowered:
            return subagents[name], None
    return None, f"Subagent '{target}' not found. Type /agent to view available subagents."


def resolve_subagent_command(
    user_input: str, subagents: dict[str, Subagent]
) -> SubagentCommandResult:
    """Parses and resolves a ``/agent`` command into a listing, inspection, or delegation prompt."""
    parts = user_input.strip().split(maxsplit=2)
    if len(parts) == 1:
        if not subagents:
            return SubagentCommandResult(
                "skip",
                "ℹ️ No subagents found. Place definitions in "
                ".microharness/agents/<name>.md or ~/.microharness/agents/<name>.md.",
            )
        return SubagentCommandResult("skip", _list_subagents(subagents))

    agent, error = _select_subagent(parts[1].strip(), subagents)
    if agent is None:
        return SubagentCommandResult("skip", error or "Unknown subagent.", is_error=True)

    instruction = parts[2].strip() if len(parts) > 2 else ""
    if not instruction:
        return SubagentCommandResult("skip", _describe_subagent(agent))

    return SubagentCommandResult(
        action="turn",
        message=f"Delegating to subagent: {agent.name} [{agent.scope}]",
        prompt=compose_subagent_prompt(agent, instruction),
    )


class SubagentTool:
    """Tool that runs an isolated child AgentLoop with its own Memory and scoped tools."""

    name = "subagent"
    parallel_safe = True
    description = (
        "Delegate a self-contained task to an isolated subagent with a clean context window. "
        "Supports built-in or custom agent presets (e.g. 'explore', 'reviewer', 'worker'), "
        "optional model/effort/max_steps overrides, and parallel execution via `tasks`."
    )
    parameters: dict[str, Any] = {
        "type": "object",
        "properties": {
            "task": {
                "type": "string",
                "description": "Self-contained instruction for the subagent (omit when passing `tasks`).",
            },
            "agent": {
                "type": "string",
                "description": "Subagent preset name (default: 'worker').",
            },
            "model": {
                "type": "string",
                "description": "Optional model alias or provider/model override.",
            },
            "effort": {
                "type": "string",
                "description": "Optional reasoning effort override ('low', 'medium', 'high', 'xhigh', 'max', 'off', or token budget).",
            },
            "max_steps": {
                "type": "integer",
                "description": "Optional maximum steps the subagent may run.",
            },
            "tasks": {
                "type": "array",
                "description": "Optional batch of read-only subagent tasks to run concurrently in parallel.",
                "items": {
                    "type": "object",
                    "properties": {
                        "task": {"type": "string"},
                        "agent": {"type": "string"},
                        "model": {"type": "string"},
                        "effort": {"type": "string"},
                        "max_steps": {"type": "integer"},
                    },
                    "required": ["task"],
                },
            },
        },
    }

    def __init__(
        self,
        *,
        env: Environment,
        parent_registry: ToolRegistry,
        subagents: dict[str, Subagent],
        model_getter: Callable[[], Model],
        model_builder: Callable[[str, str | None], tuple[bool, str, Model]] | None = None,
        effort_getter: Callable[[], str | None] | None = None,
        run_dir_getter: Callable[[], Path | None] | None = None,
        max_context_tokens_getter: Callable[[], int] | None = None,
        trigger_ratio: float = 0.80,
        max_retries: int = 3,
        retry_delay: float = 5.0,
        sandboxed: bool = False,
        agents_instructions: str = "",
        skills_index: str = "",
        default_max_steps: int | None = DEFAULT_SUBAGENT_MAX_STEPS,
        doom_loop_threshold: int = DEFAULT_DOOM_LOOP_THRESHOLD,
        doom_loop_max_nudges: int = DEFAULT_DOOM_LOOP_MAX_NUDGES,
        on_file_modified: Callable[[str], None] | None = None,
        on_step: Callable[[str, StepTrace], None] | None = None,
    ):
        self.env = env
        self.parent_registry = parent_registry
        self.subagents = dict(subagents)
        self.model_getter = model_getter
        self.model_builder = model_builder
        self.effort_getter = effort_getter
        self.run_dir_getter = run_dir_getter
        self.max_context_tokens_getter = max_context_tokens_getter
        self.trigger_ratio = trigger_ratio
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.sandboxed = sandboxed
        self.agents_instructions = agents_instructions
        self.skills_index = skills_index
        if default_max_steps is not None and (
            isinstance(default_max_steps, bool)
            or not isinstance(default_max_steps, int)
            or default_max_steps <= 0
        ):
            raise ValueError("default_max_steps must be a positive integer or None.")
        self.default_max_steps = default_max_steps
        self.doom_loop_threshold = doom_loop_threshold
        self.doom_loop_max_nudges = doom_loop_max_nudges
        self.on_file_modified = on_file_modified
        self.on_step = on_step
        self._write_lock = threading.Lock()
        self._model_lock = threading.RLock()
        self._usage_lock = threading.Lock()
        self.total_usage = TokenUsage()

    def execute(
        self,
        task: str = "",
        agent: str = "worker",
        model: str | None = None,
        effort: str | None = None,
        max_steps: int | None = None,
        tasks: list[dict[str, Any]] | None = None,
    ) -> tuple[int, str]:
        """Executes a single subagent task or a parallel batch of subagent tasks."""
        if tasks is not None:
            return self._run_parallel(tasks)
        return self._run_single(
            task=task,
            agent=agent,
            model=model,
            effort=effort,
            max_steps=max_steps,
        )

    def _run_parallel(self, tasks: Any) -> tuple[int, str]:
        if not isinstance(tasks, list) or not tasks:
            return 1, "Error: `tasks` must be a non-empty list of task objects."

        allowed_keys = {"task", "agent", "model", "effort", "max_steps"}
        normalized_items: list[dict[str, Any]] = []
        for idx, item in enumerate(tasks, start=1):
            if not isinstance(item, dict):
                return 1, f"Error: Task entry #{idx} must be an object."
            unknown = set(item) - allowed_keys
            if unknown:
                return (
                    1,
                    f"Error: Task entry #{idx} contains unsupported key(s): {', '.join(sorted(unknown))}",
                )
            raw_task = item.get("task")
            if not isinstance(raw_task, str) or not raw_task.strip():
                return 1, f"Error: Task entry #{idx} must include a non-empty string `task`."
            normalized_items.append(
                {
                    "task": raw_task,
                    "agent": item.get("agent") if item.get("agent") is not None else "worker",
                    "model": item.get("model"),
                    "effort": item.get("effort"),
                    "max_steps": item.get("max_steps"),
                }
            )

        if len(normalized_items) == 1:
            return self._run_single(**normalized_items[0])

        resolved_specs: list[Subagent] = []
        for item in normalized_items:
            raw_agent = item["agent"]
            if not isinstance(raw_agent, str):
                return 1, "Error: `agent` must be a string."
            spec, agent_err = self._resolve_agent(raw_agent)
            if spec is None:
                return 1, agent_err
            resolved_specs.append(spec)

        write_capable = sorted(
            {spec.name for spec in resolved_specs if self._is_write_capable(spec)}
        )
        if write_capable:
            names = ", ".join(f"'{name}'" for name in write_capable)
            return (
                1,
                f"Error: Parallel `tasks` batch cannot run write-capable subagent(s) ({names}) "
                "concurrently on the shared workspace. Run write-capable subagents sequentially, "
                "or use read-only presets (e.g. 'explore', 'reviewer') for parallel `tasks`.",
            )

        with ThreadPoolExecutor(max_workers=min(len(normalized_items), 4)) as pool:
            results = list(pool.map(lambda kw: self._run_single(**kw), normalized_items))

        overall_exit = 0 if all(code == 0 for code, _ in results) else 1
        combined = "\n\n---\n\n".join(output for _, output in results)
        return overall_exit, combined

    def _resolve_agent(self, agent_name: str) -> tuple[Subagent | None, str]:
        clean = (agent_name or "worker").strip().lower() or "worker"
        for key, spec in self.subagents.items():
            if key.lower() == clean:
                return spec, ""
        available = ", ".join(sorted(self.subagents)) or "none"
        return None, f"Error: Unknown subagent '{agent_name}'. Available: {available}"

    def _resolve_effort(
        self, arg_effort: str | None, spec: Subagent
    ) -> tuple[bool, bool, str | None, str]:
        """Returns ``(ok, has_explicit_override, normalized_effort_or_none, error)``."""
        if isinstance(arg_effort, str) and arg_effort.strip():
            lowered = arg_effort.strip().lower()
            if lowered not in INHERIT_VALUES:
                if lowered in DISABLED_VALUES:
                    return True, True, None, ""
                try:
                    return True, True, normalize_reasoning_effort(arg_effort), ""
                except ValueError as exc:
                    return False, False, None, f"Error: {exc}"
        elif arg_effort is not None and not isinstance(arg_effort, str):
            return False, False, None, "Error: `effort` must be a string."

        if spec.effort is not None:
            resolved = None if spec.effort == "off" else spec.effort
            return True, True, resolved, ""

        inherited = self.effort_getter() if self.effort_getter else None
        return True, False, inherited, ""

    def _resolve_model_spec(
        self, arg_model: str | None, spec: Subagent
    ) -> tuple[bool, str | None, str]:
        if arg_model is not None:
            if not isinstance(arg_model, str):
                return False, None, "Error: `model` must be a string."
            clean = arg_model.strip()
            if clean and clean.lower() not in INHERIT_VALUES:
                return True, clean, ""
        if spec.model:
            return True, spec.model, ""
        return True, None, ""

    def _is_write_capable(self, spec: Subagent) -> bool:
        """Returns True if the subagent may mutate workspace files."""
        if spec.read_only:
            return False
        if spec.tools is None:
            return True
        registered = self.parent_registry.registered_tools()
        for tool_name in spec.tools:
            if tool_name in BLOCKED_CHILD_TOOLS:
                continue
            if tool_name == "bash":
                return True
            tool = registered.get(tool_name)
            if tool is not None and getattr(tool, "modifies_files", False):
                return True
        return False

    def _bind_tool_to_env(self, tool: Tool, child_env: Environment) -> Tool:
        """Returns a tool bound to ``child_env`` without mutating ``parent_registry`` tools."""
        if child_env is self.env:
            return tool
        if isinstance(tool, BashTool):
            return BashTool(child_env, timeout=tool.timeout, max_timeout=tool.max_timeout)
        if isinstance(tool, ReadFileTool):
            return ReadFileTool(child_env)
        if isinstance(tool, WriteFileTool):
            return WriteFileTool(child_env)
        if isinstance(tool, EditFileTool):
            return EditFileTool(child_env)
        if getattr(tool, "env", None) is self.env:
            cloned = copy.copy(tool)
            setattr(cloned, "env", child_env)
            return cloned
        return tool

    def _build_child_registry(self, spec: Subagent, child_env: Environment) -> ToolRegistry:
        child_registry = ToolRegistry(
            env=child_env,
            max_tool_tokens=self.parent_registry.max_tool_tokens,
            hooks=self.parent_registry.hooks,
        )
        for name, tool in self.parent_registry.registered_tools().items():
            if name in BLOCKED_CHILD_TOOLS:
                continue
            if spec.read_only and getattr(tool, "modifies_files", False):
                continue
            if spec.tools is not None and name not in spec.tools:
                continue
            child_registry.register(self._bind_tool_to_env(tool, child_env))
        return child_registry

    def _record_usage(
        self,
        child_usage: TokenUsage,
        parent_model: Model,
        child_model: Model,
    ) -> None:
        with self._usage_lock:
            self.total_usage = self.total_usage + child_usage
            if child_model is not parent_model and hasattr(parent_model, "total_prompt_tokens"):
                cur_prompt = int(getattr(parent_model, "total_prompt_tokens", 0) or 0)
                cur_comp = int(getattr(parent_model, "total_completion_tokens", 0) or 0)
                cur_cached = int(getattr(parent_model, "total_cached_tokens", 0) or 0)
                setattr(parent_model, "total_prompt_tokens", cur_prompt + child_usage.prompt_tokens)
                setattr(
                    parent_model,
                    "total_completion_tokens",
                    cur_comp + child_usage.completion_tokens,
                )
                setattr(parent_model, "total_cached_tokens", cur_cached + child_usage.cached_tokens)

    def _run_single(
        self,
        task: str = "",
        agent: str = "worker",
        model: str | None = None,
        effort: str | None = None,
        max_steps: int | None = None,
    ) -> tuple[int, str]:
        if not isinstance(task, str) or not task.strip():
            return 1, "Error: `task` must be a non-empty string."
        if not isinstance(agent, str):
            return 1, "Error: `agent` must be a string."

        spec, agent_err = self._resolve_agent(agent)
        if spec is None:
            return 1, agent_err

        if max_steps is not None:
            if isinstance(max_steps, bool) or not isinstance(max_steps, int) or max_steps <= 0:
                return 1, "Error: `max_steps` must be a positive integer."
            effective_max_steps = max_steps
        elif spec.max_steps is not None:
            effective_max_steps = spec.max_steps
        else:
            effective_max_steps = self.default_max_steps

        ok_eff, has_explicit_effort, resolved_effort, eff_err = self._resolve_effort(effort, spec)
        if not ok_eff:
            return 1, eff_err

        ok_mod, target_model_spec, mod_err = self._resolve_model_spec(model, spec)
        if not ok_mod:
            return 1, mod_err

        parent_model = self.model_getter()
        if target_model_spec is not None:
            if self.model_builder is None:
                return (
                    1,
                    f"Error: Cannot resolve subagent model '{target_model_spec}': no model builder configured.",
                )
            builder_effort = (
                "off" if (has_explicit_effort and resolved_effort is None) else resolved_effort
            )
            ok_build, build_err, child_model = self.model_builder(target_model_spec, builder_effort)
            if not ok_build:
                return 1, f"Error resolving subagent model '{target_model_spec}': {build_err}"
        elif isinstance(parent_model, EndpointAwareModel):
            child_model = copy.copy(parent_model)
            if hasattr(child_model, "total_prompt_tokens"):
                setattr(child_model, "total_prompt_tokens", 0)
            if hasattr(child_model, "total_completion_tokens"):
                setattr(child_model, "total_completion_tokens", 0)
            if hasattr(child_model, "total_cached_tokens"):
                setattr(child_model, "total_cached_tokens", 0)
            if hasattr(child_model, "last_usage"):
                setattr(child_model, "last_usage", {})
            if has_explicit_effort and isinstance(child_model, ReasoningAwareModel):
                child_model.set_reasoning_effort(resolved_effort)
        else:
            child_model = parent_model

        is_write_capable = self._is_write_capable(spec)
        child_env: Environment = ReadOnlyEnvironment(self.env) if spec.read_only else self.env
        child_registry = self._build_child_registry(spec, child_env)
        child_system_prompt = build_system_prompt(
            spec.system_prompt,
            tool_docs=child_registry.to_system_prompt_snippet(),
            workspace=child_env.workspace_dir,
            sandboxed=self.sandboxed,
            agents_instructions=self.agents_instructions,
            skills_index=self.skills_index,
        )
        max_ctx = (
            self.max_context_tokens_getter()
            if self.max_context_tokens_getter is not None
            else 64000
        )
        child_memory = Memory(
            system_prompt=child_system_prompt,
            user_prompt=task.strip(),
            max_context_tokens=max_ctx,
            trigger_ratio=self.trigger_ratio,
        )

        base_run_dir = (
            self.run_dir_getter() if self.run_dir_getter is not None else None
        ) or self.env.logs_dir.parent
        child_run_dir = base_run_dir / "subagents" / f"{spec.name}_{uuid.uuid4().hex[:6]}"
        child_tracer = Tracer(
            task=Task(
                id=f"subagent_{spec.name}",
                instruction=task.strip(),
                max_steps=effective_max_steps or 0,
            ),
            output_dir=child_run_dir,
            workspace=self.env.workspace_dir,
            verbose=False,
        )
        if self.on_step is not None:
            on_step_cb = self.on_step
            orig_record_step = child_tracer.record_step

            def _record_and_notify(step: StepTrace) -> None:
                orig_record_step(step)
                on_step_cb(spec.name, step)

            child_tracer.record_step = _record_and_notify  # type: ignore[method-assign]

        parent_deadline = getattr(parent_model, "deadline", None)
        if child_model is not parent_model:
            if isinstance(child_model, DeadlineAwareModel) and parent_deadline is not None:
                child_model.set_deadline(parent_deadline)
            parent_probe = getattr(parent_model, "_cancel_probe", None)
            if isinstance(child_model, CancellableModel) and callable(parent_probe):
                child_model.set_cancel_probe(parent_probe)

        child_loop = AgentLoop(
            model=child_model,
            memory=child_memory,
            tools=child_registry,
            doom_loop_threshold=self.doom_loop_threshold,
            doom_loop_max_nudges=self.doom_loop_max_nudges,
        )

        with contextlib.ExitStack() as stack:
            if is_write_capable:
                stack.enter_context(self._write_lock)
            if (
                child_model is parent_model
                and has_explicit_effort
                and isinstance(parent_model, ReasoningAwareModel)
            ):
                stack.enter_context(self._model_lock)
                prev_effort = parent_model.get_reasoning_effort()
                parent_model.set_reasoning_effort(resolved_effort)
                stack.callback(parent_model.set_reasoning_effort, prev_effort)

            try:
                outcome = run_steps(
                    child_model,
                    child_loop,
                    child_tracer,
                    start_step=1,
                    max_steps=effective_max_steps,
                    deadline=parent_deadline,
                    max_retries=self.max_retries,
                    retry_delay=self.retry_delay,
                    require_finish_tool=False,
                    propagate_interrupts=True,
                )
            finally:
                if child_model is not parent_model:
                    if isinstance(child_model, DeadlineAwareModel):
                        child_model.set_deadline(None)
                    if isinstance(child_model, CancellableModel):
                        child_model.set_cancel_probe(None)

            if self.on_file_modified is not None:
                for path in sorted(child_memory.modified_files):
                    self.on_file_modified(path)

        child_c_stats = (
            child_model.get_cache_stats() if isinstance(child_model, CacheAwareModel) else None
        )
        child_usage = summarize_run_usage(
            (step.usage for step in child_tracer.steps), child_c_stats
        )
        self._record_usage(child_usage, parent_model, child_model)
        child_tracer.finalize(finish_reason=outcome.reason, telemetry=child_usage.as_dict())

        if outcome.reason == INTERRUPTED:
            raise ModelCancelledError("Subagent interrupted.")

        final_text = ""
        if child_tracer.steps:
            final_text = _final_response_text(child_tracer.steps[-1]).strip()
        if not final_text:
            final_text = f"(Subagent exited with status '{outcome.reason}' without final text.)"

        header = f"[Subagent: {spec.name}] ({outcome.reason} in {len(child_tracer.steps)} step(s))"
        exit_code = 0 if outcome.finished else 1
        return exit_code, f"{header}\n{final_text}"
