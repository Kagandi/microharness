"""Named agent config presets (bundles of ``agent.md``, ``subagents/``, ``tools/``, ``skills/``).

A preset is a self-contained directory describing a runnable agent configuration:

- ``agent.md`` (or ``AGENT.md``): Optional YAML frontmatter (name, description, model,
  reasoning effort, tool allowlist, isolation flags, etc.) + base system prompt body.
- ``config.json``: Optional JSONC overrides for ``HarnessConfig`` / hooks / models.
- ``subagents/`` (or ``agents/``): Preset-scoped subagent definitions.
- ``tools/``: Preset-scoped Python tool plugins (subject to ``trust_project_tools``
  when the preset is not from ``<global_dir>/presets``).
- ``skills/``: Preset-scoped progressive-disclosure skills.
- ``commands/``: Optional preset-scoped custom slash commands.
"""

from __future__ import annotations

import os
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.commands import Command, _load_command
from core.config import (
    _coerce_bool,
    _coerce_string_list,
    _parse_optional_positive_number,
    read_jsonc,
)
from core.discovery import discover_resources
from core.environment import Environment
from core.skills import Skill, _first_heading, _load_skill, _split_frontmatter
from core.subagents import (
    INHERIT_VALUES,
    TOOL_NAME_ALIASES,
    Subagent,
    _extract_frontmatter_fields,
    _load_subagent,
    _normalize_tool_token,
)
from core.tools import Tool, load_tool_plugin

PRESET_MANIFEST_NAMES: tuple[str, str] = ("agent.md", "AGENT.md")
PRESET_CONFIG_FILENAME = "config.json"

_FRONTMATTER_BOOL_KEYS: frozenset[str] = frozenset(
    {
        "sandbox",
        "docker",
        "auto_discover_tools",
        "allow_network_tools",
        "disable_default_tools",
        "include_default_hooks",
        "checkpoints",
    }
)
_FRONTMATTER_INT_KEYS: frozenset[str] = frozenset(
    {
        "max_tool_tokens",
        "max_context_tokens",
        "max_step_retries",
        "subagent_max_steps",
        "docker_pids_limit",
        "max_steps",
    }
)
_FRONTMATTER_FLOAT_KEYS: frozenset[str] = frozenset(
    {
        "trigger_ratio",
        "step_retry_delay",
        "bash_timeout",
        "docker_cpus",
    }
)
_FRONTMATTER_STR_KEYS: frozenset[str] = frozenset(
    {
        "model",
        "provider",
        "endpoint",
        "api_key",
        "cache_ttl",
        "permission_mode",
        "docker_image",
        "docker_network",
        "docker_memory",
        "runs_dir",
    }
)
_PRESET_CONTROL_KEYS: frozenset[str] = frozenset(
    {
        "name",
        "description",
        "tools",
        "isolated",
        "include_builtin_subagents",
        "inherit_global",
        "inherit_project",
        "include_workspace_agents_md",
    }
)


@dataclass(frozen=True)
class AgentPreset:
    """A loaded agent preset bundle and its resolved metadata."""

    id: str
    name: str
    description: str
    path: Path
    scope: str  # 'global', 'project', or 'path'
    system_prompt: str | None = None
    config: dict[str, Any] = field(default_factory=dict)
    tools: tuple[str, ...] | None = None
    isolated: bool = False
    include_builtin_subagents: bool = True
    inherit_global: bool = True
    inherit_project: bool = True
    include_workspace_agents_md: bool = True

    @property
    def badge(self) -> str:
        return f"[{self.scope}] "


def _slugify(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", value.strip().lower()).strip("-")


def _find_agent_doc(preset_dir: Path) -> Path | None:
    for filename in PRESET_MANIFEST_NAMES:
        candidate = preset_dir / filename
        if candidate.is_file():
            return candidate
    return None


def _normalize_preset_tools(raw_tools: Any) -> tuple[str, ...] | None:
    """Normalizes a preset's tool allow-list; None means allow all registered tools."""
    if raw_tools is None:
        return None
    if isinstance(raw_tools, str):
        stripped = raw_tools.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            stripped = stripped[1:-1].strip()
        items = [part.strip() for part in stripped.split(",") if part.strip()] if stripped else []
    elif isinstance(raw_tools, (list, tuple)):
        items = [str(item) for item in raw_tools]
    else:
        raise ValueError(
            f"Unsupported preset 'tools' value: {raw_tools!r}. Must be a list or string."
        )

    normalized: list[str] = []
    seen: set[str] = set()
    for item in items:
        token = _normalize_tool_token(item)
        if not token:
            continue
        if token.lower() in ("*", "all"):
            return None
        alias = TOOL_NAME_ALIASES.get(token.lower())
        canonical = alias if alias is not None else token
        if canonical not in seen:
            seen.add(canonical)
            normalized.append(canonical)
    return tuple(normalized)


def _optional_bool(fields: dict[str, Any], key: str) -> bool | None:
    if key not in fields:
        return None
    raw = fields[key]
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return None
    return _coerce_bool(raw)


def _extract_frontmatter_config(fields: dict[str, Any]) -> dict[str, Any]:
    """Converts scalar/list frontmatter fields into typed ``HarnessConfig`` keys."""
    cfg: dict[str, Any] = {}

    for key in _FRONTMATTER_STR_KEYS:
        if key not in fields:
            continue
        raw = str(fields[key]).strip() if fields[key] is not None else ""
        if not raw or (key in ("model", "provider", "cache_ttl") and raw.lower() in INHERIT_VALUES):
            continue
        cfg[key] = raw

    for effort_key in ("reasoning_effort", "effort", "thinking", "reasoning"):
        if effort_key not in fields:
            continue
        raw_effort = str(fields[effort_key]).strip() if fields[effort_key] is not None else ""
        if not raw_effort or raw_effort.lower() in INHERIT_VALUES:
            continue
        cfg["reasoning_effort"] = raw_effort
        break

    for key in _FRONTMATTER_BOOL_KEYS:
        val = _optional_bool(fields, key)
        if val is not None:
            cfg[key] = val

    for key in _FRONTMATTER_INT_KEYS:
        if key not in fields:
            continue
        raw_val = str(fields[key]).strip() if fields[key] is not None else ""
        if not raw_val:
            continue
        if raw_val.lower() in {"null", "none", "~"}:
            cfg[key] = None
        else:
            cfg[key] = _parse_optional_positive_number(fields[key], key)

    for key in _FRONTMATTER_FLOAT_KEYS:
        if key not in fields:
            continue
        raw_val = str(fields[key]).strip() if fields[key] is not None else ""
        if not raw_val:
            continue
        if raw_val.lower() in {"null", "none", "~"}:
            cfg[key] = None
        else:
            cfg[key] = _parse_optional_positive_number(fields[key], key, as_float=True)

    if "disabled_hooks" in fields:
        raw_disabled = fields["disabled_hooks"]
        if isinstance(raw_disabled, str):
            raw_disabled = [p.strip() for p in raw_disabled.strip("[]").split(",") if p.strip()]
        cfg["disabled_hooks"] = _coerce_string_list(raw_disabled, "disabled_hooks")

    return cfg


def _has_preset_markers(preset_dir: Path) -> bool:
    if _find_agent_doc(preset_dir) is not None:
        return True
    return (preset_dir / PRESET_CONFIG_FILENAME).is_file()


def load_preset(
    preset_dir: Path | str,
    *,
    scope: str = "path",
    preset_id: str | None = None,
) -> AgentPreset:
    """Loads an :class:`AgentPreset` from ``preset_dir``."""
    resolved = Path(preset_dir).expanduser().resolve()
    if not resolved.is_dir():
        raise FileNotFoundError(f"Agent preset directory not found: '{preset_dir}'")

    agent_doc = _find_agent_doc(resolved)
    config_file = resolved / PRESET_CONFIG_FILENAME
    has_resources = any(
        (resolved / sub).is_dir() for sub in ("subagents", "agents", "tools", "skills", "commands")
    )
    if agent_doc is None and not config_file.is_file() and not has_resources:
        raise ValueError(
            f"Directory '{resolved}' is not a valid agent preset "
            "(expected agent.md, config.json, or subagents/tools/skills)."
        )

    json_cfg = read_jsonc(config_file) if config_file.is_file() else {}

    fm_fields: dict[str, Any] = {}
    body_text: str | None = None
    body_lines: list[str] = []
    if agent_doc is not None:
        content = agent_doc.read_text(encoding="utf-8", errors="replace")
        has_fm, frontmatter, body_lines = _split_frontmatter(content.splitlines())
        if has_fm:
            fm_fields = _extract_frontmatter_fields(frontmatter)
        raw_body = "\n".join(body_lines).strip()
        body_text = raw_body or None

    fm_cfg = _extract_frontmatter_config(fm_fields)

    # Merge config.json first, then agent.md frontmatter scalars on top, stripping preset control keys
    merged_cfg: dict[str, Any] = {
        k: v for k, v in json_cfg.items() if k not in _PRESET_CONTROL_KEYS
    }
    merged_cfg.update(fm_cfg)

    raw_id = preset_id or resolved.name
    norm_id = raw_id.strip().lower()
    raw_name = fm_fields.get("name") or json_cfg.get("name") or resolved.name
    display_name = str(raw_name).strip() or resolved.name

    raw_desc = fm_fields.get("description") or json_cfg.get("description")
    description = (
        str(raw_desc).strip()
        if isinstance(raw_desc, str) and raw_desc.strip()
        else (_first_heading(body_lines) or display_name)
    )

    raw_tools = fm_fields["tools"] if "tools" in fm_fields else json_cfg.get("tools")
    tools = _normalize_preset_tools(raw_tools)

    control_merged = {**json_cfg, **fm_fields}
    isolated_val = _optional_bool(control_merged, "isolated")
    isolated = bool(isolated_val) if isolated_val is not None else False
    default_inherit = not isolated

    include_builtin_subagents = _optional_bool(control_merged, "include_builtin_subagents")
    inherit_global = _optional_bool(control_merged, "inherit_global")
    inherit_project = _optional_bool(control_merged, "inherit_project")
    include_workspace_agents_md = _optional_bool(control_merged, "include_workspace_agents_md")

    return AgentPreset(
        id=norm_id,
        name=display_name,
        description=description,
        path=resolved,
        scope=scope,
        system_prompt=body_text,
        config=merged_cfg,
        tools=tools,
        isolated=isolated,
        include_builtin_subagents=(
            include_builtin_subagents if include_builtin_subagents is not None else default_inherit
        ),
        inherit_global=inherit_global if inherit_global is not None else default_inherit,
        inherit_project=inherit_project if inherit_project is not None else default_inherit,
        include_workspace_agents_md=(
            include_workspace_agents_md
            if include_workspace_agents_md is not None
            else default_inherit
        ),
    )


def _load_preset_entry(entry: Path, scope: str) -> tuple[str, AgentPreset] | None:
    if not entry.is_dir() or not _has_preset_markers(entry):
        return None
    preset = load_preset(entry, scope=scope, preset_id=entry.name.lower())
    return preset.id, preset


def discover_presets(
    workspace: Path | str | None = None,
    global_dir: Path | str | None = None,
) -> dict[str, AgentPreset]:
    """Discovers named agent presets from global and project ``presets/`` directories."""
    discovered = discover_resources(
        resource_name="presets",
        loader=_load_preset_entry,
        workspace_dir=workspace,
        global_dir=global_dir,
    )
    return {res.id: res.item for res in discovered.values()}


def _looks_like_path(spec: str) -> bool:
    return (
        os.sep in spec
        or "/" in spec
        or "\\" in spec
        or spec.startswith((".", "~"))
        or Path(spec).expanduser().is_dir()
    )


def _infer_path_scope(
    resolved_path: Path,
    workspace: Path | str | None = None,
    global_dir: Path | str | None = None,
) -> str:
    if global_dir is not None:
        g_base = Path(global_dir).resolve()
        for cand in (g_base / "presets", g_base / ".presets"):
            if resolved_path.is_relative_to(cand):
                return "global"
    if workspace is not None:
        ws_base = Path(workspace).resolve()
        if resolved_path.is_relative_to(ws_base):
            return "project"
    return "path"


def _resolve_preset_path(
    raw_path: str | Path,
    *,
    workspace: Path | str | None = None,
    global_dir: Path | str | None = None,
) -> AgentPreset:
    candidate = Path(raw_path).expanduser()
    if not candidate.is_absolute() and workspace is not None:
        ws_candidate = Path(workspace).resolve() / candidate
        if ws_candidate.is_dir():
            candidate = ws_candidate
    resolved = candidate.resolve()
    scope = _infer_path_scope(resolved, workspace=workspace, global_dir=global_dir)
    return load_preset(resolved, scope=scope)


def resolve_preset(
    spec: AgentPreset | str | Path,
    workspace: Path | str | None = None,
    global_dir: Path | str | None = None,
) -> AgentPreset:
    """Resolves a preset instance, directory path, or preset name/slug."""
    if isinstance(spec, AgentPreset):
        return spec

    if isinstance(spec, Path):
        return _resolve_preset_path(spec, workspace=workspace, global_dir=global_dir)

    if not isinstance(spec, str) or not spec.strip():
        raise ValueError("Agent preset specification cannot be empty.")

    cleaned = spec.strip()
    if _looks_like_path(cleaned):
        return _resolve_preset_path(cleaned, workspace=workspace, global_dir=global_dir)

    presets = discover_presets(workspace=workspace, global_dir=global_dir)
    lowered = cleaned.lower()
    if lowered in presets:
        return presets[lowered]

    for preset in presets.values():
        if preset.name.strip().lower() == lowered:
            return preset

    slug = _slugify(cleaned)
    if slug:
        for preset in presets.values():
            if _slugify(preset.id) == slug or _slugify(preset.name) == slug:
                return preset

    available = ", ".join(sorted(presets)) if presets else "none found"
    raise ValueError(f"Unknown agent preset '{cleaned}'. Available presets: {available}.")


def _iter_resource_entries(folder: Path, *, pattern: str | None = None) -> list[Path]:
    if not folder.is_dir():
        return []
    entries = sorted(folder.glob(pattern)) if pattern else sorted(folder.iterdir())
    return [
        entry
        for entry in entries
        if not entry.name.startswith(("_", "."))
        and entry.name.lower() not in {"readme.md", "__pycache__"}
    ]


def discover_preset_subagents(preset: AgentPreset | None) -> dict[str, Subagent]:
    """Discovers subagents inside a preset's ``agents/`` or ``subagents/`` directory."""
    if preset is None:
        return {}
    subagents: dict[str, Subagent] = {}
    for folder_name in ("agents", "subagents"):
        for entry in _iter_resource_entries(preset.path / folder_name):
            loaded = _load_subagent(entry, "preset", None)
            if loaded is not None:
                sub_id, item = loaded
                subagents[sub_id] = item
    return subagents


def discover_preset_skills(
    preset: AgentPreset | None,
    *,
    path_base: Path | None = None,
) -> dict[str, Skill]:
    """Discovers skills inside a preset's ``skills/`` directory."""
    if preset is None:
        return {}
    base = Path(path_base).resolve() if path_base is not None else None
    skills: dict[str, Skill] = {}
    for entry in _iter_resource_entries(preset.path / "skills"):
        loaded = _load_skill(entry, "preset", base)
        if loaded is not None:
            skill_id, item = loaded
            skills[skill_id] = item
    return skills


def discover_preset_commands(preset: AgentPreset | None) -> dict[str, Command]:
    """Discovers custom slash commands inside a preset's ``commands/`` directory."""
    if preset is None:
        return {}
    commands: dict[str, Command] = {}
    for entry in _iter_resource_entries(preset.path / "commands", pattern="*.md"):
        cmd_id, item = _load_command(entry, "preset")
        commands[cmd_id] = item
    return commands


def discover_preset_tools(
    preset: AgentPreset | None,
    *,
    env: Environment | None = None,
    trust_project_tools: bool = False,
) -> list[Tool]:
    """Discovers custom tool plugins inside a preset's ``tools/`` directory.

    Presets outside ``<global_dir>/presets`` require ``trust_project_tools=True``
    before executing Python plugin modules.
    """
    if preset is None:
        return []
    entries = _iter_resource_entries(preset.path / "tools", pattern="*.py")
    if not entries:
        return []
    if preset.scope != "global" and not trust_project_tools:
        print(
            f"⚠️ Warning: Skipping untrusted preset tool plugin(s) in '{preset.path / 'tools'}'. "
            "Pass --trust-project-tools to enable.",
            file=sys.stderr,
        )
        return []

    tools_by_name: dict[str, Tool] = {}
    for entry in entries:
        for tool in load_tool_plugin(entry, env=env):
            tools_by_name[tool.name] = tool
    return list(tools_by_name.values())


def format_preset_listing(
    presets: dict[str, AgentPreset],
    *,
    active_preset: AgentPreset | None = None,
) -> str:
    """Formats discovered presets for CLI ``--list-presets`` and interactive ``/preset``."""
    if not presets:
        return (
            "No agent presets found. Create one in ~/.microharness/presets/<name>/agent.md "
            "or .microharness/presets/<name>/agent.md."
        )
    lines = ["Available Agent Presets:"]
    for idx, preset_id in enumerate(sorted(presets), start=1):
        preset = presets[preset_id]
        marker = " *" if active_preset is not None and active_preset.path == preset.path else ""
        label = (
            f"{preset_id} ({preset.name})"
            if preset.name.strip().lower() != preset_id
            else preset_id
        )
        lines.append(f"  {idx}. {label:<24} {preset.badge}- {preset.description}{marker}")
    return "\n".join(lines)
