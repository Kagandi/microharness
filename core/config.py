"""Configuration cascade: JSONC reading, global/project config, and setting resolution.

Settings resolve through a single precedence chain (highest first):

1. Explicit values passed by the caller (CLI flags / ``Harness(...)`` kwargs)
2. Project ``.microharness/config.json``
3. Global ``~/.microharness/config.json``
4. Built-in defaults declared on :class:`HarnessConfig`
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from core.tools import DEFAULT_MAX_TOOL_TOKENS

GLOBAL_CONFIG_FILENAME = "config.json"
PROJECT_CONFIG_RELPATH = ".microharness/config.json"

CACHE_TTL_CHOICES: dict[str, str] = {
    "5m": "5m",
    "5min": "5m",
    "5mins": "5m",
    "300": "5m",
    "300s": "5m",
    "1h": "1h",
    "1hour": "1h",
    "1hr": "1h",
    "60m": "1h",
    "3600": "1h",
    "3600s": "1h",
}

DISABLED_VALUES: frozenset[str] = frozenset(
    {"off", "none", "disable", "disabled", "false", "0", ""}
)

REASONING_CHOICES: frozenset[str] = frozenset({"low", "medium", "high", "xhigh", "max"})

PROJECT_TRUST_NOTICE = (
    "Project config attempted to enable 'trust_project_tools'. "
    "Project tools can only be trusted via CLI flag (--trust-project-tools) "
    "or global config (~/.microharness/config.json)."
)


def get_default_global_dir() -> Path:
    """Returns the default global directory, evaluating ``Path.home()`` at call time."""
    return Path.home() / ".microharness"


def strip_json_comments(text: str) -> str:
    """Strips ``#`` and ``//`` comments from JSON-like text while preserving string literals."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escape = False

    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
            continue

        i += 1
        if c == '"':
            in_string = True
            escape = False
            out.append(c)
        elif c == "#" or (c == "/" and i < n and text[i] == "/"):
            while i < n and text[i] != "\n":
                i += 1
        else:
            out.append(c)

    return "".join(out)


def read_jsonc(file_path: Path | None) -> dict[str, Any]:
    """Reads a JSON-with-comments file, returning ``{}`` for missing, unreadable, or non-object files."""
    if not file_path or not file_path.is_file():
        return {}
    try:
        data = json.loads(strip_json_comments(file_path.read_text(encoding="utf-8")))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_global_config(global_dir: Path | None = None) -> dict[str, Any]:
    """Loads user baseline configuration from ``~/.microharness/config.json``."""
    return read_jsonc(global_dir / GLOBAL_CONFIG_FILENAME if global_dir else None)


def load_project_config(workspace: Path | None = None) -> dict[str, Any]:
    """Loads optional project-level configuration from ``.microharness/config.json``."""
    return read_jsonc(workspace / PROJECT_CONFIG_RELPATH if workspace else None)


@dataclass(frozen=True)
class ConfigLayers:
    """The configuration layers, kept separate so trust decisions can inspect each one."""

    global_config: dict[str, Any]
    project_config: dict[str, Any]
    merged: dict[str, Any]

    @classmethod
    def of(
        cls, global_config: dict[str, Any], project_config: dict[str, Any]
    ) -> "ConfigLayers":
        """Builds the layers and their merge once.

        Project keys shadow global keys; an empty project file does not mask
        global defaults. The merge is computed here so every consumer reads the
        same dictionary instead of rebuilding it.
        """
        return cls(
            global_config=global_config,
            project_config=project_config,
            merged={**global_config, **project_config},
        )

    @classmethod
    def load(
        cls,
        global_dir: Path | None = None,
        workspace: Path | None = None,
    ) -> "ConfigLayers":
        return cls.of(
            global_config=load_global_config(global_dir),
            project_config=load_project_config(workspace),
        )


def resolve_project_tool_trust(
    explicit: bool | None,
    layers: ConfigLayers,
) -> tuple[bool, bool]:
    """Decides whether project-local tool plugins may be loaded.

    Trust is a security boundary, so it can only come from an explicit caller
    value or the user's own global config. A project config may *ask* for trust
    but can never grant it to itself.

    Returns:
        ``(trusted, project_claim_ignored)``
    """
    trusted = bool(explicit) if explicit is not None else bool(
        layers.global_config.get("trust_project_tools", False)
    )
    claimed = bool(layers.project_config.get("trust_project_tools")) and not trusted
    return trusted, claimed


class _Cascade:
    """Resolves individual settings through explicit > merged config > default."""

    def __init__(self, merged: Mapping[str, Any]):
        self._merged = merged

    def pick(
        self,
        key: str,
        explicit: Any,
        cast: Callable[[Any], Any],
        default: Any,
        *,
        require_truthy: bool = False,
    ) -> Any:
        """Returns the winning value for ``key``, cast to ``cast``.

        ``require_truthy`` treats an explicitly falsy value (``""``, ``0``) as
        "not specified", which is how an empty config entry falls back to the
        default instead of disabling the feature.
        """
        val = explicit if explicit is not None else self._merged.get(key)
        if require_truthy:
            val = val if val else default
        else:
            val = val if val is not None else default
        return cast(val) if val is not None else default


@dataclass(frozen=True)
class HarnessConfig:
    """Settings a caller may supply explicitly. ``None`` means "not specified"."""

    sandbox: bool | None = None
    docker: bool | None = None
    docker_image: str | None = None
    auto_discover_tools: bool | None = None
    disable_default_tools: bool | None = None
    runs_dir: str | Path | None = None
    max_tool_tokens: int | None = None
    max_context_tokens: int | None = None
    trigger_ratio: float | None = None
    cache_ttl: str | None = None
    reasoning_effort: str | None = None
    max_step_retries: int | None = None
    step_retry_delay: float | None = None

    def resolve(self, layers: ConfigLayers) -> "ResolvedConfig":
        """Applies the precedence cascade and normalizes validated settings."""
        c = _Cascade(layers.merged)
        cache_ttl = c.pick("cache_ttl", self.cache_ttl, str, "5m")
        reasoning = c.pick("reasoning_effort", self.reasoning_effort, str, None)

        tool_tok = c.pick("max_tool_tokens", self.max_tool_tokens, int, None)
        if tool_tok is None:
            legacy_tool_chars = c.pick("max_tool_chars", None, int, None)
            tool_tok = legacy_tool_chars // 4 if legacy_tool_chars else DEFAULT_MAX_TOOL_TOKENS

        context_tok = (
            c.pick("max_context_tokens", self.max_context_tokens, int, None)
            or c.pick("max_tokens", None, int, None)
            or c.pick("context_tokens", None, int, None)
        )
        if context_tok is None:
            legacy_context_chars = c.pick("max_context_chars", None, int, None)
            context_tok = legacy_context_chars // 4 if legacy_context_chars else 64000

        return ResolvedConfig(
            sandbox=c.pick("sandbox", self.sandbox, bool, False),
            docker=c.pick("docker", self.docker, bool, False),
            docker_image=c.pick(
                "docker_image", self.docker_image, str, "python:3.11", require_truthy=True
            ),
            auto_discover_tools=c.pick("auto_discover_tools", self.auto_discover_tools, bool, True),
            disable_default_tools=c.pick(
                "disable_default_tools", self.disable_default_tools, bool, False
            ),
            runs_dir=c.pick("runs_dir", self.runs_dir, Path, None, require_truthy=True),
            max_tool_tokens=tool_tok,
            max_context_tokens=context_tok,
            trigger_ratio=c.pick("trigger_ratio", self.trigger_ratio, float, 0.80),
            cache_ttl=normalize_cache_ttl(cache_ttl),
            reasoning_effort=normalize_reasoning_effort(reasoning),
            max_step_retries=c.pick("max_step_retries", self.max_step_retries, int, 3),
            step_retry_delay=c.pick("step_retry_delay", self.step_retry_delay, float, 5.0),
        )


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully resolved, cast, and validated settings ready for the harness to use."""

    sandbox: bool
    docker: bool
    docker_image: str
    auto_discover_tools: bool
    disable_default_tools: bool
    runs_dir: Path | None
    max_tool_tokens: int
    max_context_tokens: int
    trigger_ratio: float
    cache_ttl: str | None
    reasoning_effort: str | None
    max_step_retries: int
    step_retry_delay: float


def normalize_cache_ttl(ttl: str | None) -> str | None:
    """Normalizes a cache TTL setting to ``'5m'``, ``'1h'``, or ``None`` (disabled).

    Supported representations:
    - '5m', '5min', '5mins', '300', '300s' -> '5m' (standard ephemeral cache)
    - '1h', '1hour', '1hr', '60m', '3600', '3600s' -> '1h' (extended beta cache)
    - 'off', 'none', 'disable', 'disabled', 'false', '0', '', None -> None
    """
    if ttl is None:
        return None
    val = str(ttl).strip().lower()
    if val in DISABLED_VALUES:
        return None
    if val in CACHE_TTL_CHOICES:
        return CACHE_TTL_CHOICES[val]
    raise ValueError(
        f"Unsupported cache TTL: '{ttl}'. Valid options: '5m' (5 minutes), '1h' (1 hour), or 'off'."
    )


def normalize_reasoning_effort(effort: str | None) -> str | None:
    """Normalizes a reasoning effort level to ``'low'``/``'medium'``/``'high'``/``'xhigh'``/``'max'``, a token budget, or ``None``."""
    if effort is None:
        return None
    val = str(effort).strip().lower()
    if val in DISABLED_VALUES:
        return None
    if val in REASONING_CHOICES or val.isdigit():
        return val
    raise ValueError(
        f"Unsupported reasoning effort: '{effort}'. Valid options: 'low', 'medium', 'high', 'xhigh', 'max', numeric token budget, or 'off'."
    )