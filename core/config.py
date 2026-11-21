"""Configuration cascade: JSONC reading, global/project config, and setting resolution.

Settings resolve through a single precedence chain (highest first):

1. Explicit values passed by the caller (CLI flags / ``Harness(...)`` kwargs)
2. Project ``.microharness/config.json``
3. Global ``~/.microharness/config.json``
4. Built-in defaults declared on :class:`HarnessConfig`
"""

from __future__ import annotations

import json
import sys
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from core.hooks import parse_hook_specs
from core.tools import DEFAULT_MAX_TOOL_TOKENS
from core.types import DEFAULT_DOOM_LOOP_MAX_NUDGES, DEFAULT_DOOM_LOOP_THRESHOLD

DEFAULT_SUBAGENT_MAX_STEPS: int | None = None

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
    """Returns the default global directory, evaluating ``Path.home()`` at call time.

    Falls back to legacy ``~/.mini-harness`` if it exists and ``~/.microharness`` does not.
    """
    primary = Path.home() / ".microharness"
    if not primary.exists():
        legacy = Path.home() / ".mini-harness"
        if legacy.exists():
            return legacy
    return primary


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
    def of(cls, global_config: dict[str, Any], project_config: dict[str, Any]) -> "ConfigLayers":
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
    trusted = (
        bool(explicit)
        if explicit is not None
        else bool(layers.global_config.get("trust_project_tools", False))
    )
    claimed = bool(layers.project_config.get("trust_project_tools")) and not trusted
    return trusted, claimed


def _warn_legacy_char_keys(merged: Mapping[str, Any]) -> None:
    """Emits a one-line stderr warning for removed char-based keys."""
    phrases = []
    if "max_tool_chars" in merged:
        phrases.append("uses legacy max_tool_chars which is removed; set max_tool_tokens")
    if "max_context_chars" in merged:
        phrases.append("uses legacy max_context_chars which is removed; set max_context_tokens")
    if phrases:
        print(f"Warning: config {'; '.join(phrases)}.", file=sys.stderr)


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


def _coerce_bool(val: Any) -> bool:
    """Coerces booleans and common string representations cleanly."""
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        norm = val.strip().lower()
        if norm in {"0", "false", "no", "off", "disable", "disabled", ""}:
            return False
        if norm in {"1", "true", "yes", "on", "enable", "enabled"}:
            return True
    return bool(val)


def _coerce_non_negative_int(raw: Any, name: str) -> int:
    """Validates and converts a non-negative integer setting, rejecting booleans and floats."""
    if isinstance(raw, (bool, float)):
        raise ValueError(f"Unsupported {name}: {raw!r}. Must be a non-negative integer.")
    try:
        value = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Unsupported {name}: {raw!r}. Must be a non-negative integer.") from exc
    if value < 0:
        raise ValueError(f"Unsupported {name}: {value}. Must be a non-negative integer.")
    return value


def _coerce_string_list(raw: Any, name: str) -> list[str]:
    """Normalizes an optional string sequence config entry into a list of non-empty strings."""
    if raw is None:
        return []
    if isinstance(raw, str):
        cleaned = raw.strip()
        return [cleaned] if cleaned else []
    if not isinstance(raw, Iterable) or isinstance(raw, (bytes, Mapping)):
        raise ValueError(f"Unsupported {name}: {raw!r}. Must be a list of hook names.")
    out: list[str] = []
    for item in raw:
        if not isinstance(item, str):
            raise ValueError(f"Unsupported {name} entry: {item!r}. Must be a string.")
        cleaned = item.strip()
        if cleaned:
            out.append(cleaned)
    return out


@dataclass(frozen=True)
class HarnessConfig:
    """Settings a caller may supply explicitly. ``None`` means "not specified"."""

    sandbox: bool | None = None
    docker: bool | None = None
    docker_image: str | None = None
    docker_network: str | None = None
    auto_discover_tools: bool | None = None
    allow_network_tools: bool | None = None
    disable_default_tools: bool | None = None
    runs_dir: str | Path | None = None
    max_tool_tokens: int | None = None
    max_context_tokens: int | None = None
    trigger_ratio: float | None = None
    cache_ttl: str | None = None
    reasoning_effort: str | None = None
    max_step_retries: int | None = None
    step_retry_delay: float | None = None
    bash_timeout: float | None = None
    subagent_max_steps: int | None = None
    hooks: Mapping[str, Any] | Sequence[Mapping[str, Any]] | None = None
    disabled_hooks: Sequence[str] | None = None
    include_default_hooks: bool | None = None
    doom_loop_threshold: int | None = None
    doom_loop_max_nudges: int | None = None

    def resolve(self, layers: ConfigLayers) -> "ResolvedConfig":
        """Applies the precedence cascade and normalizes validated settings."""
        c = _Cascade(layers.merged)
        _warn_legacy_char_keys(layers.merged)
        cache_ttl = c.pick("cache_ttl", self.cache_ttl, str, "5m")
        reasoning = c.pick("reasoning_effort", self.reasoning_effort, str, None)

        tool_tok = c.pick("max_tool_tokens", self.max_tool_tokens, int, None)
        if tool_tok is None:
            tool_tok = DEFAULT_MAX_TOOL_TOKENS

        context_tok = (
            c.pick("max_context_tokens", self.max_context_tokens, int, None)
            or c.pick("max_tokens", None, int, None)
            or c.pick("context_tokens", None, int, None)
        )
        if context_tok is None:
            context_tok = 64000

        raw_network = c.pick("docker_network", self.docker_network, str, None, require_truthy=True)
        docker_network = raw_network.strip() if isinstance(raw_network, str) else None
        if docker_network == "":
            docker_network = None

        bash_timeout = c.pick("bash_timeout", self.bash_timeout, float, 120.0)
        if bash_timeout <= 0:
            raise ValueError(f"Unsupported bash_timeout: {bash_timeout}. Must be greater than 0.")

        raw_subagent_steps = (
            self.subagent_max_steps
            if self.subagent_max_steps is not None
            else layers.merged.get("subagent_max_steps")
        )
        if raw_subagent_steps is None:
            subagent_max_steps = DEFAULT_SUBAGENT_MAX_STEPS
        elif isinstance(raw_subagent_steps, (bool, float)):
            raise ValueError(
                f"Unsupported subagent_max_steps: {raw_subagent_steps!r}. Must be a positive integer."
            )
        else:
            try:
                subagent_max_steps = int(raw_subagent_steps)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Unsupported subagent_max_steps: {raw_subagent_steps!r}. Must be a positive integer."
                ) from exc
            if subagent_max_steps <= 0:
                raise ValueError(
                    f"Unsupported subagent_max_steps: {subagent_max_steps}. Must be a positive integer."
                )

        hook_specs = (
            parse_hook_specs(layers.global_config.get("hooks"), scope="global")
            + parse_hook_specs(layers.project_config.get("hooks"), scope="project")
            + parse_hook_specs(self.hooks, scope="explicit")
        )
        disabled_seen: list[str] = []
        for candidate in (
            _coerce_string_list(layers.global_config.get("disabled_hooks"), "disabled_hooks")
            + _coerce_string_list(layers.project_config.get("disabled_hooks"), "disabled_hooks")
            + _coerce_string_list(self.disabled_hooks, "disabled_hooks")
        ):
            if candidate not in disabled_seen:
                disabled_seen.append(candidate)

        raw_doom_threshold = (
            self.doom_loop_threshold
            if self.doom_loop_threshold is not None
            else layers.merged.get("doom_loop_threshold")
        )
        doom_loop_threshold = (
            DEFAULT_DOOM_LOOP_THRESHOLD
            if raw_doom_threshold is None
            else _coerce_non_negative_int(raw_doom_threshold, "doom_loop_threshold")
        )

        raw_doom_nudges = (
            self.doom_loop_max_nudges
            if self.doom_loop_max_nudges is not None
            else layers.merged.get("doom_loop_max_nudges")
        )
        doom_loop_max_nudges = (
            DEFAULT_DOOM_LOOP_MAX_NUDGES
            if raw_doom_nudges is None
            else _coerce_non_negative_int(raw_doom_nudges, "doom_loop_max_nudges")
        )

        return ResolvedConfig(
            sandbox=c.pick("sandbox", self.sandbox, _coerce_bool, False),
            docker=c.pick("docker", self.docker, _coerce_bool, False),
            docker_image=c.pick(
                "docker_image", self.docker_image, str, "python:3.11", require_truthy=True
            ),
            docker_network=docker_network,
            auto_discover_tools=c.pick(
                "auto_discover_tools", self.auto_discover_tools, _coerce_bool, True
            ),
            allow_network_tools=c.pick(
                "allow_network_tools", self.allow_network_tools, _coerce_bool, True
            ),
            disable_default_tools=c.pick(
                "disable_default_tools", self.disable_default_tools, _coerce_bool, False
            ),
            runs_dir=c.pick("runs_dir", self.runs_dir, Path, None, require_truthy=True),
            max_tool_tokens=tool_tok,
            max_context_tokens=context_tok,
            trigger_ratio=c.pick("trigger_ratio", self.trigger_ratio, float, 0.80),
            cache_ttl=normalize_cache_ttl(cache_ttl),
            reasoning_effort=normalize_reasoning_effort(reasoning),
            max_step_retries=c.pick("max_step_retries", self.max_step_retries, int, 3),
            step_retry_delay=c.pick("step_retry_delay", self.step_retry_delay, float, 5.0),
            bash_timeout=bash_timeout,
            subagent_max_steps=subagent_max_steps,
            hooks=tuple(hook_specs),
            disabled_hooks=tuple(disabled_seen),
            include_default_hooks=c.pick(
                "include_default_hooks", self.include_default_hooks, _coerce_bool, True
            ),
            doom_loop_threshold=doom_loop_threshold,
            doom_loop_max_nudges=doom_loop_max_nudges,
        )


@dataclass(frozen=True)
class ResolvedConfig:
    """Fully resolved, cast, and validated settings ready for the harness to use."""

    sandbox: bool
    docker: bool
    docker_image: str
    docker_network: str | None
    auto_discover_tools: bool
    allow_network_tools: bool
    disable_default_tools: bool
    runs_dir: Path | None
    max_tool_tokens: int
    max_context_tokens: int
    trigger_ratio: float
    cache_ttl: str | None
    reasoning_effort: str | None
    max_step_retries: int
    step_retry_delay: float
    bash_timeout: float
    subagent_max_steps: int | None
    hooks: tuple[dict[str, Any], ...]
    disabled_hooks: tuple[str, ...]
    include_default_hooks: bool
    doom_loop_threshold: int
    doom_loop_max_nudges: int


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
