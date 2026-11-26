from __future__ import annotations

import fnmatch
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import shlex
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.defaults import get_default
from core.discovery import discover_resources
from core.environment import Environment, validate_read_only_git_args
from core.hooks import POST_TOOL, PRE_TOOL, HookContext, HookRegistry
from core.protocols import ModelCancelledError
from core.types import ToolCall, ToolResult


@runtime_checkable
class Tool(Protocol):
    """Protocol defining a pluggable tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    @property
    def execute(self) -> Callable[..., tuple[int, str]]:
        """Execute tool and return (exit_code, output)."""
        ...


DEFAULT_MAX_TOOL_TOKENS: int = int(get_default("max_tool_tokens"))
DEFAULT_MAX_GREP_MATCHES: int = int(get_default("max_grep_matches"))
DEFAULT_MAX_GREP_CONTEXT_LINES: int = int(get_default("max_grep_context_lines"))
DEFAULT_MAX_GLOB_RESULTS: int = int(get_default("max_glob_results"))

NETWORK_TOOL_NAMES: frozenset[str] = frozenset(
    {"web_fetch", "web_search", "fetch_url", "http_request"}
)


def is_network_tool(tool: Any) -> bool:
    """Checks whether a tool requires outbound network access."""
    if bool(getattr(tool, "requires_network", False)):
        return True
    return getattr(tool, "name", "") in NETWORK_TOOL_NAMES


_SAFE_FILENAME_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _sanitize_filename_token(value: str, fallback: str) -> str:
    """Normalizes an identifier into a safe single-segment filename token."""
    cleaned = _SAFE_FILENAME_RE.sub("_", value or "").strip("._")
    return cleaned or fallback


def _reserve_spill_log_path(
    logs_dir: Path,
    step_index: int,
    tool_name: str,
    call_id: str,
) -> Path:
    """Atomically reserves a unique spill log path inside ``logs_dir``."""
    logs_dir.mkdir(parents=True, exist_ok=True)
    safe_tool = _sanitize_filename_token(tool_name, "tool")
    safe_call = _sanitize_filename_token(call_id, "call")
    base_prefix = f"step_{step_index:02d}_{safe_tool}_{safe_call[:6]}"

    candidates: list[str] = [f"{base_prefix}.log"]
    if len(safe_call) > 6 or call_id != safe_call:
        digest = hashlib.sha1((call_id or "").encode("utf-8")).hexdigest()[:6]
        candidates.append(f"{base_prefix}_{digest}.log")

    for name in candidates:
        candidate = logs_dir / name
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            continue
        os.close(fd)
        return candidate

    counter = 1
    while True:
        candidate = logs_dir / f"{base_prefix}_{counter}.log"
        try:
            fd = os.open(candidate, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
        except FileExistsError:
            counter += 1
            continue
        os.close(fd)
        return candidate


def format_tool_output(
    output: str,
    call_id: str,
    tool_name: str,
    step_index: int,
    logs_dir: Path,
    max_tokens: int = DEFAULT_MAX_TOOL_TOKENS,
    display_logs_dir: Path | str | None = None,
) -> tuple[str, bool, str | None]:
    """Applies head + tail truncation and spills full raw output to disk if too long.

    Returns:
        tuple[formatted_output, was_truncated, raw_log_path]
    """
    max_chars = max_tokens * 4
    if len(output) <= max_chars:
        return output, False, None

    # Spill raw output to disk safely without colliding across parallel calls
    raw_log_path: str | None = None
    log_path: Path | None = None
    try:
        log_path = _reserve_spill_log_path(logs_dir, step_index, tool_name, call_id)
        log_path.write_text(output, encoding="utf-8")
        raw_log_path = (
            f"{str(display_logs_dir).rstrip('/')}/{log_path.name}"
            if display_logs_dir is not None
            else str(log_path)
        )
    except OSError:
        if log_path is not None:
            try:
                log_path.unlink(missing_ok=True)
            except OSError:
                pass
        raw_log_path = None

    # Compute head and tail bounds
    head_size = max_chars // 3
    tail_size = (max_chars * 2) // 3
    head = output[:head_size]
    tail = output[-tail_size:]
    truncated_count = len(output) - (head_size + tail_size)

    if raw_log_path:
        marker = (
            f"\n\n[... Truncated {truncated_count} characters ({output.count(chr(10))} total lines). "
            f"Full raw output saved to {raw_log_path} ...]\n"
            f"[Tip: You can use `grep` or `read_file` to inspect the full log file if needed]\n\n"
        )
    else:
        marker = f"\n\n[... Truncated {truncated_count} characters ({output.count(chr(10))} total lines) ...]\n\n"

    truncated_output = head + marker + tail
    return truncated_output, True, raw_log_path


DEFAULT_MAX_BASH_TIMEOUT = 1800.0


class BashTool:
    name = "bash"
    description = "Execute a shell command inside the workspace directory."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run (e.g. 'pytest', 'ls -la', 'python script.py').",
            },
            "timeout": {
                "type": "number",
                "description": (
                    "Optional timeout in seconds for this command (default: harness bash_timeout; "
                    "set higher for long test suites or builds, max 1800)."
                ),
            },
        },
        "required": ["command"],
    }

    def __init__(
        self,
        env: Environment,
        timeout: float = 30.0,
        max_timeout: float = DEFAULT_MAX_BASH_TIMEOUT,
    ):
        self.env = env
        self.timeout = timeout
        self.max_timeout = max(max_timeout, timeout)

    def execute(self, command: str, timeout: float | None = None) -> tuple[int, str]:
        """Executes ``command`` using either the tool default or a validated per-call timeout."""
        if timeout is None:
            effective_timeout = self.timeout
        elif (
            isinstance(timeout, bool)
            or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout)
            or timeout <= 0
        ):
            return 1, f"Error: invalid bash timeout {timeout!r}. Must be a positive number."
        else:
            effective_timeout = min(float(timeout), self.max_timeout)
        exit_code, output = self.env.execute_bash(command, timeout=effective_timeout)
        if (
            exit_code == 124
            and effective_timeout < self.max_timeout
            and "[Command timed out after " in output
        ):
            output = (
                f"{output}\n[Tip: Pass the 'timeout' tool parameter "
                f"(up to {int(self.max_timeout)}s) for long-running test suites or builds.]"
            )
        return exit_code, output


class ReadFileTool:
    name = "read_file"
    parallel_safe = True
    description = (
        "Read file content from the workspace with line numbers and optional line slicing."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file relative to workspace root.",
            },
            "start_line": {
                "type": "integer",
                "description": "1-indexed starting line number (default: 1).",
                "default": 1,
            },
            "end_line": {
                "type": "integer",
                "description": "1-indexed ending line number inclusive (-1 for end of file).",
                "default": -1,
            },
        },
        "required": ["path"],
    }

    def __init__(self, env: Environment):
        self.env = env

    def execute(self, path: str, start_line: int = 1, end_line: int = -1) -> tuple[int, str]:
        try:
            content = self.env.read_file(path, start_line=start_line, end_line=end_line)
            return 0, content
        except Exception as exc:
            return 1, f"Error reading file '{path}': {exc}"


class WriteFileTool:
    name = "write_file"
    modifies_files = True
    description = "Create or overwrite a file inside the workspace with given content."
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file relative to workspace root.",
            },
            "content": {
                "type": "string",
                "description": "The full text content to write to the file.",
            },
        },
        "required": ["path", "content"],
    }

    def __init__(self, env: Environment):
        self.env = env

    def execute(self, path: str, content: str) -> tuple[int, str]:
        try:
            self.env.write_file(path, content)
            return 0, f"Successfully wrote {len(content.encode('utf-8'))} bytes to '{path}'"
        except Exception as exc:
            return 1, f"Error writing file '{path}': {exc}"


class EditFileTool:
    name = "edit_file"
    modifies_files = True
    description = (
        "Atomically apply one or more sequential {old_str, new_str} edits to a file in one call. "
        "Combine all edits for the same file into `edits`, and batch independent `edit_file` calls "
        "across different files in the same turn."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file relative to workspace root.",
            },
            "edits": {
                "type": "array",
                "description": (
                    "List of sequential {old_str, new_str} replacements applied atomically "
                    "to this file in order."
                ),
                "items": {
                    "type": "object",
                    "properties": {
                        "old_str": {
                            "type": "string",
                            "description": "Exact unique string or code block to replace.",
                        },
                        "new_str": {
                            "type": "string",
                            "description": "Replacement string or code block.",
                        },
                    },
                    "required": ["old_str", "new_str"],
                },
            },
        },
        "required": ["path", "edits"],
    }

    def __init__(self, env: Environment):
        self.env = env

    def execute(
        self,
        path: str,
        old_str: str | None = None,
        new_str: str | None = None,
        edits: list[dict[str, Any]] | None = None,
    ) -> tuple[int, str]:
        try:
            if edits is not None:
                if old_str is not None or new_str is not None:
                    return (
                        1,
                        f"Error editing file '{path}': provide either (old_str, new_str) or edits, not both.",
                    )
                if not isinstance(edits, list) or not edits:
                    return (
                        1,
                        f"Error editing file '{path}': edits must be a non-empty list of {{old_str, new_str}} objects.",
                    )
                parsed_edits: list[tuple[str, str]] = []
                for idx, item in enumerate(edits, start=1):
                    if not isinstance(item, dict):
                        return 1, f"Error editing file '{path}': edit #{idx} must be an object."
                    item_old = item.get("old_str")
                    item_new = item.get("new_str")
                    if not isinstance(item_old, str) or not item_old:
                        return (
                            1,
                            f"Error editing file '{path}': edit #{idx} requires a non-empty string 'old_str'.",
                        )
                    if not isinstance(item_new, str):
                        return (
                            1,
                            f"Error editing file '{path}': edit #{idx} requires a string 'new_str'.",
                        )
                    parsed_edits.append((item_old, item_new))
                _, message = self.env.edit_file(path, edits=parsed_edits)
                return 0, message

            if not isinstance(old_str, str) or not isinstance(new_str, str):
                return (
                    1,
                    f"Error editing file '{path}': provide (old_str, new_str) or a non-empty edits list.",
                )
            _, message = self.env.edit_file(path, old_str, new_str)
            return 0, message
        except Exception as exc:
            return 1, f"Error editing file '{path}': {exc}"


class FinishTool:
    name = "finish"
    terminates_loop = True
    description = (
        "Signal that the entire task is complete and verified. "
        "Call this only when you are completely finished and ready to submit your final answer."
    )
    parameters = {
        "type": "object",
        "properties": {
            "summary": {
                "type": "string",
                "description": "Concise final summary of the changes made and verification results.",
            }
        },
        "required": ["summary"],
    }

    def execute(self, summary: str = "") -> tuple[int, str]:
        return 0, summary.strip() or "Task completed."


_SEARCH_SKIP_DIRS: frozenset[str] = frozenset(
    {
        ".git",
        "__pycache__",
        "node_modules",
        ".venv",
        ".pytest_cache",
        ".mypy_cache",
        ".ruff_cache",
        ".tox",
    }
)

_GIT_READ_ACTIONS: tuple[str, ...] = (
    "status",
    "diff",
    "log",
    "show",
    "blame",
    "ls-files",
    "branch",
    "rev-parse",
)


def _split_brace_alternatives(body: str) -> list[str]:
    """Splits comma-separated brace alternatives while respecting nested ``{...}``."""
    alts: list[str] = []
    depth = 0
    start = 0
    for idx, ch in enumerate(body):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth = max(0, depth - 1)
        elif ch == "," and depth == 0:
            alts.append(body[start:idx])
            start = idx + 1
    alts.append(body[start:])
    return alts


def _glob_fragment_to_regex(fragment: str) -> str:
    """Translates a POSIX glob fragment (including ``**`` and ``{a,b}``) to regex source."""
    parts: list[str] = []
    i, n = 0, len(fragment)
    while i < n:
        if fragment.startswith("**/", i):
            parts.append("(?:.+/)?")
            i += 3
        elif fragment.startswith("**", i):
            parts.append(".*")
            i += 2
        elif fragment[i] == "*":
            parts.append("[^/]*")
            i += 1
        elif fragment[i] == "?":
            parts.append("[^/]")
            i += 1
        elif fragment[i] == "[":
            end = fragment.find("]", i + 1)
            if end == -1:
                parts.append(re.escape("["))
                i += 1
            else:
                body = fragment[i + 1 : end]
                if body.startswith("!"):
                    body = "^" + body[1:]
                parts.append(f"[{body}]")
                i = end + 1
        elif fragment[i] == "{":
            depth = 1
            j = i + 1
            while j < n and depth > 0:
                if fragment[j] == "{":
                    depth += 1
                elif fragment[j] == "}":
                    depth -= 1
                j += 1
            if depth == 0:
                inner = fragment[i + 1 : j - 1]
                alts = [_glob_fragment_to_regex(alt) for alt in _split_brace_alternatives(inner)]
                parts.append(f"(?:{'|'.join(alts)})")
                i = j
            else:
                parts.append(re.escape("{"))
                i += 1
        else:
            parts.append(re.escape(fragment[i]))
            i += 1
    return "".join(parts)


def _compile_glob_pattern(pattern: str) -> re.Pattern[str]:
    """Compiles a POSIX-style glob pattern (supporting ``**`` and ``{a,b}``) into a regex."""
    normalized = pattern.strip().lstrip("./")
    if not normalized:
        return re.compile(r"^$")
    return re.compile(f"^{_glob_fragment_to_regex(normalized)}$")


def _iter_searchable_files(base: Path) -> list[Path]:
    """Collects sorted regular files under ``base``, skipping noisy VCS and cache directories."""
    files: list[Path] = []
    for root, dirnames, filenames in os.walk(base):
        dirnames[:] = sorted(d for d in dirnames if d not in _SEARCH_SKIP_DIRS)
        root_path = Path(root)
        for name in sorted(filenames):
            candidate = root_path / name
            if candidate.is_file():
                files.append(candidate)
    return files


def _display_path(target: Path, env: Environment, base: Path) -> str:
    """Formats a resolved path relative to the workspace when inside it, otherwise relative to ``base``."""
    if target.is_relative_to(env.workspace_dir):
        rel = target.relative_to(env.workspace_dir).as_posix()
        return rel or "."
    if target.is_relative_to(base):
        rel = target.relative_to(base).as_posix()
        return rel or target.name
    return str(target)


def _validate_positive_int(value: int | None, default: int, name: str) -> tuple[int | None, str]:
    """Validates an optional positive integer parameter."""
    if value is None:
        return default, ""
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        return None, f"Error: '{name}' must be a positive integer."
    return value, ""


class GlobTool:
    """Read-only file discovery tool matching glob patterns inside the workspace."""

    name = "glob"
    parallel_safe = True
    description = (
        "Find files in the workspace matching a glob pattern (e.g. '**/*.py' or 'tests/*.py')."
    )
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Glob pattern relative to `path` (supports `**`, `*`, `?`).",
            },
            "path": {
                "type": "string",
                "description": "Directory relative to workspace root to search (default: '.').",
                "default": ".",
            },
            "max_results": {
                "type": "integer",
                "description": f"Maximum matching file paths to return (default: {DEFAULT_MAX_GLOB_RESULTS}).",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, env: Environment):
        self.env = env

    def execute(
        self,
        pattern: str,
        path: str = ".",
        max_results: int | None = None,
    ) -> tuple[int, str]:
        if not isinstance(pattern, str) or not pattern.strip():
            return 1, "Error: 'pattern' must be a non-empty string."
        limit, limit_err = _validate_positive_int(
            max_results, DEFAULT_MAX_GLOB_RESULTS, "max_results"
        )
        if limit is None:
            return 1, limit_err

        try:
            base = self.env.resolve_read_path(path)
        except Exception as exc:
            return 1, f"Error resolving path '{path}': {exc}"
        if not base.exists():
            return 1, f"Error: Directory not found: '{path}'"
        if not base.is_dir():
            return 1, f"Error: Path is not a directory: '{path}'"

        regex = _compile_glob_pattern(pattern)
        matches: list[str] = []
        for file_path in _iter_searchable_files(base):
            rel = file_path.relative_to(base).as_posix()
            if regex.match(rel):
                matches.append(_display_path(file_path, self.env, base))

        if not matches:
            return 0, "No files matched."
        total = len(matches)
        if total > limit:
            shown = "\n".join(matches[:limit])
            return 0, f"{shown}\n\n[... showing first {limit} of {total} matched files]"
        return 0, "\n".join(matches)


class GrepTool:
    """Read-only content search tool using Python regex/literal matching."""

    name = "grep"
    parallel_safe = True
    description = "Search file contents in the workspace (or allowed log directories) for a regex or literal pattern."
    parameters = {
        "type": "object",
        "properties": {
            "pattern": {
                "type": "string",
                "description": "Regular expression (or literal string when `literal=true`) to search for.",
            },
            "path": {
                "type": "string",
                "description": "File or directory path relative to workspace root (default: '.').",
                "default": ".",
            },
            "include": {
                "type": "string",
                "description": "Optional filename/path glob filter (e.g. '*.py' or 'src/**/*.ts').",
            },
            "case_insensitive": {
                "type": "boolean",
                "description": "Perform case-insensitive matching (default: false).",
                "default": False,
            },
            "literal": {
                "type": "boolean",
                "description": "Treat `pattern` as a literal substring instead of a regex (default: false).",
                "default": False,
            },
            "context": {
                "type": "integer",
                "description": f"Context lines before and after each match (0..{DEFAULT_MAX_GREP_CONTEXT_LINES}, default: 0).",
                "default": 0,
            },
            "max_matches": {
                "type": "integer",
                "description": f"Maximum matching lines to return (default: {DEFAULT_MAX_GREP_MATCHES}).",
            },
        },
        "required": ["pattern"],
    }

    def __init__(self, env: Environment):
        self.env = env

    def execute(
        self,
        pattern: str,
        path: str = ".",
        include: str | None = None,
        case_insensitive: bool = False,
        literal: bool = False,
        context: int = 0,
        max_matches: int | None = None,
    ) -> tuple[int, str]:
        if not isinstance(pattern, str) or not pattern:
            return 1, "Error: 'pattern' must be a non-empty string."
        if isinstance(context, bool) or not isinstance(context, int) or context < 0:
            return 1, "Error: 'context' must be a non-negative integer."
        limit, limit_err = _validate_positive_int(
            max_matches, DEFAULT_MAX_GREP_MATCHES, "max_matches"
        )
        if limit is None:
            return 1, limit_err

        effective_context = min(context, DEFAULT_MAX_GREP_CONTEXT_LINES)
        flags = re.IGNORECASE if case_insensitive else 0
        raw_pattern = re.escape(pattern) if literal else pattern
        try:
            compiled = re.compile(raw_pattern, flags)
        except re.error as exc:
            return 1, f"Error: Invalid regex pattern {pattern!r}: {exc}"

        try:
            target = self.env.resolve_read_path(path)
        except Exception as exc:
            return 1, f"Error resolving path '{path}': {exc}"
        if not target.exists():
            return 1, f"Error: Path not found: '{path}'"

        include_re = (
            _compile_glob_pattern(include)
            if isinstance(include, str) and include.strip() and "/" in include
            else None
        )
        include_name = include.strip() if isinstance(include, str) and include.strip() else None

        if target.is_file():
            candidates = [target]
            base = target.parent
        else:
            candidates = []
            base = target
            for file_path in _iter_searchable_files(target):
                if include_re is not None:
                    if not include_re.match(file_path.relative_to(target).as_posix()):
                        continue
                elif include_name is not None and not fnmatch.fnmatch(file_path.name, include_name):
                    continue
                candidates.append(file_path)

        output_lines: list[str] = []
        total_matches = 0
        truncated = False

        for file_path in candidates:
            try:
                raw_bytes = file_path.read_bytes()
            except OSError:
                continue
            if b"\x00" in raw_bytes[:8192]:
                continue
            text = raw_bytes.decode("utf-8", errors="replace")
            lines = text.splitlines()
            matched_indices = [idx for idx, line in enumerate(lines) if compiled.search(line)]
            if not matched_indices:
                continue

            display = _display_path(file_path, self.env, base)
            selected: set[int] = set()
            matched_set: set[int] = set()
            for idx in matched_indices:
                if total_matches >= limit:
                    truncated = True
                    break
                total_matches += 1
                matched_set.add(idx)
                start = max(0, idx - effective_context)
                end = min(len(lines) - 1, idx + effective_context)
                selected.update(range(start, end + 1))

            prev_idx: int | None = None
            for idx in sorted(selected):
                if prev_idx is not None and idx > prev_idx + 1:
                    output_lines.append("--")
                sep = ":" if idx in matched_set else "-"
                output_lines.append(f"{display}{sep}{idx + 1}{sep}{lines[idx]}")
                prev_idx = idx

            if truncated:
                break

        if not output_lines:
            return 0, "No matches found."
        body = "\n".join(output_lines)
        if truncated:
            return 0, f"{body}\n\n[... reached max_matches={limit}; refine pattern or path]"
        return 0, body


class GitReadTool:
    """Read-only Git inspection tool backed by ``validate_read_only_git_args``."""

    name = "git_read"
    parallel_safe = True
    description = (
        "Run a read-only git inspection command (`status`, `diff`, `log`, `show`, "
        "`blame`, `ls-files`, `branch`, `rev-parse`)."
    )
    parameters = {
        "type": "object",
        "properties": {
            "action": {
                "type": "string",
                "enum": list(_GIT_READ_ACTIONS),
                "description": "Read-only git subcommand to execute.",
            },
            "args": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Optional flags/refs/paths for the git subcommand (e.g. ['--stat'] or ['-n', '5', '--oneline']).",
            },
        },
        "required": ["action"],
    }

    def __init__(self, env: Environment):
        self.env = env

    def execute(self, action: str, args: list[str] | None = None) -> tuple[int, str]:
        if not isinstance(action, str) or not action.strip():
            return 1, "Error: 'action' must be a non-empty string."
        clean_action = action.strip()
        if clean_action not in _GIT_READ_ACTIONS:
            allowed = ", ".join(_GIT_READ_ACTIONS)
            return 1, f"Error: Unsupported git_read action '{clean_action}'. Allowed: {allowed}"

        if args is None:
            clean_args: list[str] = []
        elif isinstance(args, list) and all(isinstance(item, str) for item in args):
            clean_args = list(args)
        else:
            return 1, "Error: 'args' must be a list of strings."

        ok, reason = validate_read_only_git_args([clean_action, *clean_args])
        if not ok:
            return 1, f"Error: {reason}"

        cmd = " ".join(shlex.quote(tok) for tok in ["git", clean_action, *clean_args])
        return self.env.execute_bash(cmd)


class ToolRegistry:
    """Manages available tools, formats schemas, and dispatches calls."""

    def __init__(
        self,
        env: Environment,
        max_tool_tokens: int = DEFAULT_MAX_TOOL_TOKENS,
        hooks: HookRegistry | None = None,
    ):
        self.env = env
        self.max_tool_tokens = max_tool_tokens
        self.hooks = hooks if hooks is not None else HookRegistry.with_defaults()
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get(self, tool_name: str) -> Tool | None:
        return self._tools.get(tool_name)

    def registered_tools(self) -> dict[str, Tool]:
        """Returns a shallow mapping of registered tool names to tool instances."""
        return dict(self._tools)

    def is_parallel_safe(self, tool_name: str) -> bool:
        """Checks if a registered tool can be dispatched concurrently across multiple calls in one step."""
        tool = self._tools.get(tool_name)
        return bool(getattr(tool, "parallel_safe", False))

    def modifies_files(self, tool_name: str) -> bool:
        """Checks if a registered tool is marked as modifying workspace files."""
        tool = self._tools.get(tool_name)
        return bool(getattr(tool, "modifies_files", False))

    def terminates_loop(self, tool_name: str) -> bool:
        """Checks if a registered tool signals end-of-task when called successfully."""
        tool = self._tools.get(tool_name)
        return bool(getattr(tool, "terminates_loop", False))

    def requires_network(self, tool_name: str) -> bool:
        """Checks if a registered tool requires outbound network access."""
        tool = self._tools.get(tool_name)
        return is_network_tool(tool) if tool is not None else False

    def to_system_prompt_snippet(self) -> str:
        """Generates dynamic documentation for all registered tools."""
        if not self._tools:
            return ""
        lines = ["You have access to tools to interact with the environment:"]
        for i, tool in enumerate(self._tools.values(), 1):
            props = tool.parameters.get("properties", {})
            params = [f"{p_name}: {p_val.get('type', 'any')}" for p_name, p_val in props.items()]
            params_str = ", ".join(params)
            lines.append(f"{i}. {tool.name}({params_str}): {tool.description}")
        return "\n".join(lines)

    def to_openai_schemas(self) -> list[dict[str, Any]]:
        return [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                },
            }
            for tool in self._tools.values()
        ]

    def dispatch(self, tool_call: ToolCall, step_index: int) -> ToolResult:
        tool = self._tools.get(tool_call.name)
        if not tool:
            return ToolResult(
                call_id=tool_call.id,
                output=f"Error: Unknown tool '{tool_call.name}'. Available: {list(self._tools.keys())}",
                exit_code=1,
            )

        if not isinstance(tool_call.arguments, dict):
            return ToolResult(
                call_id=tool_call.id,
                output=(
                    f"Error: Tool arguments for '{tool_call.name}' must be a dictionary/object, "
                    f"got: {type(tool_call.arguments).__name__}"
                ),
                exit_code=1,
            )

        if "_raw_arguments" in tool_call.arguments:
            return ToolResult(
                call_id=tool_call.id,
                output=_format_malformed_json_error(
                    tool_call.name, tool_call.arguments["_raw_arguments"]
                ),
                exit_code=1,
            )

        raw_path = tool_call.arguments.get("path")
        target_path = str(raw_path) if raw_path is not None else None
        modifies = self.modifies_files(tool_call.name)

        pre_outcome = self.hooks.run(
            PRE_TOOL,
            HookContext(
                event=PRE_TOOL,
                env=self.env,
                tool_name=tool_call.name,
                tool_call=tool_call,
                tool_args=dict(tool_call.arguments),
                step_index=step_index,
                modifies_files=modifies,
                path=target_path,
            ),
        )
        if pre_outcome.blocked:
            return ToolResult(
                call_id=tool_call.id,
                output=pre_outcome.combined_message or f"Tool '{tool_call.name}' blocked by hook.",
                exit_code=1,
            )

        if tool_call.name not in ("write_file", "edit_file") and modifies:
            confirm_fn = getattr(self.env, "confirm_action", None)
            if callable(confirm_fn):
                res = confirm_fn(tool_call.name, str(tool_call.arguments))
                if isinstance(res, tuple) and len(res) == 2:
                    allowed, reason = res
                    if not allowed:
                        return ToolResult(
                            call_id=tool_call.id,
                            output=f"Error: {reason}",
                            exit_code=1,
                        )

        try:
            exit_code, raw_output = tool.execute(**tool_call.arguments)
        except ModelCancelledError:
            raise
        except Exception as exc:
            exit_code = 1
            raw_output = f"Error executing tool '{tool_call.name}': {exc}"

        raw_output_str = str(raw_output) if raw_output is not None else ""

        post_outcome = self.hooks.run(
            POST_TOOL,
            HookContext(
                event=POST_TOOL,
                env=self.env,
                tool_name=tool_call.name,
                tool_call=tool_call,
                tool_args=dict(tool_call.arguments),
                exit_code=exit_code,
                output=raw_output_str,
                step_index=step_index,
                modifies_files=modifies,
                path=target_path,
            ),
        )
        if post_outcome.messages:
            hook_text = post_outcome.combined_message
            raw_output_str = f"{raw_output_str}\n\n{hook_text}" if raw_output_str else hook_text

        try:
            raw_agent_logs_dir = getattr(self.env, "agent_logs_dir", None)
            display_logs_dir = (
                raw_agent_logs_dir if isinstance(raw_agent_logs_dir, (str, Path)) else None
            )
            formatted_output, was_truncated, raw_log_path = format_tool_output(
                output=raw_output_str,
                call_id=tool_call.id,
                tool_name=tool_call.name,
                step_index=step_index,
                logs_dir=self.env.logs_dir,
                max_tokens=self.max_tool_tokens,
                display_logs_dir=display_logs_dir,
            )
        except Exception:
            max_chars = self.max_tool_tokens * 4
            formatted_output = raw_output_str[:max_chars]
            was_truncated = len(raw_output_str) > max_chars
            raw_log_path = None

        return ToolResult(
            call_id=tool_call.id,
            output=formatted_output,
            exit_code=exit_code,
            was_truncated=was_truncated,
            raw_log_path=raw_log_path,
        )


def _format_malformed_json_error(tool_name: str, raw_args: Any) -> str:
    """Formats an actionable JSON parse error when model tool-call arguments failed to decode."""
    parse_detail = ""
    if isinstance(raw_args, str):
        try:
            json.loads(raw_args)
        except json.JSONDecodeError as exc:
            parse_detail = f" ({exc})"
    preview = repr(raw_args)
    if len(preview) > 200:
        preview = f"{preview[:197]}..."
    return (
        f"Error: Invalid JSON arguments for tool '{tool_name}'{parse_detail}. "
        "Arguments must be a valid JSON object matching the tool schema; "
        f"received: {preview}"
    )


def _is_tool_like(obj: Any) -> bool:
    """Checks if an instantiated object conforms to the Tool protocol."""
    if not isinstance(obj, Tool):
        return False
    return (
        isinstance(obj.name, str)
        and bool(obj.name)
        and isinstance(obj.description, str)
        and isinstance(obj.parameters, dict)
        and callable(obj.execute)
    )


def _as_tools(val: Any) -> list[Tool]:
    """Flattens and validates tool instances conforming to the Tool protocol."""
    candidates = val if isinstance(val, (list, tuple)) else [val]
    return [t for t in candidates if _is_tool_like(t)]


def _accepts_env(target: Any) -> bool:
    """Whether a plugin hook or class can be handed the environment.

    Read from the signature rather than discovered by catching ``TypeError``: a
    ``TypeError`` raised inside a plugin's own ``__init__`` would otherwise be
    misread as "wrong signature" and silently retried with fewer arguments.
    """
    try:
        parameters = inspect.signature(target).parameters
    except (TypeError, ValueError):
        return False
    env_param = parameters.get("env")
    if env_param is not None and env_param.kind is not inspect.Parameter.POSITIONAL_ONLY:
        return True
    return any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())


def _call_plugin_hook(target: Any, env: Environment | None) -> Any:
    """Invokes a plugin hook, passing the environment only when it accepts one."""
    if env is not None and _accepts_env(target):
        return target(env=env)
    return target()


def _tools_from_classes(module: Any, env: Environment | None) -> list[Tool]:
    """Instantiates every Tool-protocol class the plugin module defines."""
    tools: list[Tool] = []
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ != module.__name__ or not hasattr(cls, "execute"):
            continue
        try:
            instance = _call_plugin_hook(cls, env)
        except Exception as exc:
            print(
                f"⚠️ Warning: tool plugin class '{cls.__name__}' failed to construct: {exc}",
                file=sys.stderr,
            )
            continue
        if _is_tool_like(instance):
            tools.append(instance)
    return tools


def load_tool_plugin(plugin_path: Path | str, env: Environment | None = None) -> list[Tool]:
    """Dynamically loads tools from a standalone Python plugin file.

    Supported plugin conventions, in the order they are tried:
    1. Factory function: ``get_tools(env)`` or ``get_tools()``
    2. Module attribute: ``tools`` or ``TOOLS = [MyTool()]``
    3. Direct classes: any class defined in the file conforming to Tool protocol
    """
    path = Path(plugin_path).resolve()
    if not path.is_file() or path.suffix != ".py":
        return []

    module = _import_plugin_module(path)
    if module is None:
        return []

    factory = getattr(module, "get_tools", None)
    if callable(factory):
        try:
            return _as_tools(_call_plugin_hook(factory, env))
        except Exception as exc:
            print(f"⚠️ Warning: get_tools() in '{path}' failed: {exc}", file=sys.stderr)
            return []

    for attr in ("tools", "TOOLS"):
        if tools := _as_tools(getattr(module, attr, None)):
            return tools

    return _tools_from_classes(module, env)


def _import_plugin_module(path: Path) -> Any | None:
    """Executes a plugin file as a module, or reports why it could not be loaded."""
    try:
        spec = importlib.util.spec_from_file_location(
            f"_mh_tool_{path.stem}_{abs(hash(str(path)))}", path
        )
        if not spec or not spec.loader:
            return None
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module
    except Exception as exc:
        print(f"⚠️ Warning: Failed to load tool plugin '{path}': {exc}", file=sys.stderr)
        return None


def discover_tool_plugins(
    workspace_dir: Path | str | None = None,
    global_dir: Path | str | None = None,
    env: Environment | None = None,
    trust_project_tools: bool = False,
) -> list[Tool]:
    """Discovers and loads tool plugins from global and project directories.

    Locations searched:
    1. Global: `~/.microharness/tools/*.py` (or `global_dir / "tools"`)
    2. Project: `<workspace>/.microharness/tools/*.py`

    Project tools require `trust_project_tools=True` before module execution to
    prevent untrusted host code execution from cloned repositories.
    Project-level tools cleanly shadow global tools sharing the same tool name.
    """

    def _loader(path: Path, scope: str):
        if scope == "project" and not trust_project_tools:
            print(
                f"⚠️ Warning: Skipping untrusted project tool plugin '{path}'. "
                "Pass --trust-project-tools to enable.",
                file=sys.stderr,
            )
            return
        for tool in load_tool_plugin(path, env=env):
            yield tool.name, tool

    resources = discover_resources(
        resource_name="tools",
        loader=_loader,
        workspace_dir=workspace_dir,
        global_dir=global_dir,
        file_pattern="*.py",
    )
    return [r.item for r in resources.values()]
