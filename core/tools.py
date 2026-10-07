from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.discovery import discover_resources
from core.environment import Environment
from core.types import ToolCall, ToolResult


@runtime_checkable
class Tool(Protocol):
    """Protocol defining a pluggable tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    def execute(self, **kwargs: Any) -> tuple[int, str]:
        """Execute tool and return (exit_code, output)."""
        ...


DEFAULT_MAX_TOOL_TOKENS = 8000
DEFAULT_MAX_TOOL_CHARS = DEFAULT_MAX_TOOL_TOKENS * 4  # 32,000 chars (~8,000 tokens)


def format_tool_output(
    output: str,
    call_id: str,
    tool_name: str,
    step_index: int,
    logs_dir: Path,
    max_chars: int = DEFAULT_MAX_TOOL_CHARS,
) -> tuple[str, bool, str | None]:
    """Applies head + tail truncation and spills full raw output to disk if too long.

    Returns:
        tuple[formatted_output, was_truncated, raw_log_path]
    """
    if len(output) <= max_chars:
        return output, False, None

    # Spill raw output to disk safely
    raw_log_path: str | None = None
    log_filename = f"step_{step_index:02d}_{tool_name}_{call_id[:6]}.log"
    log_path = logs_dir / log_filename
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        log_path.write_text(output, encoding="utf-8")
        raw_log_path = str(log_path)
    except OSError:
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
            f"Full raw output saved to {log_filename} ...]\n"
            f"[Tip: You can use `grep` or `read_file` to inspect the full log file if needed]\n\n"
        )
    else:
        marker = f"\n\n[... Truncated {truncated_count} characters ({output.count(chr(10))} total lines) ...]\n\n"

    truncated_output = head + marker + tail
    return truncated_output, True, raw_log_path


class BashTool:
    name = "bash"
    description = "Execute a shell command inside the workspace directory."
    parameters = {
        "type": "object",
        "properties": {
            "command": {
                "type": "string",
                "description": "The shell command to run (e.g. 'pytest', 'ls -la', 'python script.py').",
            }
        },
        "required": ["command"],
    }

    def __init__(self, env: Environment, timeout: float = 30.0):
        self.env = env
        self.timeout = timeout

    def execute(self, command: str) -> tuple[int, str]:
        return self.env.execute_bash(command, timeout=self.timeout)


class ReadFileTool:
    name = "read_file"
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
        "Replaces an exact, unique occurrence of old_str with new_str in a file. "
        "The old_str must match exactly (including whitespace/indentation) and be unique."
    )
    parameters = {
        "type": "object",
        "properties": {
            "path": {
                "type": "string",
                "description": "Path to the file relative to workspace root.",
            },
            "old_str": {
                "type": "string",
                "description": "The exact unique string or code block to replace.",
            },
            "new_str": {
                "type": "string",
                "description": "The replacement string or code block.",
            },
        },
        "required": ["path", "old_str", "new_str"],
    }

    def __init__(self, env: Environment):
        self.env = env

    def execute(self, path: str, old_str: str, new_str: str) -> tuple[int, str]:
        try:
            _, message = self.env.edit_file(path, old_str, new_str)
            return 0, message
        except Exception as exc:
            return 1, f"Error editing file '{path}': {exc}"


class ToolRegistry:
    """Manages available tools, formats schemas, and dispatches calls."""

    def __init__(self, env: Environment, max_tool_chars: int = DEFAULT_MAX_TOOL_CHARS):
        self.env = env
        self.max_tool_chars = max_tool_chars
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def __contains__(self, tool_name: str) -> bool:
        return tool_name in self._tools

    def get(self, tool_name: str) -> Tool | None:
        return self._tools.get(tool_name)

    def modifies_files(self, tool_name: str) -> bool:
        """Checks if a registered tool is marked as modifying workspace files."""
        tool = self._tools.get(tool_name)
        return bool(getattr(tool, "modifies_files", False))

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

        try:
            exit_code, raw_output = tool.execute(**tool_call.arguments)
        except Exception as exc:
            exit_code = 1
            raw_output = f"Error executing tool '{tool_call.name}': {exc}"

        raw_output_str = str(raw_output) if raw_output is not None else ""

        try:
            formatted_output, was_truncated, raw_log_path = format_tool_output(
                output=raw_output_str,
                call_id=tool_call.id,
                tool_name=tool_call.name,
                step_index=step_index,
                logs_dir=self.env.logs_dir,
                max_chars=self.max_tool_chars,
            )
        except Exception:
            formatted_output = raw_output_str[: self.max_tool_chars]
            was_truncated = len(raw_output_str) > self.max_tool_chars
            raw_log_path = None

        return ToolResult(
            call_id=tool_call.id,
            output=formatted_output,
            exit_code=exit_code,
            was_truncated=was_truncated,
            raw_log_path=raw_log_path,
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


