from __future__ import annotations

import importlib.util
import inspect
import sys
from pathlib import Path
from typing import Any, Protocol

from core.discovery import discover_resources
from core.environment import Environment
from core.types import ToolCall, ToolResult


class Tool(Protocol):
    """Protocol defining a pluggable tool."""

    name: str
    description: str
    parameters: dict[str, Any]

    def execute(self, **kwargs: Any) -> tuple[int, str]:
        """Execute tool and return (exit_code, output)."""
        ...


DEFAULT_MAX_TOOL_CHARS = 2000


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
    return (
        hasattr(obj, "name")
        and isinstance(getattr(obj, "name"), str)
        and bool(getattr(obj, "name"))
        and hasattr(obj, "description")
        and isinstance(getattr(obj, "description"), str)
        and hasattr(obj, "parameters")
        and isinstance(getattr(obj, "parameters"), dict)
        and hasattr(obj, "execute")
        and callable(getattr(obj, "execute"))
    )


def _is_tool_class(cls: type) -> bool:
    """Checks if a class looks like a Tool implementation."""
    return isinstance(cls, type) and hasattr(cls, "execute") and callable(getattr(cls, "execute"))


def load_tool_plugin(plugin_path: Path | str, env: Environment | None = None) -> list[Tool]:
    """Dynamically loads tools from a standalone Python plugin file.

    Supported plugin conventions:
    1. Factory function: `get_tools(env)` or `get_tools()` returning list[Tool] or single Tool.
    2. Module attribute: `tools` or `TOOLS` containing list[Tool] or Tool.
    3. Tool classes: Any class defined in the module conforming to the Tool protocol.
       If __init__ accepts `env`, it is instantiated with `env`, otherwise with zero arguments.
    """
    path = Path(plugin_path).resolve()
    if not path.is_file() or path.suffix != ".py":
        return []

    module_name = f"_mh_tool_plugin_{path.stem}_{abs(hash(str(path)))}"
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        return []

    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        print(f"⚠️ Warning: Failed to load tool plugin '{path}': {exc}", file=sys.stderr)
        return []

    tools: list[Tool] = []

    # 1. Factory function: get_tools() or get_tools(env)
    if hasattr(module, "get_tools") and callable(module.get_tools):
        try:
            sig = inspect.signature(module.get_tools)
            if len(sig.parameters) >= 1 and env is not None:
                res = module.get_tools(env)
            else:
                res = module.get_tools()
            if isinstance(res, (list, tuple)):
                tools.extend([t for t in res if _is_tool_like(t)])
            elif _is_tool_like(res):
                tools.append(res)
            return tools
        except Exception as exc:
            print(f"⚠️ Warning: Error calling get_tools() in '{path}': {exc}", file=sys.stderr)
            return []

    # 2. Module attributes: tools or TOOLS
    for attr_name in ("tools", "TOOLS"):
        if hasattr(module, attr_name):
            val = getattr(module, attr_name)
            if isinstance(val, (list, tuple)):
                tools.extend([t for t in val if _is_tool_like(t)])
            elif _is_tool_like(val):
                tools.append(val)
            if tools:
                return tools

    # 3. Tool classes defined in this module
    for _, cls in inspect.getmembers(module, inspect.isclass):
        if cls.__module__ == module_name and _is_tool_class(cls):
            instance = None
            try:
                sig = inspect.signature(cls)
                if "env" in sig.parameters and env is not None:
                    try:
                        instance = cls(env=env)
                    except TypeError:
                        instance = cls(env)
                else:
                    instance = cls()
            except TypeError:
                if env is not None:
                    try:
                        instance = cls(env=env)
                    except TypeError:
                        try:
                            instance = cls(env)
                        except Exception:
                            instance = None
                else:
                    instance = None
            except Exception:
                instance = None

            if instance is not None and _is_tool_like(instance):
                tools.append(instance)

    return tools


def discover_tool_plugins(
    workspace_dir: Path | str | None = None,
    global_dir: Path | str | None = None,
    env: Environment | None = None,
) -> list[Tool]:
    """Discovers and loads tool plugins from global and project directories.

    Locations searched:
    1. Global: `~/.microharness/tools/*.py` (or `global_dir / "tools"`)
    2. Project: `<workspace>/.microharness/tools/*.py`, `<workspace>/tools/*.py`

    Project-level tools cleanly shadow global tools sharing the same tool name.
    """
    def _loader(path: Path, scope: str):
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


