from __future__ import annotations

from pathlib import Path
from typing import Any, Protocol

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


def format_tool_output(
    output: str,
    call_id: str,
    tool_name: str,
    step_index: int,
    logs_dir: Path,
    max_chars: int = 3000,
) -> tuple[str, bool, str | None]:
    """Applies head + tail truncation and spills full raw output to disk if too long.

    Returns:
        tuple[formatted_output, was_truncated, raw_log_path]
    """
    if len(output) <= max_chars:
        return output, False, None

    # Spill raw output to disk
    log_filename = f"step_{step_index:02d}_{tool_name}_{call_id[:6]}.log"
    log_path = logs_dir / log_filename
    log_path.write_text(output, encoding="utf-8")

    # Compute head and tail bounds
    head_size = max_chars // 3
    tail_size = (max_chars * 2) // 3
    head = output[:head_size]
    tail = output[-tail_size:]
    truncated_count = len(output) - (head_size + tail_size)

    marker = (
        f"\n\n[... Truncated {truncated_count} characters ({output.count(chr(10))} total lines). "
        f"Full raw output saved to {log_filename} ...]\n"
        f"[Tip: You can use `grep` or `read_file` to inspect the full log file if needed]\n\n"
    )

    truncated_output = head + marker + tail
    return truncated_output, True, str(log_path)


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

    def __init__(self, env: Environment, max_tool_chars: int = 3000):
        self.env = env
        self.max_tool_chars = max_tool_chars
        self._tools: dict[str, Tool] = {}

    def register(self, tool: Tool) -> None:
        self._tools[tool.name] = tool

    def get_tool(self, name: str) -> Tool | None:
        return self._tools.get(name)

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

        try:
            exit_code, raw_output = tool.execute(**tool_call.arguments)
        except Exception as exc:
            exit_code = 1
            raw_output = f"Error executing tool '{tool_call.name}': {exc}"

        formatted_output, was_truncated, raw_log_path = format_tool_output(
            output=raw_output,
            call_id=tool_call.id,
            tool_name=tool_call.name,
            step_index=step_index,
            logs_dir=self.env.logs_dir,
            max_chars=self.max_tool_chars,
        )

        return ToolResult(
            call_id=tool_call.id,
            output=formatted_output,
            exit_code=exit_code,
            was_truncated=was_truncated,
            raw_log_path=raw_log_path,
        )
