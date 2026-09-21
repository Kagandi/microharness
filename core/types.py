from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: Optional[str] = None
    tool_calls: Optional[list[ToolCall]] = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content}
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            data["tool_calls"] = [
                {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": tc.arguments if isinstance(tc.arguments, str) else None,
                    },
                }
                for tc in self.tool_calls
            ]
        return data


@dataclass
class ToolResult:
    call_id: str
    output: str
    exit_code: int = 0
    was_truncated: bool = False
    raw_log_path: Optional[str] = None


@dataclass
class Task:
    id: str
    instruction: str
    system_prompt: Optional[str] = None
    initial_files: dict[str, str] = field(default_factory=dict)
    max_steps: int = 15
    timeout_seconds: float = 60.0


@dataclass
class StepTrace:
    step_index: int
    prompt_messages: list[dict[str, Any]]
    model_response: dict[str, Any]
    tool_results: list[dict[str, Any]]
    compaction_applied: bool = False
    duration_s: float = 0.0


@dataclass
class EvalResult:
    passed: bool
    score: float
    feedback: str


@dataclass
class RunResult:
    task_id: str
    passed: bool
    score: float
    total_steps: int
    duration_s: float
    finish_reason: str  # "completed" | "max_steps_exceeded" | "timeout" | "error"
    trace_path: str
    workspace_path: str
    eval_result: Optional[EvalResult] = None
