from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any]
    extra_content: dict[str, Any] | None = None


@dataclass
class Message:
    role: str  # "system" | "user" | "assistant" | "tool"
    content: str
    tool_call_id: str | None = None
    tool_calls: list[ToolCall] | None = None
    raw_log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"role": self.role, "content": self.content or ""}
        if self.tool_call_id:
            data["tool_call_id"] = self.tool_call_id
        if self.tool_calls:
            calls_list: list[dict[str, Any]] = []
            for tc in self.tool_calls:
                call_data: dict[str, Any] = {
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.name,
                        "arguments": (
                            json.dumps(tc.arguments)
                            if isinstance(tc.arguments, dict)
                            else str(tc.arguments or "{}")
                        ),
                    },
                }
                if tc.extra_content:
                    call_data["extra_content"] = tc.extra_content
                calls_list.append(call_data)
            data["tool_calls"] = calls_list
        return data


@dataclass
class ToolResult:
    call_id: str
    output: str
    exit_code: int = 0
    was_truncated: bool = False
    raw_log_path: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "call_id": self.call_id,
            "output": self.output,
            "exit_code": self.exit_code,
            "was_truncated": self.was_truncated,
            "raw_log_path": self.raw_log_path,
        }


@dataclass
class Task:
    id: str
    instruction: str
    system_prompt: str | None = None
    initial_files: dict[str, str] = field(default_factory=dict)
    max_steps: int = 15
    timeout_seconds: float = 60.0
    docker_image: str | None = None
    init_workdir: str | None = None


@dataclass
class StepTrace:
    step_index: int
    prompt_messages: list[dict[str, Any]]
    model_response: dict[str, Any]
    tool_results: list[dict[str, Any]]
    compaction_applied: bool = False
    duration_s: float = 0.0
    usage: dict[str, Any] = field(default_factory=dict)


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
    finish_reason: str  # "completed" | "max_steps_exceeded" | "timeout" | "error" | "interrupted"
    trace_path: str
    workspace_path: str
    eval_result: EvalResult | None = None
    patch_path: str | None = None
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0
    cache_hit_rate_pct: float = 0.0
