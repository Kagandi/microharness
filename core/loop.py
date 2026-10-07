from __future__ import annotations

import time
from typing import Any

from core.memory import Memory
from core.model import Model, UsageReportingModel
from core.tools import ToolRegistry
from core.types import StepTrace, ToolResult


class AgentLoop:
    """The native tool-calling agent loop (~30 LOC).

    Agnostic to sandboxing, benchmarks, and evaluation.
    Drives the structured LLM generation -> tool dispatch -> observation cycle.
    """

    def __init__(self, model: Model, memory: Memory, tools: ToolRegistry):
        self.model = model
        self.memory = memory
        self.tools = tools

    def step(self, step_index: int) -> tuple[bool, StepTrace]:
        """Executes a single tool-calling turn.

        Returns:
            (is_finished, step_trace)
        """
        start_time = time.time()

        # 1. Compaction check before sending context to LLM
        compaction_applied = self.memory.compact_if_needed()

        # 2. Snapshot current prompt state
        prompt_snapshot = [msg.to_dict() for msg in self.memory.get_messages()]

        # 3. Model generation
        content, tool_calls = self.model.generate(
            messages=self.memory.get_messages(),
            tools=self.tools.to_openai_schemas(),
        )

        # 4. Record assistant thought / intent into memory
        self.memory.add_assistant(content=content, tool_calls=tool_calls)

        step_usage: dict[str, Any] = {}
        if isinstance(self.model, UsageReportingModel):
            step_usage = self.model.get_last_step_usage()
        elif hasattr(self.model, "last_usage"):
            step_usage = dict(getattr(self.model, "last_usage", {}) or {})

        # 5. Stop condition: If no tools called, the agent has finished reasoning
        if not tool_calls:
            duration = time.time() - start_time
            trace = StepTrace(
                step_index=step_index,
                prompt_messages=prompt_snapshot,
                model_response={"content": content, "tool_calls": []},
                tool_results=[],
                compaction_applied=compaction_applied,
                duration_s=duration,
                usage=step_usage,
            )
            return True, trace

        # 6. Execute actions and record observations
        tool_results: list[ToolResult] = []
        for call in tool_calls:
            result = self.tools.dispatch(call, step_index=step_index)
            if result.exit_code == 0 and self.tools.modifies_files(call.name):
                target_path = call.arguments.get("path")
                if target_path:
                    self.memory.record_modified_file(str(target_path))
            self.memory.add_tool_result(result)
            tool_results.append(result)

        duration = time.time() - start_time
        trace = StepTrace(
            step_index=step_index,
            prompt_messages=prompt_snapshot,
            model_response={
                "content": content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
                ],
            },
            tool_results=[tr.to_dict() for tr in tool_results],
            compaction_applied=compaction_applied,
            duration_s=duration,
            usage=step_usage,
        )

        return False, trace
