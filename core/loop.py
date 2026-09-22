from __future__ import annotations

import time

from core.memory import Memory
from core.model import Model
from core.tools import ToolRegistry
from core.types import StepTrace, ToolResult


class AgentLoop:
    """The pure ReAct execution engine (~30 LOC).

    Agnostic to sandboxing, benchmarks, and evaluation.
    Only manages the Thought -> Action -> Observation cycle.
    """

    def __init__(self, model: Model, memory: Memory, tools: ToolRegistry):
        self.model = model
        self.memory = memory
        self.tools = tools

    def step(self, step_index: int) -> tuple[bool, StepTrace]:
        """Executes a single ReAct turn.

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
            )
            return True, trace

        # 6. Execute actions and record observations
        tool_results: list[ToolResult] = []
        for call in tool_calls:
            result = self.tools.dispatch(call, step_index=step_index)
            self.memory.add_tool_result(result)
            tool_results.append(result)

        duration = time.time() - start_time
        trace = StepTrace(
            step_index=step_index,
            prompt_messages=prompt_snapshot,
            model_response={
                "content": content,
                "tool_calls": [{"name": tc.name, "arguments": tc.arguments} for tc in tool_calls],
            },
            tool_results=[tr.to_dict() for tr in tool_results],
            compaction_applied=compaction_applied,
            duration_s=duration,
        )

        return False, trace
