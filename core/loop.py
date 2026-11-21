from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any

from core.hooks import POST_STEP, PRE_STEP, HookContext, HookOutcome, HookRegistry
from core.memory import Memory
from core.model import AssistantMetadataModel, Model, UsageReportingModel
from core.tools import ToolRegistry
from core.types import HOOK_BLOCKED, StepTrace, ToolResult
from core.usage import normalize_usage


class AgentLoop:
    """The native tool-calling agent loop.

    Agnostic to sandboxing, benchmarks, and evaluation.
    Drives the structured LLM generation -> tool dispatch -> observation cycle.
    """

    def __init__(
        self,
        model: Model,
        memory: Memory,
        tools: ToolRegistry,
        *,
        hooks: HookRegistry | None = None,
    ):
        self.model = model
        self.memory = memory
        self.tools = tools
        resolved_hooks = hooks if hooks is not None else getattr(tools, "hooks", None)
        self.hooks: HookRegistry | None = (
            resolved_hooks if isinstance(resolved_hooks, HookRegistry) else None
        )
        self._hook_state: dict[str, Any] = {}
        self.last_post_step_outcome: HookOutcome | None = None

    def step(self, step_index: int) -> tuple[bool, StepTrace]:
        """Executes a single tool-calling turn.

        Returns:
            (is_finished, step_trace)
        """
        start_time = time.time()
        self.last_post_step_outcome = None
        env = getattr(self.tools, "env", None)

        if self.hooks is not None:
            pre_step = self.hooks.run(
                PRE_STEP,
                HookContext(
                    event=PRE_STEP,
                    env=env,
                    memory=self.memory,
                    step_index=step_index,
                    state=self._hook_state,
                ),
            )
            if pre_step.blocked:
                trace = StepTrace(
                    step_index=step_index,
                    prompt_messages=[msg.to_dict() for msg in self.memory.get_messages()],
                    model_response={"content": pre_step.combined_message, "tool_calls": []},
                    tool_results=[],
                    compaction_applied=False,
                    duration_s=time.time() - start_time,
                    stop_reason=pre_step.stop_reason or HOOK_BLOCKED,
                    stop_detail=pre_step.combined_message or "Step blocked by pre_step hook.",
                )
                return True, trace

        # 1. Compaction check before sending context to LLM
        compaction_applied = self.memory.compact_if_needed()

        # 2. Snapshot current prompt state & raw token estimate for calibration
        prompt_messages = self.memory.get_messages()
        prompt_raw_tokens = self.memory.raw_estimated_tokens()
        prompt_snapshot = [msg.to_dict() for msg in prompt_messages]

        # 3. Model generation
        content, tool_calls = self.model.generate(
            messages=prompt_messages,
            tools=self.tools.to_openai_schemas(),
        )
        assistant_fields = (
            self.model.get_last_assistant_fields()
            if isinstance(self.model, AssistantMetadataModel)
            else {}
        )

        # 4. Record assistant thought / intent into memory
        self.memory.add_assistant(
            content=content,
            tool_calls=tool_calls,
            extra_fields=assistant_fields or None,
        )

        step_usage: dict[str, Any] = {}
        if isinstance(self.model, UsageReportingModel):
            step_usage = self.model.get_last_step_usage()
        elif hasattr(self.model, "last_usage"):
            step_usage = dict(getattr(self.model, "last_usage", {}) or {})

        actual_prompt_tokens = normalize_usage(step_usage).prompt_tokens
        if actual_prompt_tokens > 0:
            self.memory.calibrate_tokens(
                actual_prompt_tokens,
                baseline_raw_tokens=prompt_raw_tokens,
            )

        # 5. Stop condition: If no tools called, the agent has finished reasoning
        if not tool_calls:
            if self.hooks is not None:
                self.last_post_step_outcome = self.hooks.run(
                    POST_STEP,
                    HookContext(
                        event=POST_STEP,
                        env=env,
                        memory=self.memory,
                        step_index=step_index,
                        state=self._hook_state,
                        metadata={"terminated": True},
                    ),
                )
            trace = StepTrace(
                step_index=step_index,
                prompt_messages=prompt_snapshot,
                model_response={"content": content, "tool_calls": [], **assistant_fields},
                tool_results=[],
                compaction_applied=compaction_applied,
                duration_s=time.time() - start_time,
                usage=step_usage,
            )
            return True, trace

        # 6. Execute actions and record observations
        tool_results: list[ToolResult] = []
        terminated = False
        stop_reason: str | None = None
        stop_detail: str | None = None

        if len(tool_calls) > 1 and all(self.tools.is_parallel_safe(c.name) for c in tool_calls):
            with ThreadPoolExecutor(max_workers=min(len(tool_calls), 4)) as pool:
                dispatched = list(
                    pool.map(lambda c: self.tools.dispatch(c, step_index=step_index), tool_calls)
                )
        else:
            dispatched = [self.tools.dispatch(call, step_index=step_index) for call in tool_calls]

        for call, result in zip(tool_calls, dispatched):
            if result.exit_code == 0 and self.tools.modifies_files(call.name):
                target_path = call.arguments.get("path")
                if target_path:
                    self.memory.record_modified_file(str(target_path))
            if result.exit_code == 0 and self.tools.terminates_loop(call.name):
                terminated = True
            self.memory.add_tool_result(result)
            tool_results.append(result)

        if not terminated and self.hooks is not None:
            post_outcome = self.hooks.run(
                POST_STEP,
                HookContext(
                    event=POST_STEP,
                    env=env,
                    memory=self.memory,
                    step_index=step_index,
                    tool_calls=tool_calls,
                    tool_results=tool_results,
                    modifies_files_fn=self.tools.modifies_files,
                    state=self._hook_state,
                ),
            )
            self.last_post_step_outcome = post_outcome
            if post_outcome.blocked:
                terminated = True
                stop_reason = post_outcome.stop_reason or HOOK_BLOCKED
                stop_detail = post_outcome.combined_message

        trace = StepTrace(
            step_index=step_index,
            prompt_messages=prompt_snapshot,
            model_response={
                "content": content,
                "tool_calls": [
                    {"id": tc.id, "name": tc.name, "arguments": tc.arguments} for tc in tool_calls
                ],
                **assistant_fields,
            },
            tool_results=[tr.to_dict() for tr in tool_results],
            compaction_applied=compaction_applied,
            duration_s=time.time() - start_time,
            usage=step_usage,
            stop_reason=stop_reason,
            stop_detail=stop_detail,
        )

        return terminated, trace
