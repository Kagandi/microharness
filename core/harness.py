from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional

from core.environment import Environment
from core.evaluator import Evaluator
from core.loop import AgentLoop
from core.memory import Memory
from core.model import Model
from core.prompts import DEFAULT_SYSTEM_PROMPT
from core.tools import BashTool, ReadFileTool, Tool, ToolRegistry, WriteFileTool
from core.tracer import Tracer
from core.types import EvalResult, RunResult, Task


class Harness:
    """The orchestrator / chassis that hosts, constrains, traces, and evaluates the agent loop."""

    def __init__(
        self,
        model: Model,
        tools: Optional[list[Tool]] = None,
        evaluator: Optional[Evaluator] = None,
        runs_dir: Optional[Path | str] = None,
        max_tool_chars: int = 3000,
        max_context_chars: int = 15000,
        verbose: bool = True,
    ):
        self.model = model
        self.custom_tools = tools
        self.evaluator = evaluator
        self.runs_dir = Path(runs_dir or ".runs").resolve()
        self.max_tool_chars = max_tool_chars
        self.max_context_chars = max_context_chars
        self.verbose = verbose

    def run(self, task: Task) -> RunResult:
        run_start = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_dir / f"{task.id}_{timestamp}"
        workspace_dir = run_dir / "workspace"
        workspace_dir.mkdir(parents=True, exist_ok=True)

        if self.verbose:
            print(f"\n🚀 Starting Harness for Task: {task.id}")
            print(f"📁 Workspace: {workspace_dir}")
            print(f"📝 Instruction: {task.instruction.strip()}\n")

        # 1. Setup Isolated Environment & Sandbox
        env = Environment(workspace_dir=workspace_dir, cleanup_on_teardown=False)
        if task.initial_files:
            env.populate_files(task.initial_files)

        # 2. Setup Tool Registry
        tool_registry = ToolRegistry(env=env, max_tool_chars=self.max_tool_chars)
        if self.custom_tools:
            for t in self.custom_tools:
                tool_registry.register(t)
        else:
            # Register standard SWE tools
            tool_registry.register(BashTool(env))
            tool_registry.register(ReadFileTool(env))
            tool_registry.register(WriteFileTool(env))

        # 3. Setup Memory & Context Management
        system_prompt = task.system_prompt or DEFAULT_SYSTEM_PROMPT
        memory = Memory(
            system_prompt=system_prompt,
            user_prompt=task.instruction,
            max_context_chars=self.max_context_chars,
        )

        # 4. Setup Telemetry & Tracer
        tracer = Tracer(task=task, output_dir=run_dir, verbose=self.verbose)

        # 5. Initialize Pure Agent Loop
        loop = AgentLoop(model=self.model, memory=memory, tools=tool_registry)

        finish_reason = "max_steps_exceeded"
        step_idx = 1

        # 6. Execute Loop under Guardrails & Budgets
        while step_idx <= task.max_steps:
            # Check timeout budget
            elapsed = time.time() - run_start
            if elapsed > task.timeout_seconds:
                finish_reason = "timeout"
                tracer.log(f"\n⚠️ Task reached maximum allowed timeout ({task.timeout_seconds}s).")
                break

            try:
                is_finished, step_trace = loop.step(step_index=step_idx)
                tracer.record_step(step_trace)

                if is_finished:
                    finish_reason = "completed"
                    break
            except Exception as exc:
                finish_reason = "error"
                tracer.log(f"\n❌ Exception occurred during step {step_idx}: {exc}")
                break

            step_idx += 1

        total_duration = time.time() - run_start
        total_steps = len(tracer.steps)

        # 7. Post-Run Evaluation
        eval_result: Optional[EvalResult] = None
        if self.evaluator:
            if self.verbose:
                print(f"\n{'='*20} Running Evaluation {'='*20}")
            eval_result = self.evaluator.evaluate(env, tracer.steps)
            if self.verbose:
                status_emoji = "✅" if eval_result.passed else "❌"
                print(f"{status_emoji} Evaluation Passed: {eval_result.passed} (Score: {eval_result.score})")
                print(f"Feedback: {eval_result.feedback.strip()}")

        # 8. Finalize Telemetry and Traces
        json_trace_path, md_trace_path = tracer.finalize(
            finish_reason=finish_reason, eval_result=eval_result
        )

        passed = eval_result.passed if eval_result else (finish_reason == "completed")
        score = eval_result.score if eval_result else (1.0 if passed else 0.0)

        if self.verbose:
            print(f"\n🏁 Finished run in {total_duration:.2f}s across {total_steps} steps.")
            print(f"📄 Trace saved to: {md_trace_path}\n")

        return RunResult(
            task_id=task.id,
            passed=passed,
            score=score,
            total_steps=total_steps,
            duration_s=total_duration,
            finish_reason=finish_reason,
            trace_path=str(json_trace_path),
            workspace_path=str(workspace_dir),
            eval_result=eval_result,
        )
