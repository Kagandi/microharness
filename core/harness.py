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
from core.types import EvalResult, Message, RunResult, Task


class Harness:
    """The orchestrator / chassis that hosts, constrains, traces, and evaluates the agent loop."""

    def __init__(
        self,
        model: Model,
        workspace_dir: Optional[Path | str] = None,
        sandbox: bool = False,
        tools: Optional[list[Tool]] = None,
        evaluator: Optional[Evaluator] = None,
        runs_dir: Optional[Path | str] = None,
        max_tool_chars: int = 2000,
        max_context_chars: int = 8000,
        verbose: bool = True,
    ):
        self.model = model
        self.explicit_workspace_dir = Path(workspace_dir).resolve() if workspace_dir else None
        self.sandbox = sandbox
        self.custom_tools = tools
        self.evaluator = evaluator
        self.runs_dir = Path(runs_dir or ".runs").resolve()
        self.max_tool_chars = max_tool_chars
        self.max_context_chars = max_context_chars
        self.verbose = verbose

    def _setup_run(
        self,
        task_id: str,
        system_prompt: str,
        initial_instruction: str,
        initial_files: dict[str, str],
        force_sandbox: bool = False,
    ) -> tuple[Environment, ToolRegistry, Memory, Tracer, AgentLoop, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_dir / f"{task_id}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        use_sandbox = self.sandbox or force_sandbox or (self.explicit_workspace_dir is None)

        if use_sandbox:
            workspace = run_dir / "workspace"
            workspace.mkdir(parents=True, exist_ok=True)
        else:
            workspace = self.explicit_workspace_dir

        env = Environment(workspace_dir=workspace, logs_dir=logs_dir, cleanup_on_teardown=False)
        if initial_files:
            env.populate_files(initial_files)

        tool_registry = ToolRegistry(env=env, max_tool_chars=self.max_tool_chars)
        if self.custom_tools:
            for t in self.custom_tools:
                tool_registry.register(t)
        else:
            tool_registry.register(BashTool(env))
            tool_registry.register(ReadFileTool(env))
            tool_registry.register(WriteFileTool(env))

        # Augment system prompt with current working directory context if operating in an existing project
        active_system_prompt = system_prompt
        if not use_sandbox:
            active_system_prompt += (
                f"\n\nContext:\nYou are working directly in an existing project directory at `{workspace}`.\n"
                "Inspect existing files and git history to understand the codebase before making changes."
            )

        memory = Memory(
            system_prompt=active_system_prompt,
            user_prompt=initial_instruction,
            max_context_chars=self.max_context_chars,
        )

        task = Task(id=task_id, instruction=initial_instruction, initial_files=initial_files)
        tracer = Tracer(task=task, output_dir=run_dir, verbose=self.verbose)
        loop = AgentLoop(model=self.model, memory=memory, tools=tool_registry)

        return env, tool_registry, memory, tracer, loop, run_dir

    def run(self, task: Task) -> RunResult:
        """One-shot batch execution mode (e.g. for benchmarks or tests)."""
        run_start = time.time()
        system_prompt = task.system_prompt or DEFAULT_SYSTEM_PROMPT

        env, _, _, tracer, loop, run_dir = self._setup_run(
            task_id=task.id,
            system_prompt=system_prompt,
            initial_instruction=task.instruction,
            initial_files=task.initial_files,
            force_sandbox=bool(task.initial_files),
        )

        if self.verbose:
            print(f"\n🚀 Starting Harness for Task: {task.id}")
            print(f"📁 Workspace: {env.workspace_dir}")
            print(f"📝 Instruction: {task.instruction.strip()}\n")

        finish_reason = "max_steps_exceeded"
        step_idx = 1

        while step_idx <= task.max_steps:
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

        eval_result: Optional[EvalResult] = None
        if self.evaluator:
            if self.verbose:
                print(f"\n{'='*20} Running Evaluation {'='*20}")
            eval_result = self.evaluator.evaluate(env, tracer.steps)
            if self.verbose:
                status_emoji = "✅" if eval_result.passed else "❌"
                print(f"{status_emoji} Evaluation Passed: {eval_result.passed} (Score: {eval_result.score})")
                print(f"Feedback: {eval_result.feedback.strip()}")

        json_trace_path, md_trace_path = tracer.finalize(
            finish_reason=finish_reason, eval_result=eval_result
        )

        passed = eval_result.passed if eval_result else (finish_reason == "completed")
        score = eval_result.score if eval_result else (1.0 if passed else 0.0)

        return RunResult(
            task_id=task.id,
            passed=passed,
            score=score,
            total_steps=total_steps,
            duration_s=total_duration,
            finish_reason=finish_reason,
            trace_path=str(json_trace_path),
            workspace_path=str(env.workspace_dir),
            eval_result=eval_result,
        )

    def run_interactive(
        self,
        initial_instruction: Optional[str] = None,
        task_id: str = "interactive_session",
    ) -> None:
        """Interactive multi-turn session mode allowing continuous conversation."""
        first_instruction = initial_instruction or input("👤 You: ").strip()
        if not first_instruction:
            print("No instruction provided. Exiting.")
            return

        env, _, memory, tracer, loop, run_dir = self._setup_run(
            task_id=task_id,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            initial_instruction=first_instruction,
            initial_files={},
        )

        print(f"\n🚀 Interactive Session Started")
        print(f"📁 Project Workspace: {env.workspace_dir}")
        print("💡 Type 'exit' or 'quit' anytime to end the session.\n")

        step_idx = 1
        current_instruction: Optional[str] = first_instruction

        while current_instruction:
            while True:
                try:
                    is_finished, step_trace = loop.step(step_index=step_idx)
                    tracer.record_step(step_trace)
                    step_idx += 1

                    if is_finished:
                        agent_reply = step_trace.model_response.get("content", "")
                        print(f"\n🤖 Agent:\n{agent_reply.strip()}\n")
                        break
                except Exception as exc:
                    print(f"\n❌ Error during turn: {exc}")
                    break

            try:
                user_followup = input("👤 You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSession interrupted.")
                break

            if not user_followup or user_followup.lower() in ("exit", "quit", "/exit", "/quit"):
                print("\nEnding session.")
                break

            memory.messages.append(Message(role="user", content=user_followup))
            current_instruction = user_followup

        json_path, md_path = tracer.finalize(finish_reason="session_ended")
        print(f"\n📄 Complete session trace saved to:\n   {md_path}\n")
