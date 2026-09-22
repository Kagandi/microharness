from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

from core.environment import Environment
from core.evaluator import Evaluator
from core.loop import AgentLoop
from core.memory import Memory
from core.model import Model
from core.prompts import DEFAULT_SYSTEM_PROMPT
from core.tools import BashTool, ReadFileTool, Tool, ToolRegistry, WriteFileTool
from core.tracer import Tracer
from core.types import EvalResult, Message, RunResult, Task

DEFAULT_GLOBAL_DIR = Path.home() / ".mini-harness"


def _load_agents_instructions(workspace: Path, global_dir: Path | None = None) -> str:
    """Loads global and project-level steering guidelines in cascading order."""
    sections: list[str] = []

    # 1. Global user baseline instructions (~/.mini-harness/AGENTS.md)
    if global_dir:
        global_file = global_dir / "AGENTS.md"
        if global_file.is_file():
            content = global_file.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                sections.append(
                    f"--- Global User Instructions (~/.mini-harness/AGENTS.md) ---\n"
                    f"{content}\n"
                    f"----------------------------------------------------------"
                )

    # 2. Project-level instructions (./AGENTS.md, takes precedence/specializes)
    local_file = workspace / "AGENTS.md"
    if local_file.is_file():
        content = local_file.read_text(encoding="utf-8", errors="replace").strip()
        if content:
            sections.append(
                f"--- Project Instructions (AGENTS.md) ---\n"
                f"{content}\n"
                f"--------------------------------------"
            )

    return ("\n\n" + "\n\n".join(sections)) if sections else ""


def _discover_skills_index(workspace: Path, global_dir: Path | None = None) -> str:
    """Discovers available skills via progressive disclosure (Global and Project, with shadowing)."""
    # Key: skill_name -> (summary, path_str, scope)
    skills: dict[str, tuple[str, str, str]] = {}

    def scan_dir(s_dir: Path, scope: str, relative_base: Path | None) -> None:
        if not s_dir.is_dir():
            return
        for entry in sorted(s_dir.iterdir()):
            skill_doc = (
                entry / "SKILL.md" if entry.is_dir() else (entry if entry.suffix == ".md" else None)
            )
            if skill_doc and skill_doc.is_file() and entry.name != "README.md":
                lines = skill_doc.read_text(encoding="utf-8", errors="replace").splitlines()
                summary = next(
                    (line.strip().lstrip("#").strip() for line in lines if line.strip()), entry.stem
                )
                if relative_base and skill_doc.is_relative_to(relative_base):
                    path_str = str(skill_doc.relative_to(relative_base))
                else:
                    path_str = str(skill_doc)
                skills[entry.stem] = (summary, path_str, scope)

    # 1. Global skills first
    if global_dir:
        for gd in [global_dir / "skills", global_dir / ".skills"]:
            scan_dir(gd, "global", relative_base=None)

    # 2. Project skills second (shadows/overrides global skills with matching names)
    for pd in [workspace / "skills", workspace / ".skills"]:
        scan_dir(pd, "project", relative_base=workspace)

    if not skills:
        return ""

    has_global = any(s[2] == "global" for s in skills.values())
    lines: list[str] = []
    for name, (summary, path_str, scope) in sorted(skills.items()):
        badge = f"[{scope}] " if has_global else ""
        lines.append(f"- {badge}{name}: {summary} (read `{path_str}` for full instructions)")

    return (
        "--- Available Skills (Progressive Disclosure) ---\n"
        + "\n".join(lines)
        + "\nTip: Use the `read_file` tool to inspect a skill's full instructions only when needed.\n"
        + "------------------------------------------------"
    )


class Harness:
    """The orchestrator / chassis that hosts, constrains, traces, and evaluates the agent loop."""

    def __init__(
        self,
        model: Model,
        workspace_dir: Path | str | None = None,
        global_dir: Path | str | None = None,
        sandbox: bool = False,
        tools: list[Tool] | None = None,
        evaluator: Evaluator | None = None,
        runs_dir: Path | str | None = None,
        max_tool_chars: int = 2000,
        max_context_chars: int = 8000,
        verbose: bool = True,
    ):
        self.model = model
        self.explicit_workspace_dir = Path(workspace_dir).resolve() if workspace_dir else None
        self.global_dir = (
            Path(global_dir).resolve() if global_dir is not None else DEFAULT_GLOBAL_DIR
        )
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

        allowed_reads = [self.global_dir] if self.global_dir else []
        env = Environment(
            workspace_dir=workspace,
            logs_dir=logs_dir,
            allowed_read_dirs=allowed_reads,
            cleanup_on_teardown=False,
        )
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

        active_system_prompt = system_prompt
        if not use_sandbox:
            active_system_prompt += (
                f"\n\nContext:\nYou are working directly in an existing project directory at `{workspace}`.\n"
                "Inspect existing files and git history to understand the codebase before making changes."
            )

        # 1. Project steering instructions (AGENTS.md)
        agents_instructions = _load_agents_instructions(workspace, self.global_dir)
        if agents_instructions:
            active_system_prompt += f"\n\n{agents_instructions}"

        # 2. Progressive skills index (skills/ or .skills/)
        skills_index = _discover_skills_index(workspace, self.global_dir)
        if skills_index:
            active_system_prompt += f"\n\n{skills_index}"

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

        eval_result: EvalResult | None = None
        if self.evaluator:
            if self.verbose:
                print(f"\n{'=' * 20} Running Evaluation {'=' * 20}")
            eval_result = self.evaluator.evaluate(env, tracer.steps)
            if self.verbose:
                status_emoji = "✅" if eval_result.passed else "❌"
                print(
                    f"{status_emoji} Evaluation Passed: {eval_result.passed} (Score: {eval_result.score})"
                )
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
        initial_instruction: str | None = None,
        task_id: str = "interactive_session",
    ) -> None:
        """Interactive multi-turn session with slash commands and context usage tracking."""
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

        print("\n🚀 Interactive Session Started")
        print(f"📁 Project Workspace: {env.workspace_dir}")
        print(
            "💡 Commands: /compact (prune memory) | /status (context info) | /new (reset) | /exit\n"
        )

        step_idx = 1
        current_instruction: str | None = first_instruction

        while True:
            # If we have an active instruction to execute, run the agent loop
            if current_instruction:
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
                current_instruction = None

            # Render minimal context usage footer
            stats = memory.stats()
            print(
                f"─── 📊 Context: {stats['total_chars']} / {stats['max_context_chars']} chars ({stats['usage_pct']}%) | Messages: {stats['total_messages']} ───"
            )

            try:
                user_input = input("👤 You: ").strip()
            except (KeyboardInterrupt, EOFError):
                print("\nSession interrupted.")
                break

            if not user_input:
                continue

            # Handle Meta / Slash Commands without calling the model
            cmd = user_input.lower().strip()
            if cmd in ("/exit", "exit", "/quit", "quit"):
                print("Ending session.")
                break

            elif cmd in ("/help", "help"):
                print("\nAvailable meta-commands:")
                print("  /status   - Show context window size, message breakdown, and budget")
                print("  /compact  - Manually collapse older observations to free context")
                print("  /new      - Clear conversation history while keeping workspace files")
                print("  /exit     - Save session trace and exit\n")
                continue

            elif cmd in ("/status", "/context", "status"):
                s = memory.stats()
                print("\n📊 Context Breakdown:")
                print(
                    f"  • Usage: {s['total_chars']} / {s['max_context_chars']} chars ({s['usage_pct']}%)"
                )
                print(
                    f"  • Messages ({s['total_messages']}): {s['by_role']['system']} sys, {s['by_role']['user']} user, {s['by_role']['assistant']} assistant, {s['by_role']['tool']} tools\n"
                )
                continue

            elif cmd in ("/compact", "compact"):
                old_chars = memory.total_chars()
                did_compact = memory.compact_if_needed(force=True)
                new_chars = memory.total_chars()
                if did_compact:
                    print(
                        f"\n🧹 Compaction applied: freed {old_chars - new_chars} characters ({old_chars} -> {new_chars} chars).\n"
                    )
                else:
                    print("\nℹ️ No older observations eligible for compaction at this time.\n")
                continue

            elif cmd in ("/new", "/reset", "new", "reset"):
                memory.reset()
                print("\n✨ Started a fresh conversation context (workspace files preserved).\n")
                continue

            # Normal user message: feed into memory and execute
            memory.messages.append(Message(role="user", content=user_input))
            current_instruction = user_input

        json_path, md_path = tracer.finalize(finish_reason="session_ended")
        print(f"\n📄 Complete session trace saved to:\n   {md_path}\n")
