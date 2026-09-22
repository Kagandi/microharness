from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

from core.environment import Environment
from core.evaluator import Evaluator
from core.loop import AgentLoop
from core.memory import Memory
from core.model import Model
from core.prompts import DEFAULT_SYSTEM_PROMPT
from core.tools import BashTool, EditFileTool, ReadFileTool, Tool, ToolRegistry, WriteFileTool
from core.tracer import Tracer
from core.types import EvalResult, RunResult, Task

DEFAULT_GLOBAL_DIR = Path.home() / ".mini-harness"


def load_global_config(global_dir: Path | None = None) -> dict[str, Any]:
    """Loads user baseline configuration from ~/.mini-harness/config.json."""
    if not global_dir:
        return {}
    config_file = global_dir / "config.json"
    if not config_file.is_file():
        return {}
    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


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


def _discover_commands(
    workspace: Path, global_dir: Path | None = None
) -> dict[str, tuple[str, str, str]]:
    """Discovers custom slash commands from global and project commands/ directories.

    Returns dict mapping: command_name -> (description, template, scope)
    where scope is 'global' or 'project'. Project commands cleanly shadow global ones.
    """
    commands: dict[str, tuple[str, str, str]] = {}

    def scan_dir(cmd_dir: Path, scope: str) -> None:
        if not cmd_dir.is_dir():
            return
        for file in sorted(cmd_dir.glob("*.md")):
            if file.name.lower() == "readme.md":
                continue
            content = file.read_text(encoding="utf-8", errors="replace")
            lines = content.splitlines()
            desc = next(
                (line.strip().lstrip("#").strip() for line in lines if line.strip()), file.stem
            )
            commands[file.stem.lower()] = (desc, content, scope)

    # 1. Global commands (~/.mini-harness/commands/)
    if global_dir:
        for gd in [global_dir / "commands", global_dir / ".commands"]:
            scan_dir(gd, "global")

    # 2. Project commands (./commands/ or ./.commands/, shadows global)
    for pd in [workspace / "commands", workspace / ".commands"]:
        scan_dir(pd, "project")

    return commands


def _expand_command(template: str, user_args: str) -> str:
    """Expands a command template with user arguments."""
    args = user_args.strip()
    result = template
    if "$ARGUMENTS" in result:
        result = result.replace("$ARGUMENTS", args)
    elif "{args}" in result:
        result = result.replace("{args}", args)
    elif args:
        result = f"{result.strip()}\n\nAdditional Arguments / Target:\n{args}"
    return result.strip()


class Harness:
    """The orchestrator / chassis that hosts, constrains, traces, and evaluates the agent loop."""

    def __init__(
        self,
        model: Model,
        workspace_dir: Path | str | None = None,
        global_dir: Path | str | None = None,
        sandbox: bool | None = None,
        tools: list[Tool] | None = None,
        evaluator: Evaluator | None = None,
        runs_dir: Path | str | None = None,
        max_tool_chars: int | None = None,
        max_context_chars: int | None = None,
        trigger_ratio: float | None = None,
        verbose: bool = True,
    ):
        self.model = model
        self.explicit_workspace_dir = Path(workspace_dir).resolve() if workspace_dir else None
        self.global_dir = (
            Path(global_dir).resolve() if global_dir is not None else DEFAULT_GLOBAL_DIR
        )
        self.global_config = load_global_config(self.global_dir)

        self.sandbox = (
            sandbox if sandbox is not None else bool(self.global_config.get("sandbox", False))
        )
        self.custom_tools = tools
        self.evaluator = evaluator
        self.runs_dir = Path(runs_dir or ".runs").resolve()
        self.max_tool_chars = (
            max_tool_chars
            if max_tool_chars is not None
            else int(self.global_config.get("max_tool_chars", 2000))
        )
        self.max_context_chars = (
            max_context_chars
            if max_context_chars is not None
            else int(self.global_config.get("max_context_chars", 8000))
        )
        self.trigger_ratio = (
            trigger_ratio
            if trigger_ratio is not None
            else float(self.global_config.get("trigger_ratio", 0.80))
        )
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
            tool_registry.register(EditFileTool(env))

        active_system_prompt = system_prompt
        tool_docs = tool_registry.to_system_prompt_snippet()
        if (
            tool_docs
            and "You have access to tools to interact with the environment:"
            not in active_system_prompt
        ):
            active_system_prompt += f"\n\n{tool_docs}"

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

        # Expand custom command if initial instruction is a slash command
        custom_commands = _discover_commands(workspace, self.global_dir)
        effective_instruction = initial_instruction
        if initial_instruction.startswith("/"):
            cmd_part, _, args_part = initial_instruction[1:].partition(" ")
            cmd_key = cmd_part.lower().strip()
            if cmd_key in custom_commands:
                _, tmpl, scope = custom_commands[cmd_key]
                effective_instruction = _expand_command(tmpl, args_part)
                if self.verbose:
                    print(f"⚡ Expanded custom command /{cmd_key} [{scope}]")

        memory = Memory(
            system_prompt=active_system_prompt,
            user_prompt=effective_instruction,
            max_context_chars=self.max_context_chars,
            trigger_ratio=self.trigger_ratio,
        )

        task = Task(id=task_id, instruction=effective_instruction, initial_files=initial_files)
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

        patch_diff = env.get_git_patch()
        patch_path: str | None = None
        if patch_diff:
            patch_file = run_dir / "patch.diff"
            patch_file.write_text(patch_diff, encoding="utf-8")
            patch_path = str(patch_file)
            if self.verbose:
                print(f"📦 Git patch saved to: {patch_file}")

        json_trace_path, md_trace_path = tracer.finalize(
            finish_reason=finish_reason, eval_result=eval_result, patch_diff=patch_diff
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
            patch_path=patch_path,
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

        custom_commands = _discover_commands(env.workspace_dir, self.global_dir)

        print("\n🚀 Interactive Session Started")
        print(f"📁 Project Workspace: {env.workspace_dir}")
        print(
            "💡 Commands: /compact (prune memory) | /status (context info) | /new (reset) | /help | /exit\n"
        )

        step_idx = 1
        has_pending_turn = True
        max_steps_per_turn = 30

        while True:
            # If we have an active instruction to execute, run the agent loop
            if has_pending_turn:
                turn_steps = 0
                while turn_steps < max_steps_per_turn:
                    try:
                        is_finished, step_trace = loop.step(step_index=step_idx)
                        tracer.record_step(step_trace)
                        step_idx += 1
                        turn_steps += 1

                        if is_finished:
                            agent_reply = step_trace.model_response.get("content", "")
                            print(f"\n🤖 Agent:\n{agent_reply.strip()}\n")
                            break
                    except KeyboardInterrupt:
                        print("\n⚠️ Step interrupted by user (Ctrl+C). Returning to prompt.\n")
                        break
                    except Exception as exc:
                        print(f"\n❌ Error during turn: {exc}")
                        break
                else:
                    print(
                        f"\n⚠️ Reached safety limit of {max_steps_per_turn} steps for this turn. Returning to prompt.\n"
                    )
                has_pending_turn = False

            # Render minimal context usage footer
            stats = memory.stats()
            print(
                f"─── 📊 Context: {stats['total_chars']:,} chars (~{stats['estimated_tokens']:,} tokens) / "
                f"{stats['max_context_chars']:,} ({stats['usage_pct']}%) | Messages: {stats['total_messages']} ───"
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
                print("\nAvailable commands:")
                print("  /status   - Show context window size, message breakdown, and budget")
                print("  /compact  - Manually collapse older observations to free context")
                print("  /new      - Clear conversation history while keeping workspace files")
                print("  /exit     - Save session trace and exit")
                if custom_commands:
                    print("\nCustom slash commands (from commands/):")
                    for c_name, (c_desc, _, c_scope) in sorted(custom_commands.items()):
                        print(f"  /{c_name:<10} - [{c_scope}] {c_desc}")
                print()
                continue

            elif cmd in ("/status", "/context", "status", "context"):
                s = memory.stats()
                print("\n📊 Context Breakdown:")
                print(
                    f"  • Usage: {s['total_chars']:,} chars (~{s['estimated_tokens']:,} tokens) / "
                    f"{s['max_context_chars']:,} chars ({s['usage_pct']}%)"
                )
                print(
                    f"  • Messages ({s['total_messages']}): {s['by_role']['system']} sys, "
                    f"{s['by_role']['user']} user, {s['by_role']['assistant']} assistant, {s['by_role']['tool']} tools\n"
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

            # Check for custom slash commands
            elif user_input.startswith("/"):
                cmd_part, _, args_part = user_input[1:].partition(" ")
                cmd_key = cmd_part.lower().strip()
                if cmd_key in custom_commands:
                    _, template, scope = custom_commands[cmd_key]
                    expanded_prompt = _expand_command(template, args_part)
                    print(f"\n⚡ Executing /{cmd_key} [{scope}] ...\n")
                    memory.add_user(expanded_prompt)
                    has_pending_turn = True
                    continue
                else:
                    print(
                        f"\n❌ Unknown command '/{cmd_key}'. Type /help to see available commands.\n"
                    )
                    continue

            # Normal user message: feed into memory and execute
            memory.add_user(user_input)
            has_pending_turn = True

        patch_diff = env.get_git_patch()
        if patch_diff:
            patch_file = run_dir / "patch.diff"
            patch_file.write_text(patch_diff, encoding="utf-8")
            print(f"📦 Git patch saved to: {patch_file}")

        json_path, md_path = tracer.finalize(finish_reason="session_ended", patch_diff=patch_diff)
        print(f"\n📄 Complete session trace saved to:\n   {md_path}\n")
