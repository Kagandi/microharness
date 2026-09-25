from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from core.environment import DockerEnvironment, Environment
from core.evaluator import Evaluator
from core.loop import AgentLoop
from core.memory import Memory
from core.model import Model, normalize_cache_ttl
from core.prompts import DEFAULT_SYSTEM_PROMPT
from core.tools import BashTool, EditFileTool, ReadFileTool, Tool, ToolRegistry, WriteFileTool
from core.tracer import Tracer
from core.types import EvalResult, RunResult, Task


def get_default_global_dir() -> Path:
    """Returns the default global directory, evaluating Path.home() at call time."""
    home = Path.home()
    if (home / ".microharness").exists() or not (home / ".mini-harness").exists():
        return home / ".microharness"
    return home / ".mini-harness"


DEFAULT_GLOBAL_DIR = get_default_global_dir()


def _load_json_config(config_file: Path | None) -> dict[str, Any]:
    if not config_file or not config_file.is_file():
        return {}
    try:
        data = json.loads(config_file.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_global_config(global_dir: Path | None = None) -> dict[str, Any]:
    """Loads user baseline configuration from ~/.microharness/config.json."""
    return _load_json_config(global_dir / "config.json" if global_dir else None)


def load_project_config(workspace: Path | None = None) -> dict[str, Any]:
    """Loads optional project-level configuration from .microharness/config.json."""
    return _load_json_config(workspace / ".microharness" / "config.json" if workspace else None)


def _load_agents_instructions(workspace: Path, global_dir: Path | None = None) -> str:
    """Loads global and project-level steering guidelines in cascading order."""
    sections: list[str] = []

    # 1. Global user baseline instructions (~/.microharness/AGENTS.md)
    if global_dir:
        global_file = global_dir / "AGENTS.md"
        if global_file.is_file():
            content = global_file.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                sections.append(
                    f"--- Global User Instructions (~/.microharness/AGENTS.md) ---\n"
                    f"{content}\n"
                    f"----------------------------------------------------------"
                )

    # 2. Project-level instructions (./.microharness/AGENTS.md or ./AGENTS.md, takes precedence/specializes)
    for candidate in [workspace / ".microharness" / "AGENTS.md", workspace / "AGENTS.md"]:
        if candidate.is_file():
            content = candidate.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                sections.append(
                    f"--- Project Instructions ({candidate.name}) ---\n"
                    f"{content}\n"
                    f"--------------------------------------"
                )
                break

    return ("\n\n" + "\n\n".join(sections)) if sections else ""


def _extract_skill_summary(content: str, default: str) -> str:
    """Extracts a 1-line summary from a SKILL.md document, supporting YAML frontmatter."""
    lines = [line.strip() for line in content.splitlines()]
    non_empty = [line for line in lines if line]
    if not non_empty:
        return default

    if non_empty[0] == "---":
        in_frontmatter = True
        desc_line = None
        after_frontmatter: list[str] = []
        for line in non_empty[1:]:
            if line == "---":
                in_frontmatter = False
                continue
            if in_frontmatter:
                if line.startswith("description:"):
                    _, _, val = line.partition(":")
                    val = val.strip().strip("\"'")
                    if val:
                        desc_line = val
            else:
                after_frontmatter.append(line)
        if desc_line:
            return desc_line
        if after_frontmatter:
            return after_frontmatter[0].lstrip("#").strip() or default
        return default

    return non_empty[0].lstrip("#").strip() or default


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
                content = skill_doc.read_text(encoding="utf-8", errors="replace")
                summary = _extract_skill_summary(content, entry.stem)
                if relative_base and skill_doc.is_relative_to(relative_base):
                    path_str = str(skill_doc.relative_to(relative_base))
                else:
                    path_str = str(skill_doc)
                skills[entry.stem] = (summary, path_str, scope)

    # 1. Global skills first (~/.microharness/skills/ or ~/.microharness/.skills/)
    if global_dir:
        for gd in [global_dir / "skills", global_dir / ".skills"]:
            scan_dir(gd, "global", relative_base=None)

    # 2. Project skills second (shadows/overrides global skills with matching names)
    # Order: legacy folders first, then .microharness/skills so .microharness takes priority
    for pd in [workspace / "skills", workspace / ".skills", workspace / ".microharness" / "skills"]:
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
    """Discovers custom slash commands from global and project commands directories.

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

    # 1. Global commands (~/.microharness/commands/)
    if global_dir:
        for gd in [global_dir / "commands", global_dir / ".commands"]:
            scan_dir(gd, "global")

    # 2. Project commands (shadows global commands)
    # Order: legacy folders first, then .microharness/commands so .microharness takes priority
    for pd in [
        workspace / "commands",
        workspace / ".commands",
        workspace / ".microharness" / "commands",
    ]:
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
        docker: bool | None = None,
        docker_image: str | None = None,
        tools: list[Tool] | None = None,
        evaluator: Evaluator | None = None,
        runs_dir: Path | str | None = None,
        max_tool_chars: int | None = None,
        max_context_chars: int | None = None,
        trigger_ratio: float | None = None,
        cache_ttl: str | None = None,
        max_step_retries: int | None = None,
        step_retry_delay: float | None = None,
        verbose: bool = True,
    ):
        self.model = model
        self.explicit_workspace_dir = Path(workspace_dir).resolve() if workspace_dir else None
        self.global_dir = (
            Path(global_dir).resolve() if global_dir is not None else get_default_global_dir()
        )
        self.global_config = load_global_config(self.global_dir)
        self.project_config = load_project_config(self.explicit_workspace_dir)
        self.merged_config = {**self.global_config, **self.project_config}

        self.sandbox = (
            sandbox if sandbox is not None else bool(self.merged_config.get("sandbox", False))
        )
        self.docker = (
            docker if docker is not None else bool(self.merged_config.get("docker", False))
        )
        self.docker_image = docker_image or self.merged_config.get("docker_image", "python:3.11")
        self.custom_tools = tools
        self.evaluator = evaluator
        configured_runs_dir = self.merged_config.get("runs_dir")
        if runs_dir is not None:
            self.runs_dir = Path(runs_dir).resolve()
        elif configured_runs_dir:
            self.runs_dir = Path(configured_runs_dir).resolve()
        else:
            self.runs_dir = (self.global_dir / "runs").resolve()
        self.max_tool_chars = (
            max_tool_chars
            if max_tool_chars is not None
            else int(self.merged_config.get("max_tool_chars", 2000))
        )
        self.max_context_chars = (
            max_context_chars
            if max_context_chars is not None
            else int(self.merged_config.get("max_context_chars", 8000))
        )
        self.trigger_ratio = (
            trigger_ratio
            if trigger_ratio is not None
            else float(self.merged_config.get("trigger_ratio", 0.80))
        )
        configured_cache_ttl = (
            cache_ttl if cache_ttl is not None else self.merged_config.get("cache_ttl", "5m")
        )
        self.cache_ttl = normalize_cache_ttl(configured_cache_ttl)
        if hasattr(self.model, "set_cache_ttl"):
            self.model.set_cache_ttl(self.cache_ttl)
        self.max_step_retries = (
            max_step_retries
            if max_step_retries is not None
            else int(self.merged_config.get("max_step_retries", 3))
        )
        self.step_retry_delay = (
            step_retry_delay
            if step_retry_delay is not None
            else float(self.merged_config.get("step_retry_delay", 5.0))
        )
        self.verbose = verbose

    def set_cache_ttl(self, ttl: str | None) -> str | None:
        """Dynamically switches prompt caching duration on the fly ('5m', '1h', or 'off'/None)."""
        self.cache_ttl = normalize_cache_ttl(ttl)
        if hasattr(self.model, "set_cache_ttl"):
            return self.model.set_cache_ttl(self.cache_ttl)
        return self.cache_ttl

    def get_cache_stats(self) -> dict[str, Any]:
        """Returns prompt cache TTL and token performance metrics."""
        if hasattr(self.model, "get_cache_stats"):
            return self.model.get_cache_stats()
        return {"cache_ttl": self.cache_ttl}

    def _create_environment(
        self,
        workspace: Path,
        logs_dir: Path,
        active_docker: bool,
        active_image: str,
        init_workdir: str | None,
        initial_files: dict[str, str],
    ) -> Environment:
        allowed_reads: list[Path] = [logs_dir, self.runs_dir]
        if self.global_dir:
            for sub in ["skills", ".skills", "commands", ".commands"]:
                sub_path = self.global_dir / sub
                allowed_reads.append(sub_path)
            allowed_reads.append(self.global_dir / "AGENTS.md")
        if active_docker:
            env: Environment = DockerEnvironment(
                image=active_image,
                workspace_dir=workspace,
                logs_dir=logs_dir,
                allowed_read_dirs=allowed_reads,
                cleanup_on_teardown=True,
                init_workdir=init_workdir,
            )
        else:
            env = Environment(
                workspace_dir=workspace,
                logs_dir=logs_dir,
                allowed_read_dirs=allowed_reads,
                cleanup_on_teardown=False,
            )
        if initial_files:
            env.populate_files(initial_files)
        return env

    def _init_tools(self, env: Environment) -> ToolRegistry:
        registry = ToolRegistry(env=env, max_tool_chars=self.max_tool_chars)
        if self.custom_tools:
            for t in self.custom_tools:
                registry.register(t)
        else:
            registry.register(BashTool(env))
            registry.register(ReadFileTool(env))
            registry.register(WriteFileTool(env))
            registry.register(EditFileTool(env))
        return registry

    def _build_system_prompt(
        self,
        workspace: Path,
        base_prompt: str,
        tool_registry: ToolRegistry,
        use_sandbox: bool,
    ) -> str:
        prompt = base_prompt
        tool_docs = tool_registry.to_system_prompt_snippet()
        if tool_docs and "You have access to tools to interact with the environment:" not in prompt:
            prompt += f"\n\n{tool_docs}"

        if not use_sandbox:
            prompt += (
                f"\n\nContext:\nYou are working directly in an existing project directory at `{workspace}`.\n"
                "Inspect existing files and git history to understand the codebase before making changes."
            )

        agents_instructions = _load_agents_instructions(workspace, self.global_dir)
        if agents_instructions:
            prompt += f"\n\n{agents_instructions}"

        skills_index = _discover_skills_index(workspace, self.global_dir)
        if skills_index:
            prompt += f"\n\n{skills_index}"

        return prompt

    def _setup_run(
        self,
        task_id: str,
        system_prompt: str,
        initial_instruction: str,
        initial_files: dict[str, str],
        force_sandbox: bool = False,
        docker_image: str | None = None,
        init_workdir: str | None = None,
    ) -> tuple[Environment, ToolRegistry, Memory, Tracer, AgentLoop, Path]:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = self.runs_dir / f"{task_id}_{timestamp}"
        run_dir.mkdir(parents=True, exist_ok=True)
        logs_dir = run_dir / "logs"
        logs_dir.mkdir(parents=True, exist_ok=True)

        use_sandbox = self.sandbox or force_sandbox or (self.explicit_workspace_dir is None)
        workspace = (run_dir / "workspace") if use_sandbox else self.explicit_workspace_dir
        workspace.mkdir(parents=True, exist_ok=True)

        env = self._create_environment(
            workspace=workspace,
            logs_dir=logs_dir,
            active_docker=self.docker or bool(docker_image),
            active_image=docker_image or self.docker_image,
            init_workdir=init_workdir,
            initial_files=initial_files,
        )
        tool_registry = self._init_tools(env)
        active_system_prompt = self._build_system_prompt(
            workspace, system_prompt, tool_registry, use_sandbox
        )

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

    def _execute_steps(
        self,
        loop: AgentLoop,
        tracer: Tracer,
        start_step: int,
        max_steps: int,
        deadline: float | None = None,
        on_finish: Callable[[str], None] | None = None,
    ) -> tuple[bool, str, int]:
        """Drives the agent loop for up to max_steps with retries.

        Returns:
            (is_finished, finish_reason, next_step_index)
        """
        step_idx = start_step
        step_attempts = 0
        finish_reason = "max_steps_exceeded"

        while (step_idx - start_step) < max_steps:
            if deadline and time.time() > deadline:
                tracer.log("\n⚠️ Task reached maximum allowed timeout.")
                return False, "timeout", step_idx

            try:
                is_finished, step_trace = loop.step(step_index=step_idx)
                tracer.record_step(step_trace)
                step_attempts = 0
                step_idx += 1

                if is_finished:
                    if on_finish:
                        on_finish(step_trace.model_response.get("content", ""))
                    return True, "completed", step_idx

            except KeyboardInterrupt:
                tracer.log("\n⚠️ Interrupted by user (Ctrl+C).")
                return False, "interrupted", step_idx

            except Exception as exc:
                step_attempts += 1
                if step_attempts <= self.max_step_retries:
                    sleep_s = self.step_retry_delay * step_attempts
                    tracer.log(
                        f"\n⚠️ Step {step_idx} encountered an error: {exc}\n"
                        f"⏳ Sleeping {sleep_s:.1f}s and retrying step ({step_attempts}/{self.max_step_retries})..."
                    )
                    time.sleep(sleep_s)
                    continue
                tracer.log(
                    f"\n❌ Step {step_idx} failed after {self.max_step_retries} retries: {exc}"
                )
                return False, "error", step_idx

        return False, finish_reason, step_idx

    def run(self, task: Task) -> RunResult:
        """One-shot batch execution mode (e.g. for benchmarks or tests)."""
        run_start = time.time()
        system_prompt = task.system_prompt or DEFAULT_SYSTEM_PROMPT

        active_docker = self.docker or bool(task.docker_image)
        active_image = task.docker_image or self.docker_image

        env, _, _, tracer, loop, run_dir = self._setup_run(
            task_id=task.id,
            system_prompt=system_prompt,
            initial_instruction=task.instruction,
            initial_files=task.initial_files,
            force_sandbox=bool(task.initial_files) or bool(task.init_workdir),
            docker_image=task.docker_image,
            init_workdir=task.init_workdir,
        )

        if self.verbose:
            print(f"\n🚀 Starting Harness for Task: {task.id}")
            print(f"📁 Workspace: {env.workspace_dir}")
            if active_docker:
                print(f"🐳 Docker Sandbox: ACTIVE ({active_image})")
            print(f"📝 Instruction: {task.instruction.strip()}\n")

        try:
            is_finished, finish_reason, _ = self._execute_steps(
                loop=loop,
                tracer=tracer,
                start_step=1,
                max_steps=task.max_steps,
                deadline=run_start + task.timeout_seconds,
            )

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
        finally:
            env.teardown()

    def _print_interactive_help(self, custom_commands: dict[str, tuple[str, str, str]]) -> None:
        print("\nAvailable commands:")
        print("  /cache [5m|1h|off] - View or switch prompt caching duration on the fly")
        print("  /status            - Show context window size, message breakdown, and budget")
        print("  /compact           - Manually collapse older observations to free context")
        print("  /new               - Clear conversation history while keeping workspace files")
        print("  /exit              - Save session trace and exit")
        if custom_commands:
            print("\nCustom slash commands (from commands/):")
            for c_name, (c_desc, _, c_scope) in sorted(custom_commands.items()):
                print(f"  /{c_name:<10} - [{c_scope}] {c_desc}")
        print()

    def _print_context_status(self, memory: Memory) -> None:
        s = memory.stats()
        print("\n📊 Context Breakdown:")
        print(
            f"  • Usage: {s['total_chars']:,} chars (~{s['estimated_tokens']:,} tokens) / "
            f"{s['max_context_chars']:,} chars ({s['usage_pct']}%)"
        )
        print(
            f"  • Messages ({s['total_messages']}): {s['by_role']['system']} sys, "
            f"{s['by_role']['user']} user, {s['by_role']['assistant']} assistant, {s['by_role']['tool']} tools"
        )
        c_stats = self.get_cache_stats()
        ttl_val = c_stats.get("cache_ttl")
        print(f"  • Prompt Caching: {ttl_val.upper() if ttl_val else 'OFF'}")
        if c_stats.get("total_prompt_tokens"):
            cached = c_stats.get("total_cached_tokens", 0)
            total = c_stats.get("total_prompt_tokens", 0)
            rate = c_stats.get("cache_hit_rate_pct", 0.0)
            print(f"    Cache Hits: {cached:,} / {total:,} prompt tokens ({rate}% hit rate)\n")
        else:
            print()

    def _handle_cache_command(self, user_input: str) -> None:
        parts = user_input.strip().split()
        if len(parts) == 1:
            stats = self.get_cache_stats()
            curr_ttl = stats.get("cache_ttl")
            if curr_ttl:
                desc = (
                    "5 minutes (Standard Ephemeral)"
                    if curr_ttl == "5m"
                    else "1 hour (Extended Beta)"
                )
                print(f"\n⚡ Prompt Caching: {curr_ttl.upper()} ({desc})")
            else:
                print("\n⚡ Prompt Caching: DISABLED (off)")
            if stats.get("total_prompt_tokens"):
                cached = stats.get("total_cached_tokens", 0)
                total = stats.get("total_prompt_tokens", 0)
                rate = stats.get("cache_hit_rate_pct", 0.0)
                print(f"  • Cached Tokens: {cached:,} / {total:,} ({rate}% hit rate)")
            print("  • Usage: /cache 5m | /cache 1h | /cache off\n")
            return

        target_ttl = parts[1].lower().strip()
        try:
            new_ttl = self.set_cache_ttl(target_ttl)
            if new_ttl:
                desc = "5-minute ephemeral" if new_ttl == "5m" else "1-hour extended beta"
                print(
                    f"\n⚡ Prompt caching switched to: {new_ttl.upper()} ({desc}) for subsequent turns.\n"
                )
            else:
                print("\n⚡ Prompt caching disabled for subsequent turns.\n")
        except ValueError as exc:
            print(f"\n❌ {exc}\n")

    def _dispatch_interactive_input(
        self,
        user_input: str,
        memory: Memory,
        custom_commands: dict[str, tuple[str, str, str]],
    ) -> str:
        """Processes an interactive user input line.

        Returns:
            "exit" to terminate session,
            "turn" if an agent turn should be executed,
            "skip" if command was handled internally without calling the model.
        """
        cmd = user_input.lower().strip()
        if cmd in ("/exit", "exit", "/quit", "quit"):
            print("Ending session.")
            return "exit"

        if cmd in ("/help", "help"):
            self._print_interactive_help(custom_commands)
            return "skip"

        if cmd in ("/status", "/context", "status", "context"):
            self._print_context_status(memory)
            return "skip"

        if cmd.startswith("/cache") or cmd == "cache":
            self._handle_cache_command(user_input)
            return "skip"

        if cmd in ("/compact", "compact"):
            old_chars = memory.total_chars()
            did_compact = memory.compact_if_needed(force=True)
            new_chars = memory.total_chars()
            if did_compact:
                print(
                    f"\n🧹 Compaction applied: freed {old_chars - new_chars} characters ({old_chars} -> {new_chars} chars).\n"
                )
            else:
                print("\nℹ️ No older observations eligible for compaction at this time.\n")
            return "skip"

        if cmd in ("/new", "/reset", "new", "reset"):
            memory.reset()
            print("\n✨ Started a fresh conversation context (workspace files preserved).\n")
            return "skip"

        if user_input.startswith("/"):
            cmd_part, _, args_part = user_input[1:].partition(" ")
            cmd_key = cmd_part.lower().strip()
            if cmd_key in custom_commands:
                _, template, scope = custom_commands[cmd_key]
                expanded_prompt = _expand_command(template, args_part)
                print(f"\n⚡ Executing /{cmd_key} [{scope}] ...\n")
                memory.add_user(expanded_prompt)
                return "turn"
            print(f"\n❌ Unknown command '/{cmd_key}'. Type /help to see available commands.\n")
            return "skip"

        memory.add_user(user_input)
        return "turn"

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
        if self.docker:
            print(f"🐳 Docker Sandbox: ACTIVE ({self.docker_image})")
        cache_label = self.cache_ttl.upper() if self.cache_ttl else "OFF"
        print(f"⚡ Prompt Caching: {cache_label}")
        print("💡 Commands: /cache [5m|1h|off] | /compact | /status | /new | /help | /exit\n")

        step_idx = 1
        has_pending_turn = True
        max_steps_per_turn = 30

        try:
            while True:
                # If we have an active instruction to execute, run the agent loop
                if has_pending_turn:
                    is_finished, reason, step_idx = self._execute_steps(
                        loop=loop,
                        tracer=tracer,
                        start_step=step_idx,
                        max_steps=max_steps_per_turn,
                        on_finish=lambda reply: print(f"\n🤖 Agent:\n{reply.strip()}\n"),
                    )
                    if not is_finished and reason == "max_steps_exceeded":
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

                action = self._dispatch_interactive_input(user_input, memory, custom_commands)
                if action == "exit":
                    break
                elif action == "turn":
                    has_pending_turn = True

            patch_diff = env.get_git_patch()
            if patch_diff:
                patch_file = run_dir / "patch.diff"
                patch_file.write_text(patch_diff, encoding="utf-8")
                print(f"📦 Git patch saved to: {patch_file}")

            json_path, md_path = tracer.finalize(
                finish_reason="session_ended", patch_diff=patch_diff
            )
            print(f"\n📄 Complete session trace saved to:\n   {md_path}\n")
        finally:
            env.teardown()
