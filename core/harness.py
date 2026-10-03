from __future__ import annotations

import sys
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

from core.commands import (
    Command,
    SlashOutcome,
    discover_commands,
    expand_command,
    resolve_slash_input,
    split_slash_input,
)
from core.config import (
    PROJECT_TRUST_NOTICE,
    ConfigLayers,
    HarnessConfig,
    get_default_global_dir,
    load_global_config,
    load_project_config,
    normalize_cache_ttl,
    normalize_reasoning_effort,
    resolve_project_tool_trust,
)
from core.discovery import discover_workspace_files, expand_tagged_files
from core.environment import DockerEnvironment, Environment
from core.evaluator import Evaluator
from core.loop import AgentLoop
from core.memory import Memory
from core.model import (
    CacheAwareModel,
    CancellableModel,
    DeadlineAwareModel,
    Model,
    ModelCancelledError,
    NamedModel,
    ProviderAwareModel,
    ReasoningAwareModel,
)
from core.prompts import DEFAULT_SYSTEM_PROMPT, build_system_prompt, load_agents_instructions
from core.providers import (
    build_model_from_spec,
    get_available_models_from_config,
    resolve_model_spec,
)
from core.session import (
    find_sessions,
    load_session,
    rate_limit_wait,
    run_steps,
)
from core.skills import (
    Skill,
    SkillCommandResult,
    compose_skill_prompt,
    discover_skills,
    extract_skill_summary,
    render_skills_index,
    resolve_skill_command,
)
from core.tools import (
    BashTool,
    EditFileTool,
    ReadFileTool,
    Tool,
    ToolRegistry,
    WriteFileTool,
    discover_tool_plugins,
)
from core.tracer import ActionFormatter, Tracer, _dim
from core.types import EvalResult, Message, RunResult, Task
from core.usage import summarize_run_usage

__all__ = [
    "DEFAULT_GLOBAL_DIR",
    "Command",
    "Harness",
    "Skill",
    "SkillCommandResult",
    "SlashOutcome",
    "compose_skill_prompt",
    "discover_commands",
    "discover_skills",
    "expand_command",
    "get_default_global_dir",
    "load_global_config",
    "load_project_config",
    "render_skills_index",
    "resolve_skill_command",
    "resolve_slash_input",
    "split_slash_input",
]


DEFAULT_GLOBAL_DIR = get_default_global_dir()


SKILL_COMMAND_ALIASES = ("/skill", "/skills", "skill", "skills")
SKILL_COMMAND_PREFIXES = ("/skill ", "/skills ")

# Skill and command discovery moved to core/skills.py and core/commands.py. These
# aliases keep the historical import paths working for the CLI and tests.
_discover_skills = discover_skills
_discover_skills_index = render_skills_index
_extract_skill_summary = extract_skill_summary
_discover_commands = discover_commands
_expand_command = expand_command
_resolve_skill_command = resolve_skill_command


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
        custom_tools: list[Tool] | None = None,
        extra_tools: list[Tool] | None = None,
        auto_discover_tools: bool | None = None,
        disable_default_tools: bool | None = None,
        trust_project_tools: bool | None = None,
        evaluator: Evaluator | None = None,
        runs_dir: Path | str | None = None,
        max_tool_chars: int | None = None,
        max_context_chars: int | None = None,
        trigger_ratio: float | None = None,
        cache_ttl: str | None = None,
        reasoning_effort: str | None = None,
        max_step_retries: int | None = None,
        step_retry_delay: float | None = None,
        action_formatters: dict[str, ActionFormatter] | None = None,
        verbose: bool = True,
    ):
        self.model = model
        self.explicit_workspace_dir = Path(workspace_dir).resolve() if workspace_dir else None
        self.global_dir = (
            Path(global_dir).resolve() if global_dir is not None else get_default_global_dir()
        )
        self.layers = ConfigLayers.load(self.global_dir, self.explicit_workspace_dir)
        self.global_config = self.layers.global_config
        self.project_config = self.layers.project_config
        self.merged_config = self.layers.merged

        cfg = HarnessConfig(
            sandbox=sandbox,
            docker=docker,
            docker_image=docker_image,
            auto_discover_tools=auto_discover_tools,
            disable_default_tools=disable_default_tools,
            runs_dir=runs_dir,
            max_tool_chars=max_tool_chars,
            max_context_chars=max_context_chars,
            trigger_ratio=trigger_ratio,
            cache_ttl=cache_ttl,
            reasoning_effort=reasoning_effort,
            max_step_retries=max_step_retries,
            step_retry_delay=step_retry_delay,
        ).resolve(self.layers)

        self.sandbox = cfg.sandbox
        self.docker = cfg.docker
        self.docker_image = cfg.docker_image
        self.auto_discover_tools = cfg.auto_discover_tools
        self.disable_default_tools = cfg.disable_default_tools
        self.runs_dir = cfg.runs_dir.resolve() if cfg.runs_dir else (self.global_dir / "runs").resolve()
        self.max_tool_chars = cfg.max_tool_chars
        self.max_context_chars = cfg.max_context_chars
        self.trigger_ratio = cfg.trigger_ratio
        self.cache_ttl = cfg.cache_ttl
        self.reasoning_effort = cfg.reasoning_effort
        self.max_step_retries = cfg.max_step_retries
        self.step_retry_delay = cfg.step_retry_delay

        self.trust_project_tools, project_claim_ignored = resolve_project_tool_trust(
            trust_project_tools, self.layers
        )
        if project_claim_ignored:
            print(f"⚠️ Notice: {PROJECT_TRUST_NOTICE}", file=sys.stderr)

        if isinstance(self.model, CacheAwareModel):
            self.model.set_cache_ttl(self.cache_ttl)
        if isinstance(self.model, ReasoningAwareModel):
            self.model.set_reasoning_effort(self.reasoning_effort)

        self.custom_tools = tools if tools is not None else custom_tools
        self.extra_tools = list(extra_tools or [])
        self.evaluator = evaluator
        self.action_formatters = action_formatters
        self.verbose = verbose

        from core.model import load_provider_registry

        self.provider_registry = load_provider_registry(
            global_dir=self.global_dir,
            project_dir=self.explicit_workspace_dir,
            inline_providers=self.merged_config.get("providers"),
        )
        self.active_loop: AgentLoop | None = None

    def _evaluate(self, env: Environment, tracer: Tracer) -> EvalResult | None:
        """Grades a finished run when an evaluator is configured."""
        if not self.evaluator:
            return None
        if self.verbose:
            print(f"\n{'=' * 20} Running Evaluation {'=' * 20}")
        result = self.evaluator.evaluate(env, tracer.steps)
        if self.verbose:
            emoji = "✅" if result.passed else "❌"
            print(f"{emoji} Evaluation Passed: {result.passed} (Score: {result.score})")
            print(f"Feedback: {result.feedback.strip()}")
        return result

    def _write_patch(self, patch_diff: str, run_dir: Path) -> str | None:
        """Persists the workspace diff next to the trace, returning its path."""
        if not patch_diff:
            return None
        patch_file = run_dir / "patch.diff"
        patch_file.write_text(patch_diff, encoding="utf-8")
        if self.verbose:
            print(f"📦 Git patch saved to: {patch_file}")
        return str(patch_file)

    def _rate_limit_wait(self) -> float:
        """Cumulative seconds this model has spent honouring provider rate limits."""
        return rate_limit_wait(self.model)

    @property
    def workspace_dir(self) -> Path:
        return self.explicit_workspace_dir if self.explicit_workspace_dir is not None else Path.cwd()

    def set_cache_ttl(self, ttl: str | None) -> str | None:
        """Dynamically switches prompt caching duration on the fly ('5m', '1h', or 'off'/None)."""
        self.cache_ttl = normalize_cache_ttl(ttl)
        if isinstance(self.model, CacheAwareModel):
            return self.model.set_cache_ttl(self.cache_ttl)
        return self.cache_ttl

    def get_cache_stats(self) -> dict[str, Any]:
        """Returns prompt cache TTL and token performance metrics."""
        if isinstance(self.model, CacheAwareModel):
            stats = self.model.get_cache_stats()
            if isinstance(stats, dict):
                return stats
        return {"cache_ttl": self.cache_ttl}

    def set_reasoning_effort(self, effort: str | None) -> str | None:
        """Dynamically switches reasoning effort level ('low', 'medium', 'high', numeric budget, or 'off'/None)."""
        self.reasoning_effort = normalize_reasoning_effort(effort)
        if isinstance(self.model, ReasoningAwareModel):
            return self.model.set_reasoning_effort(self.reasoning_effort)
        return self.reasoning_effort

    def get_reasoning_effort(self) -> str | None:
        """Returns the current reasoning effort level or None."""
        if isinstance(self.model, ReasoningAwareModel):
            effort = self.model.get_reasoning_effort()
            if isinstance(effort, str) or effort is None:
                return effort
        return getattr(self, "reasoning_effort", None)

    def get_available_models(self) -> list[dict[str, Any]]:
        """Returns configured and available models for dynamic /model switching."""
        return get_available_models_from_config(self.merged_config, self.provider_registry)

    def active_model_name(self) -> str:
        """Best-effort display name for the model currently in use."""
        return getattr(self.model, "model", "") or getattr(self.model, "model_name", "") or ""

    def resolve_model_spec(self, spec: str) -> tuple[bool, str, str]:
        """Resolves a user-supplied model name to a canonical spec.

        Accepts a configured alias from ``models``, the underlying model id of
        one, or a provider name (``openai`` / ``openai/gpt-4o``) present in the
        provider registry. Returns ``(ok, canonical_spec, error)``.
        """
        return resolve_model_spec(spec, self.get_available_models(), self.provider_registry)

    def _canonical_model_spec(
        self, spec: str | dict[str, Any]
    ) -> tuple[bool, str | dict[str, Any], str]:
        """Turns a /model argument (name or 1-based index) into a buildable spec."""
        if not spec:
            return False, None, "Model specification cannot be empty."
        if not isinstance(spec, (str, dict)):
            return (
                False,
                None,
                f"Expected model spec as str or dict, got {type(spec).__name__}",
            )

        if isinstance(spec, dict):
            return True, spec, ""

        clean = spec.strip()
        if not clean:
            return False, None, "Model specification cannot be empty."

        available = self.get_available_models()
        if clean.isdigit():
            index = int(clean) - 1
            if 0 <= index < len(available):
                return True, available[index]["name"], ""
            return (
                False,
                None,
                f"Invalid model selection: index {spec} out of range (1-{len(available)})",
            )

        ok, canonical, error = self.resolve_model_spec(clean)
        return (True, canonical, "") if ok else (False, None, error)

    def switch_model(self, spec: str | dict[str, Any]) -> tuple[bool, str, Model]:
        """Switches active model dynamically using config and provider registry."""
        ok, target_spec, error = self._canonical_model_spec(spec)
        if not ok or target_spec is None:
            return False, error, self.model

        try:
            new_model = build_model_from_spec(
                spec=target_spec,
                config=self.merged_config,
                registry=self.provider_registry,
                cache_ttl=self.cache_ttl,
                reasoning_effort=self.reasoning_effort,
            )
        except Exception as exc:
            return False, f"Failed to switch model to '{spec}': {exc}", self.model

        self.model = new_model
        if self.active_loop is not None:
            self.active_loop.model = new_model

        provider = getattr(new_model, "provider", "") or ""
        suffix = f" [{provider}]" if provider else ""
        name = self.active_model_name() or str(target_spec)
        return True, f"Switched model to {name}{suffix}", new_model

    def find_recent_sessions(self, limit: int = 10) -> list[dict[str, Any]]:
        """Finds recent runs in runs_dir with workspace affinity."""
        return find_sessions(self.runs_dir, self.explicit_workspace_dir, limit=limit)

    def load_session(
        self, session_id_or_path: str | Path | None = None
    ) -> tuple[list[Message], dict[str, Any]]:
        """Loads previous session messages and metadata by id, index, path, or recency."""
        selector = None if session_id_or_path is True else session_id_or_path
        return load_session(self.runs_dir, self.explicit_workspace_dir, selector)

    def _create_environment(
        self,
        workspace: Path,
        logs_dir: Path,
        active_docker: bool,
        active_image: str,
        init_workdir: str | None,
        initial_files: dict[str, str],
    ) -> Environment:
        allowed_reads: list[Path] = [logs_dir]
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

        # 1. Base tools (unless disabled or explicit custom_tools override)
        if not self.disable_default_tools and not self.custom_tools:
            registry.register(BashTool(env))
            registry.register(ReadFileTool(env))
            registry.register(WriteFileTool(env))
            registry.register(EditFileTool(env))
        elif self.custom_tools:
            for t in self.custom_tools:
                registry.register(t)

        # 2. Additive extra_tools explicitly provided
        for t in self.extra_tools:
            registry.register(t)

        # 3. Dynamic plugins auto-discovered from global and project directories
        if self.auto_discover_tools:
            ws_dir = self.explicit_workspace_dir or env.workspace_dir
            discovered = discover_tool_plugins(
                workspace_dir=ws_dir,
                global_dir=self.global_dir,
                env=env,
                trust_project_tools=self.trust_project_tools,
            )
            for t in discovered:
                if t.name in registry:
                    print(
                        f"⚠️ Warning: Discovered tool plugin '{t.name}' overrides existing tool '{t.name}'.",
                        file=sys.stderr,
                    )
                registry.register(t)

        return registry

    def _build_system_prompt(
        self,
        workspace: Path,
        base_prompt: str,
        tool_registry: ToolRegistry,
        use_sandbox: bool,
    ) -> str:
        """Assembles the system prompt from tools, steering, and the skills index."""
        return build_system_prompt(
            base_prompt,
            tool_docs=tool_registry.to_system_prompt_snippet(),
            workspace=workspace,
            sandboxed=use_sandbox,
            agents_instructions=load_agents_instructions(workspace, self.global_dir),
            skills_index=render_skills_index(discover_skills(workspace, self.global_dir)),
        )

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
        run_id = f"{task_id}_{timestamp}_{uuid.uuid4().hex[:6]}"
        run_dir = self.runs_dir / run_id
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

        # A batch instruction may itself be a custom command, e.g. "/review src/auth.py".
        effective_instruction = initial_instruction
        if initial_instruction.startswith("/"):
            outcome = resolve_slash_input(
                initial_instruction,
                discover_commands(workspace, self.global_dir),
                allow_skills=False,
            )
            if outcome.prompt is not None:
                effective_instruction = outcome.prompt
                if self.verbose:
                    print(f"⚡ {outcome.message}")

        memory = Memory(
            system_prompt=active_system_prompt,
            user_prompt=effective_instruction,
            max_context_chars=self.max_context_chars,
            trigger_ratio=self.trigger_ratio,
        )
        task = Task(id=task_id, instruction=effective_instruction, initial_files=initial_files)
        tracer = Tracer(
            task=task,
            output_dir=run_dir,
            workspace=workspace,
            verbose=self.verbose,
            formatters=self.action_formatters,
        )
        loop = AgentLoop(model=self.model, memory=memory, tools=tool_registry)
        self.active_loop = loop

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
        """Drives the agent loop for up to max_steps with retries."""
        outcome = run_steps(
            self.model,
            loop,
            tracer,
            start_step,
            max_steps,
            deadline=deadline,
            max_retries=self.max_step_retries,
            retry_delay=self.step_retry_delay,
            on_finish=on_finish,
        )
        return outcome.finished, outcome.reason, outcome.next_step

    def run(self, task: Task) -> RunResult:
        """One-shot batch execution mode (e.g. for benchmarks or tests)."""
        run_start = time.time()
        start_rl_wait = self._rate_limit_wait()
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

        task_deadline = (run_start + task.timeout_seconds) if task.timeout_seconds else None
        if isinstance(self.model, DeadlineAwareModel):
            self.model.set_deadline(task_deadline)

        try:
            is_finished, finish_reason, _ = self._execute_steps(
                loop=loop,
                tracer=tracer,
                start_step=1,
                max_steps=task.max_steps,
                deadline=task_deadline,
            )

            total_duration = time.time() - run_start
            total_steps = len(tracer.steps)

            eval_result = self._evaluate(env, tracer)
            patch_diff = env.get_git_patch()
            patch_path = self._write_patch(patch_diff, run_dir)

            c_stats = self.get_cache_stats()
            if not isinstance(c_stats, dict):
                c_stats = {"cache_ttl": self.cache_ttl}
            json_trace_path, md_trace_path = tracer.finalize(
                finish_reason=finish_reason,
                eval_result=eval_result,
                patch_diff=patch_diff,
                telemetry=c_stats,
            )

            passed = eval_result.passed if eval_result else (finish_reason == "completed")
            score = eval_result.score if eval_result else (1.0 if passed else 0.0)

            usage = summarize_run_usage(
                (step.usage for step in tracer.steps), c_stats
            )
            task_rl_wait = max(0.0, self._rate_limit_wait() - start_rl_wait)


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
                prompt_tokens=usage.prompt_tokens,
                completion_tokens=usage.completion_tokens,
                cached_tokens=usage.cached_tokens,
                cache_hit_rate_pct=usage.cache_hit_rate_pct,
                rate_limit_wait_s=task_rl_wait,
            )
        finally:
            if isinstance(self.model, DeadlineAwareModel):
                self.model.set_deadline(None)
            env.teardown()

    def _print_interactive_help(
        self,
        custom_commands: dict[str, Command],
        skills: dict[str, Skill] | None = None,
    ) -> None:
        print("\nAvailable commands:")
        print("  /model [name|idx]             - View configured models or switch active model")
        print("  /resume [id|idx]              - View or resume a previous conversation session")
        print("  /reasoning [low|med|high|off] - View or switch reasoning / thinking effort")
        print("  /cache [5m|1h|off]            - View or switch prompt caching duration on the fly")
        print("  /status                       - Show context window size, message breakdown, and budget")
        print("  /compact                      - Manually collapse older observations to free context")
        print("  /new                          - Clear conversation history while keeping workspace files")
        print("  /skill [name]                 - Show available skills or activate one into conversation")
        print("  /exit                         - Save session trace and exit")
        if custom_commands:
            print("\nCustom slash commands (from commands/):")
            for name, command in sorted(custom_commands.items()):
                print(f"  /{name:<10} - {command.badge}{command.description}")
        if skills:
            print("\nAvailable skills (run /<name> or /skill <name>):")
            for name, skill in sorted(skills.items()):
                print(f"  /{name:<10} - {skill.badge}{skill.summary}")
        print()

    def _add_prompt(self, memory: Memory, prompt: str, workspace: Path) -> None:
        """Expands ``@file`` tags in a prompt and appends it as the next user message."""
        effective_prompt, attached = expand_tagged_files(prompt, workspace)
        if attached:
            names = ", ".join(f"@{name}" for name in attached)
            print(f"⚡ Attached {len(attached)} file(s): {names}")
        memory.add_user(effective_prompt)

    def _print_context_status(self, memory: Memory) -> None:
        s = memory.stats()
        print("\n📊 Context Breakdown:")
        print(
            f"  • Usage: {s['estimated_tokens']:,} / {s['max_estimated_tokens']:,} tokens ({s['usage_pct']}%)"
        )
        print(
            f"  • Messages ({s['total_messages']}): {s['by_role']['system']} sys, "
            f"{s['by_role']['user']} user, {s['by_role']['assistant']} assistant, {s['by_role']['tool']} tools"
        )
        curr_reasoning = self.get_reasoning_effort()
        print(f"  • Reasoning Effort: {curr_reasoning.upper() if curr_reasoning else 'OFF'}")
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

    def _handle_reasoning_command(self, user_input: str) -> None:
        parts = user_input.strip().split()
        if len(parts) == 1:
            curr = self.get_reasoning_effort()
            status_str = curr.upper() if curr else "OFF (Disabled)"
            print(f"\n🧠 Reasoning / Thinking Effort: {status_str}")
            print(
                "  • Usage: /reasoning low | /reasoning medium | /reasoning high | /reasoning off | /reasoning <tokens>\n"
            )
            return

        target = parts[1].strip()
        try:
            new_effort = self.set_reasoning_effort(target)
            if new_effort:
                print(
                    f"\n🧠 Reasoning effort switched to: {new_effort.upper()} for subsequent turns.\n"
                )
            else:
                print("\n🧠 Reasoning effort disabled (off) for subsequent turns.\n")
        except ValueError as exc:
            print(f"\n❌ {exc}\n")

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
        custom_commands: dict[str, Command],
        skills: dict[str, Skill] | None = None,
        workspace_dir: Path | None = None,
        tracer: Tracer | None = None,
        step_holder: list[int] | None = None,
    ) -> str:
        """Processes an interactive user input line.

        Returns:
            "exit" to terminate session,
            "turn" if an agent turn should be executed,
            "skip" if command was handled internally without calling the model.
        """
        ws = workspace_dir or self.workspace_dir
        cmd = user_input.lower().strip()
        if cmd in ("/exit", "exit", "/quit", "quit"):
            print("Ending session.")
            return "exit"

        if cmd in ("/help", "help"):
            self._print_interactive_help(custom_commands, skills)
            return "skip"

        if cmd in ("/status", "/context", "status", "context"):
            self._print_context_status(memory)
            return "skip"

        if cmd.startswith("/reasoning") or cmd == "reasoning":
            self._handle_reasoning_command(user_input)
            return "skip"

        if cmd.startswith("/cache") or cmd == "cache":
            self._handle_cache_command(user_input)
            return "skip"

        if cmd in ("/model", "/models", "model", "models") or cmd.startswith(
            ("/model ", "/models ")
        ):
            arg = ""
            if " " in user_input.strip():
                arg = user_input.strip().split(" ", 1)[1].strip()
            if not arg:
                models = self.get_available_models()
                curr_model = (
                    getattr(self.model, "model", "")
                    or getattr(self.model, "model_name", "")
                    or ""
                )
                print("\n🤖 Available Models (from config):")
                if not models:
                    print(f"  • {curr_model} [active]")
                else:
                    for i, m in enumerate(models, 1):
                        is_active = (
                            m["name"].lower() == curr_model.lower()
                            or m.get("model", "").lower() == curr_model.lower()
                        )
                        active_lbl = " [active]" if is_active else ""
                        prov = f"[{m['provider']}] " if m.get("provider") else ""
                        print(f"  {i}. {m['name']}{active_lbl} - {prov}{m.get('description', '')}")
                print("\nTip: Type /model <name|number> to switch active model.\n")
                return "skip"

            ok, msg, _ = self.switch_model(arg)
            if ok:
                print(f"\n✅ {msg}\n")
            else:
                print(f"\n❌ {msg}\n")
            return "skip"

        if cmd in ("/resume", "/resumes", "resume", "resumes") or cmd.startswith(
            ("/resume ", "/resumes ")
        ):
            arg = ""
            if " " in user_input.strip():
                arg = user_input.strip().split(" ", 1)[1].strip()
            if not arg:
                sessions = self.find_recent_sessions(limit=8)
                print("\n📂 Recent Sessions:")
                if not sessions:
                    print("  No previous sessions found in runs directory.")
                else:
                    for i, s in enumerate(sessions, 1):
                        ws_badge = " [this repo]" if s.get("matches_workspace") else ""
                        print(f"  {i}. {s['run_id']}{ws_badge} ({s['total_steps']} steps) - {s['preview']}")
                print("\nTip: Type /resume <number|session_id> to load a previous session.\n")
                return "skip"

            messages, meta = self.load_session(arg)
            if not messages:
                err = meta.get("error", f"Could not load session '{arg}'")
                print(f"\n❌ {err}\n")
                return "skip"

            memory.restore_messages(messages, refresh_system_prompt=memory.system_prompt)
            if tracer is not None:
                tracer.steps.clear()
            if step_holder is not None:
                step_holder[0] = meta.get("total_steps", 0) + 1
            print(f"\n🔄 Resumed session '{meta.get('run_id')}' ({len(messages)} messages loaded).\n")
            return "skip"

        if cmd in ("/compact", "compact"):
            old_chars = memory.total_chars()
            did_compact = memory.compact_if_needed(force=True)
            new_chars = memory.total_chars()
            if did_compact:
                old_tokens = old_chars // 4
                new_tokens = new_chars // 4
                freed_tokens = old_tokens - new_tokens
                print(
                    f"\n🧹 Compaction applied: freed ~{freed_tokens:,} tokens (~{old_tokens:,} -> ~{new_tokens:,} tokens).\n"
                )
            else:
                print("\nℹ️ No older observations eligible for compaction at this time.\n")
            return "skip"

        if cmd in ("/new", "/reset", "new", "reset"):
            memory.reset()
            print("\n✨ Started a fresh conversation context (workspace files preserved).\n")
            return "skip"

        if cmd in SKILL_COMMAND_ALIASES or cmd.startswith(SKILL_COMMAND_PREFIXES):
            outcome = resolve_skill_command(user_input, skills or {})
            if outcome.is_error:
                print(f"\n❌ {outcome.message}\n")
                return "skip"
            if outcome.action == "skip":
                print(f"\n{outcome.message}\n")
                return "skip"
            print(f"\n⚡ {outcome.message} ...\n")
            self._add_prompt(memory, outcome.prompt or "", ws)
            return "turn"

        if user_input.startswith("/"):
            outcome = resolve_slash_input(user_input, custom_commands, skills)
            if outcome.is_error or outcome.prompt is None:
                print(f"\n❌ {outcome.message}\n")
                return "skip"
            print(f"\n⚡ {outcome.message} ...\n")
            self._add_prompt(memory, outcome.prompt, ws)
            return "turn"

        effective_prompt, attached = expand_tagged_files(user_input, ws)
        if attached:
            print(f"\n⚡ Attached {len(attached)} file(s): " + ", ".join(f"@{f}" for f in attached) + "\n")
        memory.add_user(effective_prompt)
        return "turn"

    def run_interactive(
        self,
        initial_instruction: str | None = None,
        task_id: str = "interactive_session",
        resume: bool | str | Path | None = None,
    ) -> None:
        """Interactive multi-turn session with slash commands and context usage tracking."""
        env, _, memory, tracer, loop, run_dir = self._setup_run(
            task_id=task_id,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            initial_instruction="",
            initial_files={},
        )

        step_idx = 1
        if resume:
            res_target = None if resume is True else resume
            messages, meta = self.load_session(res_target)
            if messages:
                memory.restore_messages(messages, refresh_system_prompt=memory.system_prompt)
                step_idx = meta.get("total_steps", 0) + 1
                print(
                    f"\n🔄 Resumed session '{meta.get('run_id')}' ({len(messages)} messages loaded, continuing at step {step_idx}).\n"
                )
            else:
                print(
                    f"\n⚠️ Could not resume session ({meta.get('error', 'not found')}). Starting fresh session.\n"
                )

        custom_commands = _discover_commands(env.workspace_dir, self.global_dir)
        skills = _discover_skills(env.workspace_dir, self.global_dir)
        tracer.compact = True

        print("\n🚀 Interactive Session Started")
        print(f"📁 Project Workspace: {env.workspace_dir}")
        if self.docker:
            print(f"🐳 Docker Sandbox: ACTIVE ({self.docker_image})")
        curr_model = getattr(self.model, "model", "") or getattr(self.model, "model_name", "") or ""
        curr_prov = getattr(self.model, "provider", "") or ""
        prov_lbl = f" [{curr_prov}]" if curr_prov else ""
        print(f"🤖 Active Model: {curr_model}{prov_lbl}")
        cache_label = self.cache_ttl.upper() if self.cache_ttl else "OFF"
        print(f"⚡ Prompt Caching: {cache_label}")
        reasoning_label = self.get_reasoning_effort()
        print(f"🧠 Reasoning Effort: {reasoning_label.upper() if reasoning_label else 'OFF'}")
        print(
            "💡 Commands: /model | /resume | /reasoning | /cache | /compact | /status | /new | /skill | /help | /exit\n"
        )

        has_pending_turn = False
        max_steps_per_turn = 30

        step_idx_holder = [step_idx]

        if initial_instruction and initial_instruction.strip():
            action = self._dispatch_interactive_input(
                initial_instruction.strip(),
                memory,
                custom_commands,
                skills,
                tracer=tracer,
                step_holder=step_idx_holder,
            )
            step_idx = step_idx_holder[0]
            if action == "turn":
                has_pending_turn = True
            elif action == "exit":
                env.teardown()
                return

        try:
            while True:
                # If we have an active instruction to execute, run the agent loop
                if has_pending_turn:
                    is_finished, reason, step_idx = self._execute_steps(
                        loop=loop,
                        tracer=tracer,
                        start_step=step_idx,
                        max_steps=max_steps_per_turn,
                    )
                    step_idx_holder[0] = step_idx
                    if not is_finished and reason == "max_steps_exceeded":
                        print(
                            f"\n⚠️ Reached safety limit of {max_steps_per_turn} steps for this turn. Returning to prompt.\n"
                        )
                    has_pending_turn = False

                # Render minimal context usage footer in pure tokens
                stats = memory.stats()
                print(
                    _dim(
                        f"─── 📊 Context: {stats['estimated_tokens']:,} / {stats['max_estimated_tokens']:,} tokens "
                        f"({stats['usage_pct']}%) | Messages: {stats['total_messages']} ───"
                    )
                )

                try:
                    user_input = input("👤 You: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nSession interrupted.")
                    break

                if not user_input:
                    continue

                action = self._dispatch_interactive_input(
                    user_input,
                    memory,
                    custom_commands,
                    skills,
                    tracer=tracer,
                    step_holder=step_idx_holder,
                )
                step_idx = step_idx_holder[0]
                if action == "exit":
                    break
                elif action == "turn":
                    has_pending_turn = True

            patch_diff = env.get_git_patch()
            if patch_diff:
                patch_file = run_dir / "patch.diff"
                patch_file.write_text(patch_diff, encoding="utf-8")
                print(f"📦 Git patch saved to: {patch_file}")

            c_stats = self.get_cache_stats()
            json_path, md_path = tracer.finalize(
                finish_reason="session_ended", patch_diff=patch_diff, telemetry=c_stats
            )
            print(f"\n📄 Complete session trace saved to:\n   {md_path}\n")
        finally:
            env.teardown()

    def run_tui(
        self,
        initial_instruction: str | None = None,
        task_id: str = "tui_session",
        resume: bool | str | Path | None = None,
    ) -> None:
        """Full-screen interactive TUI with bottom-anchored chat and scrollable activity."""
        from core.tracer import _format_action_summary, _pair_calls_with_results
        from core.tui import CursesTUI, TUIState, compact_path, get_git_branch

        env, _, memory, tracer, loop, run_dir = self._setup_run(
            task_id=task_id,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            initial_instruction="",
            initial_files={},
        )
        tracer.verbose = False
        tracer.compact = True

        custom_commands = _discover_commands(env.workspace_dir, self.global_dir)
        skills = _discover_skills(env.workspace_dir, self.global_dir)

        state = TUIState(title="microharness")
        if isinstance(self.model, ProviderAwareModel):
            state.provider_name = self.model.provider or ""
        else:
            state.provider_name = getattr(self.model, "provider", "") or ""

        if isinstance(self.model, NamedModel):
            state.model_name = self.model.model or ""
        else:
            state.model_name = getattr(self.model, "model", "") or getattr(self.model, "model_name", "") or ""

        state.cache_ttl = self.cache_ttl.upper() if self.cache_ttl else ""
        curr_reasoning = self.get_reasoning_effort()
        state.reasoning = curr_reasoning.upper() if curr_reasoning else ""
        state.cwd = compact_path(str(env.workspace_dir))
        state.git_branch = get_git_branch(env.workspace_dir)
        state.set_custom_commands(custom_commands)
        state.set_skills(skills)
        state.set_workspace_files(discover_workspace_files(env.workspace_dir))
        state.set_available_models(self.get_available_models())
        state.set_available_sessions(self.find_recent_sessions(limit=15))

        step_idx = 1
        if resume:
            res_target = None if resume is True else resume
            res_msgs, res_meta = self.load_session(res_target)
            if res_msgs:
                memory.restore_messages(res_msgs, refresh_system_prompt=memory.system_prompt)
                step_idx = res_meta.get("total_steps", 0) + 1
                for m in res_msgs:
                    if m.role == "user" and m.content:
                        state.add_user(m.content)
                    elif m.role == "assistant":
                        if m.content:
                            state.add_agent(m.content)
                        if m.tool_calls:
                            for tc in m.tool_calls:
                                act = _format_action_summary(tc.name, tc.arguments, color=False)
                                state.add_action(act)
                state.add_system(
                    f"🔄 Resumed session '{res_meta.get('run_id')}' ({len(res_msgs)} messages loaded, continuing at step {step_idx})."
                )
            else:
                state.add_system(
                    f"⚠️ Could not resume session ({res_meta.get('error', 'not found')}). Starting fresh."
                )

        state.update_status(memory.stats(), self.get_cache_stats())

        max_steps_per_turn = 30
        tui = CursesTUI(state)

        def on_user_submit(user_text: str) -> None:
            nonlocal step_idx

            cmd = user_text.lower().strip()
            if cmd in ("/help", "help"):
                state.add_system("Available commands: /help, /model, /resume, /status, /compact, /new, /skill, /exit")
                state.add_system("Press Esc on an empty prompt to exit; Esc clears text or dismisses menus.")
                if custom_commands:
                    state.add_system("Custom commands: " + ", ".join(f"/{c}" for c in sorted(custom_commands.keys())))
                if skills:
                    state.add_system("Available skills (run /<name>): " + ", ".join(f"/{s}" for s in sorted(skills.keys())))
                return

            if cmd in ("/model", "/models", "model", "models") or cmd.startswith(
                ("/model ", "/models ")
            ):
                arg = ""
                if " " in user_text.strip():
                    arg = user_text.strip().split(" ", 1)[1].strip()
                if not arg:
                    models = self.get_available_models()
                    curr_model = (
                        getattr(self.model, "model", "")
                        or getattr(self.model, "model_name", "")
                        or ""
                    )
                    state.add_system("🤖 Available Models (from config):")
                    for i, m in enumerate(models, 1):
                        is_active = (
                            m["name"].lower() == curr_model.lower()
                            or m.get("model", "").lower() == curr_model.lower()
                        )
                        active_lbl = " [active]" if is_active else ""
                        prov = f"[{m['provider']}] " if m.get("provider") else ""
                        state.add_system(f"  {i}. {m['name']}{active_lbl} - {prov}{m.get('description', '')}")
                    state.add_system("Type /model <name|number> to switch active model.")
                    return

                ok, msg, new_m = self.switch_model(arg)
                if ok:
                    loop.model = new_m
                    state.model_name = (
                        getattr(new_m, "model", "")
                        or getattr(new_m, "model_name", "")
                        or ""
                    )
                    state.provider_name = getattr(new_m, "provider", "") or ""
                    state.add_system(f"✅ {msg}")
                    state.set_available_models(self.get_available_models())
                    state.update_status(memory.stats(), self.get_cache_stats())
                else:
                    state.add_error(f"Failed to switch model: {msg}")
                return

            if cmd in ("/resume", "/resumes", "resume", "resumes") or cmd.startswith(
                ("/resume ", "/resumes ")
            ):
                arg = ""
                if " " in user_text.strip():
                    arg = user_text.strip().split(" ", 1)[1].strip()
                if not arg:
                    sessions = self.find_recent_sessions(limit=8)
                    state.add_system("📂 Recent Sessions:")
                    if not sessions:
                        state.add_system("  No previous sessions found in runs directory.")
                    else:
                        for i, s in enumerate(sessions, 1):
                            ws_badge = " [this repo]" if s.get("matches_workspace") else ""
                            state.add_system(f"  {i}. {s['run_id']}{ws_badge} ({s['total_steps']} steps) - {s['preview']}")
                    state.add_system("Type /resume <number|session_id> to load a previous session.")
                    return

                messages, meta = self.load_session(arg)
                if not messages:
                    err = meta.get("error", f"Could not load session '{arg}'")
                    state.add_error(f"Failed to resume session: {err}")
                    return

                memory.restore_messages(messages, refresh_system_prompt=memory.system_prompt)
                step_idx = meta.get("total_steps", 0) + 1
                tracer.steps.clear()
                state.lines.clear()
                for m in messages:
                    if m.role == "user" and m.content:
                        state.add_user(m.content)
                    elif m.role == "assistant":
                        if m.content:
                            state.add_agent(m.content)
                        if m.tool_calls:
                            for tc in m.tool_calls:
                                act = _format_action_summary(tc.name, tc.arguments, color=False)
                                state.add_action(act)
                state.add_system(
                    f"🔄 Resumed session '{meta.get('run_id')}' ({len(messages)} messages loaded, continuing at step {step_idx})."
                )
                state.set_available_sessions(self.find_recent_sessions(limit=15))
                state.update_status(memory.stats(), self.get_cache_stats())
                return

            if cmd in ("/status", "status"):
                s = memory.stats()
                state.add_system(
                    f"Context Tokens: {s['estimated_tokens']:,} / {s['max_estimated_tokens']:,} ({s['usage_pct']:.1f}%) | "
                    f"Messages: {s['total_messages']}"
                )
                return

            if cmd in ("/compact", "compact"):
                old_chars = memory.total_chars()
                did_compact = memory.compact_if_needed(force=True)
                new_chars = memory.total_chars()
                if did_compact:
                    freed = (old_chars - new_chars) // 4
                    state.add_system(f"Compaction applied: freed ~{freed:,} tokens.")
                else:
                    state.add_system("No older observations eligible for compaction.")
                state.update_status(memory.stats(), self.get_cache_stats())
                return

            if cmd in ("/new", "/reset", "new", "reset"):
                memory.reset()
                state.lines.clear()
                state.add_system("Fresh session started (workspace preserved).")
                state.update_status(memory.stats(), self.get_cache_stats())
                return

            if cmd in SKILL_COMMAND_ALIASES or cmd.startswith(SKILL_COMMAND_PREFIXES):
                outcome = resolve_skill_command(user_text, skills)
                if outcome.is_error:
                    state.add_error(outcome.message)
                    return
                if outcome.action == "skip":
                    state.add_system(outcome.message)
                    return
                state.add_system(f"⚡ {outcome.message}")
                effective_prompt = outcome.prompt or user_text
            elif user_text.startswith("/"):
                outcome = resolve_slash_input(
                    user_text, custom_commands, skills, unknown_message="Type /help for commands."
                )
                if outcome.is_error or outcome.prompt is None:
                    state.add_error(outcome.message)
                    return
                state.add_system(outcome.message)
                effective_prompt = outcome.prompt
            else:
                effective_prompt = user_text

            effective_prompt, attached_files = expand_tagged_files(effective_prompt, env.workspace_dir)
            state.add_user(user_text)
            if attached_files:
                names = ", ".join(f"@{f}" for f in attached_files)
                state.add_system(f"⚡ Attached {len(attached_files)} file(s): {names}")
            memory.add_user(effective_prompt)
            if tui.stdscr:
                tui.draw(tui.stdscr)

            turn_start = step_idx
            state.turn_active = True
            def _cancel_probe() -> bool:
                """Non-blocking Esc check invoked from the model's retry/backoff path.

                Surfaces progress the moment it fires, so a user who pressed Esc
                during a long rate-limit backoff sees the request is aborting
                rather than assuming the key did nothing. Every caller treats a
                True result as terminal, so this runs at most once per turn.
                """
                if not tui.check_cancel_requested():
                    return False
                state.add_system(
                    "⚠️ Cancelling… aborting the pending model request "
                    "(Ctrl+C cancels immediately, even mid-request)."
                )
                state.update_status(memory.stats(), self.get_cache_stats())
                if tui.stdscr:
                    tui.draw(tui.stdscr)
                return True

            if isinstance(self.model, CancellableModel):
                self.model.set_cancel_probe(_cancel_probe)

            try:
                while (step_idx - turn_start) < max_steps_per_turn:
                    # Shared by the boundary cancel and the mid-request probe so both
                    # report identically and leave the session in the same state.
                    def _abort_turn(reason: str) -> None:
                        dropped = memory.drop_incomplete_tool_turn()
                        state.add_system(
                            f"⚠️ Turn {reason} at step {step_idx}. Session is still active."
                            + (f" Dropped {dropped} incomplete message(s)." if dropped else "")
                        )
                        state.update_status(memory.stats(), self.get_cache_stats())
                        if tui.stdscr:
                            tui.draw(tui.stdscr)

                    if tui.check_cancel_requested():
                        _abort_turn("cancelled by user (Esc)")
                        break

                    try:
                        is_finished, step_trace = loop.step(step_index=step_idx)
                    except ModelCancelledError:
                        _abort_turn("cancelled by user (Esc)")
                        break
                    except KeyboardInterrupt:
                        _abort_turn("interrupted by user (Ctrl+C)")
                        break
                    except Exception as exc:
                        state.add_error(f"Step {step_idx} error: {exc}")
                        break

                    tracer.record_step(step_trace)
                    step_idx += 1

                    model_text = step_trace.model_response.get("content") or ""
                    tool_calls = step_trace.model_response.get("tool_calls") or []

                    if tool_calls:
                        if model_text.strip():
                            first_line = model_text.strip().splitlines()[0]
                            preview = (first_line[:80] + "...") if len(first_line) > 80 else first_line
                            state.add_thought(preview)

                        for tc, tr in _pair_calls_with_results(tool_calls, step_trace.tool_results):
                            name = tc.get("name", "") if tc else ""
                            args = tc.get("arguments", {}) if tc else {}
                            action_text = _format_action_summary(name, args, color=False)
                            exit_code = tr.get("exit_code", 0) if tr else 0
                            if exit_code == 0:
                                state.add_action(f"{action_text} ({step_trace.duration_s:.1f}s)")
                            else:
                                out_lines = tr.get("output", "").strip().splitlines() if tr else []
                                err_msg = out_lines[0] if out_lines else "Failed"
                                state.add_error(f"{action_text} (exit {exit_code}: {err_msg[:60]})")
                    else:
                        if model_text.strip():
                            state.add_agent(model_text.strip())

                    state.update_status(memory.stats(), self.get_cache_stats())
                    if tui.stdscr:
                        tui.draw(tui.stdscr)

                    if is_finished:
                        break

                    if tui.check_cancel_requested():
                        dropped = memory.drop_incomplete_tool_turn()
                        state.add_system(
                            f"⚠️ Turn cancelled by user (Esc) at step {step_idx}. Session is still active."
                            + (f" Dropped {dropped} incomplete message(s)." if dropped else "")
                        )
                        state.update_status(memory.stats(), self.get_cache_stats())
                        if tui.stdscr:
                            tui.draw(tui.stdscr)
                        break
            finally:
                state.turn_active = False
                if isinstance(self.model, CancellableModel):
                    self.model.set_cancel_probe(None)
                state.update_status(memory.stats(), self.get_cache_stats())
                if tui.stdscr:
                    tui.draw(tui.stdscr)

        try:
            tui.run(
                on_user_submit=on_user_submit,
                initial_instruction=initial_instruction,
            )
        finally:
            env.teardown()
            c_stats = self.get_cache_stats()
            patch_diff = env.get_git_patch()
            json_path, md_path = tracer.finalize(
                finish_reason="session_ended", patch_diff=patch_diff, telemetry=c_stats
            )
            print(f"\nComplete session trace saved to:\n   {md_path}\n")
