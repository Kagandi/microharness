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
from core.discovery import discover_workspace_files
from core.environment import DockerEnvironment, Environment
from core.evaluator import Evaluator
from core.interactive import (
    EXIT,
    SKIP,
    TURN,
    CommandContext,
    CommandRouter,
    ConsoleUI,
    SessionUI,
    TuiUI,
    TurnState,
)
from core.loop import AgentLoop
from core.memory import Memory
from core.model import (
    CacheAwareModel,
    DeadlineAwareModel,
    Model,
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
    MAX_STEPS_EXCEEDED,
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
from core.tui_session import TuiTurnRunner, replay_messages
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

    def _make_router(
        self,
        memory: Memory,
        custom_commands: dict[str, Command],
        skills: dict[str, Skill],
        ui: SessionUI,
        workspace_dir: Path | None,
        tracer: Tracer | None,
        turns: TurnState,
        *,
        allow_bare_words: bool = True,
    ) -> CommandRouter:
        """Builds a command router bound to this harness and the caller's session state."""
        return CommandRouter(
            CommandContext(
                harness=self,
                memory=memory,
                ui=ui,
                workspace=workspace_dir or self.workspace_dir,
                commands=custom_commands,
                skills=skills,
                tracer=tracer,
                turns=turns,
            ),
            allow_bare_words=allow_bare_words,
        )

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
        turns = TurnState(step_index=step_holder[0] if step_holder else 1)
        router = self._make_router(
            memory, custom_commands, skills or {}, ConsoleUI(), workspace_dir, tracer, turns
        )
        action = router.dispatch(user_input)
        if step_holder is not None:
            step_holder[0] = turns.step_index
        return action

    def _restore_session(self, memory: Memory, resume: bool | str | Path | None) -> int:
        """Restores a previous session into memory, returning the step index to resume at."""
        if not resume:
            return 1
        messages, meta = self.load_session(None if resume is True else resume)
        if not messages:
            print(f"\n⚠️ Could not resume session ({meta.get('error', 'not found')}). Starting fresh session.\n")
            return 1
        memory.restore_messages(messages, refresh_system_prompt=memory.system_prompt)
        step_index = meta.get("total_steps", 0) + 1
        print(
            f"\n🔄 Resumed session '{meta.get('run_id')}' "
            f"({len(messages)} messages loaded, continuing at step {step_index}).\n"
        )
        return step_index

    def _print_session_banner(self, env: Environment) -> None:
        """Prints the session header: workspace, model, and the command reminder."""
        print("\n🚀 Interactive Session Started")
        print(f"📁 Project Workspace: {env.workspace_dir}")
        if self.docker:
            print(f"🐳 Docker Sandbox: ACTIVE ({self.docker_image})")
        provider = getattr(self.model, "provider", "") or ""
        suffix = f" [{provider}]" if provider else ""
        print(f"🤖 Active Model: {self.active_model_name()}{suffix}")
        print(f"⚡ Prompt Caching: {self.cache_ttl.upper() if self.cache_ttl else 'OFF'}")
        reasoning = self.get_reasoning_effort()
        print(f"🧠 Reasoning Effort: {reasoning.upper() if reasoning else 'OFF'}")
        print(
            "💡 Commands: /model | /resume | /reasoning | /cache | /compact | /status | "
            "/new | /skill | /help | /exit\n"
        )

    def _print_context_footer(self, memory: Memory) -> None:
        """Prints the single-line context usage footer shown before each prompt."""
        stats = memory.stats()
        print(
            _dim(
                f"─── 📊 Context: {stats['estimated_tokens']:,} / {stats['max_estimated_tokens']:,} "
                f"tokens ({stats['usage_pct']}%) | Messages: {stats['total_messages']} ───"
            )
        )

    def _finalize_session(self, env: Environment, tracer: Tracer, run_dir: Path) -> None:
        """Writes the patch and trace for an interactive session that has ended."""
        patch_diff = env.get_git_patch()
        if patch_diff:
            patch_file = run_dir / "patch.diff"
            patch_file.write_text(patch_diff, encoding="utf-8")
            print(f"📦 Git patch saved to: {patch_file}")
        _, md_path = tracer.finalize(
            finish_reason="session_ended",
            patch_diff=patch_diff,
            telemetry=self.get_cache_stats(),
        )
        print(f"\n📄 Complete session trace saved to:\n   {md_path}\n")

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

        turns = TurnState(step_index=self._restore_session(memory, resume))
        commands = discover_commands(env.workspace_dir, self.global_dir)
        skills = discover_skills(env.workspace_dir, self.global_dir)
        router = self._make_router(
            memory, commands, skills, ConsoleUI(), env.workspace_dir, tracer, turns
        )
        tracer.compact = True

        self._print_session_banner(env)
        max_steps_per_turn = 30
        pending_turn = self._submit(router, initial_instruction)
        if pending_turn == EXIT:
            env.teardown()
            return

        try:
            while True:
                if pending_turn == TURN:
                    pending_turn = self._run_pending_turn(
                        loop, tracer, turns, max_steps_per_turn
                    )

                self._print_context_footer(memory)
                try:
                    user_input = input("👤 You: ").strip()
                except (KeyboardInterrupt, EOFError):
                    print("\nSession interrupted.")
                    break

                if not user_input:
                    continue
                pending_turn = self._submit(router, user_input)
                if pending_turn == EXIT:
                    break

            self._finalize_session(env, tracer, run_dir)
        finally:
            env.teardown()

    def _submit(self, router: CommandRouter, user_input: str | None) -> str:
        """Dispatches one line of input; returns the resulting action."""
        if not user_input or not user_input.strip():
            return SKIP
        return router.dispatch(user_input)

    def _run_pending_turn(
        self,
        loop: AgentLoop,
        tracer: Tracer,
        turns: TurnState,
        max_steps: int,
    ) -> str:
        """Executes the queued agent turn and returns the action to take next."""
        outcome = run_steps(
            self.model,
            loop,
            tracer,
            turns.step_index,
            max_steps,
            max_retries=self.max_step_retries,
            retry_delay=self.step_retry_delay,
        )
        turns.step_index = outcome.next_step
        if not outcome.finished and outcome.reason == MAX_STEPS_EXCEEDED:
            print(f"\n⚠️ Reached safety limit of {max_steps} steps for this turn. Returning to prompt.\n")
        return SKIP

    def run_tui(
        self,
        initial_instruction: str | None = None,
        task_id: str = "tui_session",
        resume: bool | str | Path | None = None,
    ) -> None:
        """Full-screen interactive TUI with bottom-anchored chat and scrollable activity."""
        from core.tui import CursesTUI, TUIState

        env, _, memory, tracer, loop, run_dir = self._setup_run(
            task_id=task_id,
            system_prompt=DEFAULT_SYSTEM_PROMPT,
            initial_instruction="",
            initial_files={},
        )
        tracer.verbose = False
        tracer.compact = True

        commands = discover_commands(env.workspace_dir, self.global_dir)
        skills = discover_skills(env.workspace_dir, self.global_dir)
        state = self._build_tui_state(TUIState(title="microharness"), env, commands, skills)
        turns = TurnState(step_index=self._replay_resumed_session(state, memory, resume))

        tui = CursesTUI(state)
        ui = TuiUI(state, redraw=lambda: tui.draw(tui.stdscr) if tui.stdscr else None)
        router = self._make_router(
            memory, commands, skills, ui, env.workspace_dir, tracer, turns, allow_bare_words=False
        )
        runner = TuiTurnRunner(
            model=self.model,
            loop=loop,
            memory=memory,
            tracer=tracer,
            state=state,
            tui=tui,
            cache_stats=self.get_cache_stats,
            step_index=turns.step_index,
        )

        def on_user_submit(user_text: str) -> None:
            """Handles one submitted line, running a turn when a prompt was queued."""
            if not user_text.strip():
                return
            runner.step_index = turns.step_index
            if router.dispatch(user_text) == TURN:
                runner.run_turn()
                turns.step_index = runner.step_index

        state.update_status(memory.stats(), self.get_cache_stats())
        try:
            tui.run(on_user_submit=on_user_submit, initial_instruction=initial_instruction)
        finally:
            env.teardown()
            _, md_path = tracer.finalize(
                finish_reason="session_ended",
                patch_diff=env.get_git_patch(),
                telemetry=self.get_cache_stats(),
            )
            print(f"\nComplete session trace saved to:\n   {md_path}\n")

    def _build_tui_state(
        self,
        state: Any,
        env: Environment,
        commands: dict[str, Command],
        skills: dict[str, Skill],
    ) -> Any:
        """Seeds the TUI state with session metadata and autocomplete catalogues."""
        state.provider_name = (
            self.model.provider if isinstance(self.model, ProviderAwareModel) else ""
        ) or getattr(self.model, "provider", "") or ""
        state.model_name = self.active_model_name()
        state.cache_ttl = self.cache_ttl.upper() if self.cache_ttl else ""
        reasoning = self.get_reasoning_effort()
        state.reasoning = reasoning.upper() if reasoning else ""
        from core.tui import compact_path, get_git_branch

        state.cwd = compact_path(str(env.workspace_dir))
        state.git_branch = get_git_branch(env.workspace_dir)
        state.set_custom_commands(commands)
        state.set_skills(skills)
        state.set_workspace_files(discover_workspace_files(env.workspace_dir))
        state.set_available_models(self.get_available_models())
        state.set_available_sessions(self.find_recent_sessions(limit=15))
        return state

    def _replay_resumed_session(self, state: Any, memory: Memory, resume: Any) -> int:
        """Restores a previous session and replays its transcript into the TUI."""
        if not resume:
            return 1
        messages, meta = self.load_session(None if resume is True else resume)
        if not messages:
            state.add_system(
                f"⚠️ Could not resume session ({meta.get('error', 'not found')}). Starting fresh."
            )
            return 1

        memory.restore_messages(messages, refresh_system_prompt=memory.system_prompt)
        step_index = meta.get("total_steps", 0) + 1
        replay_messages(state, messages)
        state.add_system(
            f"🔄 Resumed session '{meta.get('run_id')}' "
            f"({len(messages)} messages loaded, continuing at step {step_index})."
        )
        return step_index
