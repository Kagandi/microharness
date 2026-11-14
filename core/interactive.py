"""Interactive command routing for microharness.

Handlers process slash commands (``/model``, ``/resume``, ``/compact``,
``/skill``, ...) and report through a :class:`SessionUI` sink, returning
one of three actions for the caller to act on (:data:`EXIT`, :data:`TURN`, or
:data:`SKIP`).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol

from core.commands import Command, SlashOutcome, resolve_slash_input
from core.discovery import expand_tagged_files
from core.memory import Memory
from core.skills import Skill, resolve_skill_command
from core.tracer import Tracer

if TYPE_CHECKING:
    from core.harness import Harness

# What the caller should do after dispatching one line of input.
EXIT = "exit"
TURN = "turn"
SKIP = "skip"

REASONING_USAGE = (
    "  • Usage: /reasoning low | /reasoning medium | /reasoning high | "
    "/reasoning xhigh | /reasoning max | /reasoning off | /reasoning <tokens>"
)
CACHE_USAGE = "  • Usage: /cache 5m | /cache 1h | /cache off"

CACHE_TTL_DESCRIPTIONS = {"5m": "5 minutes (Standard Ephemeral)", "1h": "1 hour (Extended Beta)"}


class SessionUI(Protocol):
    """Rendering sink for command output."""

    def info(self, text: str) -> None:
        """Reports neutral output (listings, confirmations)."""

    def success(self, text: str) -> None:
        """Reports a command that changed session state."""

    def error(self, text: str) -> None:
        """Reports a failed command."""

    def status(self, text: str) -> None:
        """Reports one undecorated line, used for lists and footers."""

    def echo(self, text: str) -> None:
        """Records what the user typed, for frontends that do not echo natively."""

    def refresh(self) -> None:
        """Re-renders any live status display."""

    def render_context(
        self, stats: dict[str, Any], reasoning: str | None, cache: dict[str, Any]
    ) -> list[str]:
        """Renders the context report in this frontend's preferred shape."""


class NullUI:
    """Headless no-op UI sink for silent execution, headless batch runs, and tests.

    Suppresses all terminal output and UI notifications (info, success, error, status).
    Errors are not printed to stdout/stderr; debugging traces should inspect the Tracer.
    """

    def info(self, text: str) -> None:
        pass

    def success(self, text: str) -> None:
        pass

    def error(self, text: str) -> None:
        pass

    def status(self, text: str) -> None:
        pass

    def echo(self, text: str) -> None:
        pass

    def refresh(self) -> None:
        pass

    def render_context(
        self, stats: dict[str, Any], reasoning: str | None, cache: dict[str, Any]
    ) -> list[str]:
        return render_context_breakdown(stats, reasoning, cache)


class TuiUI:
    """Sink backed by a :class:`~core.tui.TUIState`."""

    def __init__(self, state: Any, redraw: Callable[[], None] | None = None):
        self.state = state
        self._redraw = redraw or (lambda: None)

    def info(self, text: str) -> None:
        self.state.add_system(text)

    def success(self, text: str) -> None:
        self.state.add_system(f"✅ {text}")

    def error(self, text: str) -> None:
        self.state.add_error(text)

    def status(self, text: str) -> None:
        self.state.add_system(text)

    def echo(self, text: str) -> None:
        self.state.add_user(text)

    def refresh(self) -> None:
        self._redraw()

    def render_context(
        self, stats: dict[str, Any], reasoning: str | None, cache: dict[str, Any]
    ) -> list[str]:
        return render_context_summary(stats)


@dataclass
class TurnState:
    """The step index the next turn starts from; ``/resume`` moves it forward."""

    step_index: int = 1


@dataclass
class CommandContext:
    """Everything a command handler is allowed to read or change."""

    harness: Harness
    memory: Memory
    ui: SessionUI
    workspace: Path
    commands: dict[str, Command] = field(default_factory=dict)
    skills: dict[str, Skill] = field(default_factory=dict)
    tracer: Tracer | None = None
    turns: TurnState = field(default_factory=TurnState)
    runner: Any = None


@dataclass(frozen=True)
class CommandSpec:
    """One routable command: how it is spelled, what it is for, and its handler.

    A spec is the single source of truth for a command: the router matches
    ``names``, ``/help`` prints ``usage`` with ``description``, and the TUI
    autocomplete catalog lists ``canonical`` with ``description``.

    ``names`` may be invoked with arguments; ``bare_names`` (the slash-less
    spellings) only match exactly, so ordinary prose like "model the user
    database" is never swallowed by a command.
    """

    key: str
    names: tuple[str, ...]
    usage: str
    description: str
    handler: str
    takes_args: bool = False
    bare_names: tuple[str, ...] = ()

    @property
    def canonical(self) -> str:
        """The spelling shown in help text and autocomplete."""
        return self.names[0]

    def matches(self, lowered: str) -> bool:
        """Whether ``lowered`` (already trimmed) invokes this command."""
        if lowered in self.bare_names:
            return True
        for name in self.names:
            if lowered == name:
                return True
            if self.takes_args and lowered.startswith(f"{name} "):
                return True
        return False


HELP_USAGE_WIDTH = 29
HELP_HIDDEN = frozenset({"help"})  # /help does not list itself

COMMAND_SPECS: tuple[CommandSpec, ...] = (
    CommandSpec(
        "model",
        ("/model", "/models"),
        "/model [name|idx]",
        "View configured models or switch active model",
        "_cmd_model",
        takes_args=True,
    ),
    CommandSpec(
        "resume",
        ("/resume", "/resumes"),
        "/resume [id|idx]",
        "View or resume a previous conversation session",
        "_cmd_resume",
        takes_args=True,
    ),
    CommandSpec(
        "reasoning",
        ("/reasoning",),
        "/reasoning [low|med|high|off]",
        "View or switch reasoning / thinking effort",
        "_cmd_reasoning",
        takes_args=True,
    ),
    CommandSpec(
        "cache",
        ("/cache",),
        "/cache [5m|1h|off]",
        "View or switch prompt caching duration on the fly",
        "_cmd_cache",
        takes_args=True,
    ),
    CommandSpec(
        "status",
        ("/status", "/context"),
        "/status",
        "Show context window size, message breakdown, and budget",
        "_cmd_status",
    ),
    CommandSpec(
        "compact",
        ("/compact",),
        "/compact",
        "Manually collapse older observations to free context",
        "_cmd_compact",
    ),
    CommandSpec(
        "new",
        ("/new", "/reset"),
        "/new",
        "Clear conversation history while keeping workspace files",
        "_cmd_new",
    ),
    CommandSpec(
        "skill",
        ("/skill", "/skills"),
        "/skill [name]",
        "Show available skills or activate one into conversation",
        "_cmd_skill",
        takes_args=True,
    ),
    CommandSpec(
        "exit",
        ("/exit", "/quit"),
        "/exit",
        "Save session trace and exit",
        "_cmd_exit",
    ),
    CommandSpec(
        "help",
        ("/help",),
        "/help",
        "List built-in, custom, and discovered commands",
        "_cmd_help",
    ),
)

# Spellings not derivable from a command's own names (which yield the slash-less form).
EXTRA_ALIASES: dict[str, tuple[str, ...]] = {
    "exit": ("quit",),
    "status": ("context",),
}


def expand_specs(specs: tuple[CommandSpec, ...]) -> tuple[CommandSpec, ...]:
    """Adds slash-less aliases, so ``status`` and ``/status`` reach the same handler."""
    return tuple(
        replace(
            spec,
            bare_names=(
                *(name.lstrip("/") for name in spec.names),
                *EXTRA_ALIASES.get(spec.key, ()),
            ),
        )
        for spec in specs
    )


def command_argument(user_input: str) -> str:
    """Returns everything after the command word, or ``""`` when there is none."""
    _, _, args = user_input.strip().partition(" ")
    return args.strip()


def command_catalog() -> list[tuple[str, str]]:
    """Built-in commands as ``(canonical, description)``, for autocomplete menus."""
    return [(spec.canonical, spec.description) for spec in COMMAND_SPECS]


def render_help_lines(
    commands: dict[str, Command],
    skills: dict[str, Skill],
    footer: str = "",
) -> list[str]:
    """Renders ``/help``: built-in commands first, then discovered commands and skills."""
    lines = ["Available commands:"]
    lines += [
        f"  {spec.usage:<{HELP_USAGE_WIDTH}} - {spec.description}"
        for spec in COMMAND_SPECS
        if spec.key not in HELP_HIDDEN
    ]
    if commands:
        lines.append("\nCustom slash commands (from commands/):")
        lines += [
            f"  /{name:<10} - {command.badge}{command.description}"
            for name, command in sorted(commands.items())
        ]
    if skills:
        lines.append("\nAvailable skills (run /<name> or /skill <name>):")
        lines += [
            f"  /{name:<10} - {skill.badge}{skill.summary}"
            for name, skill in sorted(skills.items())
        ]
    if footer:
        lines.append(footer)
    return lines


def format_model_desc(model: dict[str, Any]) -> str:
    """One-line description of a model entry, shared by listings and autocomplete."""
    provider = model.get("provider")
    label = f"[{provider}] " if provider else ""
    name = model.get("name", "")
    return f"{label}{model.get('description') or f'Model {name}'}"


def format_session_desc(session: dict[str, Any], preview_limit: int | None = None) -> str:
    """One-line session summary: step count plus preview, optionally truncated.

    The autocomplete menu passes ``preview_limit`` because its column is narrow;
    the ``/resume`` listing shows the stored preview in full.
    """
    preview = str(session.get("preview") or "")
    if preview_limit is not None and len(preview) > preview_limit:
        preview = preview[:preview_limit]
    return f"({session.get('total_steps', 0)} steps) {preview}"


def render_model_lines(models: list[dict[str, Any]], active_model: str) -> list[str]:
    """Renders the ``/model`` listing, marking the active entry."""
    if not models:
        return [f"  • {active_model} [active]"] if active_model else ["  (none configured)"]

    lowered = active_model.lower()
    lines = []
    for index, model in enumerate(models, start=1):
        is_active = model["name"].lower() == lowered or model.get("model", "").lower() == lowered
        active = " [active]" if is_active else ""
        lines.append(f"  {index}. {model['name']}{active} - {format_model_desc(model)}")
    return lines


def render_session_lines(sessions: list[dict[str, Any]]) -> list[str]:
    """Renders the ``/resume`` listing."""
    if not sessions:
        return ["  No previous sessions found in runs directory."]
    return [
        f"  {index}. {session['run_id']}"
        f"{' [this repo]' if session.get('matches_workspace') else ''} "
        f"- {format_session_desc(session)}"
        for index, session in enumerate(sessions, start=1)
    ]


def render_cache_hits(cache_stats: dict[str, Any]) -> str | None:
    """Renders the cache hit-rate line, or ``None`` when nothing was recorded yet."""
    total = cache_stats.get("total_prompt_tokens")
    if not total:
        return None
    cached = cache_stats.get("total_cached_tokens", 0)
    rate = cache_stats.get("cache_hit_rate_pct", 0.0)
    return f"  • Cached Tokens: {cached:,} / {total:,} ({rate}% hit rate)"


def render_context_breakdown(
    stats: dict[str, Any],
    reasoning: str | None,
    cache: dict[str, Any],
) -> list[str]:
    """Renders the detailed multi-line context report."""
    roles = stats["by_role"]
    lines = [
        "📊 Context Breakdown:",
        f"  • Usage: {stats['estimated_tokens']:,} / {stats['max_estimated_tokens']:,} tokens "
        f"({stats['usage_pct']}%)",
        f"  • Messages ({stats['total_messages']}): {roles['system']} sys, {roles['user']} user, "
        f"{roles['assistant']} assistant, {roles['tool']} tools",
        f"  • Reasoning Effort: {reasoning.upper() if reasoning else 'OFF'}",
        f"  • Prompt Caching: {(cache.get('cache_ttl') or 'OFF').upper()}",
    ]
    total = cache.get("total_prompt_tokens")
    if total:
        lines.append(
            f"    Cache Hits: {cache.get('total_cached_tokens', 0):,} / {total:,} "
            f"prompt tokens ({cache.get('cache_hit_rate_pct', 0.0)}% hit rate)"
        )
    return [*lines, ""]


def render_context_summary(stats: dict[str, Any]) -> list[str]:
    """Renders the one-line context report used by the TUI."""
    return [
        f"Context Tokens: {stats['estimated_tokens']:,} / {stats['max_estimated_tokens']:,} "
        f"({stats['usage_pct']:.1f}%) | Messages: {stats['total_messages']}"
    ]


class CommandRouter:
    """Routes one line of interactive input to a handler and renders the outcome."""

    def __init__(self, ctx: CommandContext, *, allow_bare_words: bool = True):
        self.ctx = ctx
        self.specs = expand_specs(COMMAND_SPECS) if allow_bare_words else COMMAND_SPECS

    def dispatch(self, user_input: str) -> str:
        """Handles one input line, returning :data:`EXIT`, :data:`TURN`, or :data:`SKIP`."""
        stripped = user_input.strip()
        spec = next((s for s in self.specs if s.matches(stripped.lower())), None)
        if spec is not None:
            return getattr(self, spec.handler)(command_argument(stripped))
        return self._handle_input(stripped)

    def _handle_input(self, user_input: str) -> str:
        """Runs a custom command, a slash-invoked skill, or ordinary user input."""
        if user_input.startswith("/"):
            outcome: SlashOutcome = resolve_slash_input(
                user_input,
                self.ctx.commands,
                self.ctx.skills,
                unknown_message="Type /help to see available commands.",
            )
            if outcome.is_error or outcome.prompt is None:
                self.ctx.ui.error(outcome.message)
                return SKIP
            self.ctx.ui.info(f"⚡ {outcome.message} ...")
        else:
            outcome = SlashOutcome(TURN, "", False, user_input)

        self.ctx.ui.refresh()
        return self._add_user_prompt(outcome.prompt or "", echo=user_input)

    def _add_user_prompt(self, prompt: str, echo: str = "") -> str:
        """Expands ``@file`` tags in a prompt and appends it to memory."""
        effective_prompt, attached = expand_tagged_files(prompt, self.ctx.workspace)
        self.ctx.ui.echo(echo or prompt)
        if attached:
            names = ", ".join(f"@{name}" for name in attached)
            self.ctx.ui.info(f"⚡ Attached {len(attached)} file(s): {names}")
        self.ctx.memory.add_user(effective_prompt)
        return TURN

    # --- handlers -------------------------------------------------------

    def _cmd_exit(self, _arg: str) -> str:
        self.ctx.ui.info("Ending session.")
        return EXIT

    def _cmd_help(self, _arg: str) -> str:
        for line in render_help_lines(self.ctx.commands, self.ctx.skills):
            self.ctx.ui.status(line)
        self.ctx.ui.status("")
        return SKIP

    def _cmd_status(self, _arg: str) -> str:
        ui = self.ctx.ui
        for line in ui.render_context(
            self.ctx.memory.stats(),
            self.ctx.harness.get_reasoning_effort(),
            self.ctx.harness.get_cache_stats(),
        ):
            ui.status(line)
        ui.refresh()
        return SKIP

    def _cmd_compact(self, _arg: str) -> str:
        memory = self.ctx.memory
        before = memory.estimated_tokens()
        compacted = memory.compact_if_needed(force=True)
        after = memory.estimated_tokens()

        if compacted:
            self.ctx.ui.info(
                f"🧹 Compaction applied: freed ~{before - after:,} tokens "
                f"(~{before:,} -> ~{after:,} tokens)."
            )
        else:
            self.ctx.ui.info("ℹ️ No older observations eligible for compaction at this time.")
        self.ctx.ui.refresh()
        return SKIP

    def _cmd_new(self, _arg: str) -> str:
        self.ctx.memory.reset()
        self.ctx.ui.info("✨ Started a fresh conversation context (workspace files preserved).")
        self.ctx.ui.refresh()
        return SKIP

    def _cmd_model(self, arg: str) -> str:
        if not arg:
            harness = self.ctx.harness
            self.ctx.ui.info("🤖 Available Models (from config):")
            for line in render_model_lines(
                harness.get_available_models(), harness.active_model_name()
            ):
                self.ctx.ui.status(line)
            self.ctx.ui.status("Tip: Type /model <name|number> to switch active model.")
            return SKIP

        ok, message, new_model = self.ctx.harness.switch_model(arg)
        if ok:
            cap = getattr(self.ctx.harness, "max_context_tokens", None)
            mem = getattr(self.ctx, "memory", None)
            if cap is not None and mem is not None:
                mem.max_context_tokens = cap
            self.ctx.ui.success(message)
            state = getattr(self.ctx.ui, "state", None)
            if state is not None:
                state.model_name = self.ctx.harness.active_model_name()
                state.provider_name = getattr(new_model, "provider", "") or ""
            runner = getattr(self.ctx, "runner", None)
            if runner is not None:
                runner.model = new_model
                refresh = getattr(runner, "refresh", None)
                if callable(refresh):
                    refresh()
            else:
                state = getattr(self.ctx.ui, "state", None)
                update = getattr(state, "update_status", None) if state is not None else None
                if callable(update):
                    update(
                        self.ctx.memory.stats(),
                        self.ctx.harness.get_cache_stats(),
                    )
        else:
            self.ctx.ui.error(message)
        self.ctx.ui.refresh()
        return SKIP

    def _cmd_resume(self, arg: str) -> str:
        if not arg:
            self.ctx.ui.info("📂 Recent Sessions:")
            for line in render_session_lines(self.ctx.harness.find_recent_sessions(limit=8)):
                self.ctx.ui.status(line)
            self.ctx.ui.status("Tip: Type /resume <number|session_id> to load a previous session.")
            return SKIP
        return self._resume_session(arg)

    def _resume_session(self, selector: str) -> str:
        messages, meta = self.ctx.harness.load_session(selector)
        if not messages:
            self.ctx.ui.error(meta.get("error", f"Could not load session '{selector}'"))
            return SKIP

        memory = self.ctx.memory
        memory.restore_messages(messages, refresh_system_prompt=memory.system_prompt)
        if self.ctx.tracer is not None:
            self.ctx.tracer.steps.clear()
        self.ctx.turns.step_index = meta.get("total_steps", 0) + 1
        self.ctx.ui.success(
            f"🔄 Resumed session '{meta.get('run_id')}' ({len(messages)} messages loaded)."
        )
        self.ctx.ui.refresh()
        return SKIP

    def _cmd_skill(self, arg: str) -> str:
        invocation = f"/skill {arg}".strip()
        outcome = resolve_skill_command(invocation, self.ctx.skills)
        if outcome.is_error:
            self.ctx.ui.error(outcome.message)
            return SKIP
        if outcome.action == SKIP:
            self.ctx.ui.info(outcome.message)
            return SKIP

        self.ctx.ui.info(f"⚡ {outcome.message} ...")
        self.ctx.ui.refresh()
        return self._add_user_prompt(outcome.prompt or "", echo=invocation)

    def _cmd_reasoning(self, arg: str) -> str:
        harness = self.ctx.harness
        if not arg:
            current = harness.get_reasoning_effort()
            self.ctx.ui.info(
                f"🧠 Reasoning / Thinking Effort: {current.upper() if current else 'OFF (Disabled)'}"
            )
            self.ctx.ui.status(REASONING_USAGE)
            return SKIP

        try:
            effort = harness.set_reasoning_effort(arg)
        except ValueError as exc:
            self.ctx.ui.error(str(exc))
            return SKIP

        if effort:
            self.ctx.ui.success(
                f"🧠 Reasoning effort switched to: {effort.upper()} for subsequent turns."
            )
        else:
            self.ctx.ui.success("🧠 Reasoning effort disabled (off) for subsequent turns.")
        state = getattr(self.ctx.ui, "state", None)
        if state is not None:
            state.reasoning = effort.upper() if effort else ""
        self.ctx.ui.refresh()
        return SKIP

    def _cmd_cache(self, arg: str) -> str:
        harness = self.ctx.harness
        if not arg:
            self._show_cache()
            return SKIP

        try:
            ttl = harness.set_cache_ttl(arg)
        except ValueError as exc:
            self.ctx.ui.error(str(exc))
            return SKIP

        if ttl:
            description = CACHE_TTL_DESCRIPTIONS.get(ttl, ttl)
            self.ctx.ui.success(
                f"⚡ Prompt caching switched to: {ttl.upper()} ({description}) "
                "for subsequent turns."
            )
        else:
            self.ctx.ui.success("⚡ Prompt caching disabled for subsequent turns.")
        state = getattr(self.ctx.ui, "state", None)
        if state is not None:
            state.cache_ttl = ttl.upper() if ttl else ""
        self.ctx.ui.refresh()
        return SKIP

    def _show_cache(self) -> None:
        stats = self.ctx.harness.get_cache_stats()
        ttl = stats.get("cache_ttl")
        label = (
            f"{ttl.upper()} ({CACHE_TTL_DESCRIPTIONS.get(ttl, '')})" if ttl else "DISABLED (off)"
        )
        self.ctx.ui.info(f"⚡ Prompt Caching: {label}")
        if hits := render_cache_hits(stats):
            self.ctx.ui.status(hits)
        self.ctx.ui.status(CACHE_USAGE)
