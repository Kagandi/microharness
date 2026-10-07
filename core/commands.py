"""Custom slash commands and the shared slash-input resolver.

A command is a Markdown file whose body is a prompt template. ``$ARGUMENTS`` /
``{args}`` are substituted, otherwise trailing input is appended as context.

:func:`resolve_slash_input` is the single place that turns a leading-slash user
input into a prompt, shared across batch runs and interactive sessions.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.discovery import discover_resources
from core.skills import Skill, compose_skill_prompt

ARGUMENT_PLACEHOLDERS = ("$ARGUMENTS", "{args}")


@dataclass(frozen=True)
class Command:
    """A discovered slash command and the prompt template it expands to."""

    name: str
    description: str
    template: str
    scope: str  # 'global' or 'project'

    @property
    def badge(self) -> str:
        return f"[{self.scope}] "


def _load_command(path: Path, scope: str) -> tuple[str, Command]:
    content = path.read_text(encoding="utf-8", errors="replace")
    description = next(
        (line.strip().lstrip("#").strip() for line in content.splitlines() if line.strip()),
        path.stem,
    )
    name = path.stem.lower()
    return name, Command(name=name, description=description, template=content, scope=scope)


def discover_commands(
    workspace: Path | None,
    global_dir: Path | None = None,
) -> dict[str, Command]:
    """Discovers custom slash commands from global and project command directories.

    Returns a mapping of command name to :class:`Command`; project commands
    cleanly shadow global ones with the same name.
    """
    discovered = discover_resources(
        resource_name="commands",
        loader=_load_command,
        workspace_dir=workspace,
        global_dir=global_dir,
        file_pattern="*.md",
    )
    return {res.id: res.item for res in discovered.values()}


def expand_command(template: str, user_args: str) -> str:
    """Expands a command template with user arguments."""
    args = user_args.strip()
    for placeholder in ARGUMENT_PLACEHOLDERS:
        if placeholder in template:
            return template.replace(placeholder, args).strip()
    if args:
        return f"{template.strip()}\n\nAdditional Arguments / Target:\n{args}"
    return template.strip()


@dataclass(frozen=True)
class SlashOutcome:
    """Result of resolving a leading-slash input into something runnable."""

    action: str  # 'turn' (run the prompt) or 'skip' (nothing to run)
    message: str
    is_error: bool = False
    prompt: str | None = None


def split_slash_input(user_input: str) -> tuple[str, str]:
    """Splits ``/name rest of input`` into a lowercase command key and its arguments."""
    name, _, args = user_input[1:].partition(" ")
    return name.lower().strip(), args


def resolve_slash_input(
    user_input: str,
    commands: dict[str, Command],
    skills: dict[str, Skill] | None = None,
    *,
    allow_skills: bool = True,
    unknown_message: str = "Type /help to see available commands.",
) -> SlashOutcome:
    """Resolves a leading-slash input against custom commands and skills.

    Args:
        allow_skills: When False, only custom commands resolve. Batch runs use
            this so a plain model id or path starting with ``/`` is left alone.
        unknown_message: Trailing hint appended to the unknown-command error.
    """
    key, args = split_slash_input(user_input)

    command = commands.get(key)
    if command is not None:
        return SlashOutcome(
            action="turn",
            message=f"Executing /{key} [{command.scope}]",
            prompt=expand_command(command.template, args),
        )

    skill = (skills or {}).get(key) if allow_skills else None
    if skill is not None:
        return SlashOutcome(
            action="turn",
            message=f"⚡ Activated skill: /{key} [{skill.scope}]",
            prompt=compose_skill_prompt(skill, args.strip()),
        )

    return SlashOutcome(
        action="skip",
        message=f"Unknown command '/{key}'. {unknown_message}",
        is_error=True,
    )