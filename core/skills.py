"""Progressive-disclosure skills: discovery, one-line summaries, and activation.

A skill is a ``SKILL.md`` document. Only its summary is placed in the system
prompt; the full body is injected into the conversation when the user activates
it, which keeps the prompt small regardless of how many skills exist.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.discovery import discover_resources

BLOCK_SCALAR_MARKERS = frozenset({">", "|", ">-", "|-"})
DESCRIPTION_KEY = "description:"


@dataclass(frozen=True)
class Skill:
    """A discovered skill document and where it came from."""

    name: str
    summary: str
    path: str  # workspace-relative when the skill lives inside the workspace
    scope: str  # 'global' or 'project'
    content: str

    @property
    def badge(self) -> str:
        return f"[{self.scope}] "


def _first_content_index(lines: list[str]) -> int | None:
    """Index of the first non-blank line, or ``None`` for an empty document."""
    for index, line in enumerate(lines):
        if line.strip():
            return index
    return None


def _split_frontmatter(lines: list[str]) -> tuple[bool, list[str], list[str]]:
    """Splits a document into ``(has_frontmatter, frontmatter_lines, body_lines)``.

    Leading blank lines are skipped, and an unterminated ``---`` block is treated
    as frontmatter extending to the end of the document.
    """
    start = _first_content_index(lines)
    if start is None:
        return False, [], []
    if lines[start].strip() != "---":
        return False, [], lines
    for index in range(start + 1, len(lines)):
        if lines[index].strip() == "---":
            return True, lines[start + 1 : index], lines[index + 1 :]
    return True, lines[start + 1 :], []


def _description_from_frontmatter(frontmatter: list[str]) -> str | None:
    """Reads the ``description`` field, folding a YAML block scalar into one line."""
    parts: list[str] = []
    capturing_scalar = False

    for line in frontmatter:
        stripped = line.strip()
        if capturing_scalar:
            if line.startswith(("  ", "\t")):
                if stripped:
                    parts.append(stripped)
                continue
            capturing_scalar = False

        if stripped.startswith(DESCRIPTION_KEY):
            value = stripped.partition(":")[2].strip().strip("\"'")
            if value in BLOCK_SCALAR_MARKERS:
                capturing_scalar = True
            elif value:
                return value
    return " ".join(parts) if parts else None


def _first_heading(body: list[str]) -> str:
    """Returns the first non-blank body line, stripped of Markdown heading marks."""
    for line in body:
        stripped = line.strip()
        if stripped:
            return stripped.lstrip("#").strip()
    return ""


def extract_skill_summary(content: str, default: str) -> str:
    """Extracts a 1-line summary from a SKILL.md document.

    Looks for a YAML frontmatter ``description`` first, then falls back to the
    document's first heading.
    """
    has_frontmatter, frontmatter, body = _split_frontmatter(content.splitlines())
    described = _description_from_frontmatter(frontmatter) if has_frontmatter else None
    return described or _first_heading(body) or default


def _load_skill(entry: Path, scope: str, workspace: Path | None) -> tuple[str, Skill] | None:
    """Loads one skill directory (or standalone ``.md`` file) into a :class:`Skill`."""
    doc = entry / "SKILL.md" if entry.is_dir() else (entry if entry.suffix == ".md" else None)
    if not doc or not doc.is_file() or entry.name == "README.md":
        return None

    content = doc.read_text(encoding="utf-8", errors="replace")
    resolved = doc.resolve()
    path_str = (
        str(resolved.relative_to(workspace))
        if workspace and resolved.is_relative_to(workspace)
        else str(doc)
    )
    skill = Skill(
        name=entry.stem,
        summary=extract_skill_summary(content, entry.stem),
        path=path_str,
        scope=scope,
        content=content,
    )
    return skill.name, skill


def discover_skills(
    workspace: Path | None,
    global_dir: Path | None = None,
) -> dict[str, Skill]:
    """Discovers skills from global and project directories.

    Returns a mapping of skill name to :class:`Skill`; project skills cleanly
    shadow global ones with the same name.
    """
    ws = Path(workspace).resolve() if workspace else None

    def loader(entry: Path, scope: str) -> tuple[str, Skill] | None:
        return _load_skill(entry, scope, ws)

    discovered = discover_resources(
        resource_name="skills",
        loader=loader,
        workspace_dir=workspace,
        global_dir=global_dir,
    )
    return {res.id: res.item for res in discovered.values()}


def _index_line(skill: Skill, show_scope: bool) -> str:
    badge = skill.badge if show_scope else ""
    return f"- {badge}{skill.name}: {skill.summary} (read `{skill.path}` for full instructions)"


def render_skills_index(skills: dict[str, Skill]) -> str:
    """Renders the skill index injected into the system prompt.

    The ``[global]``/``[project]`` badge is only shown when the two scopes are
    actually mixed, so a purely project-scoped setup stays uncluttered.
    """
    if not skills:
        return ""
    show_scope = any(skill.scope == "global" for skill in skills.values())
    entries = "\n".join(_index_line(skill, show_scope) for _, skill in sorted(skills.items()))
    return (
        "--- Available Skills (Progressive Disclosure) ---\n"
        f"{entries}\n"
        "Tip: Use the `read_file` tool to inspect a skill's full instructions only when needed.\n"
        "------------------------------------------------"
    )


def compose_skill_prompt(skill: Skill, instruction: str = "") -> str:
    """Builds the user prompt that activates a skill, with an optional task."""
    if instruction:
        return f"[Active Skill: {skill.name}]\n{skill.content}\n\nTask:\n{instruction}"
    return (
        f"[Active Skill: {skill.name}]\n{skill.content}\n\n"
        f"Please follow the instructions from the '{skill.name}' skill above."
    )


@dataclass(frozen=True)
class SkillCommandResult:
    """Outcome of a ``/skill`` command: either a message to show, or a prompt to run."""

    action: str  # 'skip' (info/listing/error) or 'turn' (run a model turn)
    message: str
    is_error: bool = False
    prompt: str | None = None


def _list_skills(skills: dict[str, Skill]) -> str:
    names = sorted(skills)
    lines = ["Available Skills:"]
    lines += [
        f"  {i}. {name:<14} [{skills[name].scope}] - {skills[name].summary}"
        for i, name in enumerate(names, start=1)
    ]
    lines.append(
        "\nTo activate a skill into the conversation, type /skill <name|number> [optional instruction]"
    )
    lines.append(f"Example: /skill {names[0]}")
    return "\n".join(lines)


def _select_skill(target: str, skills: dict[str, Skill]) -> tuple[Skill | None, str | None]:
    """Resolves a skill by 1-based index or case-insensitive name."""
    names = sorted(skills)
    if target.isdigit():
        index = int(target) - 1
        if not 0 <= index < len(names):
            return None, f"Invalid skill number {target}. Type /skill to view available skills."
        return skills[names[index]], None

    lowered = target.lower()
    for name in names:
        if name.lower() == lowered:
            return skills[name], None
    return None, f"Skill '{target}' not found. Type /skill to view available skills."


def resolve_skill_command(user_input: str, skills: dict[str, Skill]) -> SkillCommandResult:
    """Parses and resolves a ``/skill`` command into a message or an activated prompt."""
    parts = user_input.strip().split(maxsplit=2)
    if len(parts) == 1:
        if not skills:
            return SkillCommandResult(
                "skip",
                "ℹ️ No skills found in workspace or global directory. Place skills in "
                ".microharness/skills/<name>/SKILL.md or ~/.microharness/skills/<name>/SKILL.md.",
            )
        return SkillCommandResult("skip", _list_skills(skills))

    skill, error = _select_skill(parts[1].strip(), skills)
    if skill is None:
        return SkillCommandResult("skip", error or "Unknown skill.", is_error=True)

    instruction = parts[2].strip() if len(parts) > 2 else ""
    return SkillCommandResult(
        action="turn",
        message=f"Activated skill: {skill.name} {skill.badge.rstrip()}",
        prompt=compose_skill_prompt(skill, instruction),
    )
