"""System prompt templates and prompt assembly.

The system prompt is assembled from independent, individually optional pieces:
the base prompt, the tool catalogue, project steering (``AGENTS.md``), and the
skills index. Assembly order is what defines precedence, so it lives in one
place rather than at each call site.
"""

from __future__ import annotations

from pathlib import Path

DEFAULT_SYSTEM_PROMPT = """You are an autonomous software engineering assistant.

Guidelines:
- First inspect the directory and existing tests or code to understand the issue.
- Formulate a hypothesis and test it.
- Edit the necessary files to implement the fix.
- Run tests to verify your fix completely passes.
- Once the task is fully satisfied and verified, provide a final response summarizing your solution without calling any more tools.
"""

TOOL_CATALOGUE_ANCHOR = "You have access to tools to interact with the environment:"

GLOBAL_INSTRUCTIONS_HEADER = "--- Global User Instructions (~/.microharness/AGENTS.md) ---"
PROJECT_INSTRUCTIONS_HEADER = "--- Project Instructions ({name}) ---"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def load_agents_instructions(workspace: Path, global_dir: Path | None = None) -> str:
    """Loads global and project-level steering guidelines in cascading order.

    Global rules come first; the project file (``.microharness/AGENTS.md`` or
    ``./AGENTS.md``) specializes them and therefore comes last.
    """
    sections: list[str] = []

    if global_dir:
        global_file = global_dir / "AGENTS.md"
        if global_file.is_file() and (content := _read(global_file)):
            sections.append(f"{GLOBAL_INSTRUCTIONS_HEADER}\n{content}\n" + "-" * 58)

    for candidate in (workspace / ".microharness" / "AGENTS.md", workspace / "AGENTS.md"):
        if candidate.is_file() and (content := _read(candidate)):
            header = PROJECT_INSTRUCTIONS_HEADER.format(name=candidate.name)
            sections.append(f"{header}\n{content}\n" + "-" * 38)
            break

    return ("\n\n" + "\n\n".join(sections)) if sections else ""


def build_system_prompt(
    base_prompt: str,
    *,
    tool_docs: str = "",
    workspace: Path | None = None,
    sandboxed: bool = True,
    agents_instructions: str = "",
    skills_index: str = "",
) -> str:
    """Assembles the full system prompt from its optional sections.

    Args:
        sandboxed: When False, adds guidance for working inside a real project
            directory (inspect existing files and git history first).
    """
    sections = [base_prompt]

    if tool_docs and TOOL_CATALOGUE_ANCHOR not in base_prompt:
        sections.append(tool_docs)

    if not sandboxed and workspace is not None:
        sections.append(
            f"Context:\nYou are working directly in an existing project directory at `{workspace}`.\n"
            "Inspect existing files and git history to understand the codebase before making changes."
        )

    for extra in (agents_instructions, skills_index):
        if extra:
            sections.append(extra)

    return "\n\n".join(sections)
