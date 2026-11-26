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

DEFAULT_FINISH_TOOL_BULLET = (
    "- Once the task is fully satisfied and verified, provide a final response summarizing "
    "your solution without calling any more tools."
)
FINISH_TOOL_BULLET = (
    "- Once the task is fully satisfied and verified, call the `finish` tool with a concise "
    "summary of your solution and verification results. Do not stop with a plain text status "
    "message while work remains."
)
FINISH_TOOL_INSTRUCTION = (
    "Completion:\n"
    "When the task is completely finished and verified, call the `finish` tool with your final "
    "summary instead of ending with a plain text message."
)

TOOL_CATALOGUE_ANCHOR = "You have access to tools to interact with the environment:"

GLOBAL_INSTRUCTIONS_HEADER = "--- Global User Instructions (~/.microharness/AGENTS.md) ---"
PROJECT_INSTRUCTIONS_HEADER = "--- Project Instructions ({name}) ---"


def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8", errors="replace").strip()


def load_agents_instructions(
    workspace: Path,
    global_dir: Path | None = None,
    *,
    include_workspace: bool = True,
) -> str:
    """Loads global and project-level steering guidelines in cascading order.

    Global rules come first; the project file (``.microharness/AGENTS.md`` or
    ``./AGENTS.md``) specializes them and therefore comes last.
    """
    sections: list[str] = []

    if global_dir:
        global_file = global_dir / "AGENTS.md"
        if global_file.is_file() and (content := _read(global_file)):
            sections.append(f"{GLOBAL_INSTRUCTIONS_HEADER}\n{content}\n" + "-" * 58)

    if include_workspace:
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
    subagents_index: str = "",
    require_finish_tool: bool = False,
) -> str:
    """Assembles the full system prompt from its optional sections.

    Args:
        sandboxed: When False, adds guidance for working inside a real project
            directory (inspect existing files and git history first).
        require_finish_tool: When True, instructs the model to terminate via
            the ``finish`` tool instead of stopping on plain text.
    """
    effective_base = base_prompt
    append_finish_section = False
    if require_finish_tool:
        if DEFAULT_FINISH_TOOL_BULLET in effective_base:
            effective_base = effective_base.replace(DEFAULT_FINISH_TOOL_BULLET, FINISH_TOOL_BULLET)
        else:
            append_finish_section = True

    sections = [effective_base]

    if tool_docs and TOOL_CATALOGUE_ANCHOR not in effective_base:
        sections.append(tool_docs)

    if append_finish_section:
        sections.append(FINISH_TOOL_INSTRUCTION)

    if not sandboxed and workspace is not None:
        sections.append(
            f"Context:\nYou are working directly in an existing project directory at `{workspace}`.\n"
            "Inspect existing files and git history to understand the codebase before making changes."
        )

    for extra in (agents_instructions, skills_index, subagents_index):
        if extra:
            sections.append(extra)

    return "\n\n".join(sections)
