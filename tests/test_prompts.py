import tempfile
import unittest
from pathlib import Path

from core.prompts import (
    DEFAULT_SYSTEM_PROMPT,
    TOOL_CATALOGUE_ANCHOR,
    build_system_prompt,
    load_agents_instructions,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestLoadAgentsInstructions(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.workspace = self.tmp / "ws"
        self.global_dir = self.tmp / "global"
        self.workspace.mkdir()
        self.global_dir.mkdir()

    def test_no_files_yields_empty_string(self):
        """Invariant: a project with no steering files adds nothing to the prompt."""
        self.assertEqual(load_agents_instructions(self.workspace, self.global_dir), "")

    def test_project_only(self):
        """Invariant: a project AGENTS.md is included under its own header."""
        _write(self.workspace / "AGENTS.md", "Use tabs.")
        result = load_agents_instructions(self.workspace, self.global_dir)
        self.assertIn("--- Project Instructions (AGENTS.md) ---", result)
        self.assertIn("Use tabs.", result)

    def test_project_namespace_file_wins_over_root(self):
        """Invariant: .microharness/AGENTS.md specializes the root file and takes precedence."""
        _write(self.workspace / "AGENTS.md", "root rules")
        _write(self.workspace / ".microharness" / "AGENTS.md", "namespaced rules")
        result = load_agents_instructions(self.workspace, self.global_dir)
        self.assertIn("namespaced rules", result)
        self.assertNotIn("root rules", result)

    def test_global_baseline_comes_before_project(self):
        """Invariant: cascading order puts the user baseline first and the project last."""
        _write(self.global_dir / "AGENTS.md", "global rules")
        _write(self.workspace / "AGENTS.md", "project rules")
        result = load_agents_instructions(self.workspace, self.global_dir)
        self.assertLess(result.index("global rules"), result.index("project rules"))
        self.assertIn("--- Global User Instructions (~/.microharness/AGENTS.md) ---", result)

    def test_works_without_a_global_dir(self):
        """Invariant: project steering loads even when no global directory exists."""
        _write(self.workspace / "AGENTS.md", "project rules")
        self.assertIn("project rules", load_agents_instructions(self.workspace, None))

    def test_empty_files_are_ignored(self):
        """Invariant: a blank steering file does not add an empty section."""
        _write(self.global_dir / "AGENTS.md", "   \n")
        self.assertEqual(load_agents_instructions(self.workspace, self.global_dir), "")


class TestBuildSystemPrompt(unittest.TestCase):
    def test_base_prompt_only(self):
        """Invariant: with no extras the prompt is exactly the base prompt."""
        self.assertEqual(build_system_prompt("BASE"), "BASE")

    def test_sections_are_joined_in_precedence_order(self):
        """Invariant: tools, then project context, then steering, then skills."""
        prompt = build_system_prompt(
            "BASE",
            tool_docs="TOOLS",
            workspace=Path("/ws"),
            sandboxed=False,
            agents_instructions="AGENTS",
            skills_index="SKILLS",
        )
        self.assertEqual(prompt, "BASE\n\nTOOLS\n\nContext:\nYou are working directly in "
                                 "an existing project directory at `/ws`.\nInspect existing "
                                 "files and git history to understand the codebase before "
                                 "making changes.\n\nAGENTS\n\nSKILLS")

    def test_tool_docs_are_not_duplicated(self):
        """Invariant: a base prompt that already catalogues its tools is left alone."""
        base = f"BASE\n{TOOL_CATALOGUE_ANCHOR}\n  - bash"
        self.assertEqual(build_system_prompt(base, tool_docs="TOOLS"), base)

    def test_sandboxed_prompts_omit_project_context(self):
        """Invariant: in a disposable sandbox there is no existing project to inspect."""
        prompt = build_system_prompt("BASE", workspace=Path("/ws"), sandboxed=True)
        self.assertEqual(prompt, "BASE")

    def test_blank_sections_are_skipped(self):
        """Invariant: empty extras never produce stray blank-line runs."""
        prompt = build_system_prompt("BASE", tool_docs="", agents_instructions="", skills_index="")
        self.assertEqual(prompt, "BASE")

    def test_default_system_prompt_is_non_empty(self):
        """Invariant: the shipped default prompt states the agent's working guidelines."""
        self.assertIn("autonomous software engineering assistant", DEFAULT_SYSTEM_PROMPT)
        self.assertIn("Run tests to verify", DEFAULT_SYSTEM_PROMPT)


if __name__ == "__main__":
    unittest.main()