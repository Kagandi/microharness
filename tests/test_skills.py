import tempfile
import unittest
from pathlib import Path

from core.skills import (
    Skill,
    compose_skill_prompt,
    discover_skills,
    extract_skill_summary,
    render_skills_index,
    resolve_skill_command,
)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


class TestExtractSummary(unittest.TestCase):
    def test_inline_frontmatter_description(self):
        """Invariant: a single-line frontmatter description is the summary."""
        content = "---\nname: s\ndescription: Run the tests\n---\n# Heading\nbody"
        self.assertEqual(extract_skill_summary(content, "fallback"), "Run the tests")

    def test_quoted_description_is_unquoted(self):
        """Invariant: surrounding quotes in the YAML value are stripped."""
        content = '---\ndescription: "Quoted summary"\n---\nbody'
        self.assertEqual(extract_skill_summary(content, "fallback"), "Quoted summary")

    def test_folded_block_scalar(self):
        """Invariant: a '>' block scalar is folded onto a single line."""
        content = "---\ndescription: >\n  first part\n  second part\n---\n# H"
        self.assertEqual(extract_skill_summary(content, "fallback"), "first part second part")

    def test_literal_block_scalar(self):
        """Invariant: a '|' block scalar is joined with spaces."""
        content = "---\ndescription: |\n  one.\n  two.\n---\n# H"
        self.assertEqual(extract_skill_summary(content, "fallback"), "one. two.")

    def test_block_scalar_ends_at_a_dedented_line(self):
        """Invariant: text after an indented block scalar does not leak into the summary."""
        content = "---\ndescription: >\n  kept\nname: other\n---\n# H"
        self.assertEqual(extract_skill_summary(content, "fallback"), "kept")

    def test_leading_blank_lines_before_frontmatter(self):
        """Invariant: frontmatter is recognized even after leading whitespace."""
        content = "\n\n---\ndescription: Late start\n---\n# H"
        self.assertEqual(extract_skill_summary(content, "fallback"), "Late start")

    def test_falls_back_to_first_heading(self):
        """Invariant: without frontmatter the document's first heading is the summary."""
        self.assertEqual(extract_skill_summary("\n\n# Linter\nRun ruff", "fb"), "Linter")

    def test_falls_back_to_first_prose_line(self):
        """Invariant: a document without headings still gets its first line as summary."""
        self.assertEqual(
            extract_skill_summary("Plain description.\nmore", "fb"), "Plain description."
        )

    def test_empty_document_uses_the_default(self):
        """Invariant: an empty skill file yields the caller-supplied fallback name."""
        self.assertEqual(extract_skill_summary("", "my-skill"), "my-skill")
        self.assertEqual(extract_skill_summary("\n\n  \n", "my-skill"), "my-skill")

    def test_frontmatter_without_description_uses_the_body(self):
        """Invariant: frontmatter carrying no description falls through to the heading."""
        content = "---\nname: s\n---\n# Deployer\nsteps"
        self.assertEqual(extract_skill_summary(content, "fb"), "Deployer")

    def test_unterminated_frontmatter_is_still_parsed(self):
        """Invariant: a truncated document does not lose its description."""
        self.assertEqual(extract_skill_summary("---\ndescription: Still here", "fb"), "Still here")


class TestDiscovery(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.workspace = self.tmp / "ws"
        self.global_dir = self.tmp / "global"
        self.workspace.mkdir()
        self.global_dir.mkdir()

    def test_finds_project_skill_with_relative_path(self):
        """Invariant: a workspace skill is indexed by a path the agent can actually read."""
        _write(
            self.workspace / ".microharness" / "skills" / "run-tests" / "SKILL.md",
            "---\ndescription: Run tests\n---\nbody",
        )
        skills = discover_skills(self.workspace, self.global_dir)
        self.assertEqual(list(skills), ["run-tests"])
        self.assertEqual(skills["run-tests"].scope, "project")
        self.assertEqual(skills["run-tests"].path, ".microharness/skills/run-tests/SKILL.md")
        self.assertEqual(skills["run-tests"].summary, "Run tests")

    def test_global_skill_uses_an_absolute_path(self):
        """Invariant: skills outside the workspace are referenced by absolute path."""
        _write(self.global_dir / "skills" / "deploy" / "SKILL.md", "# Deploy\nship it")
        skills = discover_skills(self.workspace, self.global_dir)
        self.assertEqual(skills["deploy"].scope, "global")
        self.assertTrue(Path(skills["deploy"].path).is_absolute())

    def test_project_skill_shadows_global_skill(self):
        """Invariant: a same-named project skill cleanly replaces the global one."""
        _write(self.global_dir / "skills" / "deploy" / "SKILL.md", "# Global deploy\nglobal body")
        _write(
            self.workspace / ".microharness" / "skills" / "deploy" / "SKILL.md",
            "---\ndescription: Project deploy\n---\nproject body",
        )
        skills = discover_skills(self.workspace, self.global_dir)
        self.assertEqual(len(skills), 1)
        self.assertEqual(skills["deploy"].scope, "project")
        self.assertIn("project body", skills["deploy"].content)

    def test_standalone_markdown_file_is_a_skill(self):
        """Invariant: a bare <name>.md in the skills directory is discovered too."""
        _write(self.workspace / ".microharness" / "skills" / "quick.md", "# Quick\nfast path")
        self.assertIn("quick", discover_skills(self.workspace, self.global_dir))

    def test_readme_is_never_a_skill(self):
        """Invariant: documentation files are not offered as slash commands."""
        _write(self.workspace / ".microharness" / "skills" / "README.md", "# Not a skill")
        self.assertEqual(discover_skills(self.workspace, self.global_dir), {})

    def test_no_skills_directory_yields_empty(self):
        """Invariant: a project without skills is not an error."""
        self.assertEqual(discover_skills(self.workspace, self.global_dir), {})


class TestRenderIndex(unittest.TestCase):
    def _skill(self, name, scope, summary="s"):
        return Skill(name=name, summary=summary, path=f"{name}/SKILL.md", scope=scope, content="c")

    def test_empty_index_is_empty_string(self):
        """Invariant: no skills means no index section in the prompt."""
        self.assertEqual(render_skills_index({}), "")

    def test_scope_badges_hidden_when_all_skills_are_project(self):
        """Invariant: a single-scope setup keeps the prompt uncluttered."""
        index = render_skills_index({"a": self._skill("a", "project")})
        self.assertNotIn("[project]", index)
        self.assertIn("- a: s", index)

    def test_scope_badges_shown_when_scopes_are_mixed(self):
        """Invariant: mixing scopes makes provenance visible per entry."""
        index = render_skills_index(
            {"a": self._skill("a", "project"), "b": self._skill("b", "global")}
        )
        self.assertIn("[project] a", index)
        self.assertIn("[global] b", index)

    def test_index_mentions_lazy_loading_and_reads_the_path(self):
        """Invariant: the index points at the file the agent should read on demand."""
        index = render_skills_index({"a": self._skill("a", "global")})
        self.assertIn("read `a/SKILL.md`", index)
        self.assertIn("read_file", index)


class TestComposePrompt(unittest.TestCase):
    SKILL = Skill(name="deploy", summary="s", path="p", scope="project", content="STEP 1")

    def test_with_instruction_adds_a_task_section(self):
        """Invariant: an explicit instruction becomes the task under the skill body."""
        prompt = compose_skill_prompt(self.SKILL, "deploy to staging")
        self.assertEqual(prompt, "[Active Skill: deploy]\nSTEP 1\n\nTask:\ndeploy to staging")

    def test_without_instruction_asks_the_agent_to_follow_the_skill(self):
        """Invariant: no instruction yields an explicit follow-the-skill instruction."""
        prompt = compose_skill_prompt(self.SKILL)
        self.assertIn("[Active Skill: deploy]", prompt)
        self.assertIn("Please follow the instructions from the 'deploy' skill above.", prompt)


class TestResolveSkillCommand(unittest.TestCase):
    def setUp(self):
        self.skills = {
            "deploy": Skill("deploy", "Deploy app", "d/SKILL.md", "project", "DEPLOY BODY"),
            "lint": Skill("lint", "Lint code", "l/SKILL.md", "global", "LINT BODY"),
        }

    def test_listing_mode(self):
        """Invariant: a bare /skill lists every skill with its scope and summary."""
        result = resolve_skill_command("/skill", self.skills)
        self.assertEqual(result.action, "skip")
        self.assertFalse(result.is_error)
        self.assertIsNone(result.prompt)
        self.assertIn("1. deploy", result.message)
        self.assertIn("[project]", result.message)
        self.assertIn("Example: /skill deploy", result.message)

    def test_listing_mode_without_any_skills_explains_where_to_add_them(self):
        """Invariant: an empty registry tells the user which directories to create."""
        result = resolve_skill_command("/skill", {})
        self.assertIn(".microharness/skills", result.message)
        self.assertIn("~/.microharness/skills", result.message)

    def test_activation_by_name(self):
        """Invariant: activating a skill returns a prompt carrying its full body."""
        result = resolve_skill_command("/skill deploy", self.skills)
        self.assertEqual(result.action, "turn")
        self.assertEqual(result.message, "Activated skill: deploy [project]")
        self.assertIn("DEPLOY BODY", result.prompt)

    def test_activation_by_number(self):
        """Invariant: the 1-based index in the listing selects the same skill as its name."""
        result = resolve_skill_command("/skill 1", self.skills)
        self.assertIn("[Active Skill: deploy]", result.prompt)

    def test_activation_is_case_insensitive(self):
        """Invariant: skill names are matched case-insensitively."""
        self.assertEqual(resolve_skill_command("/skill DEPLOY", self.skills).action, "turn")

    def test_activation_with_instruction(self):
        """Invariant: trailing input becomes the task for the activated skill."""
        result = resolve_skill_command("/skill deploy deploy to staging", self.skills)
        self.assertIn("Task:\ndeploy to staging", result.prompt)

    def test_unknown_skill_is_an_error(self):
        """Invariant: an unknown name is reported, never silently treated as a prompt."""
        result = resolve_skill_command("/skill nope", self.skills)
        self.assertEqual(result.action, "skip")
        self.assertTrue(result.is_error)
        self.assertEqual(
            result.message, "Skill 'nope' not found. Type /skill to view available skills."
        )

    def test_out_of_range_number_is_an_error(self):
        """Invariant: an index past the end reports the valid range."""
        result = resolve_skill_command("/skill 99", self.skills)
        self.assertTrue(result.is_error)
        self.assertIn("Invalid skill number 99", result.message)

    def test_number_zero_is_rejected(self):
        """Invariant: indices are 1-based, so zero is out of range rather than the first entry."""
        self.assertTrue(resolve_skill_command("/skill 0", self.skills).is_error)


if __name__ == "__main__":
    unittest.main()
