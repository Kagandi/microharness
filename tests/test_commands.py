import tempfile
import unittest
from pathlib import Path

from core.commands import (
    Command,
    discover_commands,
    expand_command,
    resolve_slash_input,
    split_slash_input,
)
from core.skills import Skill


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _command(name="review", description="Review PR", template="git diff", scope="project"):
    return Command(name=name, description=description, template=template, scope=scope)


class TestDiscoverCommands(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.workspace = self.tmp / "ws"
        self.global_dir = self.tmp / "global"
        self.workspace.mkdir()
        self.global_dir.mkdir()

    def test_description_comes_from_the_first_non_empty_line(self):
        """Invariant: the first line becomes the one-line description shown in /help."""
        _write(
            self.global_dir / "commands" / "explain.md", "# Explain codebase\nExplain $ARGUMENTS"
        )
        commands = discover_commands(self.workspace, self.global_dir)
        self.assertEqual(commands["explain"].description, "Explain codebase")
        self.assertEqual(commands["explain"].scope, "global")
        self.assertIn("Explain $ARGUMENTS", commands["explain"].template)

    def test_names_are_lowercased(self):
        """Invariant: a file named Review.md is reachable as /review."""
        _write(self.workspace / ".microharness" / "commands" / "Review.md", "# Review")
        self.assertIn("review", discover_commands(self.workspace, self.global_dir))

    def test_project_command_shadows_global(self):
        """Invariant: a same-named project command cleanly replaces the global one."""
        _write(self.global_dir / "commands" / "review.md", "# Global review\nglobal template")
        _write(
            self.workspace / ".microharness" / "commands" / "review.md",
            "# Project review\nproject template",
        )
        commands = discover_commands(self.workspace, self.global_dir)
        self.assertEqual(commands["review"].scope, "project")
        self.assertIn("project template", commands["review"].template)
        self.assertNotIn("global template", commands["review"].template)

    def test_no_commands_directory_yields_empty(self):
        """Invariant: a project without commands is not an error."""
        self.assertEqual(discover_commands(self.workspace, self.global_dir), {})


class TestExpandCommand(unittest.TestCase):
    def test_dollar_placeholder(self):
        """Invariant: $ARGUMENTS is replaced in place."""
        self.assertEqual(
            expand_command("Review changes in $ARGUMENTS carefully.", "auth.py"),
            "Review changes in auth.py carefully.",
        )

    def test_brace_placeholder(self):
        """Invariant: {args} is an accepted alternative spelling."""
        self.assertEqual(expand_command("Run {args} now.", "tests"), "Run tests now.")

    def test_arguments_are_appended_when_no_placeholder_exists(self):
        """Invariant: extra input is never dropped; it is appended as extra context."""
        self.assertEqual(
            expand_command("Run all benchmarks.", "--quick"),
            "Run all benchmarks.\n\nAdditional Arguments / Target:\n--quick",
        )

    def test_empty_arguments_return_the_trimmed_template(self):
        """Invariant: a bare /command expands to exactly its template."""
        self.assertEqual(expand_command("  Run all.  ", "   "), "Run all.")

    def test_placeholder_wins_over_appending(self):
        """Invariant: when a placeholder exists, nothing is appended on top of it."""
        self.assertEqual(expand_command("Do $ARGUMENTS", "x"), "Do x")


class TestSplitSlashInput(unittest.TestCase):
    def test_name_and_arguments(self):
        """Invariant: the command word is separated from its arguments on the first space."""
        self.assertEqual(split_slash_input("/review src/auth.py"), ("review", "src/auth.py"))

    def test_name_only(self):
        """Invariant: a bare command yields empty arguments."""
        self.assertEqual(split_slash_input("/review"), ("review", ""))

    def test_name_is_lowercased(self):
        """Invariant: command lookup is case-insensitive."""
        self.assertEqual(split_slash_input("/REVIEW a"), ("review", "a"))

    def test_slash_only_input(self):
        """Invariant: a lone slash parses to an empty key rather than raising."""
        self.assertEqual(split_slash_input("/"), ("", ""))


class TestResolveSlashInput(unittest.TestCase):
    def setUp(self):
        self.commands = {"review": _command()}
        self.skills = {"deploy": Skill("deploy", "Deploy", "d/SKILL.md", "project", "BODY")}

    def test_custom_command_expands_its_template(self):
        """Invariant: a custom command runs its template with the trailing arguments."""
        outcome = resolve_slash_input("/review src/auth.py", self.commands, self.skills)
        self.assertEqual(outcome.action, "turn")
        self.assertFalse(outcome.is_error)
        self.assertEqual(outcome.prompt, "git diff\n\nAdditional Arguments / Target:\nsrc/auth.py")
        self.assertIn("[project]", outcome.message)

    def test_skill_can_be_invoked_directly(self):
        """Invariant: /<skill> activates the skill, as /skill <name> does."""
        outcome = resolve_slash_input("/deploy now", self.commands, self.skills)
        self.assertEqual(outcome.action, "turn")
        self.assertIn("[Active Skill: deploy]", outcome.prompt)
        self.assertIn("Task:\nnow", outcome.prompt)

    def test_unknown_command_is_reported_as_an_error(self):
        """Invariant: an unknown slash command never becomes a model turn."""
        outcome = resolve_slash_input("/nope arg", self.commands, self.skills)
        self.assertEqual(outcome.action, "skip")
        self.assertTrue(outcome.is_error)
        self.assertIsNone(outcome.prompt)
        self.assertIn("Unknown command '/nope'", outcome.message)

    def test_custom_hint_is_configurable(self):
        """Invariant: each frontend can word its own unknown-command hint."""
        outcome = resolve_slash_input("/nope", self.commands, unknown_message="Ask for /help.")
        self.assertIn("Ask for /help.", outcome.message)

    def test_skills_can_be_disabled_for_batch_runs(self):
        """Invariant: batch runs expand custom commands only, leaving skill text alone."""
        outcome = resolve_slash_input("/deploy now", self.commands, self.skills, allow_skills=False)
        self.assertTrue(outcome.is_error)


if __name__ == "__main__":
    unittest.main()
