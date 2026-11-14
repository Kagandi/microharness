import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.commands import Command
from core.interactive import (
    COMMAND_SPECS,
    EXIT,
    SKIP,
    TURN,
    CommandContext,
    CommandRouter,
    TurnState,
    command_argument,
    expand_specs,
    render_cache_hits,
    render_context_breakdown,
    render_help_lines,
    render_model_lines,
    render_session_lines,
)
from core.memory import Memory
from core.skills import Skill

SKILL = Skill("deploy", "Deploy app", "d/SKILL.md", "project", "DEPLOY BODY")
COMMAND = Command("review", "Review PR", "Review $ARGUMENTS", "project")


class RecordingUI:
    """SessionUI sink that keeps its output instead of printing it."""

    def __init__(self):
        self.infos: list[str] = []
        self.successes: list[str] = []
        self.errors: list[str] = []
        self.statuses: list[str] = []
        self.echoes: list[str] = []
        self.refreshes = 0

    def info(self, text):
        self.infos.append(text)

    def success(self, text):
        self.successes.append(text)

    def error(self, text):
        self.errors.append(text)

    def status(self, text):
        self.statuses.append(text)

    def echo(self, text):
        self.echoes.append(text)

    def refresh(self):
        self.refreshes += 1

    def render_context(self, stats, reasoning, cache):
        return render_context_breakdown(stats, reasoning, cache)

    @property
    def all_text(self) -> str:
        return "\n".join(self.infos + self.successes + self.errors + self.statuses)


class RouterTestCase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        (self.tmp / "sample.py").write_text("print('hi')", encoding="utf-8")
        self.memory = Memory(system_prompt="sys", user_prompt="")
        self.ui = RecordingUI()
        self.turns = TurnState()
        self.harness = MagicMock()
        self.harness.get_available_models.return_value = [
            {"name": "fast", "model": "gemini", "provider": "google", "description": "Fast"}
        ]
        self.harness.active_model_name.return_value = "gemini"
        self.harness.find_recent_sessions.return_value = []
        self.harness.get_reasoning_effort.return_value = None
        self.harness.get_cache_stats.return_value = {"cache_ttl": "5m"}
        self.harness.switch_model.return_value = (True, "Switched model to gemini", MagicMock())
        self.harness.load_session.return_value = ([], {"error": "Session not found"})
        self.router = self._router()

    def _router(self, commands=None, skills=None, allow_bare_words=True):
        ctx = CommandContext(
            harness=self.harness,
            memory=self.memory,
            ui=self.ui,
            workspace=self.tmp,
            commands=commands if commands is not None else {},
            skills=skills if skills is not None else {},
            tracer=None,
            turns=self.turns,
        )
        return CommandRouter(ctx, allow_bare_words=allow_bare_words)


class TestCommandSpecs(unittest.TestCase):
    def test_slashless_aliases_are_derived_from_the_canonical_names(self):
        """Invariant: '/status' and 'status' resolve to the same handler."""
        expanded = {spec.key: spec for spec in expand_specs(COMMAND_SPECS)}
        self.assertIn("status", expanded["status"].bare_names)
        self.assertIn("exit", expanded["exit"].bare_names)

    def test_extra_aliases_are_added(self):
        """Invariant: aliases not derivable from the names are declared explicitly."""
        from core.interactive import COMMAND_SPECS

        expanded = {spec.key: spec for spec in expand_specs(COMMAND_SPECS)}
        self.assertIn("quit", expanded["exit"].bare_names)
        self.assertIn("context", expanded["status"].bare_names)

    def test_bare_words_never_absorb_arguments(self):
        """Invariant: prose like 'model the user database' is not treated as a command."""
        from core.interactive import COMMAND_SPECS

        expanded = {spec.key: spec for spec in expand_specs(COMMAND_SPECS)}
        self.assertFalse(expanded["model"].matches("model the user database"))
        self.assertTrue(expanded["model"].matches("/model fast"))

    def test_takes_args_matches_the_prefix_only(self):
        """Invariant: a longer word starting with a command name is not a command."""
        from core.interactive import COMMAND_SPECS

        expanded = {spec.key: spec for spec in expand_specs(COMMAND_SPECS)}
        self.assertTrue(expanded["skill"].matches("/skill deploy"))
        self.assertFalse(expanded["skill"].matches("/skillboard"))


class TestCommandArgument(unittest.TestCase):
    def test_returns_the_tail(self):
        """Invariant: arguments are everything after the first space."""
        self.assertEqual(command_argument("/model  fast "), "fast")

    def test_returns_empty_when_absent(self):
        """Invariant: a bare command has no argument."""
        self.assertEqual(command_argument("/model"), "")
        self.assertEqual(command_argument("  /model  "), "")


class TestRouterDispatch(RouterTestCase):
    def test_plain_text_becomes_a_turn(self):
        """Invariant: ordinary user input is appended to memory and queued as a turn."""
        self.assertEqual(self.router.dispatch("Run the tests"), TURN)
        self.assertEqual(self.memory.messages[-1].content, "Run the tests")
        self.assertEqual(self.ui.echoes, ["Run the tests"])

    def test_file_tags_are_expanded_into_the_prompt(self):
        """Invariant: '@file' references are inlined and reported back to the user."""
        self.router.dispatch("Review @sample.py")
        self.assertIn("[Attached File: @sample.py]", self.memory.messages[-1].content)
        self.assertIn("Attached 1 file(s)", self.ui.all_text)

    def test_exit_command(self):
        """Invariant: /exit ends the session."""
        self.assertEqual(self.router.dispatch("/exit"), EXIT)

    def test_help_lists_builtins_and_discovered_resources(self):
        """Invariant: /help shows built-in commands plus discovered commands and skills."""
        router = self._router(commands={"review": COMMAND}, skills={"deploy": SKILL})
        router.dispatch("/help")
        text = self.ui.all_text
        self.assertIn("/model", text)
        self.assertIn("/review", text)
        self.assertIn("[project] Review PR", text)
        self.assertIn("/deploy", text)

    def test_compact_reports_freed_tokens(self):
        """Invariant: a compaction that did work reports how much context it freed."""
        with (
            patch.object(Memory, "compact_if_needed", return_value=True),
            patch.object(Memory, "estimated_tokens", side_effect=[2000, 1000]),
        ):
            self.assertEqual(self.router.dispatch("/compact"), SKIP)
        self.assertIn("freed ~1,000 tokens", self.ui.all_text)

    def test_compact_reports_when_nothing_was_eligible(self):
        """Invariant: a no-op compaction is reported, not silently ignored."""
        with patch.object(Memory, "compact_if_needed", return_value=False):
            self.router.dispatch("/compact")
        self.assertIn("No older observations eligible", self.ui.all_text)

    def test_new_resets_memory(self):
        """Invariant: /new clears conversation history without touching the workspace."""
        self.router.dispatch("something")
        self.assertEqual(self.router.dispatch("/new"), SKIP)
        self.assertEqual([m.role for m in self.memory.messages], ["system"])
        self.assertIn("fresh conversation", self.ui.all_text)

    def test_model_without_argument_lists_models(self):
        """Invariant: a bare /model lists the configured models."""
        self.router.dispatch("/model")
        self.assertIn("fast", self.ui.all_text)
        self.assertIn("[google]", self.ui.all_text)

    def test_model_with_argument_switches(self):
        """Invariant: /model <name> switches the active model and reports success."""
        self.assertEqual(self.router.dispatch("/model fast"), SKIP)
        self.harness.switch_model.assert_called_once_with("fast")
        self.assertIn("Switched model to gemini", self.ui.successes)

    def test_model_switch_synchronizes_state_and_runner(self):
        """Invariant: /model synchronizes state.model_name, state.provider_name, and runner.model."""
        new_model = MagicMock()
        new_model.provider = "openrouter"
        self.harness.switch_model.return_value = (True, "Switched model to claude", new_model)
        self.harness.active_model_name.return_value = "claude-5.5-sonnet"

        mock_state = MagicMock()
        self.ui.state = mock_state
        mock_runner = MagicMock()
        self.router.ctx.runner = mock_runner

        self.router.dispatch("/model smart")
        self.assertEqual(mock_state.model_name, "claude-5.5-sonnet")
        self.assertEqual(mock_state.provider_name, "openrouter")
        self.assertEqual(mock_runner.model, new_model)

    def test_model_switch_failure_is_reported_as_an_error(self):
        """Invariant: a rejected switch surfaces as an error, not a success."""
        self.harness.switch_model.return_value = (False, "Unknown model 'x'", MagicMock())
        self.router.dispatch("/model x")
        self.assertEqual(self.ui.errors, ["Unknown model 'x'"])

    def test_resume_without_argument_lists_sessions(self):
        """Invariant: a bare /resume lists recent sessions."""
        self.harness.find_recent_sessions.return_value = [
            {
                "run_id": "abc",
                "run_dir": "/x",
                "total_steps": 3,
                "preview": "hello",
                "matches_workspace": True,
            }
        ]
        self.router.dispatch("/resume")
        self.assertIn("abc", self.ui.all_text)
        self.assertIn("[this repo]", self.ui.all_text)

    def test_resume_failure_reports_the_harness_error(self):
        """Invariant: the harness's own error text is surfaced verbatim."""
        self.router.dispatch("/resume 9")
        self.assertEqual(self.ui.errors, ["Session not found"])

    def test_resume_restores_memory_and_advances_the_step_index(self):
        """Invariant: restoring a session continues numbering after its recorded steps."""
        from core.types import Message

        self.harness.load_session.return_value = (
            [Message(role="user", content="old"), Message(role="assistant", content="reply")],
            {"run_id": "abc", "total_steps": 7},
        )
        self.assertEqual(self.router.dispatch("/resume 1"), SKIP)
        self.assertEqual(self.turns.step_index, 8)
        self.assertEqual([m.content for m in self.memory.messages][1:], ["old", "reply"])

    def test_skill_listing_and_activation(self):
        """Invariant: /skill lists skills and then activates one into memory."""
        router = self._router(skills={"deploy": SKILL})
        self.assertEqual(router.dispatch("/skill"), SKIP)
        self.assertIn("deploy", self.ui.all_text)

        self.assertEqual(router.dispatch("/skill deploy ship it"), TURN)
        self.assertIn("[Active Skill: deploy]", self.memory.messages[-1].content)
        self.assertIn("Task:\nship it", self.memory.messages[-1].content)

    def test_skill_error_is_reported(self):
        """Invariant: an unknown skill name is an error, not a silent turn."""
        router = self._router(skills={"deploy": SKILL})
        self.assertEqual(router.dispatch("/skill nope"), SKIP)
        self.assertIn("not found", self.ui.errors[0])

    def test_custom_command_runs_its_template(self):
        """Invariant: a custom command expands its template into a turn."""
        router = self._router(commands={"review": COMMAND})
        self.assertEqual(router.dispatch("/review src/auth.py"), TURN)
        self.assertIn("Review src/auth.py", self.memory.messages[-1].content)
        self.assertIn("/review [project]", self.ui.infos[0])

    def test_direct_skill_invocation_runs_the_skill(self):
        """Invariant: '/deploy' activates the skill without the /skill prefix."""
        router = self._router(skills={"deploy": SKILL})
        self.assertEqual(router.dispatch("/deploy ship"), TURN)
        self.assertIn("[Active Skill: deploy]", self.memory.messages[-1].content)

    def test_unknown_slash_command_is_an_error(self):
        """Invariant: an unrecognized slash command never reaches the model."""
        self.assertEqual(self.router.dispatch("/nope"), SKIP)
        self.assertIn("Unknown command '/nope'", self.ui.errors[0])
        self.assertEqual(len(self.memory.messages), 1)

    def test_status_renders_the_context_report(self):
        """Invariant: /status reports context usage through the sink."""
        self.assertEqual(self.router.dispatch("/status"), SKIP)
        self.assertIn("Context Breakdown", self.ui.all_text)
        self.assertEqual(self.ui.refreshes, 1)

    def test_reasoning_view_and_change(self):
        """Invariant: /reasoning reports the level and switches it on request."""
        self.harness.get_reasoning_effort.return_value = "low"
        self.router.dispatch("/reasoning")
        self.assertIn("LOW", self.ui.infos[0])

        self.harness.set_reasoning_effort.return_value = "high"
        self.router.dispatch("/reasoning high")
        self.assertIn("switched to: HIGH", self.ui.successes[0])

    def test_reasoning_switch_synchronizes_state(self):
        """Invariant: /reasoning synchronizes state.reasoning."""
        mock_state = MagicMock()
        self.ui.state = mock_state
        self.harness.set_reasoning_effort.return_value = "high"
        self.router.dispatch("/reasoning high")
        self.assertEqual(mock_state.reasoning, "HIGH")

    def test_reasoning_invalid_value_is_an_error(self):
        """Invariant: a rejected effort level surfaces the validation message."""
        self.harness.set_reasoning_effort.side_effect = ValueError("bad level")
        self.assertEqual(self.router.dispatch("/reasoning wild"), SKIP)
        self.assertEqual(self.ui.errors, ["bad level"])

    def test_cache_view_and_change(self):
        """Invariant: /cache reports the TTL and switches it on request."""
        self.router.dispatch("/cache")
        self.assertIn("5M", self.ui.infos[0])

        self.harness.set_cache_ttl.return_value = "1h"
        self.router.dispatch("/cache 1h")
        self.assertIn("switched to: 1H", self.ui.successes[0])

    def test_cache_invalid_value_is_an_error(self):
        """Invariant: a rejected TTL surfaces the validation message."""
        self.harness.set_cache_ttl.side_effect = ValueError("bad ttl")
        self.assertEqual(self.router.dispatch("/cache 17m"), SKIP)
        self.assertEqual(self.ui.errors, ["bad ttl"])

    def test_bare_words_can_be_disabled_for_strict_frontends(self):
        """Invariant: a frontend that requires a slash is not hijacked by plain prose."""
        router = self._router(allow_bare_words=False)
        self.assertEqual(router.dispatch("status"), TURN)
        self.assertEqual(router.dispatch("/status"), SKIP)


class TestRenderers(unittest.TestCase):
    def test_help_footer_is_appended_when_requested(self):
        """Invariant: frontends can add their own hints to the shared help block."""
        lines = render_help_lines({}, {}, footer="Press Esc to exit.")
        self.assertEqual(lines[-1], "Press Esc to exit.")

    def test_model_listing_marks_the_active_entry(self):
        """Invariant: the active model is identifiable in the /model listing."""
        lines = render_model_lines(
            [{"name": "fast", "model": "gemini", "provider": "google", "description": "Fast"}],
            "gemini",
        )
        self.assertEqual(lines, ["  1. fast [active] - [google] Fast"])

    def test_model_listing_without_configured_models(self):
        """Invariant: with nothing configured the active model is still shown."""
        self.assertEqual(render_model_lines([], "current"), ["  • current [active]"])
        self.assertEqual(render_model_lines([], ""), ["  (none configured)"])

    def test_session_listing_without_sessions(self):
        """Invariant: an empty runs directory is stated, not rendered as a blank list."""
        self.assertEqual(
            render_session_lines([]), ["  No previous sessions found in runs directory."]
        )

    def test_cache_hits_hidden_until_tokens_are_recorded(self):
        """Invariant: no prompt tokens means no cache line to show."""
        self.assertIsNone(render_cache_hits({}))
        self.assertIsNone(render_cache_hits({"total_prompt_tokens": 0}))

    def test_context_breakdown_includes_roles_and_settings(self):
        """Invariant: the report covers usage, message mix, reasoning, and caching."""
        stats = {
            "estimated_tokens": 100,
            "max_estimated_tokens": 1000,
            "usage_pct": 10.0,
            "total_messages": 4,
            "by_role": {"system": 1, "user": 1, "assistant": 1, "tool": 1},
        }
        lines = render_context_breakdown(stats, "low", {"cache_ttl": "5m"})
        text = "\n".join(lines)
        self.assertIn("100 / 1,000 tokens (10.0%)", text)
        self.assertIn("1 sys, 1 user, 1 assistant, 1 tools", text)
        self.assertIn("Reasoning Effort: LOW", text)
        self.assertIn("Prompt Caching: 5M", text)

    def test_context_breakdown_reports_disabled_settings(self):
        """Invariant: unset reasoning and caching are shown as OFF, not blank."""
        stats = {
            "estimated_tokens": 0,
            "max_estimated_tokens": 100,
            "usage_pct": 0.0,
            "total_messages": 1,
            "by_role": {"system": 1, "user": 0, "assistant": 0, "tool": 0},
        }
        text = "\n".join(render_context_breakdown(stats, None, {}))
        self.assertIn("Reasoning Effort: OFF", text)
        self.assertIn("Prompt Caching: OFF", text)


class TestRealMemoryIntegration(RouterTestCase):
    """The router against a real Memory instead of a double."""

    def test_turn_adds_exactly_one_user_message(self):
        """Invariant: dispatching a turn appends exactly one user message."""
        before = len(self.memory.messages)
        self.router.dispatch("do the thing")
        self.assertEqual(len(self.memory.messages), before + 1)
        self.assertEqual(self.memory.messages[-1].role, "user")

    def test_skip_commands_never_touch_memory(self):
        """Invariant: session-control commands leave the conversation untouched."""
        for command in ("/help", "/status", "/compact", "/new", "/model", "/resume"):
            self.router.dispatch(command)
        self.assertEqual([m.content for m in self.memory.messages], ["sys"])


if __name__ == "__main__":
    unittest.main()
