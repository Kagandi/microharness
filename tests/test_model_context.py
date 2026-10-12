import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from core.harness import Harness
from core.interactive import CommandContext, CommandRouter, TurnState
from core.memory import Memory
from core.model import MockModel
from core.providers import resolve_model_context_tokens


class StubUI:
    """Minimal SessionUI sink for router tests."""

    def __init__(self):
        self.successes: list[str] = []
        self.errors: list[str] = []
        self.refreshes = 0

    def info(self, text):
        pass

    def success(self, text):
        self.successes.append(text)

    def error(self, text):
        self.errors.append(text)

    def status(self, text):
        pass

    def echo(self, text):
        pass

    def refresh(self):
        self.refreshes += 1

    def render_context(self, stats, reasoning, cache):
        return []


def _harness_with_models(models: dict, **kwargs) -> Harness:
    tmp = tempfile.mkdtemp()
    ws = Path(tmp)
    cfg_dir = ws / ".microharness"
    cfg_dir.mkdir()
    (cfg_dir / "config.json").write_text(json.dumps({"models": models}), encoding="utf-8")
    runs = Path(tempfile.mkdtemp())
    return Harness(
        model=MockModel([("hi", [])]),
        workspace_dir=ws,
        runs_dir=runs,
        verbose=False,
        **kwargs,
    )


class TestResolveModelContextTokens(unittest.TestCase):
    def test_direct_dict_wins_over_everything(self):
        """Invariant: a mapping spec's own max_context_tokens beats alias, profile, globals."""
        cfg = {
            "models": {"big": {"model": "openai/gpt", "max_context_tokens": 200000}},
            "max_context_tokens": 64000,
            "max_tokens": 32000,
            "context_tokens": 16000,
        }
        registry = {"openai": {"context_tokens": 128000}}
        self.assertEqual(
            resolve_model_context_tokens(
                {"model": "openai/gpt", "max_context_tokens": 5000}, cfg, registry
            ),
            5000,
        )

    def test_alias_entry_beats_profile_and_globals(self):
        """Invariant: a config alias dict entry beats the provider profile and globals."""
        cfg = {
            "models": {"big": {"model": "openai/gpt", "max_context_tokens": 200000}},
            "max_context_tokens": 64000,
        }
        registry = {"openai": {"context_tokens": 128000}}
        self.assertEqual(resolve_model_context_tokens("big", cfg, registry), 200000)

    def test_profile_beats_globals(self):
        """Invariant: provider-profile context_tokens beats the global triple."""
        cfg = {"max_context_tokens": 64000}
        registry = {"openai": {"context_tokens": 128000}}
        self.assertEqual(resolve_model_context_tokens("openai/gpt-4o", cfg, registry), 128000)

    def test_bare_provider_name_falls_to_profile(self):
        """Invariant: a bare provider name resolves to that profile's context_tokens."""
        cfg = {"max_context_tokens": 64000}
        registry = {"openai": {"context_tokens": 128000}}
        self.assertEqual(resolve_model_context_tokens("openai", cfg, registry), 128000)

    def test_string_alias_entry_falls_through_to_profile(self):
        """Invariant: a string alias entry is not a cap; resolution falls to the profile."""
        cfg = {"models": {"s": "openai/gpt-4o"}, "max_context_tokens": 64000}
        registry = {"openai": {"context_tokens": 128000}}
        self.assertEqual(resolve_model_context_tokens("s", cfg, registry), 128000)

    def test_startup_dict_with_alias_model_honors_alias_entry(self):
        """Invariant: a startup-style dict naming an alias resolves that entry's cap."""
        cfg = {
            "models": {"big": {"model": "openai/gpt", "max_context_tokens": 200000}},
            "max_context_tokens": 64000,
        }
        registry = {"openai": {"context_tokens": 128000}}
        self.assertEqual(
            resolve_model_context_tokens({"model": "big", "provider": None}, cfg, registry),
            200000,
        )

    def test_globals_triple_order(self):
        """Invariant: globals resolve max_context_tokens > max_tokens > context_tokens."""
        registry: dict = {}
        self.assertEqual(
            resolve_model_context_tokens(
                "unknown-xyz",
                {"max_context_tokens": 111, "max_tokens": 222, "context_tokens": 333},
                registry,
            ),
            111,
        )
        self.assertEqual(
            resolve_model_context_tokens(
                "unknown-xyz", {"max_tokens": 222, "context_tokens": 333}, registry
            ),
            222,
        )
        self.assertEqual(
            resolve_model_context_tokens("unknown-xyz", {"context_tokens": 333}, registry),
            333,
        )

    def test_skip_shapes_fall_through(self):
        """Invariant: bool/str/0/negative/None/list/float never coerce; each level skips."""
        bad = [True, False, "100", 0, -5, None, [100], 1.5]
        registry = {"openai": {"context_tokens": 128000}}
        for value in bad:
            with self.subTest(value=value):
                self.assertEqual(
                    resolve_model_context_tokens(
                        {"model": "x", "max_context_tokens": value},
                        {"max_context_tokens": 777},
                        registry,
                    ),
                    777,
                )
                self.assertEqual(
                    resolve_model_context_tokens(
                        "big",
                        {
                            "models": {"big": {"model": "x", "max_context_tokens": value}},
                            "max_context_tokens": 777,
                        },
                        registry,
                    ),
                    777,
                )
                self.assertEqual(
                    resolve_model_context_tokens(
                        "openai/gpt",
                        {"openai": value, "max_context_tokens": 777},
                        {"openai": {"context_tokens": value}},
                    ),
                    777,
                )
                self.assertEqual(
                    resolve_model_context_tokens(
                        "unknown-xyz",
                        {
                            "max_context_tokens": value,
                            "max_tokens": value,
                            "context_tokens": value,
                        },
                        {},
                    ),
                    None,
                )

    def test_unknown_alias_returns_none_without_globals(self):
        """Invariant: an unresolvable spec with nothing declared yields None."""
        self.assertEqual(resolve_model_context_tokens("nope", {}, {}), None)
        self.assertEqual(resolve_model_context_tokens("nope", None, None), None)


class TestApplyModelContextTokens(unittest.TestCase):
    def test_switch_applies_cap_to_harness_and_live_memory_in_place(self):
        """Invariant: switch_model sets the harness cap and mutates live memory, not a copy."""
        harness = _harness_with_models(
            {"big": {"model": "m-big", "provider": "nope", "max_context_tokens": 200000}}
        )
        memory = Memory(system_prompt="s", user_prompt="")
        loop = MagicMock()
        loop.memory = memory
        harness.active_loop = loop

        ok, _, _ = harness.switch_model("big")
        self.assertEqual(ok, True)
        self.assertEqual(harness.max_context_tokens, 200000)
        self.assertEqual(memory.max_context_tokens, 200000)
        self.assertIs(harness.active_loop.memory, memory)

    def test_switch_with_direct_dict_applies_cap(self):
        """Invariant: a direct-dict switch honors its own max_context_tokens."""
        harness = _harness_with_models({})
        before = harness.max_context_tokens
        self.assertNotEqual(before, 9000)
        ok, _, _ = harness.switch_model({"model": "x-model", "max_context_tokens": 9000})
        self.assertEqual(ok, True)
        self.assertEqual(harness.max_context_tokens, 9000)

    def test_unknown_alias_leaves_caps_untouched(self):
        """Invariant: a failed or cap-less switch never mutates the current budget."""
        harness = _harness_with_models(
            {"big": {"model": "m-big", "provider": "nope", "max_context_tokens": 200000}}
        )
        memory = Memory(system_prompt="s", user_prompt="")
        loop = MagicMock()
        loop.memory = memory
        harness.active_loop = loop
        before_harness = harness.max_context_tokens
        before_memory = memory.max_context_tokens

        ok, _, _ = harness.switch_model("nonexistent-xyz")
        self.assertEqual(ok, False)
        self.assertEqual(harness.max_context_tokens, before_harness)
        self.assertEqual(memory.max_context_tokens, before_memory)

    def test_startup_apply_is_noop_when_nothing_declared(self):
        """Invariant: a startup spec dict without caps applies None and changes nothing."""
        harness = _harness_with_models({})
        before = harness.max_context_tokens
        applied = harness.apply_model_context_tokens(
            {"model": None, "provider": None, "endpoint": None, "api_key": None,
             "headers": None},
            cfg={},
            registry={},
        )
        self.assertEqual(applied, None)
        self.assertEqual(harness.max_context_tokens, before)


class TestModelSwitchFooterRefresh(unittest.TestCase):
    def _router(self):
        harness = MagicMock()
        harness.switch_model.return_value = (True, "Switched", MagicMock())
        memory = Memory(system_prompt="s", user_prompt="")
        ui = StubUI()
        ctx = CommandContext(
            harness=harness,
            memory=memory,
            ui=ui,
            workspace=Path(tempfile.mkdtemp()),
            commands={},
            skills={},
            tracer=None,
            turns=TurnState(),
        )
        return CommandRouter(ctx), harness

    def test_successful_switch_refreshes_runner_footer(self):
        """Invariant: a successful /model switch refreshes the token footer immediately."""
        router, _ = self._router()
        runner = MagicMock()
        router.ctx.runner = runner
        router.dispatch("/model fast")
        self.assertEqual(runner.refresh.call_count, 1)

    def test_failed_switch_does_not_refresh_footer(self):
        """Invariant: a rejected /model switch never touches the footer."""
        router, harness = self._router()
        harness.switch_model.return_value = (False, "Unknown model 'x'", MagicMock())
        runner = MagicMock()
        router.ctx.runner = runner
        router.dispatch("/model x")
        self.assertEqual(runner.refresh.call_count, 0)


if __name__ == "__main__":
    unittest.main()
