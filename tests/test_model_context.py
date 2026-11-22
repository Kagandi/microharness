import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.harness import Harness
from core.interactive import CommandContext, CommandRouter, TurnState
from core.memory import Memory
from core.model import HttpChatModel, MockModel
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
            {"model": None, "provider": None, "endpoint": None, "api_key": None, "headers": None},
            cfg={},
            registry={},
        )
        self.assertEqual(applied, None)
        self.assertEqual(harness.max_context_tokens, before)

    def test_switch_big_to_small_reverts_to_default_on_both(self):
        """Invariant: switching to a cap-less model resets sticky caps to the default."""
        tmp = tempfile.mkdtemp()
        ws = Path(tmp)
        cfg_dir = ws / ".microharness"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(
            json.dumps(
                {
                    "models": {
                        "big": {
                            "model": "m-big",
                            "provider": "nope",
                            "max_context_tokens": 200000,
                        },
                        "small": {"model": "m-small", "provider": "nope"},
                    }
                }
            ),
            encoding="utf-8",
        )
        harness = Harness(
            model=MockModel([("hi", [])]),
            workspace_dir=ws,
            global_dir=Path(tempfile.mkdtemp()),
            runs_dir=Path(tempfile.mkdtemp()),
            verbose=False,
        )
        memory = Memory(system_prompt="s", user_prompt="")
        loop = MagicMock()
        loop.memory = memory
        harness.active_loop = loop

        ok, _, _ = harness.switch_model("big")
        self.assertEqual(ok, True)
        self.assertEqual(harness.max_context_tokens, 200000)
        self.assertEqual(memory.max_context_tokens, 200000)

        ok, _, _ = harness.switch_model("small")
        self.assertEqual(ok, True)
        self.assertEqual(harness.max_context_tokens, 262144)
        self.assertEqual(memory.max_context_tokens, 262144)

    def test_switch_big_to_small_reverts_to_global_triple(self):
        """Invariant: a cap-less switch falls back to the global triple, first hit wins."""
        tmp = tempfile.mkdtemp()
        ws = Path(tmp)
        cfg_dir = ws / ".microharness"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(
            json.dumps(
                {
                    "max_context_tokens": 50000,
                    "models": {
                        "big": {
                            "model": "m-big",
                            "provider": "nope",
                            "max_context_tokens": 200000,
                        },
                        "small": {"model": "m-small", "provider": "nope"},
                    },
                }
            ),
            encoding="utf-8",
        )
        runs = Path(tempfile.mkdtemp())
        harness = Harness(
            model=MockModel([("hi", [])]),
            workspace_dir=ws,
            global_dir=Path(tempfile.mkdtemp()),
            runs_dir=runs,
            verbose=False,
        )
        memory = Memory(system_prompt="s", user_prompt="")
        loop = MagicMock()
        loop.memory = memory
        harness.active_loop = loop

        ok, _, _ = harness.switch_model("big")
        self.assertEqual(ok, True)
        self.assertEqual(harness.max_context_tokens, 200000)

        ok, _, _ = harness.switch_model("small")
        self.assertEqual(ok, True)
        self.assertEqual(harness.max_context_tokens, 50000)
        self.assertEqual(memory.max_context_tokens, 50000)


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

    def test_dispatch_sync_updates_ctx_memory_with_no_active_loop(self):
        """Invariant: /model success syncs ctx.memory cap even when active_loop is None."""
        harness = _harness_with_models(
            {
                "big": {"model": "m-big", "provider": "nope", "max_context_tokens": 200000},
            }
        )
        harness.active_loop = None
        memory = Memory(system_prompt="s", user_prompt="")
        self.assertNotEqual(memory.max_context_tokens, 200000)
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
        router = CommandRouter(ctx)
        router.dispatch("/model big")
        self.assertEqual(harness.max_context_tokens, 200000)
        self.assertEqual(memory.max_context_tokens, 200000)


class TestContextWindowAutoDetection(unittest.TestCase):
    def test_harness_init_autodetects_context_window_when_unset(self):
        """Invariant: HttpChatModel probes /models at Harness init when no cap is configured."""
        resp = MagicMock()
        resp.read.return_value = json.dumps(
            {"data": [{"id": "LFM3-1.7B-MLX-8bit", "max_model_len": 262144}]}
        ).encode("utf-8")
        resp.__enter__.return_value = resp

        with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
            http_model = HttpChatModel(
                base_url="http://localhost:8000/v1", model="LFM3-1.7B-MLX-8bit"
            )
            harness = Harness(
                model=http_model,
                workspace_dir=Path(tempfile.mkdtemp()),
                global_dir=Path(tempfile.mkdtemp()),
                runs_dir=Path(tempfile.mkdtemp()),
                verbose=False,
            )

        self.assertEqual(harness.max_context_tokens, 262144)
        self.assertEqual(mock_urlopen.call_count, 1)

    def test_explicit_cap_beats_models_endpoint_probe(self):
        """Invariant: explicit Harness kwarg or alias cap skips /models auto-detection."""
        with patch("urllib.request.urlopen") as mock_urlopen:
            http_model = HttpChatModel(
                base_url="http://localhost:8000/v1", model="LFM3-1.7B-MLX-8bit"
            )
            harness = Harness(
                model=http_model,
                workspace_dir=Path(tempfile.mkdtemp()),
                global_dir=Path(tempfile.mkdtemp()),
                runs_dir=Path(tempfile.mkdtemp()),
                max_context_tokens=32000,
                verbose=False,
            )

        self.assertEqual(harness.max_context_tokens, 32000)
        mock_urlopen.assert_not_called()

    def test_apply_model_context_tokens_uses_live_probe_and_falls_back_cleanly(self):
        """Invariant: switch_model applies /models cap when static config has none, else 262144."""
        tmp = tempfile.mkdtemp()
        ws = Path(tmp)
        cfg_dir = ws / ".microharness"
        cfg_dir.mkdir()
        (cfg_dir / "config.json").write_text(
            json.dumps(
                {
                    "models": {
                        "vercel-hy4": {"model": "tencent/hy4-flash", "provider": "vercel"},
                        "local-gemma": {"model": "gemma-4-E4B", "provider": "vllm"},
                    }
                }
            ),
            encoding="utf-8",
        )
        harness = Harness(
            model=MockModel([("hi", [])]),
            workspace_dir=ws,
            global_dir=Path(tempfile.mkdtemp()),
            runs_dir=Path(tempfile.mkdtemp()),
            verbose=False,
        )
        memory = Memory(system_prompt="s", user_prompt="")
        loop = MagicMock()
        loop.memory = memory
        harness.active_loop = loop

        resp_hy4 = MagicMock()
        resp_hy4.read.return_value = json.dumps(
            {"data": [{"id": "tencent/hy4-flash", "context_window": 1024000}]}
        ).encode("utf-8")
        resp_hy4.__enter__.return_value = resp_hy4

        with patch("urllib.request.urlopen", return_value=resp_hy4):
            ok, _, _ = harness.switch_model("vercel-hy4")
        self.assertEqual(ok, True)
        self.assertEqual(harness.max_context_tokens, 1024000)
        self.assertEqual(memory.max_context_tokens, 1024000)

        # Negative case: probe failure reverts to 262144 default instead of keeping previous 1024000
        with patch("urllib.request.urlopen", side_effect=OSError("down")):
            ok2, _, _ = harness.switch_model("local-gemma")
        self.assertEqual(ok2, True)
        self.assertEqual(harness.max_context_tokens, 262144)
        self.assertEqual(memory.max_context_tokens, 262144)

    def test_anthropic_max_input_tokens_and_builtin_openai_profile_fallback(self):
        """Invariant: /models parses max_input_tokens and built-in openai profile provides 128k fallback."""
        from core.providers import load_builtin_providers

        builtins = load_builtin_providers()
        self.assertEqual(resolve_model_context_tokens("openai/gpt-4o", {}, builtins), 128000)
        self.assertEqual(
            resolve_model_context_tokens("deepseek/deepseek-flash", {}, builtins), None
        )

        resp_anthropic = MagicMock()
        resp_anthropic.read.return_value = json.dumps(
            {"data": [{"id": "claude-sonnet-5-5", "max_input_tokens": 1000000}]}
        ).encode("utf-8")
        resp_anthropic.__enter__.return_value = resp_anthropic

        with patch("urllib.request.urlopen", return_value=resp_anthropic):
            m = HttpChatModel(
                base_url="https://api.anthropic.com/v1",
                model="claude-sonnet-5-5",
                api_key="test-key",
            )
            self.assertEqual(m.get_context_window(), 1000000)

    def test_subagent_resolves_its_own_context_window_without_mutating_parent(self):
        """Invariant: subagents overriding model get their own context cap, while inheriting subagents share parent's."""
        from core.subagents import Subagent

        tmp = tempfile.mkdtemp()
        ws = Path(tmp)
        agents_dir = ws / ".microharness" / "agents"
        agents_dir.mkdir(parents=True)
        (agents_dir / "capped.md").write_text(
            "---\nname: capped\ndescription: Capped model\nmodel: capped_alias\n---\nCheck.",
            encoding="utf-8",
        )
        (agents_dir / "probed.md").write_text(
            "---\nname: probed\ndescription: Probed model\nmodel: probed_alias\n---\nCheck.",
            encoding="utf-8",
        )
        (agents_dir / "fallback.md").write_text(
            "---\nname: fallback\ndescription: Fallback model\nmodel: fallback_alias\n---\nCheck.",
            encoding="utf-8",
        )
        (ws / ".microharness" / "config.json").write_text(
            json.dumps(
                {
                    "models": {
                        "capped_alias": {
                            "model": "m-capped",
                            "provider": "openai",
                            "max_context_tokens": 65536,
                        },
                        "probed_alias": {
                            "model": "m-probed",
                            "provider": "vllm",
                        },
                        "fallback_alias": {
                            "model": "m-unknown",
                            "provider": "vllm",
                        },
                    }
                }
            ),
            encoding="utf-8",
        )

        parent_model = MockModel([("parent", [])])
        harness = Harness(
            model=parent_model,
            workspace_dir=ws,
            global_dir=Path(tempfile.mkdtemp()),
            runs_dir=Path(tempfile.mkdtemp()),
            max_context_tokens=1024000,
            verbose=False,
        )
        env, reg, parent_mem, _, _, _ = harness._setup_run(
            task_id="sub_ctx",
            system_prompt="SYS",
            initial_instruction="hi",
            initial_files={},
        )
        try:
            sub_tool = reg.get("subagent")
            sub_tool.subagents["inherited"] = Subagent(
                name="inherited",
                description="Inherited",
                system_prompt="Check.",
                scope="project",
                model=None,
            )

            seen_caps: list[int] = []

            class FakeLoop:
                def __init__(self, model, memory, tools):
                    self.model = model
                    self.memory = memory
                    self.tools = tools
                    seen_caps.append(memory.max_context_tokens)

            with (
                patch("core.subagents.AgentLoop", FakeLoop),
                patch(
                    "core.subagents.run_steps",
                    return_value=MagicMock(finished=True, reason="completed"),
                ),
            ):
                # 1. Inherited model shares parent's 1,024,000 cap
                c0, _ = sub_tool.execute(agent="inherited", task="t0")
                self.assertEqual(c0, 0)

                # 2. Config-capped model uses its own 65,536 cap, not parent's 1,024,000
                c1, _ = sub_tool.execute(agent="capped", task="t1")
                self.assertEqual(c1, 0)

                # 3. Live /models-probed child model uses its 131,072 cap
                resp_probe = MagicMock()
                resp_probe.read.return_value = json.dumps(
                    {"data": [{"id": "m-probed", "max_model_len": 131072}]}
                ).encode("utf-8")
                resp_probe.__enter__.return_value = resp_probe
                with patch("urllib.request.urlopen", return_value=resp_probe):
                    c2, _ = sub_tool.execute(agent="probed", task="t2")
                self.assertEqual(c2, 0)

                # 4. Negative case: failed probe + no profile cap uses default 262,144 (never parent's 1,024,000)
                with patch("urllib.request.urlopen", side_effect=OSError("down")):
                    c3, _ = sub_tool.execute(agent="fallback", task="t3")
                self.assertEqual(c3, 0)

            self.assertEqual(seen_caps, [1024000, 65536, 131072, 262144])
            self.assertEqual(harness.max_context_tokens, 1024000)
            self.assertEqual(parent_mem.max_context_tokens, 1024000)
        finally:
            env.teardown()


if __name__ == "__main__":
    unittest.main()
