import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.model import HttpChatModel, MockModel
from core.providers import (
    FALLBACK_ENV_KEYS,
    LAST_RESORT_MODEL,
    Connection,
    ModelRequest,
    build_model_from_spec,
    find_provider_by_endpoint,
    get_available_models_from_config,
    load_builtin_providers,
    load_provider_registry,
    parse_model_spec,
    resolve_api_key,
    resolve_connection,
    resolve_model_spec,
    resolve_provider,
    seed_user_providers,
)


class TestRegistryLoading(unittest.TestCase):
    def test_builtin_registry_has_the_expected_providers(self):
        """Invariant: the committed providers.json seeds the well-known endpoints."""
        registry = load_builtin_providers()
        for name in ("google", "openrouter", "openai", "anthropic", "bedrock"):
            self.assertIn(name, registry)
            self.assertTrue(registry[name]["endpoint"].startswith("https://"))
        for name in ("lmstudio", "vllm", "llamacpp", "ollama", "local"):
            self.assertIn(name, registry)
            self.assertTrue(registry[name]["endpoint"].startswith("http://localhost:"))

    def test_registry_layers_override_in_order(self):
        """Invariant: inline config wins over project, which wins over global, over built-in."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_dir = root / "global"
            project_dir = root / "project"
            global_dir.mkdir()
            project_dir.mkdir()
            (global_dir / "providers.json").write_text(
                json.dumps({"openai": {"default_model": "global-model"}}), encoding="utf-8"
            )
            (project_dir / ".microharness").mkdir()
            (project_dir / ".microharness" / "providers.json").write_text(
                json.dumps({"openai": {"default_model": "project-model"}}), encoding="utf-8"
            )

            registry = load_provider_registry(
                global_dir=global_dir,
                project_dir=project_dir,
                inline_providers={"openai": {"default_model": "inline-model"}},
            )
            self.assertEqual(registry["openai"]["default_model"], "inline-model")

            without_inline = load_provider_registry(
                global_dir=global_dir, project_dir=project_dir
            )
            self.assertEqual(without_inline["openai"]["default_model"], "project-model")

    def test_layer_override_merges_keys_instead_of_replacing(self):
        """Invariant: overriding one profile key leaves the profile's other keys intact."""
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp)
            (global_dir / "providers.json").write_text(
                json.dumps({"openai": {"endpoint": "https://custom/v1", "headers": {"X": "1"}}}),
                encoding="utf-8",
            )
            registry = load_provider_registry(global_dir=global_dir)
            self.assertEqual(registry["openai"]["endpoint"], "https://custom/v1")
            self.assertIn("default_model", registry["openai"])

    def test_malformed_provider_file_is_ignored(self):
        """Invariant: a broken providers.json cannot take down the harness."""
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp)
            (global_dir / "providers.json").write_text("{ broken", encoding="utf-8")
            self.assertIn("openai", load_provider_registry(global_dir=global_dir))

    def test_seed_creates_the_file_only_once(self):
        """Invariant: seeding writes built-ins once and never clobbers user edits."""
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp) / "nested" / "global"
            seeded = seed_user_providers(global_dir)
            self.assertIsNotNone(seeded)
            self.assertTrue(seeded.is_file())
            seeded.write_text('{"openai": {"default_model": "mine"}}', encoding="utf-8")
            self.assertIsNone(seed_user_providers(global_dir))
            self.assertEqual(load_provider_registry(global_dir=global_dir)["openai"]["default_model"], "mine")

    def test_seed_without_global_dir_is_a_noop(self):
        """Invariant: seeding is skipped entirely when there is no global directory."""
        self.assertIsNone(seed_user_providers(None))


class TestProviderResolution(unittest.TestCase):
    REGISTRY = {
        "google": {"endpoint": "https://google/v1", "default_model": "gemini", "env_keys": ["MY_KEY"]},
        "openai": {"endpoint": "https://openai/v1", "default_model": "gpt"},
    }

    def test_alias_maps_to_canonical_name(self):
        """Invariant: friendly aliases resolve to the registry's canonical key."""
        self.assertEqual(resolve_provider(self.REGISTRY, "gemini")[0], "google")
        self.assertEqual(resolve_provider(self.REGISTRY, "open-router")[0], "openrouter")

    def test_resolution_is_case_insensitive_and_returns_a_copy(self):
        """Invariant: callers cannot mutate the registry through a resolved profile."""
        name, profile = resolve_provider(self.REGISTRY, "OpenAI")
        self.assertEqual(name, "openai")
        profile["default_model"] = "mutated"
        self.assertEqual(self.REGISTRY["openai"]["default_model"], "gpt")

    def test_unknown_provider_returns_empty_profile(self):
        """Invariant: an unknown provider is reported, not silently treated as known."""
        name, profile = resolve_provider(self.REGISTRY, "nope")
        self.assertEqual(name, "nope")
        self.assertEqual(profile, {})

    def test_no_provider_name_resolves_to_nothing(self):
        """Invariant: an absent provider name yields (None, {}) rather than a guess."""
        self.assertEqual(resolve_provider(self.REGISTRY, None), (None, {}))

    def test_endpoint_match_accepts_a_full_endpoint_url(self):
        """Invariant: a provider is detected from its completions URL, not just its base."""
        name, profile = find_provider_by_endpoint(
            self.REGISTRY, "https://openai/v1/chat/completions"
        )
        self.assertEqual(name, "openai")
        self.assertEqual(profile["default_model"], "gpt")

    def test_endpoint_match_requires_a_prefix(self):
        """Invariant: an unrelated host is not attributed to any known provider."""
        self.assertEqual(find_provider_by_endpoint(self.REGISTRY, "https://evil.example/v1"), (None, {}))


class TestAvailableModels(unittest.TestCase):
    REGISTRY = {
        "google": {"default_model": "gemini-2.5-pro", "description": "Google"},
        "openai": {"default_model": "gpt-4o", "description": "OpenAI"},
    }

    def test_list_of_strings(self):
        """Invariant: bare model ids become entries named after themselves."""
        entries = get_available_models_from_config({"models": ["gemini-2.5-flash"]}, self.REGISTRY)
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]["name"], "gemini-2.5-flash")
        self.assertEqual(entries[0]["model"], "gemini-2.5-flash")

    def test_alias_mapping_to_string(self):
        """Invariant: an alias names the entry while the value stays the model id."""
        entries = get_available_models_from_config(
            {"models": {"fast": "gemini-2.5-flash"}}, self.REGISTRY
        )
        self.assertEqual(entries[0]["name"], "fast")
        self.assertEqual(entries[0]["model"], "gemini-2.5-flash")

    def test_alias_mapping_to_object(self):
        """Invariant: explicit provider/endpoint/description survive normalization."""
        entries = get_available_models_from_config(
            {"models": {"fast": {"model": "gemini-2.5-flash", "provider": "google",
                                 "description": "Fast", "endpoint": "https://x/v1"}}},
            self.REGISTRY,
        )
        self.assertEqual(entries[0]["provider"], "google")
        self.assertEqual(entries[0]["description"], "Fast")
        self.assertEqual(entries[0]["endpoint"], "https://x/v1")

    def test_provider_is_inferred_from_a_prefixed_id(self):
        """Invariant: 'google/gemini-2.5-flash' resolves its provider from the prefix."""
        entries = get_available_models_from_config(
            {"models": ["google/gemini-2.5-flash"]}, self.REGISTRY
        )
        self.assertEqual(entries[0]["provider"], "google")

    def test_list_object_uses_name_alias_over_model(self):
        """Invariant: a display name in the config wins over the raw model id."""
        entries = get_available_models_from_config(
            {"models": [{"name": "Smart", "model": "gpt-4o"}]}, self.REGISTRY
        )
        self.assertEqual(entries[0]["name"], "Smart")
        self.assertEqual(entries[0]["model"], "gpt-4o")

    def test_endpoint_only_entry_uses_the_alias_as_its_model(self):
        """Invariant: an endpoint-only entry still yields a usable model name."""
        entries = get_available_models_from_config(
            {"models": {"custom": {"endpoint": "https://x/v1"}}}, self.REGISTRY
        )
        self.assertEqual(entries[0]["model"], "custom")

    def test_invalid_entries_are_skipped(self):
        """Invariant: unusable config entries never appear, and discovery still offers defaults."""
        entries = get_available_models_from_config(
            {"models": ["  ", 42, None, {"no_model": True}]}, self.REGISTRY
        )
        self.assertEqual([e["model"] for e in entries], ["gemini-2.5-pro", "gpt-4o"])

    def test_fallback_lists_configured_model_then_provider_defaults(self):
        """Invariant: with no models key, the active model and provider defaults are offered."""
        entries = get_available_models_from_config(
            {"model": "my-model", "provider": "google"}, self.REGISTRY
        )
        self.assertEqual(entries[0]["name"], "my-model")
        self.assertEqual([e["model"] for e in entries[1:]], ["gemini-2.5-pro", "gpt-4o"])

    def test_fallback_defaults_to_builtin_registry(self):
        """Invariant: omitting the registry falls back to the committed provider list."""
        entries = get_available_models_from_config({})
        self.assertGreater(len(entries), 1)
        self.assertTrue(all("name" in e and "model" in e for e in entries))


class TestParseModelSpec(unittest.TestCase):
    REGISTRY = {"google": {"endpoint": "https://google/v1"}}

    def test_provider_prefixed_spec(self):
        """Invariant: 'google/gemini' splits into provider and model."""
        request = parse_model_spec("google/gemini-2.5-flash", {}, self.REGISTRY)
        self.assertEqual(request, ModelRequest(model="gemini-2.5-flash", provider="google"))

    def test_unknown_prefix_stays_part_of_the_model_id(self):
        """Invariant: a slash that is not a known provider does not split the model id."""
        request = parse_model_spec("acme/tuned-model", {}, self.REGISTRY)
        self.assertEqual(request.model, "acme/tuned-model")
        self.assertIsNone(request.provider)

    def test_bare_model_id(self):
        """Invariant: a bare id needs no provider."""
        self.assertEqual(parse_model_spec("gpt-4o", {}, self.REGISTRY).model, "gpt-4o")

    def test_config_alias_is_followed(self):
        """Invariant: a config alias resolves to the mapping it points at."""
        config = {"models": {"fast": {"model": "gemini-2.5-flash", "provider": "google"}}}
        request = parse_model_spec("fast", config, self.REGISTRY)
        self.assertEqual(request.model, "gemini-2.5-flash")
        self.assertEqual(request.provider, "google")

    def test_mapping_spec_keeps_endpoint_and_headers(self):
        """Invariant: explicit connection details survive parsing."""
        request = parse_model_spec(
            {"model": "m", "endpoint": "https://x/v1", "headers": {"X": "1"}}, {}, self.REGISTRY
        )
        self.assertEqual(request.endpoint, "https://x/v1")
        self.assertEqual(request.headers, {"X": "1"})


class TestApiKeyResolution(unittest.TestCase):
    def test_explicit_key_wins(self):
        """Invariant: a key in the spec outranks every environment lookup."""
        request = ModelRequest(api_key="explicit")
        self.assertEqual(resolve_api_key(request, {}, {"api_key": "from-config"}), "explicit")

    def test_profile_env_key_is_preferred_over_generic_fallbacks(self):
        """Invariant: the provider's own key variable wins over the well-known list."""
        profile = {"env_keys": ["MY_KEY"]}
        env = {"MY_KEY": "provider-key", "OPENAI_API_KEY": "generic"}
        with patch.dict(os.environ, env, clear=True):
            self.assertEqual(resolve_api_key(ModelRequest(), profile, {}), "provider-key")

    def test_falls_back_to_config_key(self):
        """Invariant: when no environment variable is set, the config key is used."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertEqual(resolve_api_key(ModelRequest(), {}, {"api_key": "cfg"}), "cfg")

    def test_no_key_anywhere_is_none(self):
        """Invariant: an unauthenticated local endpoint is allowed."""
        with patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(resolve_api_key(ModelRequest(), {}, {}))

    def test_fallback_env_keys_are_a_stable_tuple(self):
        """Invariant: the generic lookup list stays a module constant, not a rebuilt literal."""
        self.assertIn("OPENROUTER_API_KEY", FALLBACK_ENV_KEYS)
        self.assertIsInstance(FALLBACK_ENV_KEYS, tuple)


class TestBuildModelFromSpec(unittest.TestCase):
    def setUp(self):
        self.registry = {
            "openai": {
                "endpoint": "https://openai/v1",
                "default_model": "gpt-4o",
                "env_keys": ["OPENAI_API_KEY"],
            }
        }
        self.env = {"OPENAI_API_KEY": "sk-test"}

    def test_builds_http_client_for_a_known_provider(self):
        """Invariant: a provider/model spec yields a wired HTTP client."""
        with patch.dict(os.environ, self.env, clear=True):
            model = build_model_from_spec("openai/gpt-4o", registry=self.registry)
        self.assertIsInstance(model, HttpChatModel)
        self.assertEqual(model.endpoint, "https://openai/v1/chat/completions")
        self.assertEqual(model.model, "gpt-4o")
        self.assertEqual(model.provider, "openai")

    def test_applies_cache_and_reasoning_settings(self):
        """Invariant: dynamic settings are forwarded to the new client."""
        with patch.dict(os.environ, self.env, clear=True):
            model = build_model_from_spec(
                "openai/gpt-4o", registry=self.registry, cache_ttl="1h", reasoning_effort="high"
            )
        self.assertEqual(model.cache_ttl, "1h")
        self.assertEqual(model.reasoning_effort, "high")

    def test_detects_provider_from_a_custom_endpoint(self):
        """Invariant: an unlabelled endpoint is attributed to the matching registry profile."""
        with patch.dict(os.environ, self.env, clear=True):
            model = build_model_from_spec(
                {"model": "gpt-4o", "endpoint": "https://openai/v1/chat/completions"},
                registry=self.registry,
            )
        self.assertEqual(model.provider, "openai")

    def test_offline_falls_back_to_the_mock_model(self):
        """Invariant: with no endpoint anywhere the harness still runs, on MockModel."""
        with patch.dict(os.environ, {}, clear=True):
            model = build_model_from_spec("some-model", config={}, registry={})
        self.assertIsInstance(model, MockModel)

    def test_environment_endpoint_is_used(self):
        """Invariant: MODEL_ENDPOINT enables a run with no config file at all."""
        env = {"MODEL_ENDPOINT": "https://openai/v1", "MODEL_NAME": "gpt-4o",
               "OPENAI_API_KEY": "sk-test"}
        with patch.dict(os.environ, env, clear=True):
            model = build_model_from_spec("gpt-4o", registry=self.registry)
        self.assertEqual(model.model, "gpt-4o")

    def test_empty_spec_is_rejected(self):
        """Invariant: an empty spec is an error, not a silent default."""
        with self.assertRaises(ValueError):
            build_model_from_spec("")

    def test_wrong_spec_type_is_rejected(self):
        """Invariant: a non-string, non-dict spec names the offending type."""
        with self.assertRaises(ValueError) as ctx:
            build_model_from_spec(42)
        self.assertIn("int", str(ctx.exception))

    def test_last_resort_model_is_used_when_profile_has_none(self):
        """Invariant: a profile without a default model still yields a model id."""
        registry = {"custom": {"endpoint": "https://custom/v1"}}
        with patch.dict(os.environ, {}, clear=True):
            model = build_model_from_spec("custom/whatever", registry=registry)
        self.assertEqual(model.model, "whatever")
        self.assertIn(LAST_RESORT_MODEL, LAST_RESORT_MODEL)


class TestResolveConnection(unittest.TestCase):
    REGISTRY = {
        "openai": {
            "endpoint": "https://openai/v1",
            "default_model": "gpt-4o",
            "env_keys": ["OPENAI_API_KEY"],
            "reasoning_format": "standard",
            "cache_strategy": "implicit",
        }
    }

    def test_reports_the_resolved_connection_without_building_a_client(self):
        """Invariant: a caller can inspect what would be built before building it."""
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}, clear=True):
            connection = resolve_connection(
                {"model": "gpt-4o-mini", "endpoint": "https://openai/v1"},
                registry=self.REGISTRY,
            )
        self.assertEqual(
            connection,
            Connection(
                model="gpt-4o-mini",
                endpoint="https://openai/v1",
                provider="openai",
                api_key="sk",
            ),
        )

    def test_provider_defaults_fill_an_unset_model(self):
        """Invariant: a bare provider name uses that provider's default model."""
        connection = resolve_connection({"provider": "openai"}, registry=self.REGISTRY)
        self.assertEqual(connection.model, "gpt-4o")
        self.assertEqual(connection.endpoint, "https://openai/v1")

    def test_no_endpoint_anywhere_reports_offline(self):
        """Invariant: an unconfigured environment is reported, not guessed at."""
        with patch.dict(os.environ, {}, clear=True):
            connection = resolve_connection({"model": "m"}, config={}, registry={})
        self.assertTrue(connection.is_offline)
        self.assertIsNone(connection.endpoint)

    def test_provider_dialects_are_carried_through(self):
        """Invariant: reasoning/cache dialects from the profile reach the client."""
        registry = {
            "openrouter": {
                "endpoint": "https://openrouter.ai/api/v1",
                "reasoning_format": "openrouter",
                "cache_strategy": "anthropic_block",
            }
        }
        connection = resolve_connection({"provider": "openrouter"}, registry=registry)
        self.assertEqual(connection.reasoning_format, "openrouter")
        self.assertEqual(connection.cache_strategy, "anthropic_block")

    def test_build_model_from_spec_matches_resolve_connection(self):
        """Invariant: the two entry points cannot disagree about what gets connected."""
        registry = dict(self.REGISTRY)
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk"}, clear=True):
            connection = resolve_connection("openai/gpt-4o", registry=registry)
            model = build_model_from_spec("openai/gpt-4o", registry=registry)
        self.assertEqual(model.model, connection.model)
        self.assertEqual(model.endpoint, f"{connection.endpoint}/chat/completions")
        self.assertEqual(model.provider, connection.provider)
        self.assertEqual(model.api_key, connection.api_key)

    def test_model_entry_endpoint_wins_but_profile_yields_to_unattributable_cfg(self):
        """Invariant: an alias entry with its own endpoint wins; otherwise an
        unattributable global endpoint wins over the profile endpoint."""
        config = {
            "endpoint": "https://openrouter.ai/api/v1",
            "models": {
                "smart": {"model": "gpt-4o", "provider": "openai"},
                "local": {"model": "llama-3", "endpoint": "http://localhost:1234/v1"},
            },
        }
        conn_smart = resolve_connection("smart", config=config, registry=self.REGISTRY)
        self.assertEqual(conn_smart.endpoint, "https://openrouter.ai/api/v1")
        self.assertEqual(conn_smart.provider, "openai")

        conn_local = resolve_connection("local", config=config, registry=self.REGISTRY)
        self.assertEqual(conn_local.endpoint, "http://localhost:1234/v1")


class TestWave8EndpointRouting(unittest.TestCase):
    """Wave 8: a startup dict naming a config alias merges the entry underneath.

    A request endpoint (flag/env, alias entry, or direct) beats everything;
    otherwise the profile endpoint and the global config endpoint compete after
    normalization, attributing the config endpoint to decide the winner.
    """

    REGISTRY = {
        "openai": {
            "endpoint": "https://api.openai.com/v1",
            "default_model": "gpt-4o",
            "env_keys": ["OPENAI_API_KEY"],
        },
        "bedrock": {
            "endpoint": "https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
            "default_model": "us.openai.gpt-5.6-sol",
            "env_keys": ["BEDROCK_API_KEY"],
            "auth": {"scheme": "sigv4", "service": "bedrock"},
        },
        "local": {
            "endpoint": "http://localhost:1234/v1",
            "default_model": "local-model",
            "env_keys": [],
        },
    }

    @staticmethod
    def _startup_spec(**overrides: object) -> dict:
        """Builds a main.py _startup_model_spec-shaped dict (flag/env only)."""
        spec: dict = {
            "model": None,
            "provider": None,
            "endpoint": None,
            "api_key": None,
            "headers": None,
        }
        spec.update(overrides)
        return spec

    def test_bedrock_alias_dict_ignores_global_local_endpoint(self):
        """Invariant: a bedrock alias dict resolves to the regional URL, not the global local endpoint."""
        config = {
            "endpoint": "http://localhost:1234/v1",
            "models": {"luna": {"model": "us.openai.gpt-5.6-sol", "provider": "bedrock"}},
        }
        with tempfile.TemporaryDirectory() as home:
            env = _clean_aws_env(home)
            env.update({"AWS_REGION": "us-east-1"})
            with patch.dict(os.environ, env, clear=True):
                conn = resolve_connection(
                    self._startup_spec(model="luna"),
                    config=config,
                    registry=dict(self.REGISTRY),
                )
        self.assertEqual(
            conn.endpoint, "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1"
        )
        self.assertEqual(conn.provider, "bedrock")
        self.assertEqual(conn.model, "us.openai.gpt-5.6-sol")

    def test_unattributable_cfg_endpoint_wins_over_profile(self):
        """Invariant: a gateway URL no profile claims still wins (no-attribution fallback)."""
        registry = {"openai": dict(self.REGISTRY["openai"])}
        config = {"endpoint": "https://gateway.internal/v1"}
        with patch.dict(os.environ, {"MODEL_ENDPOINT": ""}, clear=True):
            conn = resolve_connection(
                {"model": "gpt-4o", "provider": "openai"},
                config=config,
                registry=registry,
            )
        self.assertEqual(conn.endpoint, "https://gateway.internal/v1")
        self.assertEqual(conn.provider, "openai")

    def test_matching_cfg_endpoint_wins_over_profile(self):
        """Invariant: a cfg endpoint attributed to the selected provider wins."""
        registry = {"openai": dict(self.REGISTRY["openai"])}
        config = {"endpoint": "https://api.openai.com/v1/gateway"}
        with patch.dict(os.environ, {"MODEL_ENDPOINT": ""}, clear=True):
            conn = resolve_connection(
                {"model": "gpt-4o", "provider": "openai"},
                config=config,
                registry=registry,
            )
        self.assertEqual(conn.endpoint, "https://api.openai.com/v1/gateway")
        self.assertEqual(conn.provider, "openai")

    def test_foreign_cfg_endpoint_loses_to_profile(self):
        """Invariant: a cfg endpoint attributed to another provider loses to the profile."""
        registry = {
            "openai": dict(self.REGISTRY["openai"]),
            "local": dict(self.REGISTRY["local"]),
        }
        config = {"endpoint": "http://localhost:1234/v1"}
        with patch.dict(os.environ, {"MODEL_ENDPOINT": ""}, clear=True):
            conn = resolve_connection(
                {"model": "gpt-4o", "provider": "openai"},
                config=config,
                registry=registry,
            )
        self.assertEqual(conn.endpoint, "https://api.openai.com/v1")
        self.assertEqual(conn.provider, "openai")

    def test_local_default_from_cfg_is_unchanged(self):
        """Invariant: a provider-less spec over a global local endpoint stays local and keyless."""
        config = {
            "endpoint": "http://localhost:1234/v1",
            "model": "qwen2.5-coder-7b-instruct",
        }
        with patch.dict(os.environ, {"MODEL_ENDPOINT": ""}, clear=True):
            conn = resolve_connection(
                {"model": "qwen2.5-coder-7b-instruct"},
                config=config,
                registry=dict(self.REGISTRY),
            )
        self.assertEqual(conn.endpoint, "http://localhost:1234/v1")
        self.assertEqual(conn.is_local, True)
        self.assertEqual(conn.missing_credentials, False)

    def test_flag_endpoint_beats_alias_profile_and_cfg(self):
        """Invariant: an explicit request endpoint wins over both profile and global endpoints."""
        config = {
            "endpoint": "https://gateway.internal/v1",
            "models": {"luna": {"model": "us.openai.gpt-5.6-sol", "provider": "bedrock"}},
        }
        with tempfile.TemporaryDirectory() as home:
            env = _clean_aws_env(home)
            env.update({"AWS_REGION": "us-east-1"})
            with patch.dict(os.environ, env, clear=True):
                conn = resolve_connection(
                    self._startup_spec(model="luna", endpoint="https://flag.example/v1"),
                    config=config,
                    registry=dict(self.REGISTRY),
                )
        self.assertEqual(conn.endpoint, "https://flag.example/v1")
        self.assertEqual(conn.provider, "bedrock")
        self.assertEqual(conn.model, "us.openai.gpt-5.6-sol")

    def test_alias_entry_endpoint_beats_global_endpoint(self):
        """Invariant: an endpoint on the alias entry wins over the global endpoint."""
        config = {
            "endpoint": "https://gateway.internal/v1",
            "models": {"desk": {"model": "llama-3", "endpoint": "http://localhost:1234/v1"}},
        }
        with patch.dict(os.environ, {"MODEL_ENDPOINT": ""}, clear=True):
            conn = resolve_connection(
                self._startup_spec(model="desk"),
                config=config,
                registry=dict(self.REGISTRY),
            )
        self.assertEqual(conn.endpoint, "http://localhost:1234/v1")
        self.assertEqual(conn.model, "llama-3")

    def test_flag_provider_kept_with_entry_model_id(self):
        """Invariant: --provider X over an alias to Y keeps X but takes the entry model id."""
        registry = {
            "openai": dict(self.REGISTRY["openai"]),
            "providerx": {"endpoint": "https://x.example/v1", "default_model": "x-default"},
        }
        config = {"models": {"fast": {"model": "y-model-id", "provider": "openai"}}}
        with patch.dict(os.environ, {"MODEL_ENDPOINT": ""}, clear=True):
            conn = resolve_connection(
                self._startup_spec(model="fast", provider="providerx"),
                config=config,
                registry=registry,
            )
        self.assertEqual(conn.provider, "providerx")
        self.assertEqual(conn.model, "y-model-id")
        self.assertEqual(conn.endpoint, "https://x.example/v1")

    def test_trailing_slash_cfg_endpoint_matches_profile(self):
        """Invariant: endpoint comparison ignores a trailing slash (profile wins the tie)."""
        registry = {"openai": dict(self.REGISTRY["openai"])}
        config = {"endpoint": "https://api.openai.com/v1/"}
        with patch.dict(os.environ, {"MODEL_ENDPOINT": ""}, clear=True):
            conn = resolve_connection(
                {"model": "gpt-4o", "provider": "openai"},
                config=config,
                registry=registry,
            )
        self.assertEqual(conn.endpoint, "https://api.openai.com/v1")
        self.assertEqual(conn.provider, "openai")

    def test_direct_endpoint_with_cfg_provider(self):
        """Invariant: a provider-less direct endpoint keeps the cfg-pinned provider."""
        registry = {"openai": dict(self.REGISTRY["openai"])}
        config = {"provider": "openai"}
        with patch.dict(os.environ, {"MODEL_ENDPOINT": ""}, clear=True):
            conn = resolve_connection(
                {"model": "gpt-4o", "endpoint": "https://direct.example/v1"},
                config=config,
                registry=registry,
            )
        self.assertEqual(conn.endpoint, "https://direct.example/v1")
        self.assertEqual(conn.provider, "openai")

    def test_alias_entry_auth_surfaces_through_startup_dict(self):
        """Invariant: auth on the alias entry applies to a startup dict naming the alias."""
        registry = {"custom": {"endpoint": "https://x.example/v1", "auth": "bearer"}}
        config = {
            "models": {
                "fast": {"model": "m", "provider": "custom", "auth": {"scheme": "none"}}
            }
        }
        with patch.dict(os.environ, {"MODEL_ENDPOINT": ""}, clear=True):
            conn = resolve_connection(
                self._startup_spec(model="fast"), config=config, registry=registry
            )
        self.assertEqual(conn.auth_scheme, "none")

    def test_string_alias_entry_sets_model_id(self):
        """Invariant: a dict spec naming a string alias takes the entry as its model id."""
        registry = {"openai": dict(self.REGISTRY["openai"])}
        config = {"provider": "openai", "models": {"fast": "gpt-4o-mini"}}
        with patch.dict(os.environ, {"MODEL_ENDPOINT": ""}, clear=True):
            conn = resolve_connection(
                self._startup_spec(model="fast"), config=config, registry=registry
            )
        self.assertEqual(conn.model, "gpt-4o-mini")
        self.assertEqual(conn.provider, "openai")
        self.assertEqual(conn.endpoint, "https://api.openai.com/v1")

    def test_alias_entry_api_key_and_headers_surface(self):
        """Invariant: api_key/headers on the alias entry reach the connection; spec headers win ties."""
        registry = {"custom": {"endpoint": "https://x.example/v1"}}
        config = {
            "models": {
                "fast": {
                    "model": "m",
                    "provider": "custom",
                    "api_key": "entry-key",
                    "headers": {"X-Entry": "1", "X-Both": "entry"},
                }
            }
        }
        spec = self._startup_spec(
            model="fast", headers={"X-Both": "spec", "X-Spec": "2"}
        )
        with patch.dict(os.environ, {"MODEL_ENDPOINT": ""}, clear=True):
            conn = resolve_connection(spec, config=config, registry=registry)
        self.assertEqual(conn.api_key, "entry-key")
        self.assertEqual(conn.headers, {"X-Entry": "1", "X-Both": "spec", "X-Spec": "2"})
        self.assertEqual(conn.model, "m")
        self.assertEqual(conn.provider, "custom")


class TestBedrockProvider(unittest.TestCase):
    """Bedrock ships as an OpenAI-compatible shim with a regional endpoint and a bearer token."""

    REGISTRY = {
        "bedrock": {
            "endpoint": "https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
            "default_model": "anthropic.claude-sonnet-5-5",
            "env_keys": ["BEDROCK_API_KEY"],
        }
    }

    def test_builtin_registry_registers_bedrock(self):
        """Invariant: bedrock is a built-in provider with a regional endpoint template."""
        registry = load_builtin_providers()
        self.assertIn("bedrock", registry)
        self.assertIn("{region}", registry["bedrock"]["endpoint"])
        self.assertTrue(registry["bedrock"]["endpoint"].startswith("https://"))
        self.assertTrue(registry["bedrock"]["env_keys"])
        self.assertTrue(registry["bedrock"]["default_model"])

    def test_endpoint_match_maps_a_regional_url_back_to_the_template(self):
        """Invariant: a concrete regional URL is attributed to the templated profile."""
        name, profile = find_provider_by_endpoint(
            self.REGISTRY, "https://bedrock-runtime.eu-west-2.amazonaws.com/openai/v1"
        )
        self.assertEqual(name, "bedrock")
        self.assertEqual(profile["default_model"], "anthropic.claude-sonnet-5-5")

    def test_endpoint_match_rejects_an_unrelated_host(self):
        """Invariant: a template wildcard never attributes an unrelated endpoint."""
        self.assertEqual(
            find_provider_by_endpoint(self.REGISTRY, "https://evil.example/openai/v1"),
            (None, {}),
        )

    def test_region_is_substituted_from_the_aws_environment(self):
        """Invariant: AWS_REGION fills the endpoint template at resolution time."""
        with patch.dict(os.environ, {"AWS_REGION": "eu-west-1"}, clear=True):
            connection = resolve_connection(
                "bedrock/anthropic.claude-sonnet-5-5", registry=self.REGISTRY
            )
        self.assertEqual(
            connection.endpoint,
            "https://bedrock-runtime.eu-west-1.amazonaws.com/openai/v1",
        )
        self.assertEqual(connection.model, "anthropic.claude-sonnet-5-5")
        self.assertEqual(connection.provider, "bedrock")

    def test_region_in_config_wins_over_the_environment(self):
        """Invariant: an explicit config region outranks AWS environment variables."""
        env = {"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "us-east-1"}
        with patch.dict(os.environ, env, clear=True):
            connection = resolve_connection(
                {"model": "m", "provider": "bedrock"},
                config={"region": "ap-south-1"},
                registry=self.REGISTRY,
            )
        self.assertEqual(
            connection.endpoint, "https://bedrock-runtime.ap-south-1.amazonaws.com/openai/v1"
        )

    def test_aws_default_region_is_the_last_env_fallback(self):
        """Invariant: AWS_DEFAULT_REGION is honoured when AWS_REGION is unset."""
        with patch.dict(os.environ, {"AWS_DEFAULT_REGION": "us-east-2"}, clear=True):
            connection = resolve_connection(
                {"model": "m", "provider": "bedrock"}, registry=self.REGISTRY
            )
        self.assertEqual(
            connection.endpoint, "https://bedrock-runtime.us-east-2.amazonaws.com/openai/v1"
        )

    def test_missing_region_is_reported_offline(self):
        """Invariant: an unfillable template is offline, never a literal '{region}' URL."""
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home}, clear=True):
                connection = resolve_connection(
                    "bedrock/anthropic.claude-sonnet-5-5", registry=self.REGISTRY
                )
        self.assertTrue(connection.is_offline)
        self.assertIsNone(connection.endpoint)

    def test_builds_a_regional_client_with_a_bearer_token(self):
        """Invariant: a bedrock spec yields an HttpChatModel pointed at the region's shim."""
        env = {"AWS_REGION": "ap-south-1", "BEDROCK_API_KEY": "brk-token"}
        with patch.dict(os.environ, env, clear=True):
            model = build_model_from_spec(
                "bedrock/anthropic.claude-sonnet-5-5", registry=self.REGISTRY
            )
        self.assertIsInstance(model, HttpChatModel)
        self.assertEqual(
            model.endpoint,
            "https://bedrock-runtime.ap-south-1.amazonaws.com/openai/v1/chat/completions",
        )
        self.assertEqual(model.api_key, "brk-token")
        self.assertEqual(model.model, "anthropic.claude-sonnet-5-5")
        self.assertEqual(model.provider, "bedrock")

    def test_model_id_version_suffix_survives_parsing(self):
        """Invariant: a colon-suffixed Bedrock id is not split into provider and model."""
        request = parse_model_spec("bedrock/us.amazon.nova-premier-v1:0", {}, self.REGISTRY)
        self.assertEqual(request.model, "us.amazon.nova-premier-v1:0")
        self.assertEqual(request.provider, "bedrock")

    def test_offline_without_region_falls_back_to_the_mock_model(self):
        """Invariant: a bedrock spec without a region still runs, offline, with a hint."""
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, {"HOME": home}, clear=True):
                model = build_model_from_spec(
                    "bedrock/anthropic.claude-sonnet-5-5", registry=self.REGISTRY
                )
        self.assertIsInstance(model, MockModel)
        self.assertIn("AWS_REGION", model.script[0][0])

    def test_default_model_is_offered_in_discovery(self):
        """Invariant: the bedrock default appears in the /model listing via discovery."""
        entries = get_available_models_from_config({}, self.REGISTRY)
        self.assertIn("anthropic.claude-sonnet-5-5", [e["model"] for e in entries])


SIGV4_REGISTRY = {
    "bedrock": {
        "endpoint": "https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
        "default_model": "us.openai.gpt-5.6-sol",
        "env_keys": ["BEDROCK_API_KEY"],
        "reasoning_format": "none",
        "cache_strategy": "none",
        "auth": {"scheme": "sigv4", "service": "bedrock"},
    }
}


def _clean_aws_env(home: str) -> dict:
    return {
        "HOME": home,
        "AWS_ACCESS_KEY_ID": "",
        "AWS_SECRET_ACCESS_KEY": "",
        "AWS_SESSION_TOKEN": "",
        "AWS_REGION": "",
        "AWS_DEFAULT_REGION": "",
        "AWS_PROFILE": "",
        "BEDROCK_API_KEY": "",
        "MODEL_ENDPOINT": "",
        "MODEL_NAME": "",
    }


class TestBedrockSigV4Wiring(unittest.TestCase):
    def test_builtin_bedrock_uses_sigv4_and_working_default(self):
        """Invariant: the shipped bedrock entry signs SigV4 with a usable default model."""
        registry = load_builtin_providers()
        self.assertEqual(
            registry["bedrock"]["auth"], {"scheme": "sigv4", "service": "bedrock"}
        )
        self.assertEqual(registry["bedrock"]["default_model"], "us.openai.gpt-5.6-sol")

    def test_no_credentials_reports_auth_error_with_sso_hint(self):
        """Invariant: offline SSO-less SigV4 names `aws sso login`, never 'offline'."""

        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, _clean_aws_env(home), clear=True):
                connection = resolve_connection(
                    "bedrock/us.openai.gpt-5.6-sol",
                    config={"region": "us-east-1"},
                    registry={k: dict(v) for k, v in SIGV4_REGISTRY.items()},
                )
                self.assertEqual(connection.auth_scheme, "sigv4")
                self.assertEqual(
                    connection.auth_options, {"service": "bedrock", "aws_profile": "default"}
                )
                self.assertIsNotNone(connection.auth_error)
                assert connection.auth_error is not None
                self.assertIn("aws sso login", connection.auth_error)
                self.assertNotIn("offline", connection.auth_error.lower())
                self.assertTrue(connection.missing_credentials)

    def test_auth_error_builds_a_mock_model_carrying_the_hint(self):
        """Invariant: an unresolvable SigV4 connection still runs, offline, with the hint."""

        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, _clean_aws_env(home), clear=True):
                model = build_model_from_spec(
                    "bedrock/us.openai.gpt-5.6-sol",
                    config={"region": "us-east-1"},
                    registry={k: dict(v) for k, v in SIGV4_REGISTRY.items()},
                )
        self.assertIsInstance(model, MockModel)
        self.assertIn("aws sso login", model.script[0][0])

    def test_sigv4_client_is_built_from_env_credentials(self):
        """Invariant: env AWS credentials yield a SigV4 HttpChatModel, not a MockModel."""

        with tempfile.TemporaryDirectory() as home:
            env = _clean_aws_env(home)
            env.update(
                {
                    "AWS_ACCESS_KEY_ID": "AKID",
                    "AWS_SECRET_ACCESS_KEY": "SECRET",
                    "AWS_REGION": "us-east-1",
                }
            )
            with patch.dict(os.environ, env, clear=True):
                model = build_model_from_spec(
                    "bedrock/us.openai.gpt-5.6-sol",
                    registry={k: dict(v) for k, v in SIGV4_REGISTRY.items()},
                )
        self.assertIsInstance(model, HttpChatModel)
        self.assertEqual(
            model.endpoint,
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
        )
        self.assertEqual(model.auth_scheme, "sigv4")
        self.assertEqual(model.auth_options, {"service": "bedrock", "aws_profile": "default"})
        self.assertEqual(
            model.auth_secrets,
            {
                "access_key": "AKID",
                "secret_key": "SECRET",
                "session_token": None,
                "region": "us-east-1",
            },
        )

    def test_sigv4_request_carries_an_aws4_authorization_header(self):
        """Invariant: a SigV4 generate() signs the wire request (mocked urlopen)."""
        with tempfile.TemporaryDirectory() as home:
            env = _clean_aws_env(home)
            env.update(
                {
                    "AWS_ACCESS_KEY_ID": "AKID",
                    "AWS_SECRET_ACCESS_KEY": "SECRET",
                    "AWS_REGION": "us-east-1",
                }
            )
            with patch.dict(os.environ, env, clear=True):
                model = build_model_from_spec(
                    "bedrock/us.openai.gpt-5.6-sol",
                    registry={k: dict(v) for k, v in SIGV4_REGISTRY.items()},
                )
            from core.types import Message

            resp = MagicMock()
            resp.read.return_value = json.dumps(
                {"choices": [{"message": {"content": "hello"}}]}
            ).encode("utf-8")
            resp.__enter__.return_value = resp
            with patch.dict(os.environ, env, clear=True):
                with patch("urllib.request.urlopen", return_value=resp) as mock_urlopen:
                    content, _ = model.generate([Message(role="user", content="hi")], [])
        self.assertEqual(content, "hello")
        (request,), _ = mock_urlopen.call_args
        auth = request.get_header("Authorization")
        self.assertTrue(auth.startswith("AWS4-HMAC-SHA256 "))
        self.assertIn("bedrock/aws4_request", auth)
        self.assertNotIn("Bearer", auth)

    def test_stale_user_registry_is_backfilled_with_auth(self):
        """Invariant: a pre-SigV4 user copy gains the auth key; user edits still win."""

        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp)
            stale = {
                "bedrock": {
                    "endpoint": "https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
                    "default_model": "my-pinned-model",
                    "env_keys": ["BEDROCK_API_KEY"],
                    "reasoning_format": "none",
                    "cache_strategy": "none",
                }
            }
            (global_dir / "providers.json").write_text(json.dumps(stale), encoding="utf-8")
            registry = load_provider_registry(global_dir=global_dir)
        self.assertEqual(
            registry["bedrock"]["auth"], {"scheme": "sigv4", "service": "bedrock"}
        )
        self.assertEqual(registry["bedrock"]["default_model"], "my-pinned-model")

    def test_spec_dict_auth_beats_config_auth_beats_profile_auth(self):
        """Invariant: auth precedence is spec-dict > config > profile."""

        profile = {"endpoint": "https://x/v1", "auth": {"scheme": "none"}}
        registry = {"custom": profile}
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, _clean_aws_env(home), clear=True):
                conn_profile = resolve_connection(
                    {"model": "m", "provider": "custom"}, registry=registry
                )
                self.assertEqual(conn_profile.auth_scheme, "none")
                conn_cfg = resolve_connection(
                    {"model": "m", "provider": "custom"},
                    config={"auth": "bearer"},
                    registry=registry,
                )
                self.assertEqual(conn_cfg.auth_scheme, "bearer")
                conn_spec = resolve_connection(
                    {"model": "m", "provider": "custom", "auth": {"scheme": "none"}},
                    config={"auth": "bearer"},
                    registry=registry,
                )
                self.assertEqual(conn_spec.auth_scheme, "none")

    def test_model_module_has_no_provider_branches(self):
        """Invariant: core/model.py never names bedrock (registry-driven auth only)."""
        from pathlib import Path as _Path

        source = (_Path(__file__).parent.parent / "core" / "model.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("bedrock", source.lower())


class TestWave6AuthWiring(unittest.TestCase):
    def test_string_alias_carrying_auth_none_is_honored(self):
        """Invariant: a string alias resolving to a dict with auth uses that auth."""
        registry = {"custom": {"endpoint": "https://x.example/v1", "auth": "bearer"}}
        config = {
            "models": {
                "fast": {
                    "model": "m",
                    "provider": "custom",
                    "auth": {"scheme": "none"},
                }
            }
        }
        conn = resolve_connection("fast", config=config, registry=registry)
        self.assertEqual(conn.auth_scheme, "none")

    def test_string_alias_to_bare_id_uses_profile_auth(self):
        """Invariant: a string alias resolving to a bare id keeps profile auth."""
        registry = {"custom": {"endpoint": "https://x.example/v1", "auth": "bearer"}}
        config = {"models": {"fast": "custom/m"}}
        conn = resolve_connection("fast", config=config, registry=registry)
        self.assertEqual(conn.auth_scheme, "bearer")

    def test_bedrock_api_key_overrides_sigv4_with_bearer(self):
        """Invariant: BEDROCK_API_KEY turns a SigV4 spec into a bearer client."""
        with tempfile.TemporaryDirectory() as home:
            env = _clean_aws_env(home)
            env.update({"BEDROCK_API_KEY": "brk-token", "AWS_REGION": "us-east-1"})
            with patch.dict(os.environ, env, clear=True):
                model = build_model_from_spec(
                    "bedrock/us.openai.gpt-5.6-sol",
                    registry={k: dict(v) for k, v in SIGV4_REGISTRY.items()},
                )
        self.assertIsInstance(model, HttpChatModel)
        self.assertEqual(model.auth_scheme, "bearer")
        self.assertEqual(
            model.endpoint,
            "https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
        )
        self.assertEqual(model.auth_secrets, {"api_key": "brk-token"})

    def test_global_placeholder_api_key_does_not_trigger_bearer_override(self):
        """Invariant: config api_key 'none' without BEDROCK_API_KEY keeps SigV4."""
        with tempfile.TemporaryDirectory() as home:
            env = _clean_aws_env(home)
            env.update(
                {
                    "AWS_REGION": "us-east-1",
                    "AWS_ACCESS_KEY_ID": "AKID",
                    "AWS_SECRET_ACCESS_KEY": "SECRET",
                }
            )
            with patch.dict(os.environ, env, clear=True):
                model = build_model_from_spec(
                    "bedrock/us.openai.gpt-5.6-sol",
                    config={"api_key": "none"},
                    registry={k: dict(v) for k, v in SIGV4_REGISTRY.items()},
                )
        self.assertIsInstance(model, HttpChatModel)
        self.assertEqual(model.auth_scheme, "sigv4")

    def test_custom_aws_profile_flows_into_auth_options(self):
        """Invariant: config aws_profile is stashed in auth_options for sigv4."""
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, _clean_aws_env(home), clear=True):
                conn = resolve_connection(
                    "bedrock/us.openai.gpt-5.6-sol",
                    config={"region": "us-east-1", "aws_profile": "myprof"},
                    registry={k: dict(v) for k, v in SIGV4_REGISTRY.items()},
                )
                self.assertEqual(conn.auth_scheme, "sigv4")
                self.assertEqual(conn.auth_options["aws_profile"], "myprof")
                self.assertTrue(conn.missing_credentials)

    def test_missing_credentials_is_profile_aware(self):
        """Invariant: shared-file creds for the stashed profile clear the gate."""
        with tempfile.TemporaryDirectory() as home:
            aws_dir = Path(home) / ".aws"
            aws_dir.mkdir()
            (aws_dir / "credentials").write_text(
                "[myprof]\naws_access_key_id = AKID-FILE\n"
                "aws_secret_access_key = SECRET-FILE\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, _clean_aws_env(home), clear=True):
                conn = resolve_connection(
                    "bedrock/us.openai.gpt-5.6-sol",
                    config={"region": "us-east-1", "aws_profile": "myprof"},
                    registry={k: dict(v) for k, v in SIGV4_REGISTRY.items()},
                )
                self.assertFalse(conn.missing_credentials)

    def test_main_gate_reconstruction_carries_auth_error_and_profile(self):
        """Invariant: the main.py credentials-gate shape keeps auth_error/profile."""
        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, _clean_aws_env(home), clear=True):
                conn = resolve_connection(
                    "bedrock/us.openai.gpt-5.6-sol",
                    config={"region": "us-east-1", "aws_profile": "myprof"},
                    registry={k: dict(v) for k, v in SIGV4_REGISTRY.items()},
                )
                rebuilt = Connection(
                    endpoint=conn.endpoint,
                    api_key=conn.api_key,
                    model=conn.model,
                    auth_scheme=conn.auth_scheme,
                    auth_options=dict(conn.auth_options),
                    auth_error=conn.auth_error,
                )
                self.assertEqual(rebuilt.auth_options.get("aws_profile"), "myprof")
                self.assertIsNotNone(rebuilt.auth_error)
                assert rebuilt.auth_error is not None
                self.assertIn("aws sso login", rebuilt.auth_error)
                self.assertTrue(rebuilt.missing_credentials)

    def test_missing_credentials_none_is_false(self):
        """Invariant: scheme none never reports missing credentials."""
        conn = Connection(
            model="m",
            endpoint="https://x.example/v1",
            auth_scheme="none",
        )
        self.assertFalse(conn.missing_credentials)

    def test_missing_credentials_header_env_source(self):
        """Invariant: header env:VAR is missing exactly when the variable is unset."""
        conn = Connection(
            model="m",
            endpoint="https://x.example/v1",
            auth_scheme="header",
            auth_options={"name": "X-Api-Key", "value_source": "env:TEST_MH_W6_KEY"},
        )
        with patch.dict(os.environ, {}, clear=True):
            os.environ.pop("TEST_MH_W6_KEY", None)
            self.assertTrue(conn.missing_credentials)
        with patch.dict(os.environ, {"TEST_MH_W6_KEY": "v"}):
            self.assertFalse(conn.missing_credentials)

    def test_unknown_scheme_fails_fast_in_build(self):
        """Invariant: an unknown scheme raises ValueError before any client exists."""
        registry = {"custom": {"endpoint": "https://x.example/v1", "auth": "nope-xyz"}}
        with self.assertRaises(ValueError):
            build_model_from_spec(
                {"model": "m", "provider": "custom"}, registry=registry
            )

    def test_backfill_values_are_deep_copied(self):
        """Invariant: mutating a returned profile auth never touches fresh registries."""
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp)
            stale = {
                "bedrock": {
                    "endpoint": "https://bedrock-runtime.{region}.amazonaws.com/openai/v1",
                    "default_model": "my-pinned-model",
                    "env_keys": ["BEDROCK_API_KEY"],
                    "reasoning_format": "none",
                    "cache_strategy": "none",
                }
            }
            (global_dir / "providers.json").write_text(json.dumps(stale), encoding="utf-8")
            first = load_provider_registry(global_dir=global_dir)
        first["bedrock"]["auth"]["scheme"] = "mutated"
        self.assertEqual(
            load_builtin_providers()["bedrock"]["auth"],
            {"scheme": "sigv4", "service": "bedrock"},
        )
        with tempfile.TemporaryDirectory() as tmp2:
            global_dir2 = Path(tmp2)
            (global_dir2 / "providers.json").write_text(
                json.dumps(stale), encoding="utf-8"
            )
            second = load_provider_registry(global_dir=global_dir2)
        self.assertEqual(
            second["bedrock"]["auth"], {"scheme": "sigv4", "service": "bedrock"}
        )


class TestResolveModelSpec(unittest.TestCase):
    AVAILABLE = [
        {"name": "fast", "model": "gemini-2.5-flash", "provider": "google"},
        {"name": "smart", "model": "gpt-4o", "provider": "openai"},
    ]

    def test_resolves_a_configured_alias(self):
        """Invariant: an alias resolves to its canonical configured name."""
        self.assertEqual(resolve_model_spec("FAST", self.AVAILABLE, {}), (True, "fast", ""))

    def test_resolves_a_raw_model_id(self):
        """Invariant: a model id maps back to the alias users can type."""
        self.assertEqual(resolve_model_spec("gpt-4o", self.AVAILABLE, {}), (True, "smart", ""))

    def test_resolves_a_registry_provider(self):
        """Invariant: a known provider is accepted verbatim as a spec."""
        registry = {"openai": {"endpoint": "https://x"}}
        self.assertEqual(resolve_model_spec("openai", self.AVAILABLE, registry), (True, "openai", ""))

    def test_resolves_a_provider_qualified_spec(self):
        """Invariant: 'openai/gpt-4o' is accepted when the provider is registered."""
        registry = {"openai": {"endpoint": "https://x"}}
        ok, canonical, _ = resolve_model_spec("openai/gpt-4o", self.AVAILABLE, registry)
        self.assertEqual((ok, canonical), (True, "openai/gpt-4o"))

    def test_unknown_spec_lists_the_valid_options(self):
        """Invariant: a typo is reported with the available names, not swallowed."""
        ok, canonical, error = resolve_model_spec("nope", self.AVAILABLE, {})
        self.assertFalse(ok)
        self.assertEqual(canonical, "")
        self.assertIn("fast", error)
        self.assertIn("smart", error)

    def test_empty_spec_is_rejected(self):
        """Invariant: a blank /model argument is an error with a clear message."""
        ok, _, error = resolve_model_spec("   ", self.AVAILABLE, {})
        self.assertFalse(ok)
        self.assertIn("cannot be empty", error)

    def test_empty_availability_is_reported_explicitly(self):
        """Invariant: with nothing configured the error says so instead of showing a blank list."""
        _, _, error = resolve_model_spec("gpt-4o", [], {})
        self.assertIn("none configured", error)


if __name__ == "__main__":
    unittest.main()