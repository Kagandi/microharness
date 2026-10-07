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

    def test_model_under_models_scoped_endpoint_not_overridden_by_top_level_endpoint(self):
        """Invariant: selecting a provider or model with its own endpoint does not get overridden by top-level endpoint."""
        config = {
            "endpoint": "https://openrouter.ai/api/v1",
            "models": {
                "smart": {"model": "gpt-4o", "provider": "openai"},
                "local": {"model": "llama-3", "endpoint": "http://localhost:1234/v1"},
            },
        }
        conn_smart = resolve_connection("smart", config=config, registry=self.REGISTRY)
        self.assertEqual(conn_smart.endpoint, "https://openai/v1")
        self.assertEqual(conn_smart.provider, "openai")

        conn_local = resolve_connection("local", config=config, registry=self.REGISTRY)
        self.assertEqual(conn_local.endpoint, "http://localhost:1234/v1")


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
                self.assertEqual(connection.auth_options, {"service": "bedrock"})
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
        self.assertEqual(model.auth_options, {"service": "bedrock"})
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