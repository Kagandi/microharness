import json
import tempfile
import unittest
from pathlib import Path

from core.config import (
    PROJECT_TRUST_NOTICE,
    ConfigLayers,
    HarnessConfig,
    get_default_global_dir,
    load_global_config,
    load_project_config,
    normalize_cache_ttl,
    normalize_reasoning_effort,
    read_jsonc,
    resolve_project_tool_trust,
    strip_json_comments,
)


class TestJsonc(unittest.TestCase):
    def test_strips_line_comments_outside_strings(self):
        """Invariant: # and // comments are removed but string contents are preserved verbatim."""
        raw = '{\n  // a comment\n  "url": "https://x.dev/#frag", # trailing\n  "n": 1\n}'
        cleaned = strip_json_comments(raw)
        self.assertEqual(json.loads(cleaned), {"url": "https://x.dev/#frag", "n": 1})
        self.assertNotIn("a comment", cleaned)

    def test_preserves_escaped_quotes_and_backslashes(self):
        """Invariant: escaped quotes and backslashes inside strings do not end the string early."""
        raw = r'{"a": "he said \"hi\" // not a comment", "b": "C:\\path\\to"}'
        self.assertEqual(
            json.loads(strip_json_comments(raw)),
            {"a": 'he said "hi" // not a comment', "b": r"C:\path\to"},
        )

    def test_unterminated_string_is_left_alone(self):
        """Invariant: malformed input is not silently truncated into valid JSON."""
        self.assertEqual(strip_json_comments('{"a": "unterminated'), '{"a": "unterminated')

    def test_read_jsonc_missing_path_returns_empty(self):
        """Invariant: a missing file yields {} rather than raising."""
        self.assertEqual(read_jsonc(Path("/nonexistent/nope.json")), {})
        self.assertEqual(read_jsonc(None), {})

    def test_read_jsonc_non_object_payload_returns_empty(self):
        """Invariant: a JSON array at the root is rejected, not returned as config."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("[1, 2, 3]", encoding="utf-8")
            self.assertEqual(read_jsonc(path), {})

    def test_read_jsonc_invalid_json_returns_empty(self):
        """Invariant: malformed JSON degrades to {} so a bad config cannot crash startup."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text("{not valid json", encoding="utf-8")
            self.assertEqual(read_jsonc(path), {})


class TestConfigLayers(unittest.TestCase):
    def test_project_keys_shadow_global_keys(self):
        """Invariant: merged config lets the project layer win over the global one."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            global_dir = root / "global"
            workspace = root / "ws"
            (global_dir).mkdir()
            (workspace / ".microharness").mkdir(parents=True)
            (global_dir / "config.json").write_text(
                json.dumps({"sandbox": True, "max_context_tokens": 100}), encoding="utf-8"
            )
            (workspace / ".microharness" / "config.json").write_text(
                json.dumps({"max_context_tokens": 200}), encoding="utf-8"
            )

            layers = ConfigLayers.load(global_dir, workspace)
            self.assertEqual(layers.global_config["sandbox"], True)
            self.assertEqual(layers.merged["max_context_tokens"], 200)
            self.assertEqual(layers.merged["sandbox"], True)

    def test_load_helpers_return_empty_for_absent_files(self):
        """Invariant: absent config files produce {} from both loaders."""
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self.assertEqual(load_global_config(root), {})
            self.assertEqual(load_project_config(root), {})

    def test_load_global_config_reads_json_with_comments(self):
        """Invariant: the global config file supports # comments like the provider registry."""
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)
            (path / "config.json").write_text(
                '{\n  // tuned for laptops\n  "max_tool_tokens": 500\n}', encoding="utf-8"
            )
            self.assertEqual(load_global_config(path), {"max_tool_tokens": 500})


class TestHarnessConfigCascade(unittest.TestCase):
    def setUp(self):
        self.global_dir = Path(tempfile.mkdtemp())
        self.workspace = Path(tempfile.mkdtemp())

    def _resolve(self, explicit: HarnessConfig, global_cfg=None, project_cfg=None):
        (self.global_dir / "config.json").write_text(json.dumps(global_cfg or {}), encoding="utf-8")
        (self.workspace / ".microharness").mkdir(parents=True, exist_ok=True)
        (self.workspace / ".microharness" / "config.json").write_text(
            json.dumps(project_cfg or {}), encoding="utf-8"
        )
        layers = ConfigLayers.load(self.global_dir, self.workspace)
        return explicit.resolve(layers)

    def test_defaults_when_nothing_is_configured(self):
        """Invariant: an unconfigured harness resolves to the documented defaults."""
        cfg = self._resolve(HarnessConfig())
        self.assertEqual(cfg.sandbox, False)
        self.assertEqual(cfg.docker, False)
        self.assertEqual(cfg.docker_image, "python:3.11")
        self.assertEqual(cfg.auto_discover_tools, True)
        self.assertEqual(cfg.disable_default_tools, False)
        self.assertEqual(cfg.runs_dir, None)
        self.assertEqual(cfg.max_tool_tokens, 8000)
        self.assertEqual(cfg.max_context_tokens, 64000)
        self.assertEqual(cfg.trigger_ratio, 0.80)
        self.assertEqual(cfg.cache_ttl, "5m")
        self.assertEqual(cfg.reasoning_effort, None)
        self.assertEqual(cfg.max_step_retries, 3)
        self.assertEqual(cfg.step_retry_delay, 5.0)

    def test_explicit_beats_config_file(self):
        """Invariant: explicit arguments win over both config layers."""
        cfg = self._resolve(
            HarnessConfig(max_context_tokens=999),
            global_cfg={"max_context_tokens": 111},
            project_cfg={"max_context_tokens": 222},
        )
        self.assertEqual(cfg.max_context_tokens, 999)

    def test_project_beats_global(self):
        """Invariant: project config shadows the global default."""
        cfg = self._resolve(
            HarnessConfig(),
            global_cfg={"max_context_tokens": 111},
            project_cfg={"max_context_tokens": 222},
        )
        self.assertEqual(cfg.max_context_tokens, 222)

    def test_falsy_runs_dir_falls_back_to_default(self):
        """Invariant: an empty explicit value means 'unspecified' and yields the built-in default.

        An explicit value always outranks the config file, so an empty string drops
        straight to the default rather than being replaced by a configured path.
        """
        cfg = self._resolve(HarnessConfig(runs_dir=""), global_cfg={"runs_dir": "/from/config"})
        self.assertIsNone(cfg.runs_dir)

    def test_falsy_docker_image_falls_back_to_default(self):
        """Invariant: an empty docker_image cannot disable the default image."""
        cfg = self._resolve(HarnessConfig(docker_image=""))
        self.assertEqual(cfg.docker_image, "python:3.11")

    def test_falsy_explicit_false_is_honoured(self):
        """Invariant: an explicit False disables a feature that defaults to on."""
        cfg = self._resolve(HarnessConfig(auto_discover_tools=False))
        self.assertEqual(cfg.auto_discover_tools, False)

    def test_numeric_strings_are_cast(self):
        """Invariant: string config values are cast to the declared type."""
        cfg = self._resolve(
            HarnessConfig(), global_cfg={"max_step_retries": "7", "trigger_ratio": "0.5"}
        )
        self.assertEqual(cfg.max_step_retries, 7)
        self.assertEqual(cfg.trigger_ratio, 0.5)

    def test_cache_ttl_is_normalized_during_resolution(self):
        """Invariant: '300' from a config file resolves to the canonical '5m'."""
        cfg = self._resolve(HarnessConfig(), global_cfg={"cache_ttl": "300"})
        self.assertEqual(cfg.cache_ttl, "5m")

    def test_invalid_cache_ttl_raises_value_error(self):
        """Invariant: an unsupported TTL is rejected loudly rather than silently ignored."""
        with self.assertRaises(ValueError) as ctx:
            self._resolve(HarnessConfig(cache_ttl="17m"))
        self.assertIn("Unsupported cache TTL", str(ctx.exception))

    def test_legacy_tool_chars_ignored_uses_default_and_warns(self):
        """Invariant: max_tool_chars is ignored; defaults apply and a removal warning is emitted."""
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            cfg = self._resolve(HarnessConfig(), global_cfg={"max_tool_chars": 4000})
        self.assertEqual(cfg.max_tool_tokens, 8000)
        self.assertEqual(cfg.max_context_tokens, 64000)
        self.assertNotEqual(cfg.max_tool_tokens, 1000)
        warning = buf.getvalue()
        self.assertIn("max_tool_chars", warning)
        self.assertIn("max_tool_tokens", warning)

    def test_legacy_context_chars_ignored_uses_default_and_warns(self):
        """Invariant: max_context_chars is ignored; defaults apply and a removal warning is emitted."""
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            cfg = self._resolve(HarnessConfig(), global_cfg={"max_context_chars": 40000})
        self.assertEqual(cfg.max_tool_tokens, 8000)
        self.assertEqual(cfg.max_context_tokens, 64000)
        self.assertNotEqual(cfg.max_context_tokens, 10000)
        warning = buf.getvalue()
        self.assertIn("max_context_chars", warning)
        self.assertIn("max_context_tokens", warning)

    def test_legacy_chars_do_not_override_token_settings(self):
        """Invariant: legacy chars never win over explicit token settings and warn once."""
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            cfg = self._resolve(
                HarnessConfig(),
                global_cfg={
                    "max_tool_chars": 4000,
                    "max_context_chars": 40000,
                    "max_tool_tokens": 1234,
                    "max_context_tokens": 5678,
                },
            )
        self.assertEqual(cfg.max_tool_tokens, 1234)
        self.assertEqual(cfg.max_context_tokens, 5678)
        warning = buf.getvalue()
        self.assertIn("max_tool_chars", warning)
        self.assertIn("max_context_chars", warning)
        self.assertEqual(warning.strip().count("\n"), 0)

    def test_no_warning_without_legacy_chars(self):
        """Invariant: token-only configs resolve silently on stderr."""
        import contextlib
        import io

        buf = io.StringIO()
        with contextlib.redirect_stderr(buf):
            cfg = self._resolve(HarnessConfig(), global_cfg={"max_tool_tokens": 1234})
        self.assertEqual(cfg.max_tool_tokens, 1234)
        self.assertEqual(buf.getvalue(), "")

    def test_get_default_global_dir_falls_back_to_legacy_mini_harness(self):
        """Invariant: falls back to ~/.mini-harness if ~/.microharness does not exist."""
        from unittest.mock import patch

        from core.config import get_default_global_dir

        with tempfile.TemporaryDirectory() as tmp_dir:
            fake_home = Path(tmp_dir)
            legacy = fake_home / ".mini-harness"
            legacy.mkdir()
            with patch("pathlib.Path.home", return_value=fake_home):
                self.assertEqual(get_default_global_dir(), legacy)


class TestProjectToolTrust(unittest.TestCase):
    def _layers(self, global_cfg=None, project_cfg=None):
        return ConfigLayers.of(global_cfg or {}, project_cfg or {})

    def test_explicit_true_grants_trust(self):
        """Invariant: the CLI flag grants trust outright."""
        trusted, ignored = resolve_project_tool_trust(True, self._layers())
        self.assertEqual((trusted, ignored), (True, False))

    def test_explicit_false_overrides_global_grant(self):
        """Invariant: --trust-project-tools=false wins over a global grant."""
        layers = self._layers(global_cfg={"trust_project_tools": True})
        trusted, ignored = resolve_project_tool_trust(False, layers)
        self.assertEqual((trusted, ignored), (False, False))

    def test_global_config_grants_trust(self):
        """Invariant: the user's own global config may grant trust."""
        layers = self._layers(global_cfg={"trust_project_tools": True})
        self.assertEqual(resolve_project_tool_trust(None, layers), (True, False))

    def test_project_config_cannot_self_grant_trust(self):
        """Invariant: a repo cannot trust its own plugins; the claim is reported as ignored."""
        layers = self._layers(project_cfg={"trust_project_tools": True})
        self.assertEqual(resolve_project_tool_trust(None, layers), (False, True))

    def test_notice_names_the_trusted_sources(self):
        """Invariant: the warning tells the user where trust may legitimately come from."""
        self.assertIn("--trust-project-tools", PROJECT_TRUST_NOTICE)
        self.assertIn("~/.microharness/config.json", PROJECT_TRUST_NOTICE)


class TestNormalizers(unittest.TestCase):
    def test_cache_ttl_aliases(self):
        """Invariant: documented TTL spellings collapse to the two canonical values."""
        for raw, expected in [
            ("5m", "5m"),
            ("5min", "5m"),
            ("300", "5m"),
            ("300s", "5m"),
            ("1h", "1h"),
            ("1hour", "1h"),
            ("60m", "1h"),
            ("3600", "1h"),
            (" 1H ", "1h"),
        ]:
            self.assertEqual(normalize_cache_ttl(raw), expected, raw)

    def test_cache_ttl_disabled_values(self):
        """Invariant: every 'off' spelling disables caching."""
        for raw in ("off", "none", "disable", "disabled", "false", "0", "", None):
            self.assertIsNone(normalize_cache_ttl(raw))

    def test_cache_ttl_rejects_unknown(self):
        """Invariant: a typo raises instead of defaulting to a cache the user did not ask for."""
        with self.assertRaises(ValueError):
            normalize_cache_ttl("17m")

    def test_reasoning_levels_and_budgets(self):
        """Invariant: named levels and numeric token budgets pass through lowercased."""
        self.assertEqual(normalize_reasoning_effort("LOW"), "low")
        self.assertEqual(normalize_reasoning_effort("Medium"), "medium")
        self.assertEqual(normalize_reasoning_effort("high"), "high")
        self.assertEqual(normalize_reasoning_effort("xhigh"), "xhigh")
        self.assertEqual(normalize_reasoning_effort("max"), "max")
        self.assertEqual(normalize_reasoning_effort("2048"), "2048")

    def test_reasoning_disabled_values(self):
        """Invariant: 'off' spellings disable reasoning effort."""
        for raw in ("off", "none", "disable", "false", "0", "", None):
            self.assertIsNone(normalize_reasoning_effort(raw))

    def test_reasoning_rejects_unknown(self):
        """Invariant: an unsupported effort raises instead of silently disabling reasoning."""
        with self.assertRaises(ValueError):
            normalize_reasoning_effort("extreme")


class TestGlobalDir(unittest.TestCase):
    def test_returns_a_directory_ending_in_microharness_by_default(self):
        """Invariant: the resolved global dir is a Path (never cached at import time)."""
        resolved = get_default_global_dir()
        self.assertIsInstance(resolved, Path)
        self.assertEqual(resolved.parent, Path.home())
        self.assertEqual(resolved.name, ".microharness")


if __name__ == "__main__":
    unittest.main()
