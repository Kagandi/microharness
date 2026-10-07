import json
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.model import (
    HttpChatModel,
    MockModel,
    RateLimitError,
    ReasoningAwareModel,
    build_model_from_spec,
    extract_retry_delay,
    find_provider_by_endpoint,
    get_available_models_from_config,
    load_builtin_providers,
    load_provider_registry,
    normalize_cache_ttl,
    normalize_reasoning_effort,
    resolve_provider,
    seed_user_providers,
    strip_json_comments,
)
from core.types import Message, ToolCall


class TestModel(unittest.TestCase):
    def test_url_normalization(self):
        m1 = HttpChatModel(base_url="https://openrouter.ai/api/v1", model="test")
        self.assertEqual(m1.endpoint, "https://openrouter.ai/api/v1/chat/completions")

        m2 = HttpChatModel(base_url="https://openrouter.ai/api/v1/chat/completions", model="test")
        self.assertEqual(m2.endpoint, "https://openrouter.ai/api/v1/chat/completions")

        m3 = HttpChatModel(base_url="http://localhost:11434/v1/", model="test")
        self.assertEqual(m3.endpoint, "http://localhost:11434/v1/chat/completions")

    def test_openrouter_headers_default(self):
        m = HttpChatModel(base_url="https://openrouter.ai/api/v1", model="test")
        self.assertIn("HTTP-Referer", m.extra_headers)
        self.assertIn("X-Title", m.extra_headers)

    def test_mock_model_sequence_and_stop(self):
        script = [
            ("Step 1", [ToolCall(id="t1", name="bash", arguments={"cmd": "ls"})]),
            ("Step 2", []),
        ]
        mock = MockModel(script=script)

        # First call
        text1, calls1 = mock.generate([Message(role="user", content="start")], [])
        self.assertEqual(text1, "Step 1")
        self.assertEqual(len(calls1), 1)

        # Second call
        text2, calls2 = mock.generate([Message(role="user", content="next")], [])
        self.assertEqual(text2, "Step 2")
        self.assertEqual(len(calls2), 0)

        # Subsequent call when script exhausted
        text3, calls3 = mock.generate([], [])
        self.assertIn("completed", text3)
        self.assertEqual(len(calls3), 0)

    def test_message_to_dict_standard_user_and_tool(self):
        m1 = Message(role="user", content="hello")
        self.assertEqual(m1.to_dict(), {"role": "user", "content": "hello"})

        m2 = Message(role="tool", content="out", tool_call_id="call_123")
        self.assertEqual(
            m2.to_dict(),
            {"role": "tool", "content": "out", "tool_call_id": "call_123"},
        )

    def test_message_to_dict_serializes_tool_call_arguments_as_json_string(self):
        tc = ToolCall(id="c1", name="bash", arguments={"command": "ls -la"})
        msg = Message(role="assistant", content="Running command", tool_calls=[tc])
        data = msg.to_dict()

        self.assertEqual(data["role"], "assistant")
        self.assertEqual(data["content"], "Running command")
        self.assertIn("tool_calls", data)
        self.assertEqual(len(data["tool_calls"]), 1)
        self.assertEqual(data["tool_calls"][0]["id"], "c1")
        self.assertEqual(data["tool_calls"][0]["type"], "function")
        self.assertEqual(data["tool_calls"][0]["function"]["name"], "bash")
        # Invariant: OpenAI expects function arguments to be a JSON string, not a python dict or None
        self.assertEqual(
            data["tool_calls"][0]["function"]["arguments"],
            '{"command": "ls -la"}',
        )

    def test_tool_result_to_dict(self):
        from core.types import ToolResult

        tr = ToolResult(
            call_id="call_99",
            output="test output",
            exit_code=0,
            was_truncated=True,
            raw_log_path="/path/to/log.log",
        )
        self.assertEqual(
            tr.to_dict(),
            {
                "call_id": "call_99",
                "output": "test output",
                "exit_code": 0,
                "was_truncated": True,
                "raw_log_path": "/path/to/log.log",
            },
        )

    def test_normalize_cache_ttl(self):
        # 5 minute variants
        self.assertEqual(normalize_cache_ttl("5m"), "5m")
        self.assertEqual(normalize_cache_ttl("5min"), "5m")
        self.assertEqual(normalize_cache_ttl("5mins"), "5m")
        self.assertEqual(normalize_cache_ttl("300s"), "5m")

        # 1 hour variants
        self.assertEqual(normalize_cache_ttl("1h"), "1h")
        self.assertEqual(normalize_cache_ttl("1hour"), "1h")
        self.assertEqual(normalize_cache_ttl("60m"), "1h")
        self.assertEqual(normalize_cache_ttl("3600s"), "1h")

        # Disabled / off variants
        self.assertIsNone(normalize_cache_ttl("off"))
        self.assertIsNone(normalize_cache_ttl("none"))
        self.assertIsNone(normalize_cache_ttl("disabled"))
        self.assertIsNone(normalize_cache_ttl(""))
        self.assertIsNone(normalize_cache_ttl(None))

        # Unsupported raises ValueError
        with self.assertRaises(ValueError):
            normalize_cache_ttl("2h")
        with self.assertRaises(ValueError):
            normalize_cache_ttl("invalid")

    def test_http_chat_model_cache_ttl_on_the_fly(self):
        m = HttpChatModel(base_url="https://openrouter.ai/api/v1", model="test", cache_ttl="5m")
        self.assertEqual(m.cache_ttl, "5m")

        # Switch to 1h on the fly
        res = m.set_cache_ttl("1h")
        self.assertEqual(res, "1h")
        self.assertEqual(m.cache_ttl, "1h")

        # Switch to off on the fly
        res_off = m.set_cache_ttl("off")
        self.assertIsNone(res_off)
        self.assertIsNone(m.cache_ttl)

    @patch("urllib.request.urlopen")
    def test_http_chat_model_generate_payload_caching_5m(self, mock_urlopen):
        # Fake successful HTTP response
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "ok", "tool_calls": None}}],
                "usage": {
                    "prompt_tokens": 1500,
                    "completion_tokens": 50,
                    "prompt_tokens_details": {"cached_tokens": 1200},
                },
            }
        ).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        m = HttpChatModel(
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-3.5-sonnet",
            cache_ttl="5m",
        )

        messages = [
            Message(role="system", content="System guidelines"),
            Message(role="user", content="Fix the bug"),
        ]
        tools = [{"type": "function", "function": {"name": "bash"}}]

        content, calls = m.generate(messages, tools)
        self.assertEqual(content, "ok")
        self.assertEqual(len(calls), 0)

        # Inspect captured request
        req = mock_urlopen.call_args[0][0]
        sent_headers = req.headers
        self.assertEqual(sent_headers.get("X-openrouter-cache-ttl"), "300")
        self.assertNotIn("Anthropic-beta", sent_headers)

        payload = json.loads(req.data.decode("utf-8"))
        # Verify system prompt has ephemeral cache_control
        self.assertEqual(
            payload["messages"][0]["content"],
            [{"type": "text", "text": "System guidelines", "cache_control": {"type": "ephemeral"}}],
        )
        # Verify user message remains normal string
        self.assertEqual(payload["messages"][1]["content"], "Fix the bug")
        # Verify tool has cache_control on last tool
        self.assertEqual(payload["tools"][0]["cache_control"], {"type": "ephemeral"})

        # Check telemetry stats
        stats = m.get_cache_stats()
        self.assertEqual(stats["cache_ttl"], "5m")
        self.assertEqual(stats["total_prompt_tokens"], 1500)
        self.assertEqual(stats["total_cached_tokens"], 1200)
        self.assertEqual(stats["cache_hit_rate_pct"], 80.0)

    @patch("urllib.request.urlopen")
    def test_http_chat_model_generate_payload_caching_1h(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 2000, "completion_tokens": 100},
            }
        ).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        m = HttpChatModel(
            base_url="https://openrouter.ai/api/v1",
            model="anthropic/claude-5.5-sonnet",
            cache_ttl="1h",
        )

        messages = [
            Message(role="system", content="System guidelines"),
            Message(role="user", content="Hello"),
        ]
        tools = [{"type": "function", "function": {"name": "bash"}}]

        m.generate(messages, tools)

        req = mock_urlopen.call_args[0][0]
        sent_headers = req.headers
        self.assertEqual(sent_headers.get("Anthropic-beta"), "extended-cache-ttl-2025-04-11")
        self.assertEqual(sent_headers.get("X-openrouter-cache-ttl"), "3600")

        payload = json.loads(req.data.decode("utf-8"))
        self.assertEqual(
            payload["messages"][0]["content"],
            [
                {
                    "type": "text",
                    "text": "System guidelines",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
        )
        self.assertEqual(
            payload["tools"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )

    @patch("urllib.request.urlopen")
    def test_http_chat_model_generate_payload_caching_off(self, mock_urlopen):
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "choices": [{"message": {"content": "ok"}}],
                "usage": {"prompt_tokens": 500, "completion_tokens": 10},
            }
        ).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        m = HttpChatModel(
            base_url="https://openrouter.ai/api/v1",
            model="test",
            cache_ttl="off",
        )

        messages = [
            Message(role="system", content="System guidelines"),
            Message(role="user", content="Hello"),
        ]
        tools = [{"type": "function", "function": {"name": "bash"}}]

        m.generate(messages, tools)

        req = mock_urlopen.call_args[0][0]
        sent_headers = req.headers
        self.assertNotIn("Anthropic-beta", sent_headers)
        self.assertNotIn("X-openrouter-cache-ttl", sent_headers)

        payload = json.loads(req.data.decode("utf-8"))
        # Plain string, no cache_control dict
        self.assertEqual(payload["messages"][0]["content"], "System guidelines")
        self.assertNotIn("cache_control", payload["tools"][0])

    def test_mock_model_cache_controls(self):
        mock = MockModel(script=[], cache_ttl="5m")
        self.assertEqual(mock.cache_ttl, "5m")
        mock.set_cache_ttl("1h")
        self.assertEqual(mock.cache_ttl, "1h")
        stats = mock.get_cache_stats()
        self.assertEqual(stats["cache_ttl"], "1h")

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_http_chat_model_429_retry_after_success(self, mock_urlopen, mock_sleep):
        """Invariant: HTTP 429 parses Retry-After header and retries successfully."""
        import io
        import urllib.error

        # First call raises 429 with Retry-After: 5
        http_err = urllib.error.HTTPError(
            url="http://test",
            code=429,
            msg="Rate limit",
            hdrs={"Retry-After": "5"},
            fp=io.BytesIO(b'{"error": "rate limit"}'),
        )
        mock_success = MagicMock()
        mock_success.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "recovered"}}]}
        ).encode("utf-8")
        mock_success.__enter__.return_value = mock_success

        mock_urlopen.side_effect = [http_err, mock_success]

        m = HttpChatModel(base_url="http://test/v1", model="test")
        content, _ = m.generate([Message(role="user", content="hi")], [])

        self.assertEqual(content, "recovered")
        mock_sleep.assert_called_once_with(5.0)
        self.assertEqual(m.total_rate_limit_wait_s, 5.0)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_http_chat_model_429_exhausted_raises_rate_limit_error(self, mock_urlopen, mock_sleep):
        """Invariant: When HTTP 429 persists across all retries, RateLimitError is raised."""
        import io
        import urllib.error

        from core.model import RateLimitError

        http_err = urllib.error.HTTPError(
            url="http://test",
            code=429,
            msg="Rate limit",
            hdrs={},
            fp=io.BytesIO(b'{"error": "rate limit quota exceeded"}'),
        )
        mock_urlopen.side_effect = http_err

        m = HttpChatModel(base_url="http://test/v1", model="test")
        with self.assertRaises(RateLimitError) as ctx:
            m.generate([Message(role="user", content="hi")], [])

        self.assertIn("rate limit (HTTP 429) exceeded", str(ctx.exception))
        self.assertTrue(m.total_rate_limit_wait_s > 0)

    @patch("urllib.request.urlopen")
    def test_http_chat_model_google_ai_studio_endpoint_clean_payload(self, mock_urlopen):
        """Invariant: When connecting to Google AI Studio endpoint, URL is normalized and

        no unsupported Anthropic cache_control blocks are injected into messages/tools.
        """
        mock_response = MagicMock()
        mock_response.read.return_value = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": "Hello from Gemini",
                            "tool_calls": [
                                {
                                    "id": "call_1",
                                    "type": "function",
                                    "function": {"name": "bash", "arguments": '{"command": "ls"}'},
                                }
                            ],
                        }
                    }
                ],
                "usage": {"prompt_tokens": 100, "completion_tokens": 20},
            }
        ).encode("utf-8")
        mock_response.__enter__.return_value = mock_response
        mock_urlopen.return_value = mock_response

        m = HttpChatModel(
            base_url="https://generativelanguage.googleapis.com/v1beta/openai",
            model="gemini-2.5-flash",
            api_key="fake-gemini-key",
            cache_ttl="5m",
        )

        self.assertEqual(
            m.endpoint, "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )

        messages = [
            Message(role="system", content="System instruction"),
            Message(role="user", content="List files"),
        ]
        tools = [{"type": "function", "function": {"name": "bash", "parameters": {}}}]

        content, tool_calls = m.generate(messages, tools)

        self.assertEqual(content, "Hello from Gemini")
        self.assertEqual(len(tool_calls), 1)
        self.assertEqual(tool_calls[0].name, "bash")

        req = mock_urlopen.call_args[0][0]
        self.assertEqual(req.headers.get("Authorization"), "Bearer fake-gemini-key")

        payload = json.loads(req.data.decode("utf-8"))
        # System content must remain clean string, not Anthropic dict
        self.assertEqual(payload["messages"][0]["content"], "System instruction")
        # Tool definition must not have cache_control
        self.assertNotIn("cache_control", payload["tools"][0])

    def test_resolve_model_provider_gemini_defaults(self):
        """Invariant: Setting provider='gemini' in config or CLI defaults to Google AI Studio."""
        import argparse

        from main import _resolve_model

        args = argparse.Namespace(
            endpoint=None,
            model=None,
            api_key=None,
            provider="gemini",
        )
        model = _resolve_model(args, config={}, cache_ttl="off")
        self.assertIsInstance(model, HttpChatModel)
        self.assertEqual(
            model.endpoint,
            "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
        )
        self.assertEqual(model.model, "gemini-2.5-flash")

    def test_resolve_model_picks_up_gemini_api_key_env(self):
        """Invariant: GEMINI_API_KEY environment variable is detected."""
        import argparse
        import os
        from unittest.mock import patch

        from main import _resolve_model

        args = argparse.Namespace(
            endpoint=None,
            model=None,
            api_key=None,
            provider=None,
        )
        config = {
            "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
            "model": "gemini-2.5-pro",
        }
        with patch.dict(os.environ, {"GEMINI_API_KEY": "test-env-key-gemini"}, clear=False):
            model = _resolve_model(args, config=config, cache_ttl="off")
            self.assertIsInstance(model, HttpChatModel)
            self.assertEqual(model.api_key, "test-env-key-gemini")

    def test_resolve_model_warns_on_unrecognized_flag(self):
        """Invariant: an explicit --model matching no alias, model id, or provider warns on stderr."""
        import argparse
        import contextlib
        import io

        from main import _resolve_model

        config = {
            "endpoint": "https://example.invalid/v1",
            "api_key": "k",
            "models": [{"name": "good", "model": "good-model", "provider": "openai"}],
        }

        def run(model_flag: str | None) -> str:
            args = argparse.Namespace(
                endpoint=None, model=model_flag, api_key=None, provider=None
            )
            stderr = io.StringIO()
            with contextlib.redirect_stderr(stderr):
                _resolve_model(args, config=config, cache_ttl="off")
            return stderr.getvalue()

        # Typo: warned, and the available list is surfaced.
        bad = run("gpt-4o-mmp")
        self.assertIn("does not match any configured model", bad)
        self.assertIn("gpt-4o-mmp", bad)
        self.assertIn("good", bad)

        # Configured alias, underlying model id, and provider-qualified names are all accepted.
        self.assertEqual(run("good"), "")
        self.assertEqual(run("good-model"), "")
        self.assertEqual(run("openai"), "")
        self.assertEqual(run("openai/gpt-4o"), "")

        # No explicit flag means nothing to validate.
        self.assertEqual(run(None), "")

    def test_harness_resolve_model_spec_contract(self):
        """Invariant: resolve_model_spec canonicalizes known names and explains unknown ones."""
        import tempfile
        from pathlib import Path

        from core.harness import Harness
        from core.model import MockModel

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            (tmp / "g").mkdir()
            (tmp / "g" / "config.json").write_text(
                '{"models":[{"name":"Fast","model":"fast-id","provider":"openai"},'
                '{"name":"Big","model":"big-id","provider":"openai"}]}',
                encoding="utf-8",
            )
            harness = Harness(
                model=MockModel([("x", [])]),
                workspace_dir=tmp / "ws",
                global_dir=tmp / "g",
                runs_dir=tmp / "r",
                sandbox=False,
                verbose=False,
            )

            ok, canonical, err = harness.resolve_model_spec("fast")
            self.assertTrue(ok)
            self.assertEqual(canonical, "Fast")
            self.assertEqual(err, "")

            ok, canonical, _ = harness.resolve_model_spec("  BIG-ID  ")
            self.assertTrue(ok)
            self.assertEqual(canonical, "Big")

            ok, canonical, _ = harness.resolve_model_spec("openai/gpt-4o")
            self.assertTrue(ok)
            self.assertEqual(canonical, "openai/gpt-4o")

            ok, canonical, err = harness.resolve_model_spec("nope")
            self.assertFalse(ok)
            self.assertEqual(canonical, "")
            self.assertIn("Unknown model 'nope'", err)
            self.assertIn("Fast", err)
            self.assertIn("Big", err)

            ok, _, err = harness.resolve_model_spec("   ")
            self.assertFalse(ok)
            self.assertIn("cannot be empty", err)

    def test_normalize_reasoning_effort(self):
        """Invariant: Reasoning effort normalization accepts low/medium/high/digits and disables cleanly."""
        self.assertEqual(normalize_reasoning_effort("low"), "low")
        self.assertEqual(normalize_reasoning_effort("LOW"), "low")
        self.assertEqual(normalize_reasoning_effort("medium"), "medium")
        self.assertEqual(normalize_reasoning_effort("high"), "high")
        self.assertEqual(normalize_reasoning_effort("2048"), "2048")

        self.assertIsNone(normalize_reasoning_effort(None))
        self.assertIsNone(normalize_reasoning_effort("off"))
        self.assertIsNone(normalize_reasoning_effort("none"))
        self.assertIsNone(normalize_reasoning_effort("disable"))
        self.assertIsNone(normalize_reasoning_effort("disabled"))
        self.assertIsNone(normalize_reasoning_effort("false"))
        self.assertIsNone(normalize_reasoning_effort("0"))
        self.assertIsNone(normalize_reasoning_effort(""))

        with self.assertRaises(ValueError):
            normalize_reasoning_effort("extreme")
        with self.assertRaises(ValueError):
            normalize_reasoning_effort("unsupported_effort")

    def test_reasoning_protocol_and_methods(self):
        """Invariant: HttpChatModel and MockModel implement ReasoningAwareModel."""
        m_http = HttpChatModel(base_url="https://api.openai.com/v1", model="o3-mini", reasoning_effort="high")
        self.assertTrue(isinstance(m_http, ReasoningAwareModel))
        self.assertEqual(m_http.get_reasoning_effort(), "high")
        m_http.set_reasoning_effort("low")
        self.assertEqual(m_http.get_reasoning_effort(), "low")
        m_http.set_reasoning_effort("off")
        self.assertIsNone(m_http.get_reasoning_effort())

        m_mock = MockModel(script=[], reasoning_effort="medium")
        self.assertTrue(isinstance(m_mock, ReasoningAwareModel))
        self.assertEqual(m_mock.get_reasoning_effort(), "medium")
        m_mock.set_reasoning_effort("off")
        self.assertIsNone(m_mock.get_reasoning_effort())

    def test_reasoning_payload_injection(self):
        """Invariant: HttpChatModel injects reasoning payload according to effort."""
        messages = [Message(role="user", content="Think carefully")]

        # Standard OpenAI / Google AI Studio endpoint (uses standard reasoning_effort)
        m_std = HttpChatModel(base_url="https://api.openai.com/v1", model="o3-mini", reasoning_effort="low")
        payload_std = m_std._build_payload(messages, tools=[])
        self.assertEqual(payload_std.get("reasoning_effort"), "low")
        self.assertNotIn("reasoning", payload_std)

        # OpenRouter endpoint (uses both standard reasoning_effort and proprietary reasoning dict)
        m_or = HttpChatModel(base_url="https://openrouter.ai/api/v1", model="stealth/model", reasoning_effort="low")
        payload_or = m_or._build_payload(messages, tools=[])
        self.assertEqual(payload_or.get("reasoning_effort"), "low")
        self.assertEqual(payload_or.get("reasoning"), {"effort": "low"})

        # Numeric token budget on OpenRouter
        m_num = HttpChatModel(base_url="https://openrouter.ai/api/v1", model="stealth/model", reasoning_effort="4096")
        payload_num = m_num._build_payload(messages, tools=[])
        self.assertEqual(payload_num.get("reasoning"), {"max_tokens": 4096})
        self.assertNotIn("reasoning_effort", payload_num)

        # Off / None
        m_off = HttpChatModel(base_url="https://api.openai.com/v1", model="o3-mini", reasoning_effort=None)
        payload_off = m_off._build_payload(messages, tools=[])
        self.assertNotIn("reasoning", payload_off)
        self.assertNotIn("reasoning_effort", payload_off)

    def test_get_last_step_usage_and_cache_hit_rate(self):
        """Invariant: Token extraction normalizes OpenAI, Anthropic, Gemini formats and computes hit percentage."""
        m = HttpChatModel(base_url="https://api.openai.com/v1", model="test")

        # 1. OpenAI / OpenRouter format
        m._update_usage({
            "prompt_tokens": 1000,
            "completion_tokens": 200,
            "prompt_tokens_details": {"cached_tokens": 750},
        })
        step_usage = m.get_last_step_usage()
        self.assertEqual(step_usage["prompt_tokens"], 1000)
        self.assertEqual(step_usage["completion_tokens"], 200)
        self.assertEqual(step_usage["cached_tokens"], 750)
        self.assertEqual(step_usage["cache_hit_rate_pct"], 75.0)

        # 2. Anthropic format
        m._update_usage({
            "input_tokens": 2000,
            "output_tokens": 150,
            "cache_read_input_tokens": 1200,
        })
        step_usage2 = m.get_last_step_usage()
        self.assertEqual(step_usage2["prompt_tokens"], 2000)
        self.assertEqual(step_usage2["completion_tokens"], 150)
        self.assertEqual(step_usage2["cached_tokens"], 1200)
        self.assertEqual(step_usage2["cache_hit_rate_pct"], 60.0)

        # 3. Google AI Studio / Gemini format
        m._update_usage({
            "promptTokenCount": 500,
            "candidatesTokenCount": 50,
            "cachedContentTokenCount": 200,
        })
        step_usage3 = m.get_last_step_usage()
        self.assertEqual(step_usage3["prompt_tokens"], 500)
        self.assertEqual(step_usage3["completion_tokens"], 50)
        self.assertEqual(step_usage3["cached_tokens"], 200)
        self.assertEqual(step_usage3["cache_hit_rate_pct"], 40.0)

        # 4. Zero tokens
        m._update_usage({})
        step_usage4 = m.get_last_step_usage()
        self.assertEqual(step_usage4["prompt_tokens"], 0)
        self.assertEqual(step_usage4["completion_tokens"], 0)
        self.assertEqual(step_usage4["cached_tokens"], 0)
        self.assertEqual(step_usage4["cache_hit_rate_pct"], 0.0)

    def test_load_builtin_providers(self):
        """Invariant: Built-in providers.json loads correctly and defines standard providers."""
        builtin = load_builtin_providers()
        self.assertIsInstance(builtin, dict)
        self.assertIn("google", builtin)
        self.assertIn("openrouter", builtin)
        self.assertIn("openai", builtin)
        self.assertIn("anthropic", builtin)
        self.assertIn("ollama", builtin)
        self.assertIn("deepseek", builtin)
        self.assertIn("groq", builtin)

        # Check strict equality of properties
        self.assertEqual(builtin["google"]["default_model"], "gemini-2.5-flash")
        self.assertEqual(builtin["google"]["reasoning_format"], "standard")
        self.assertEqual(builtin["google"]["cache_strategy"], "implicit")
        self.assertEqual(builtin["openrouter"]["reasoning_format"], "openrouter")
        self.assertEqual(builtin["openrouter"]["cache_strategy"], "anthropic_block")

    def test_seed_user_providers(self):
        """Invariant: seed_user_providers creates providers.json in target global directory."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            global_path = Path(tmp_dir)
            created = seed_user_providers(global_path)
            self.assertIsNotNone(created)
            self.assertTrue((global_path / "providers.json").is_file())

            # Second call should not overwrite or error
            second = seed_user_providers(global_path)
            self.assertIsNone(second)

        # Negative case: None path returns None
        self.assertIsNone(seed_user_providers(None))

    def test_load_provider_registry_precedence_and_comments(self):
        """Invariant: Providers merge properly across built-in, global, project, and inline config."""
        with tempfile.TemporaryDirectory() as tmp_global, tempfile.TemporaryDirectory() as tmp_proj:
            global_dir = Path(tmp_global)
            proj_dir = Path(tmp_proj)

            # 1. Global override: customize google default model and add a custom proxy
            global_file = global_dir / "providers.json"
            global_file.write_text(
                """// Comment test
{
  "google": {
    "default_model": "gemini-custom-global"
  },
  "custom-proxy": {
    "endpoint": "https://proxy.example.com/v1",
    "default_model": "custom-v1",
    "reasoning_format": "standard",
    "cache_strategy": "implicit"
  }
}
""",
                encoding="utf-8",
            )

            # 2. Project override: project-specific .microharness/providers.json
            proj_micro = proj_dir / ".microharness"
            proj_micro.mkdir()
            (proj_micro / "providers.json").write_text(
                """# Project comment
{
  "google": {
    "default_model": "gemini-custom-project"
  }
}
""",
                encoding="utf-8",
            )

            # 3. Inline config: config.json 'providers'
            inline = {
                "google": {"default_model": "gemini-inline"},
                "inline-only": {"endpoint": "https://inline.example.com/v1"}
            }

            registry = load_provider_registry(
                global_dir=global_dir,
                project_dir=proj_dir,
                inline_providers=inline,
            )

            # Inline should win for google default_model
            self.assertEqual(registry["google"]["default_model"], "gemini-inline")
            # Google endpoint from built-in should be preserved
            self.assertEqual(
                registry["google"]["endpoint"],
                "https://generativelanguage.googleapis.com/v1beta/openai",
            )
            # Custom proxy from global is retained
            self.assertEqual(registry["custom-proxy"]["default_model"], "custom-v1")
            # Inline only provider is present
            self.assertEqual(registry["inline-only"]["endpoint"], "https://inline.example.com/v1")

    def test_resolve_provider_aliases_and_case_insensitivity(self):
        """Invariant: Provider names resolve aliases and case-insensitivity cleanly."""
        registry = load_builtin_providers()

        # Exact match
        name, prof = resolve_provider(registry, "google")
        self.assertEqual(name, "google")
        self.assertEqual(prof["default_model"], "gemini-2.5-flash")

        # Alias: gemini -> google
        name_gemini, prof_gemini = resolve_provider(registry, "gemini")
        self.assertEqual(name_gemini, "google")
        self.assertEqual(prof_gemini["default_model"], "gemini-2.5-flash")

        # Alias: open-router -> openrouter
        name_or, prof_or = resolve_provider(registry, "open-router")
        self.assertEqual(name_or, "openrouter")

        # Case-insensitivity
        name_case, prof_case = resolve_provider(registry, "GoOgLe")
        self.assertEqual(name_case, "google")

        # Unknown provider returns canonical with empty dict
        unk_name, unk_prof = resolve_provider(registry, "unknown-vendor")
        self.assertEqual(unk_name, "unknown-vendor")
        self.assertEqual(unk_prof, {})

        # None returns None, {}
        none_name, none_prof = resolve_provider(registry, None)
        self.assertIsNone(none_name)
        self.assertEqual(none_prof, {})

    def test_find_provider_by_endpoint(self):
        """Invariant: Matching endpoint finds provider profile regardless of trailing slash."""
        registry = load_builtin_providers()

        name, prof = find_provider_by_endpoint(registry, "https://openrouter.ai/api/v1")
        self.assertEqual(name, "openrouter")

        name_slash, prof_slash = find_provider_by_endpoint(registry, "https://openrouter.ai/api/v1/")
        self.assertEqual(name_slash, "openrouter")

        name_chat, prof_chat = find_provider_by_endpoint(
            registry, "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
        )
        self.assertEqual(name_chat, "google")

        # Negative case: unknown endpoint
        unk_name, unk_prof = find_provider_by_endpoint(registry, "https://unknown.ai/v1")
        self.assertIsNone(unk_name)
        self.assertEqual(unk_prof, {})

    def test_http_chat_model_explicit_capabilities_without_url_sniffing(self):
        """Invariant: HttpChatModel obeys explicit capability flags on arbitrary proxy URLs."""
        # Custom internal proxy URL with explicit openrouter reasoning and anthropic cache block
        proxy_url = "https://internal-ai-gateway.corp.net/v1"
        m = HttpChatModel(
            base_url=proxy_url,
            model="custom-model",
            reasoning_effort="high",
            reasoning_format="openrouter",
            cache_strategy="anthropic_block",
            cache_ttl="1h",
            headers={"X-Custom-Auth": "token123"},
        )

        messages = [
            Message(role="system", content="Be helpful"),
            Message(role="user", content="Test query"),
        ]
        payload = m._build_payload(messages, tools=[])

        # Verifies openrouter reasoning structure applied to arbitrary corporate gateway URL
        self.assertEqual(payload["reasoning_effort"], "high")
        self.assertEqual(payload["reasoning"], {"effort": "high"})

        # Verifies anthropic cache_control block injected without checking URL string
        self.assertIsInstance(payload["messages"][0]["content"], list)
        self.assertEqual(
            payload["messages"][0]["content"][0]["cache_control"],
            {"type": "ephemeral", "ttl": "1h"},
        )

        # Verifies headers
        hdrs = m._build_headers()
        self.assertEqual(hdrs["X-Custom-Auth"], "token123")
        self.assertEqual(hdrs["anthropic-beta"], "extended-cache-ttl-2025-04-11")
        self.assertEqual(hdrs["X-OpenRouter-Cache-TTL"], "3600")

        # Negative case: reasoning_format='none' suppresses reasoning fields completely
        m_none = HttpChatModel(
            base_url=proxy_url,
            model="custom-model",
            reasoning_effort="high",
            reasoning_format="none",
            cache_strategy="none",
        )
        payload_none = m_none._build_payload(messages, tools=[])
        self.assertNotIn("reasoning_effort", payload_none)
        self.assertNotIn("reasoning", payload_none)
        self.assertEqual(payload_none["messages"][0]["content"], "Be helpful")

    def test_extract_retry_delay_from_retry_after_header(self):
        """Invariant: the standard Retry-After header is honoured verbatim."""
        self.assertEqual(extract_retry_delay({"Retry-After": "42"}), 42.0)
        self.assertEqual(extract_retry_delay({"Retry-After": "0"}), 0.0)

    def test_extract_retry_delay_from_reset_header(self):
        """Invariant: an X-RateLimit-Reset epoch stamp becomes the remaining wait."""
        seconds = extract_retry_delay({"X-RateLimit-Reset": str(time.time() + 30)})
        self.assertAlmostEqual(seconds, 30.0, delta=1.5)

        millis = extract_retry_delay({"x-ratelimit-reset": str((time.time() + 20) * 1000)})
        self.assertAlmostEqual(millis, 20.0, delta=1.5)

    def test_extract_retry_delay_elapsed_reset_is_zero_not_negative(self):
        """Invariant: a reset stamp already in the past yields no wait at all."""
        self.assertEqual(extract_retry_delay({"X-RateLimit-Reset": str(time.time() - 60)}), 0.0)

    def test_extract_retry_delay_negative_cases(self):
        """Invariant: absent, malformed, or negative hints fall back to caller backoff."""
        self.assertIsNone(extract_retry_delay({}))
        self.assertIsNone(extract_retry_delay(None))
        self.assertIsNone(extract_retry_delay({"Retry-After": "soon"}))
        self.assertIsNone(extract_retry_delay({"Retry-After": "-5"}))

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_send_request_falls_back_to_backoff_without_delay_headers(self, mock_urlopen, mock_sleep):
        """Invariant: a 429 with no delay header uses exponential backoff, not a guessed delay."""
        import urllib.error

        err = urllib.error.HTTPError(
            url="https://generativelanguage.googleapis.com/v1beta/openai/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs=None,
            fp=MagicMock(read=MagicMock(return_value=b'{"error": {"code": 429}}')),
        )
        success_resp = MagicMock()
        success_resp.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "recovered"}}]}
        ).encode("utf-8")
        success_resp.__enter__.return_value = success_resp

        mock_urlopen.side_effect = [err, success_resp]

        m = HttpChatModel(base_url="https://generativelanguage.googleapis.com/v1beta/openai", model="gemini")
        result = m._send_request(payload={"model": "gemini"}, headers={})

        self.assertEqual(result["choices"][0]["message"]["content"], "recovered")
        mock_sleep.assert_called_once_with(2.0)

    def test_wait_or_cancel_aborts_backoff_when_probe_fires(self):
        """Invariant: an installed cancel probe cuts a retry backoff short and raises."""
        import time

        from core.model import ModelCancelledError

        m = HttpChatModel(base_url="https://example.invalid/v1", model="m")

        # No probe: behaves as a plain sleep with no early exit.
        m.set_cancel_probe(None)
        t0 = time.monotonic()
        m._wait_or_cancel(0.2)
        self.assertGreaterEqual(time.monotonic() - t0, 0.15)

        # Probe fires immediately: returns far sooner than the full delay.
        m.set_cancel_probe(lambda: True)
        t0 = time.monotonic()
        with self.assertRaises(ModelCancelledError):
            m._wait_or_cancel(30.0)
        self.assertLess(time.monotonic() - t0, 5.0)

        # Probe stays false: the full delay is honoured.
        m.set_cancel_probe(lambda: False)
        t0 = time.monotonic()
        m._wait_or_cancel(0.2)
        self.assertGreaterEqual(time.monotonic() - t0, 0.15)

        # Clearing the probe restores plain-sleep behaviour.
        m.set_cancel_probe(None)
        m._wait_or_cancel(0)

    def test_cancel_probe_exception_does_not_break_requests(self):
        """Invariant: a raising probe is treated as 'not cancelled' rather than surfacing."""
        from core.model import ModelCancelledError

        m = HttpChatModel(base_url="https://example.invalid/v1", model="m")

        def bad_probe() -> bool:
            raise RuntimeError("probe exploded")

        m.set_cancel_probe(bad_probe)
        self.assertFalse(m._cancelled())
        m._wait_or_cancel(0)  # must not raise

        # And a cancel raised as an exception still surfaces as a cancellation.
        m.set_cancel_probe(lambda: True)
        with self.assertRaises(ModelCancelledError):
            m._wait_or_cancel(30.0)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_send_request_honours_cancel_before_retry(self, mock_urlopen, mock_sleep):
        """Invariant: a pending cancel short-circuits the retry ladder instead of waiting it out."""
        import urllib.error

        from core.model import ModelCancelledError

        err = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=MagicMock(read=MagicMock(return_value=b"upstream down")),
        )
        mock_urlopen.side_effect = err

        m = HttpChatModel(base_url="https://openrouter.ai/api/v1", model="m")
        # Fail the first attempt, then report a cancel for every later check.
        calls = {"n": 0}

        def probe() -> bool:
            calls["n"] += 1
            return calls["n"] > 1

        m.set_cancel_probe(probe)

        with self.assertRaises(ModelCancelledError):
            m._send_request(payload={"model": "m"}, headers={})

        # Exactly one request was issued: no further attempts after the cancel.
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("urllib.request.urlopen")
    def test_send_request_pre_attempt_guard_when_backoff_is_a_noop(self, mock_urlopen):
        """Invariant: a cancel arriving between backoff and the next attempt still blocks that attempt."""
        import urllib.error

        from core.model import ModelCancelledError

        err = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=503,
            msg="Service Unavailable",
            hdrs=None,
            fp=MagicMock(read=MagicMock(return_value=b"upstream down")),
        )
        mock_urlopen.side_effect = err

        m = HttpChatModel(base_url="https://openrouter.ai/api/v1", model="m")
        # Neutralise the backoff so only the pre-attempt guard can observe the cancel.
        m._wait_or_cancel = lambda seconds: None

        calls = {"n": 0}

        def probe() -> bool:
            calls["n"] += 1
            return calls["n"] >= 2

        m.set_cancel_probe(probe)

        with self.assertRaises(ModelCancelledError):
            m._send_request(payload={"model": "m"}, headers={})

        # The retry was refused rather than re-issued.
        self.assertEqual(mock_urlopen.call_count, 1)

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_send_request_http_402_fails_immediately(self, mock_urlopen, mock_sleep):
        """Invariant: HTTP 402 (Insufficient credits) fails immediately without retry loop."""
        import urllib.error
        err_body = b'{"error": {"code": 402, "message": "Insufficient credits"}}'
        err = urllib.error.HTTPError(
            url="https://openrouter.ai/api/v1/chat/completions",
            code=402,
            msg="Payment Required",
            hdrs=None,
            fp=MagicMock(read=MagicMock(return_value=err_body)),
        )
        mock_urlopen.side_effect = err

        m = HttpChatModel(base_url="https://openrouter.ai/api/v1", model="test")
        with self.assertRaises(RateLimitError) as ctx:
            m._send_request(payload={"model": "test"}, headers={})

        self.assertIn("rejected the request for payment", str(ctx.exception))
        mock_sleep.assert_not_called()

    @patch("time.sleep")
    @patch("urllib.request.urlopen")
    def test_send_request_waits_the_delay_the_server_asked_for(self, mock_urlopen, mock_sleep):
        """Invariant: a throttled request waits exactly the Retry-After the server sent."""
        import urllib.error

        err = urllib.error.HTTPError(
            url="https://api.example.com/chat/completions",
            code=429,
            msg="Too Many Requests",
            hdrs={"Retry-After": "5"},
            fp=MagicMock(read=MagicMock(return_value=b'{"error":{"message":"slow down"}}')),
        )
        success_resp = MagicMock()
        success_resp.read.return_value = json.dumps(
            {"choices": [{"message": {"content": "recovered"}}]}
        ).encode("utf-8")
        success_resp.__enter__.return_value = success_resp

        mock_urlopen.side_effect = [err, success_resp]

        m = HttpChatModel(base_url="https://api.example.com", model="test-model")
        result = m._send_request(payload={"prompt": "hi"}, headers={})

        self.assertEqual(result["choices"][0]["message"]["content"], "recovered")
        mock_sleep.assert_called_once_with(5.0)
        self.assertEqual(m.total_rate_limit_wait_s, 5.0)

    def test_load_provider_registry_pure_read_only(self):
        """Invariant: load_provider_registry is strictly read-only and never writes to disk."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            non_existent_global = Path(tmp_dir) / "sub_global"
            self.assertFalse(non_existent_global.exists())
            reg = load_provider_registry(global_dir=non_existent_global)
            self.assertIn("google", reg)
            self.assertIn("openrouter", reg)
            self.assertFalse(non_existent_global.exists())

    def test_strip_json_comments_preserves_strings(self):
        """Invariant: strip_json_comments strips // and # comments while strictly preserving strings containing # and //."""
        raw = """// Top level comment
        {
            # Hash comment
            "title": "Issue #42: Check // comment",
            "url": "https://example.com/api",
            "nested": {
                "escaped": "Quote \\"with # and //\\"",
                "comment_at_end": 123 // inline comment
            } # inline hash comment
        }
        """
        cleaned = strip_json_comments(raw)
        data = json.loads(cleaned)
        self.assertEqual(data["title"], "Issue #42: Check // comment")
        self.assertEqual(data["url"], "https://example.com/api")
        self.assertEqual(data["nested"]["escaped"], 'Quote "with # and //"')
        self.assertEqual(data["nested"]["comment_at_end"], 123)

    def test_in_flight_timeout_bounds_model_request(self):
        """Invariant: HttpChatModel bounds socket requests against deadline and raises TimeoutError when expired."""
        import time
        model = HttpChatModel(
            base_url="https://api.example.com",
            model="test-model",
            timeout=90.0,
        )
        model.set_deadline(time.time() - 1.0)
        with self.assertRaises(TimeoutError):
            model._send_request({"prompt": "hi"}, {})

    @patch("urllib.request.urlopen")
    def test_send_request_clamps_socket_timeout_to_remaining_deadline(self, mock_urlopen):
        """Invariant: In-flight socket timeout is clamped to remaining task deadline when deadline < timeout."""
        import time
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        model = HttpChatModel(
            base_url="https://api.example.com",
            model="test-model",
            timeout=90.0,
        )
        # Set deadline 3.0 seconds in the future
        model.set_deadline(time.time() + 3.0)
        model._send_request(payload={"prompt": "hi"}, headers={})

        mock_urlopen.assert_called_once()
        passed_timeout = mock_urlopen.call_args.kwargs.get("timeout")
        self.assertIsNotNone(passed_timeout)
        self.assertLessEqual(passed_timeout, 3.0)
        self.assertGreater(passed_timeout, 0.0)
        # Strictly verify it did NOT pass default self.timeout (90.0)
        self.assertNotEqual(passed_timeout, 90.0)

    @patch("urllib.request.urlopen")
    def test_send_request_clamps_socket_timeout_when_remaining_exceeds_timeout(self, mock_urlopen):
        """Invariant: In-flight socket timeout is clamped to self.timeout when remaining deadline > timeout."""
        import time
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        model = HttpChatModel(
            base_url="https://api.example.com",
            model="test-model",
            timeout=15.0,
        )
        # Set deadline 100.0s in the future (remaining >> timeout)
        model.set_deadline(time.time() + 100.0)
        model._send_request(payload={"prompt": "hi"}, headers={})

        mock_urlopen.assert_called_once()
        passed_timeout = mock_urlopen.call_args.kwargs.get("timeout")
        # Invariant: Socket timeout must be clamped to self.timeout (15.0), not remaining deadline (~100.0)
        self.assertEqual(passed_timeout, 15.0)

    @patch("urllib.request.urlopen")
    def test_send_request_clamps_socket_timeout_minimum_floor(self, mock_urlopen):
        """Invariant: In-flight socket timeout enforces a 0.1s minimum floor when remaining deadline is near zero."""
        import time
        mock_resp = MagicMock()
        mock_resp.read.return_value = json.dumps({"choices": [{"message": {"content": "ok"}}]}).encode("utf-8")
        mock_resp.__enter__.return_value = mock_resp
        mock_resp.__exit__.return_value = None
        mock_urlopen.return_value = mock_resp

        model = HttpChatModel(
            base_url="https://api.example.com",
            model="test-model",
            timeout=15.0,
        )
        # Set deadline barely in the future (remaining ~0.02s < 0.1s floor)
        model.set_deadline(time.time() + 0.02)
        model._send_request(payload={"prompt": "hi"}, headers={})

        mock_urlopen.assert_called_once()
        passed_timeout = mock_urlopen.call_args.kwargs.get("timeout")
        # Invariant: Socket timeout must be clamped to 0.1s minimum floor
        self.assertEqual(passed_timeout, 0.1)

    def test_http_chat_model_reset(self):
        """Invariant: HttpChatModel.reset() zeroes cumulative token counters and clears last_usage."""
        m = HttpChatModel(base_url="https://api.example.com", model="test")
        m._update_usage({"prompt_tokens": 100, "completion_tokens": 20, "cached_tokens": 50})
        self.assertEqual(m.total_prompt_tokens, 100)
        self.assertEqual(m.total_completion_tokens, 20)
        self.assertEqual(m.total_cached_tokens, 50)
        self.assertTrue(bool(m.last_usage))

        m.reset()
        self.assertEqual(m.total_prompt_tokens, 0)
        self.assertEqual(m.total_completion_tokens, 0)
        self.assertEqual(m.total_cached_tokens, 0)
        self.assertEqual(m.last_usage, {})

    def test_capability_protocols_runtime_checkable(self):
        """Invariant: Capability protocols are @runtime_checkable and accurately identify conforming models."""
        from core.model import (
            DeadlineAwareModel,
            EndpointAwareModel,
            NamedModel,
            ProviderAwareModel,
            RateLimitAwareModel,
            UsageReportingModel,
        )

        http_model = HttpChatModel(
            base_url="https://api.example.com",
            model="gpt-4o",
            api_key="secret",
        )
        self.assertTrue(isinstance(http_model, DeadlineAwareModel))
        self.assertTrue(isinstance(http_model, RateLimitAwareModel))
        self.assertTrue(isinstance(http_model, UsageReportingModel))
        self.assertTrue(isinstance(http_model, NamedModel))
        self.assertTrue(isinstance(http_model, EndpointAwareModel))
        self.assertTrue(isinstance(http_model, ProviderAwareModel))

        mock_model = MockModel(script=[])
        self.assertFalse(isinstance(mock_model, DeadlineAwareModel))
        self.assertFalse(isinstance(mock_model, RateLimitAwareModel))
        self.assertTrue(isinstance(mock_model, UsageReportingModel))
        self.assertTrue(isinstance(mock_model, NamedModel))
        self.assertFalse(isinstance(mock_model, EndpointAwareModel))
        self.assertTrue(isinstance(mock_model, ProviderAwareModel))

        class EmptyDummy:
            pass

        dummy = EmptyDummy()
        self.assertFalse(isinstance(dummy, DeadlineAwareModel))
        self.assertFalse(isinstance(dummy, RateLimitAwareModel))
        self.assertFalse(isinstance(dummy, UsageReportingModel))
        self.assertFalse(isinstance(dummy, NamedModel))
        self.assertFalse(isinstance(dummy, EndpointAwareModel))
        self.assertFalse(isinstance(dummy, ProviderAwareModel))

    def test_get_available_models_from_config_list_and_aliases(self):
        """Invariant: get_available_models_from_config normalizes list, dictionary aliases, and falls back to provider defaults."""
        # 1. Config with list of models
        cfg_list = {
            "models": ["gemini-2.5-flash", "gpt-4o"],
            "provider": "google",
        }
        res_list = get_available_models_from_config(cfg_list)
        self.assertEqual(len(res_list), 2)
        self.assertEqual(res_list[0]["name"], "gemini-2.5-flash")
        self.assertEqual(res_list[0]["model"], "gemini-2.5-flash")
        self.assertEqual(res_list[1]["name"], "gpt-4o")

        # 2. Config with dictionary aliases
        cfg_dict = {
            "models": {
                "fast": {
                    "model": "gemini-2.5-flash",
                    "provider": "google",
                    "description": "Fast low-latency model",
                },
                "smart": "claude-3-5-sonnet",
            }
        }
        res_dict = get_available_models_from_config(cfg_dict)
        self.assertEqual(len(res_dict), 2)
        by_name = {m["name"]: m for m in res_dict}
        self.assertIn("fast", by_name)
        self.assertEqual(by_name["fast"]["model"], "gemini-2.5-flash")
        self.assertEqual(by_name["fast"]["provider"], "google")
        self.assertEqual(by_name["fast"]["description"], "Fast low-latency model")
        self.assertIn("smart", by_name)
        self.assertEqual(by_name["smart"]["model"], "claude-3-5-sonnet")

        # 3. Fallback when config has no models key: discovers active provider defaults
        mock_registry = {
            "google": {
                "name": "Google Gemini",
                "default_model": "gemini-2.5-pro",
                "api_key_env": "GEMINI_API_KEY",
                "endpoint": "https://generativelanguage.googleapis.com/v1beta/openai",
                "models": ["gemini-2.5-pro", "gemini-2.5-flash"],
            }
        }
        with patch.dict("os.environ", {"GEMINI_API_KEY": "fake_key"}):
            res_fb = get_available_models_from_config({}, registry=mock_registry)
            self.assertGreaterEqual(len(res_fb), 1)
            model_names = [m["model"] for m in res_fb]
            self.assertIn("gemini-2.5-pro", model_names)

    def test_build_model_from_spec_and_options(self):
        """Invariant: build_model_from_spec creates configured models with proper endpoints, cache_ttl, and reasoning."""
        config = {
            "provider": "openai",
            "model": "gpt-4o",
        }
        registry = {
            "openai": {
                "name": "OpenAI",
                "default_model": "gpt-4o",
                "endpoint": "https://api.openai.com/v1/chat/completions",
                "api_key_env": "OPENAI_API_KEY",
            }
        }

        # 1. Spec as string name
        with patch.dict("os.environ", {"OPENAI_API_KEY": "sk-test"}):
            model1 = build_model_from_spec(
                "gpt-4o-mini",
                config=config,
                registry=registry,
                cache_ttl="5m",
                reasoning_effort="low",
            )
            self.assertIsInstance(model1, HttpChatModel)
            self.assertEqual(model1.model, "gpt-4o-mini")
            self.assertEqual(model1.cache_ttl, "5m")
            self.assertEqual(model1.reasoning_effort, "low")

        # 2. Spec as dictionary specification
        spec_dict = {
            "model": "deepseek-coder",
            "provider": "custom_provider",
            "endpoint": "https://api.custom.com/v1/chat/completions",
            "api_key": "custom-key",
            "description": "Custom coder model",
        }
        model2 = build_model_from_spec(spec_dict, config=config, registry=registry)
        self.assertIsInstance(model2, HttpChatModel)
        self.assertEqual(model2.model, "deepseek-coder")
        self.assertEqual(model2.provider, "custom_provider")
        self.assertEqual(model2.endpoint, "https://api.custom.com/v1/chat/completions")

        # 3. Negative case: invalid spec raises ValueError
        with self.assertRaises(ValueError):
            build_model_from_spec("", config=config, registry=registry)
        with self.assertRaises(ValueError):
            build_model_from_spec(123, config=config, registry=registry)  # type: ignore


if __name__ == "__main__":
    unittest.main()

