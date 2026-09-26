import json
import unittest
from unittest.mock import MagicMock, patch

import tempfile
from pathlib import Path

from core.model import (
    HttpChatModel,
    MockModel,
    ReasoningAwareModel,
    find_provider_by_endpoint,
    load_builtin_providers,
    load_provider_registry,
    normalize_cache_ttl,
    normalize_reasoning_effort,
    resolve_provider,
    seed_user_providers,
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
            model="anthropic/claude-3.7-sonnet",
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


if __name__ == "__main__":
    unittest.main()
