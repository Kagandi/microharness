import json
import unittest

from core.model import HttpChatModel
from core.types import Message


def _client(strategy, ttl):
    return HttpChatModel(
        base_url="http://localhost:1234/v1",
        model="test",
        cache_strategy=strategy,
        cache_ttl=ttl,
    )


class TestOpenAIBreakpointCache(unittest.TestCase):
    def test_breakpoint_present_exact_payload(self):
        m = _client("openai_breakpoint", "5m")
        messages = [
            Message(role="system", content="System guidelines"),
            Message(role="user", content="Fix the bug"),
        ]
        tools = [{"type": "function", "function": {"name": "bash"}}]
        payload = m._build_payload(messages, tools)
        self.assertEqual(
            payload,
            {
                "model": "test",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "System guidelines",
                                "prompt_cache_breakpoint": {"mode": "explicit"},
                            }
                        ],
                    },
                    {"role": "user", "content": "Fix the bug"},
                ],
                "tools": [{"type": "function", "function": {"name": "bash"}}],
                "tool_choice": "auto",
            },
        )
        # Normalized "1h" enables the identical breakpoint (no TTL field on this endpoint).
        m_hour = _client("openai_breakpoint", "1h")
        payload_hour = m_hour._build_payload(messages, tools)
        self.assertEqual(payload_hour["messages"], payload["messages"])
        # Emits no cache headers.
        self.assertEqual(m._build_headers(), {"Content-Type": "application/json"})

    def test_breakpoint_absent_ttl_off(self):
        m = _client("openai_breakpoint", "off")
        messages = [
            Message(role="system", content="System guidelines"),
            Message(role="user", content="Hello"),
        ]
        payload = m._build_payload(messages, [])
        self.assertEqual(
            payload,
            {
                "model": "test",
                "messages": [
                    {"role": "system", "content": "System guidelines"},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        self.assertNotIn("prompt_cache_breakpoint", json.dumps(payload))

    def test_breakpoint_absent_no_system_message(self):
        m = _client("openai_breakpoint", "5m")
        messages = [Message(role="user", content="Hello")]
        payload = m._build_payload(messages, [])
        self.assertEqual(
            payload,
            {"model": "test", "messages": [{"role": "user", "content": "Hello"}]},
        )
        self.assertNotIn("prompt_cache_breakpoint", json.dumps(payload))

    def test_breakpoint_absent_non_string_content(self):
        m = _client("openai_breakpoint", "5m")
        listed = [{"type": "text", "text": "hi"}]
        messages = [Message.from_dict({"role": "system", "content": listed})]
        payload = m._build_payload(messages, [])
        self.assertEqual(
            payload,
            {"model": "test", "messages": [{"role": "system", "content": listed}]},
        )
        self.assertNotIn("prompt_cache_breakpoint", json.dumps(payload))

    def test_breakpoint_tools_payload_untouched(self):
        m = _client("openai_breakpoint", "5m")
        messages = [
            Message(role="system", content="System guidelines"),
            Message(role="user", content="Hello"),
        ]
        tools = [{"type": "function", "function": {"name": "bash"}}]
        payload = m._build_payload(messages, tools)
        self.assertEqual(payload["tools"], tools)
        dumped = json.dumps(payload["tools"])
        self.assertNotIn("prompt_cache_breakpoint", dumped)
        self.assertNotIn("cache_control", dumped)

    def test_breakpoint_does_not_mutate_input_message(self):
        m = _client("openai_breakpoint", "5m")
        system = Message(role="system", content="System guidelines")
        messages = [system, Message(role="user", content="Hello")]
        m._build_payload(messages, [])
        self.assertEqual(system.content, "System guidelines")
        self.assertIsInstance(system.content, str)

    def test_regression_anthropic_block_payload_unchanged(self):
        m = _client("anthropic_block", "5m")
        messages = [
            Message(role="system", content="System guidelines"),
            Message(role="user", content="Fix the bug"),
        ]
        tools = [{"type": "function", "function": {"name": "bash"}}]
        payload = m._build_payload(messages, tools)
        self.assertEqual(
            payload,
            {
                "model": "test",
                "messages": [
                    {
                        "role": "system",
                        "content": [
                            {
                                "type": "text",
                                "text": "System guidelines",
                                "cache_control": {"type": "ephemeral"},
                            }
                        ],
                    },
                    {"role": "user", "content": "Fix the bug"},
                ],
                "tools": [
                    {
                        "type": "function",
                        "function": {"name": "bash"},
                        "cache_control": {"type": "ephemeral"},
                    }
                ],
                "tool_choice": "auto",
            },
        )
        self.assertNotIn("prompt_cache_breakpoint", json.dumps(payload))

    def test_regression_implicit_payload_unchanged(self):
        m = _client("implicit", "5m")
        messages = [
            Message(role="system", content="System guidelines"),
            Message(role="user", content="Fix the bug"),
        ]
        tools = [{"type": "function", "function": {"name": "bash"}}]
        payload = m._build_payload(messages, tools)
        self.assertEqual(
            payload,
            {
                "model": "test",
                "messages": [
                    {"role": "system", "content": "System guidelines"},
                    {"role": "user", "content": "Fix the bug"},
                ],
                "tools": [{"type": "function", "function": {"name": "bash"}}],
                "tool_choice": "auto",
            },
        )
        dumped = json.dumps(payload)
        self.assertNotIn("prompt_cache_breakpoint", dumped)
        self.assertNotIn("cache_control", dumped)

    def test_regression_unknown_strategy_payload_unchanged(self):
        m = _client("does_not_exist", "5m")
        messages = [
            Message(role="system", content="System guidelines"),
            Message(role="user", content="Fix the bug"),
        ]
        tools = [{"type": "function", "function": {"name": "bash"}}]
        payload = m._build_payload(messages, tools)
        self.assertEqual(
            payload,
            {
                "model": "test",
                "messages": [
                    {"role": "system", "content": "System guidelines"},
                    {"role": "user", "content": "Fix the bug"},
                ],
                "tools": [{"type": "function", "function": {"name": "bash"}}],
                "tool_choice": "auto",
            },
        )
        self.assertNotIn("prompt_cache_breakpoint", json.dumps(payload))


class TestGatewayAutoCache(unittest.TestCase):
    def test_gateway_auto_injects_provider_options_exact_payload(self):
        m = _client("gateway_auto", "5m")
        messages = [
            Message(role="system", content="System guidelines"),
            Message(role="user", content="Fix the bug"),
        ]
        tools = [{"type": "function", "function": {"name": "bash"}}]
        payload = m._build_payload(messages, tools)
        self.assertEqual(
            payload,
            {
                "model": "test",
                "messages": [
                    {"role": "system", "content": "System guidelines"},
                    {"role": "user", "content": "Fix the bug"},
                ],
                "providerOptions": {"gateway": {"caching": "auto"}},
                "tools": [{"type": "function", "function": {"name": "bash"}}],
                "tool_choice": "auto",
            },
        )
        # Normalized 1h also enables gateway auto caching without emitting Anthropic/OpenRouter headers.
        m_hour = _client("gateway_auto", "1h")
        self.assertEqual(
            m_hour._build_payload(messages, tools),
            payload,
        )
        self.assertEqual(m_hour._build_headers(), {"Content-Type": "application/json"})

    def test_gateway_auto_absent_when_ttl_off(self):
        m = _client("gateway_auto", "off")
        messages = [
            Message(role="system", content="System guidelines"),
            Message(role="user", content="Hello"),
        ]
        payload = m._build_payload(messages, [])
        self.assertEqual(
            payload,
            {
                "model": "test",
                "messages": [
                    {"role": "system", "content": "System guidelines"},
                    {"role": "user", "content": "Hello"},
                ],
            },
        )
        self.assertNotIn("providerOptions", payload)

    def test_gateway_auto_applied_from_vercel_endpoint_without_explicit_flags(self):
        m = HttpChatModel(
            base_url="https://ai-gateway.vercel.sh/v1",
            model="openai/gpt-4o",
            api_key="vck-test",
            cache_ttl="5m",
        )
        self.assertEqual(m.provider, "vercel")
        self.assertEqual(m.cache_strategy, "gateway_auto")
        self.assertEqual(m.reasoning_format, "gateway")
        payload = m._build_payload([Message(role="user", content="hi")], [])
        self.assertEqual(payload["providerOptions"], {"gateway": {"caching": "auto"}})
        self.assertEqual(
            m._build_headers(),
            {
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/microharness",
                "X-Title": "microharness",
                "Authorization": "Bearer vck-test",
            },
        )


if __name__ == "__main__":
    unittest.main()
