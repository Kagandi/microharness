import unittest

from core.model import HttpChatModel, MockModel
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


if __name__ == "__main__":
    unittest.main()
