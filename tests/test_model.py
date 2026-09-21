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


if __name__ == "__main__":
    unittest.main()
