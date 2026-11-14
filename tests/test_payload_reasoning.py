"""Payload reasoning-format matrix (strict equality, no network)."""

import unittest

from core.model import HttpChatModel
from core.types import Message

BASE = "http://localhost:1/v1"
TOOL = {
    "type": "function",
    "function": {
        "name": "ping",
        "description": "x",
        "parameters": {"type": "object", "properties": {}},
    },
}


def _client(**kwargs: object) -> HttpChatModel:
    args: dict = {"base_url": BASE, "model": "m"}
    args.update(kwargs)
    return HttpChatModel(**args)


class TestExplicitNoneReasoning(unittest.TestCase):
    def test_explicit_none_sends_none_without_effort(self):
        """Invariant: explicit_none states reasoning off when no effort is set."""
        payload = _client(reasoning_format="explicit_none")._build_payload(
            [Message(role="user", content="hi")], [TOOL]
        )
        self.assertEqual(payload["reasoning_effort"], "none")
        self.assertNotIn("reasoning", payload)

    def test_explicit_none_honors_user_effort(self):
        """Invariant: an explicit user effort wins over the implicit none."""
        payload = _client(reasoning_format="explicit_none", reasoning_effort="low")._build_payload(
            [Message(role="user", content="hi")], [TOOL]
        )
        self.assertEqual(payload["reasoning_effort"], "low")
        self.assertNotIn("reasoning", payload)

    def test_plain_none_omits_all_reasoning_keys(self):
        """Invariant: the none strategy sends no reasoning params (local servers)."""
        payload = _client(reasoning_format="none", reasoning_effort="low")._build_payload(
            [Message(role="user", content="hi")], [TOOL]
        )
        self.assertNotIn("reasoning_effort", payload)
        self.assertNotIn("reasoning", payload)


if __name__ == "__main__":
    unittest.main()
