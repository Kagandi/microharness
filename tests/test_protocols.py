import unittest

from core.protocols import (
    CacheAwareModel,
    CancellableModel,
    DeadlineAwareModel,
    EndpointAwareModel,
    Model,
    NamedModel,
    ProviderAwareModel,
    RateLimitAwareModel,
    ReasoningAwareModel,
    ResettableModel,
    UsageReportingModel,
)
from core.types import Message, ToolCall


class FullFeaturedModel:
    """A model implementing every optional capability."""

    model = "gpt-4o"
    provider = "openai"
    endpoint = "https://api.openai.com/v1/chat/completions"
    total_rate_limit_wait_s = 12.5

    def generate(self, messages, tools):
        return "ok", []

    def reset(self):
        pass

    def set_cache_ttl(self, ttl):
        return ttl

    def get_cache_stats(self):
        return {"cache_ttl": "5m"}

    def set_reasoning_effort(self, effort):
        return effort

    def get_reasoning_effort(self):
        return "low"

    def set_deadline(self, deadline):
        pass

    def set_cancel_probe(self, probe):
        pass

    def get_last_step_usage(self):
        return {"prompt_tokens": 1}


class MinimalModel:
    """A model with nothing but the required generate() method."""

    def generate(self, messages, tools):
        return "ok", []


class TestCapabilityProtocols(unittest.TestCase):
    ALL_PROTOCOLS = (
        Model,
        ResettableModel,
        CacheAwareModel,
        ReasoningAwareModel,
        DeadlineAwareModel,
        CancellableModel,
        RateLimitAwareModel,
        UsageReportingModel,
        NamedModel,
        EndpointAwareModel,
        ProviderAwareModel,
    )

    def test_minimal_model_satisfies_only_the_base_protocol(self):
        """Invariant: a bare generate() implementation is a valid Model and nothing more."""
        minimal = MinimalModel()
        self.assertIsInstance(minimal, Model)
        for protocol in self.ALL_PROTOCOLS:
            if protocol is Model:
                continue
            self.assertNotIsInstance(minimal, protocol, f"{protocol.__name__} should not match")

    def test_full_model_satisfies_every_protocol(self):
        """Invariant: an implementation opting into all capabilities is detected as such."""
        full = FullFeaturedModel()
        for protocol in self.ALL_PROTOCOLS:
            self.assertIsInstance(full, protocol, f"{protocol.__name__} should match")

    def test_partial_capabilities_are_detected_independently(self):
        """Invariant: capabilities are orthogonal; missing one never hides the others."""
        base = FullFeaturedModel()
        partial = type("Partial", (MinimalModel,), {})()
        partial.total_rate_limit_wait_s = 0.0

        self.assertIsInstance(partial, RateLimitAwareModel)
        self.assertNotIsInstance(partial, CacheAwareModel)
        self.assertEqual(base.total_rate_limit_wait_s, 12.5)

    def test_protocol_generate_signature_matches_the_call_site(self):
        """Invariant: generate() consumes Message objects and returns (text, tool calls)."""
        model = MinimalModel()
        content, calls = model.generate([Message(role="user", content="hi")], [])
        self.assertEqual(content, "ok")
        self.assertEqual(calls, [])
        self.assertIsInstance(ToolCall(id="1", name="bash", arguments={}), ToolCall)


if __name__ == "__main__":
    unittest.main()