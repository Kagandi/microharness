import unittest

from core.usage import TokenUsage, UsageTelemetry, normalize_usage, sum_usage, summarize_run_usage


class TestNormalizeUsage(unittest.TestCase):
    def test_openai_shape(self):
        """Invariant: the OpenAI dialect maps prompt/completion tokens straight through."""
        usage = normalize_usage({"prompt_tokens": 100, "completion_tokens": 20})
        self.assertEqual(
            usage, TokenUsage(prompt_tokens=100, completion_tokens=20, cached_tokens=0)
        )

    def test_anthropic_shape(self):
        """Invariant: Anthropic's input/output naming is recognized."""
        usage = normalize_usage({"input_tokens": 30, "output_tokens": 7})
        self.assertEqual(usage.prompt_tokens, 30)
        self.assertEqual(usage.completion_tokens, 7)

    def test_google_shape(self):
        """Invariant: Google's camelCase promptTokenCount/candidatesTokenCount are recognized."""
        usage = normalize_usage({"promptTokenCount": 12, "candidatesTokenCount": 4})
        self.assertEqual(usage.prompt_tokens, 12)
        self.assertEqual(usage.completion_tokens, 4)

    def test_cache_reads_from_nested_details(self):
        """Invariant: nested prompt_tokens_details.cached_tokens takes precedence."""
        usage = normalize_usage(
            {"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 40}}
        )
        self.assertEqual(usage.cached_tokens, 40)
        self.assertEqual(usage.cache_hit_rate_pct, 40.0)

    def test_cache_reads_from_flat_keys(self):
        """Invariant: providers that report cache reads flat are also understood."""
        self.assertEqual(normalize_usage({"cache_read_input_tokens": 25}).cached_tokens, 25)
        self.assertEqual(normalize_usage({"cachedContentTokenCount": 9}).cached_tokens, 9)

    def test_missing_payload_is_all_zeroes(self):
        """Invariant: absent or malformed usage must never raise during telemetry."""
        self.assertEqual(normalize_usage(None), TokenUsage())
        self.assertEqual(normalize_usage({}), TokenUsage())
        self.assertEqual(normalize_usage("not a dict"), TokenUsage())

    def test_non_numeric_values_are_ignored(self):
        """Invariant: a garbled field counts as zero instead of crashing the run."""
        usage = normalize_usage({"prompt_tokens": "many", "completion_tokens": None})
        self.assertEqual(usage.prompt_tokens, 0)
        self.assertEqual(usage.completion_tokens, 0)

    def test_total_prefixed_keys_are_understood(self):
        """Invariant: model-side cumulative counters feed back through the same parser."""
        usage = normalize_usage({"total_prompt_tokens": 500, "total_cached_tokens": 100})
        self.assertEqual(usage.prompt_tokens, 500)
        self.assertEqual(usage.cached_tokens, 100)

    def test_hit_rate_is_zero_without_prompt_tokens(self):
        """Invariant: no prompt tokens means no meaningful hit rate, not a division error."""
        self.assertEqual(TokenUsage(cached_tokens=50).cache_hit_rate_pct, 0.0)


class TestAggregation(unittest.TestCase):
    def test_sum_usage_adds_components(self):
        """Invariant: per-step payloads add up field-wise."""
        total = sum_usage(
            [
                {"prompt_tokens": 10, "completion_tokens": 2},
                {"input_tokens": 5, "output_tokens": 1},
            ]
        )
        self.assertEqual(total, TokenUsage(prompt_tokens=15, completion_tokens=3))

    def test_run_summary_prefers_step_level_usage(self):
        """Invariant: step-level counts win, since the loop records them per step."""
        summary = summarize_run_usage(
            [{"prompt_tokens": 30, "completion_tokens": 4}],
            {"total_prompt_tokens": 999, "total_completion_tokens": 999},
        )
        self.assertEqual(summary.prompt_tokens, 30)
        self.assertEqual(summary.completion_tokens, 4)

    def test_run_summary_falls_back_to_model_counters(self):
        """Invariant: providers without per-step usage still report totals."""
        summary = summarize_run_usage([], {"total_prompt_tokens": 120, "total_cached_tokens": 60})
        self.assertEqual(summary.prompt_tokens, 120)
        self.assertEqual(summary.cached_tokens, 60)
        self.assertEqual(summary.cache_hit_rate_pct, 50.0)

    def test_run_summary_of_empty_run_is_zero(self):
        """Invariant: a run with no steps and no counters reports zeros, not None."""
        summary = summarize_run_usage([], {})
        self.assertEqual(summary, TokenUsage())


class TelemetryModel(UsageTelemetry):
    def __init__(self, cache_ttl=None):
        self._init_cache_telemetry(cache_ttl)


class TestUsageTelemetry(unittest.TestCase):
    def test_cache_ttl_is_normalized_on_init(self):
        """Invariant: the mixin validates TTL exactly like the harness cascade does."""
        self.assertEqual(TelemetryModel(cache_ttl="300").cache_ttl, "5m")
        self.assertIsNone(TelemetryModel(cache_ttl="off").cache_ttl)

    def test_set_cache_ttl_returns_the_effective_value(self):
        """Invariant: callers learn what the TTL actually became."""
        model = TelemetryModel(cache_ttl="5m")
        self.assertEqual(model.set_cache_ttl("1h"), "1h")
        self.assertEqual(model.cache_ttl, "1h")

    def test_totals_accumulate_across_steps(self):
        """Invariant: each recorded step adds to the cumulative counters."""
        model = TelemetryModel()
        model._update_usage({"prompt_tokens": 10, "completion_tokens": 2})
        model._update_usage({"prompt_tokens": 5, "completion_tokens": 1})
        stats = model.get_cache_stats()
        self.assertEqual(stats["total_prompt_tokens"], 15)
        self.assertEqual(stats["total_completion_tokens"], 3)

    def test_last_step_usage_reflects_only_the_latest_step(self):
        """Invariant: per-step reporting is not polluted by cumulative totals."""
        model = TelemetryModel()
        model._update_usage({"prompt_tokens": 10})
        model._update_usage({"prompt_tokens": 3, "completion_tokens": 1})
        self.assertEqual(
            model.get_last_step_usage(),
            {
                "prompt_tokens": 3,
                "completion_tokens": 1,
                "cached_tokens": 0,
                "cache_hit_rate_pct": 0.0,
            },
        )

    def test_reset_clears_counters_but_keeps_ttl(self):
        """Invariant: reset prepares the next task without discarding configuration."""
        model = TelemetryModel(cache_ttl="1h")
        model._update_usage({"prompt_tokens": 42})
        model.reset()
        stats = model.get_cache_stats()
        self.assertEqual(stats["total_prompt_tokens"], 0)
        self.assertEqual(stats["last_usage"], {})
        self.assertEqual(stats["cache_ttl"], "1h")

    def test_cache_hit_rate_uses_cumulative_totals(self):
        """Invariant: the reported rate is computed over all tokens seen so far."""
        model = TelemetryModel()
        model._update_usage({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 30}})
        model._update_usage({"prompt_tokens": 100, "prompt_tokens_details": {"cached_tokens": 30}})
        self.assertEqual(model.get_cache_stats()["cache_hit_rate_pct"], 30.0)


if __name__ == "__main__":
    unittest.main()
