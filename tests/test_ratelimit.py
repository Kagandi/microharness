import json
import time
import unittest

from core.ratelimit import (
    DELAY_STRATEGIES,
    RateLimitError,
    RateLimitHints,
    RetryPolicy,
    extract_retry_delay,
    format_delay,
    is_unrecoverable_quota_error,
    resolve_rate_limit_delay,
)


class TestErrors(unittest.TestCase):
    def test_rate_limit_error_carries_retry_after(self):
        """Invariant: the retry hint survives on the exception so callers can report it."""
        err = RateLimitError("slow down", retry_after=42.0)
        self.assertEqual(str(err), "slow down")
        self.assertEqual(err.retry_after, 42.0)

    def test_rate_limit_error_defaults_to_no_hint(self):
        """Invariant: a terminal quota failure reports retry_after=None."""
        self.assertIsNone(RateLimitError("quota gone").retry_after)


class TestQuotaClassification(unittest.TestCase):
    def test_payment_required_is_always_unrecoverable(self):
        """Invariant: HTTP 402 never warrants a retry, whatever the body says."""
        self.assertTrue(is_unrecoverable_quota_error(402, ""))

    def test_daily_quota_phrases_are_unrecoverable(self):
        """Invariant: provider-specific daily-limit phrasing is detected across wordings."""
        for body in (
            "Free tier limit exceeded for today",
            "Daily request limit reached",
            "You exceeded your current quota",
            "insufficient_quota",
            "Your credit balance is too low",
        ):
            self.assertTrue(is_unrecoverable_quota_error(429, body), body)

    def test_minute_level_limits_are_recoverable(self):
        """Invariant: a per-minute limit is a normal throttle, not an exhausted quota."""
        self.assertFalse(is_unrecoverable_quota_error(429, "Too many requests per minute"))

    def test_empty_body_is_recoverable(self):
        """Invariant: with no body text there is nothing claiming an unrecoverable limit."""
        self.assertFalse(is_unrecoverable_quota_error(429, ""))


class TestExtractRetryDelay(unittest.TestCase):
    def test_retry_after_header_wins(self):
        """Invariant: an explicit Retry-After header is the most authoritative signal."""
        self.assertEqual(extract_retry_delay("retry in 99s", retry_header="7"), 7.0)

    def test_non_numeric_retry_after_falls_through(self):
        """Invariant: a malformed header value is skipped, not coerced to zero."""
        self.assertIsNone(extract_retry_delay("{}", retry_header="soon"))

    def test_negative_retry_after_is_ignored(self):
        """Invariant: a negative delay would spin the retry loop, so it is rejected."""
        self.assertIsNone(extract_retry_delay("{}", retry_header="-5"))

    def test_reset_header_in_seconds(self):
        """Invariant: an epoch-seconds reset stamp yields the remaining wait."""
        delay = extract_retry_delay("", reset_header=str(time.time() + 30))
        self.assertIsNotNone(delay)
        self.assertAlmostEqual(delay, 30.0, delta=1.5)

    def test_reset_header_in_milliseconds(self):
        """Invariant: millisecond-precision stamps are detected and normalized to seconds."""
        delay = extract_retry_delay("", reset_header=str((time.time() + 20) * 1000))
        self.assertIsNotNone(delay)
        self.assertAlmostEqual(delay, 20.0, delta=1.5)

    def test_reset_header_in_the_past_is_rejected(self):
        """Invariant: a reset stamp already elapsed yields no delay."""
        self.assertIsNone(extract_retry_delay("", reset_header=str(time.time() - 60)))

    def test_google_retry_info_details(self):
        """Invariant: Google RPC RetryInfo detail blocks are parsed out of a JSON list body."""
        body = json.dumps([
            {"error": {"details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": "55s"}
            ]}}
        ])
        self.assertEqual(extract_retry_delay(body), 55.0)

    def test_openrouter_metadata_reset_header(self):
        """Invariant: gateways echoing retry headers inside metadata are understood."""
        stamp = (time.time() + 12) * 1000
        body = json.dumps({"metadata": {"headers": {"X-RateLimit-Reset": str(stamp)}}})
        delay = extract_retry_delay(body)
        self.assertIsNotNone(delay)
        self.assertAlmostEqual(delay, 12.0, delta=1.5)

    def test_explicit_retry_after_field(self):
        """Invariant: a structured retry_after field is honoured."""
        self.assertEqual(extract_retry_delay(json.dumps({"error": {"retry_after": 8}})), 8.0)

    def test_regex_fallback_on_prose(self):
        """Invariant: a human-readable sentence is the last resort signal."""
        self.assertEqual(extract_retry_delay("Quota exceeded. Please retry in 55.1s."), 55.1)
        self.assertEqual(extract_retry_delay("Resets in 5 seconds."), 5.0)

    def test_unusable_body_returns_none(self):
        """Invariant: no signal at all means the caller must choose its own backoff."""
        self.assertIsNone(extract_retry_delay("something went wrong"))
        self.assertIsNone(extract_retry_delay(""))

    def test_strategies_are_ordered_most_authoritative_first(self):
        """Invariant: the retry-after header is consulted before any body-derived guess."""
        self.assertEqual(len(DELAY_STRATEGIES), 4)
        body = json.dumps({"error": {"retry_after": 8}})
        self.assertEqual(extract_retry_delay(body, retry_header="3"), 3.0)


class TestResolveRateLimitDelay(unittest.TestCase):
    def setUp(self):
        self.policy = RetryPolicy()

    def test_header_delay_is_used_verbatim(self):
        """Invariant: a server-sent header is trusted exactly, with no safety padding."""
        hints = RateLimitHints(body="", retry_after="5")
        self.assertEqual(resolve_rate_limit_delay(hints, 0, self.policy), 5.0)

    def test_body_delay_gets_a_one_second_buffer(self):
        """Invariant: an inferred delay is padded so the client does not race the server."""
        hints = RateLimitHints(body="retry in 55s")
        self.assertEqual(resolve_rate_limit_delay(hints, 0, self.policy), 56.0)

    def test_no_signal_falls_back_to_exponential_backoff(self):
        """Invariant: with nothing to parse, the attempt number drives the wait."""
        hints = RateLimitHints(body="too many requests")
        self.assertEqual(resolve_rate_limit_delay(hints, 2, self.policy), 8.0)


class TestRetryPolicy(unittest.TestCase):
    def test_backoff_doubles_and_caps(self):
        """Invariant: delays grow exponentially but never exceed max_delay."""
        policy = RetryPolicy(base_delay=2.0, max_delay=60.0)
        self.assertEqual([policy.backoff(a) for a in range(8)],
                         [2.0, 4.0, 8.0, 16.0, 32.0, 60.0, 60.0, 60.0])

    def test_can_retry_is_false_on_the_final_attempt(self):
        """Invariant: the last attempt never schedules another one."""
        policy = RetryPolicy(max_attempts=6)
        self.assertTrue(policy.can_retry(0))
        self.assertTrue(policy.can_retry(4))
        self.assertFalse(policy.can_retry(5))

    def test_single_attempt_policy_never_retries(self):
        """Invariant: max_attempts=1 disables retrying entirely."""
        self.assertFalse(RetryPolicy(max_attempts=1).can_retry(0))


class TestFormatDelay(unittest.TestCase):
    def test_scales_units_with_magnitude(self):
        """Invariant: operator-facing durations pick a readable unit at each magnitude."""
        self.assertEqual(format_delay(5.0), "5.0s")
        self.assertEqual(format_delay(3600.0), "1.0 hours")
        self.assertEqual(format_delay(86400.0), "1.0d (24.0 hours)")
        self.assertEqual(format_delay(3600000.0), "~1.0k hours")
        self.assertEqual(format_delay(9820956445.2), "~2.7M hours")


if __name__ == "__main__":
    unittest.main()