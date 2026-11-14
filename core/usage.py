"""Token accounting: one normalizer, one telemetry mixin, one run-level summary.

Providers disagree about token field names (``input_tokens`` vs ``prompt_tokens``
vs ``promptTokenCount``) and about where cache reads are reported. All of that
knowledge lives in :func:`normalize_usage`, so the per-step mixin and the
per-run summary can never drift apart.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

from core.config import normalize_cache_ttl

PROMPT_TOKEN_KEYS: tuple[str, ...] = (
    "total_prompt_tokens",
    "prompt_tokens",
    "input_tokens",
    "promptTokenCount",
)
COMPLETION_TOKEN_KEYS: tuple[str, ...] = (
    "total_completion_tokens",
    "completion_tokens",
    "output_tokens",
    "candidatesTokenCount",
)
CACHED_TOKEN_KEYS: tuple[str, ...] = (
    "total_cached_tokens",
    "cache_read_input_tokens",
    "cached_tokens",
    "cachedContentTokenCount",
    "native_tokens_prompt_cache_read",
)


@dataclass(frozen=True)
class TokenUsage:
    """Normalized token counts for one generation step or one whole run."""

    prompt_tokens: int = 0
    completion_tokens: int = 0
    cached_tokens: int = 0

    @property
    def cache_hit_rate_pct(self) -> float:
        return (
            round(self.cached_tokens / self.prompt_tokens * 100, 1) if self.prompt_tokens else 0.0
        )

    def __add__(self, other: "TokenUsage") -> "TokenUsage":
        return TokenUsage(
            self.prompt_tokens + other.prompt_tokens,
            self.completion_tokens + other.completion_tokens,
            self.cached_tokens + other.cached_tokens,
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cached_tokens": self.cached_tokens,
            "cache_hit_rate_pct": self.cache_hit_rate_pct,
        }


def _first_int(usage: dict[str, Any], keys: Iterable[str]) -> int:
    for key in keys:
        try:
            value = int(usage.get(key) or 0)
        except (TypeError, ValueError):
            continue
        if value:
            return value
    return 0


def normalize_usage(usage: dict[str, Any] | None) -> TokenUsage:
    """Reads a provider usage payload into normalized token counts.

    Missing or malformed fields count as zero rather than raising: telemetry
    must never be the reason a run fails.
    """
    if not isinstance(usage, dict):
        return TokenUsage()

    details = usage.get("prompt_tokens_details")
    nested_cached = _first_int(details, ("cached_tokens",)) if isinstance(details, dict) else 0
    cached = nested_cached or _first_int(usage, CACHED_TOKEN_KEYS)
    return TokenUsage(
        prompt_tokens=_first_int(usage, PROMPT_TOKEN_KEYS),
        completion_tokens=_first_int(usage, COMPLETION_TOKEN_KEYS),
        cached_tokens=cached,
    )


def sum_usage(payloads: Iterable[dict[str, Any] | None]) -> TokenUsage:
    """Adds up many usage payloads (e.g. one per step of a run)."""
    total = TokenUsage()
    for payload in payloads:
        total = total + normalize_usage(payload)
    return total


def summarize_run_usage(
    step_usages: Iterable[dict[str, Any] | None],
    cache_stats: dict[str, Any] | None = None,
) -> TokenUsage:
    """Derives run-level token totals for a :class:`~core.types.RunResult`.

    Step-level usage is authoritative because it is recorded by the agent loop.
    Providers that only expose cumulative counters fall back to the model-side
    totals when no step reported any tokens.
    """
    from_steps = sum_usage(step_usages)
    if from_steps.prompt_tokens > 0:
        return from_steps
    return normalize_usage(cache_stats)


class UsageTelemetry:
    """Cumulative prompt-cache telemetry and token accounting for a model client.

    Mixed into model implementations that report usage. Provides the
    ``CacheAwareModel`` and ``UsageReportingModel`` surfaces.
    """

    cache_ttl: str | None
    last_usage: dict[str, Any]
    total_prompt_tokens: int
    total_completion_tokens: int
    total_cached_tokens: int

    def _init_cache_telemetry(self, cache_ttl: str | None = None) -> None:
        self.cache_ttl = normalize_cache_ttl(cache_ttl)
        self.last_usage = {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0

    def set_cache_ttl(self, ttl: str | None) -> str | None:
        """Dynamically switches prompt caching duration on the fly ('5m', '1h', or 'off'/None)."""
        self.cache_ttl = normalize_cache_ttl(ttl)
        return self.cache_ttl

    def get_last_step_usage(self) -> dict[str, Any]:
        """Returns normalized token counts and cache hit percentage for the most recent step."""
        return normalize_usage(self.last_usage).as_dict()

    def get_cache_stats(self) -> dict[str, Any]:
        """Returns prompt cache TTL and cumulative token utilization metrics."""
        totals = TokenUsage(
            self.total_prompt_tokens,
            self.total_completion_tokens,
            self.total_cached_tokens,
        )
        return {
            "cache_ttl": self.cache_ttl,
            "last_usage": dict(self.last_usage),
            "total_prompt_tokens": totals.prompt_tokens,
            "total_completion_tokens": totals.completion_tokens,
            "total_cached_tokens": totals.cached_tokens,
            "cache_hit_rate_pct": totals.cache_hit_rate_pct,
        }

    def reset(self) -> None:
        """Resets per-task token telemetry and usage counters."""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0
        self.last_usage = {}

    def _update_usage(self, usage: dict[str, Any]) -> None:
        """Records one provider usage payload and folds it into the cumulative totals."""
        self.last_usage = usage or {}
        step = normalize_usage(self.last_usage)
        self.total_prompt_tokens += step.prompt_tokens
        self.total_completion_tokens += step.completion_tokens
        self.total_cached_tokens += step.cached_tokens
