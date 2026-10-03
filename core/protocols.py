"""Capability protocols that model implementations may opt into.

The harness never requires a concrete model class: it checks ``isinstance``
against these runtime-checkable protocols and falls back to a capability-free
path when an implementation does not advertise one. Adding a new capability
therefore means adding one small protocol here, never editing the harness.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, Protocol, runtime_checkable

from core.types import Message, ToolCall


@runtime_checkable
class Model(Protocol):
    """The one method any pluggable LLM caller must provide."""

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]: ...


@runtime_checkable
class ResettableModel(Model, Protocol):
    """Supports resetting per-task state between benchmark tasks or test runs."""

    def reset(self) -> None: ...


@runtime_checkable
class CacheAwareModel(Model, Protocol):
    """Supports dynamic prompt cache TTL control and token telemetry."""

    def set_cache_ttl(self, ttl: str | None) -> str | None: ...

    def get_cache_stats(self) -> dict[str, Any]: ...


@runtime_checkable
class ReasoningAwareModel(Model, Protocol):
    """Supports dynamic reasoning / thinking effort control."""

    def set_reasoning_effort(self, effort: str | None) -> str | None: ...

    def get_reasoning_effort(self) -> str | None: ...


@runtime_checkable
class DeadlineAwareModel(Model, Protocol):
    """Accepts a task deadline to bound in-flight HTTP requests."""

    def set_deadline(self, deadline: float | None) -> None: ...


@runtime_checkable
class CancellableModel(Model, Protocol):
    """Can abandon a pending wait when the user cancels.

    The probe is invoked from the model's own retry/backoff path, so it must be
    cheap and must not block. ``None`` disables probing entirely, which is the
    default outside interactive sessions.
    """

    def set_cancel_probe(self, probe: Callable[[], bool] | None) -> None: ...


@runtime_checkable
class RateLimitAwareModel(Model, Protocol):
    """Tracks cumulative rate-limit throttling duration."""

    total_rate_limit_wait_s: float


@runtime_checkable
class UsageReportingModel(Model, Protocol):
    """Reports normalized token counts and cache hit rates per generation step."""

    def get_last_step_usage(self) -> dict[str, Any]: ...


@runtime_checkable
class NamedModel(Model, Protocol):
    """Exposes its model name identifier."""

    model: str


@runtime_checkable
class EndpointAwareModel(Model, Protocol):
    """Exposes its HTTP endpoint URL."""

    endpoint: str


@runtime_checkable
class ProviderAwareModel(Model, Protocol):
    """Exposes its underlying provider name."""

    provider: str