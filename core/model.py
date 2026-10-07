from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from core import config
from core.auth import get_signer
from core.protocols import (
    CacheAwareModel,
    CancellableModel,
    DeadlineAwareModel,
    EndpointAwareModel,
    Model,
    ModelCancelledError,
    NamedModel,
    ProviderAwareModel,
    RateLimitAwareModel,
    ReasoningAwareModel,
    ResettableModel,
    UsageReportingModel,
)
from core.providers import (
    build_model_from_spec,
    find_provider_by_endpoint,
    get_available_models_from_config,
    load_builtin_providers,
    load_provider_registry,
    resolve_provider,
    seed_user_providers,
)
from core.types import Message, ToolCall
from core.usage import UsageTelemetry


class RateLimitError(RuntimeError):
    """Raised when a provider returns HTTP 429, or 402 when the account is out of credit."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


# 402 means "pay before we serve you", so it never warrants a retry. 429 is a
# throttle: worth waiting out, using the delay the server asked for when it sent one.
RATE_LIMIT_STATUS = 429
PAYMENT_REQUIRED_STATUS = 402
TRANSIENT_STATUS: frozenset[int] = frozenset({500, 502, 503, 504})


@dataclass(frozen=True)
class RetryPolicy:
    """How many attempts to make and how long to wait between them."""

    max_attempts: int = 6
    base_delay: float = 2.0
    max_delay: float = 60.0

    def backoff(self, attempt: int) -> float:
        """Exponential backoff for ``attempt`` (0-based), capped at ``max_delay``."""
        return min(self.base_delay * (2**attempt), self.max_delay)

    def can_retry(self, attempt: int) -> bool:
        return attempt < self.max_attempts - 1


def extract_retry_delay(headers: Any) -> float | None:
    """Reads how long a provider asked us to wait, from its standard headers.

    Honours ``Retry-After`` (seconds) and ``X-RateLimit-Reset`` (an epoch stamp,
    in seconds or milliseconds). Returns ``None`` when the provider sent neither,
    in which case the caller falls back to its own backoff. Error bodies are not
    parsed: a provider that only reports its delay there still recovers, just with
    backoff rather than the exact figure.
    """
    if not headers:
        return None
    try:
        retry_after = float(headers.get("Retry-After"))
        if retry_after >= 0:
            return retry_after
    except (TypeError, ValueError):
        pass

    reset = headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset")
    try:
        stamp = float(reset)
    except (TypeError, ValueError):
        return None
    if stamp > 1e11:  # milliseconds, not seconds
        stamp /= 1000.0
    return max(0.0, stamp - time.time())


__all__ = [
    "CacheAwareModel",
    "CancellableModel",
    "DeadlineAwareModel",
    "EndpointAwareModel",
    "HttpChatModel",
    "MockModel",
    "Model",
    "ModelCancelledError",
    "NamedModel",
    "ProviderAwareModel",
    "RateLimitAwareModel",
    "RateLimitError",
    "ReasoningAwareModel",
    "ResettableModel",
    "UsageReportingModel",
    "RetryPolicy",
    "build_model_from_spec",
    "extract_retry_delay",
    "find_provider_by_endpoint",
    "get_available_models_from_config",
    "load_builtin_providers",
    "load_provider_registry",
    "normalize_cache_ttl",
    "normalize_reasoning_effort",
    "resolve_provider",
    "seed_user_providers",
    "strip_json_comments",
]



# Cache TTL and reasoning effort normalization live in core/config.py so the
# harness cascade and the model clients share one validation path. Re-exported
# here because they are part of the model-facing surface.
normalize_cache_ttl = config.normalize_cache_ttl
normalize_reasoning_effort = config.normalize_reasoning_effort
strip_json_comments = config.strip_json_comments


class HttpChatModel(UsageTelemetry):
    """A generic HTTP model client for any OpenAI-compatible completions endpoint.

    Accepts either:
    - A base URL (e.g. 'https://openrouter.ai/api/v1' or 'http://localhost:11434/v1')
    - A full endpoint URL (e.g. 'https://openrouter.ai/api/v1/chat/completions')

    Works identically with OpenRouter, Ollama, vLLM, LM Studio, etc.
    Supports on-the-fly prompt caching TTL control ('5m' standard ephemeral or '1h' extended beta).
    """

    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str | None = None,
        headers: dict[str, str] | None = None,
        timeout: float = 90.0,
        cache_ttl: str | None = None,
        reasoning_effort: str | None = None,
        reasoning_format: str | None = None,
        cache_strategy: str | None = None,
        deadline: float | None = None,
        provider: str | None = None,
        retry: RetryPolicy | None = None,
        auth_scheme: str = "bearer",
        auth_options: dict[str, Any] | None = None,
        auth_secrets: dict[str, Any] | None = None,
    ):
        clean_url = base_url.rstrip("/")
        # Automatically handle both base URL and full /chat/completions URL
        if not clean_url.endswith("/chat/completions"):
            self.endpoint = f"{clean_url}/chat/completions"
        else:
            self.endpoint = clean_url

        # Declarative defaults from registry if capability flags are not explicitly passed
        prov_name, profile = find_provider_by_endpoint(load_builtin_providers(), clean_url)
        default_headers = dict(profile.get("headers") or {})
        default_reasoning_format = profile.get("reasoning_format", "standard")
        default_cache_strategy = profile.get("cache_strategy", "implicit")

        self.provider = provider or prov_name
        self.model = model
        self.api_key = api_key
        self.auth_scheme = (auth_scheme or "bearer").lower()
        self.auth_options = dict(auth_options or {})
        self.auth_secrets = dict(auth_secrets or {})
        # Merge declarative profile headers with explicitly passed headers (explicit wins)
        self.extra_headers = {**default_headers, **(headers or {})}
        self.timeout = timeout
        self.deadline = deadline
        self.retry = retry or RetryPolicy()
        self.total_rate_limit_wait_s: float = 0.0
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.reasoning_format = (reasoning_format or default_reasoning_format).lower()
        self.cache_strategy = (cache_strategy or default_cache_strategy).lower()
        self._init_cache_telemetry(cache_ttl)
        self._cancel_probe: Callable[[], bool] | None = None

    def set_deadline(self, deadline: float | None) -> None:
        """Sets the task deadline timestamp to strictly bound in-flight HTTP requests."""
        self.deadline = deadline

    def set_cancel_probe(self, probe: Callable[[], bool] | None) -> None:
        """Installs a non-blocking predicate consulted while waiting between retry attempts."""
        self._cancel_probe = probe

    def _cancelled(self) -> bool:
        probe = self._cancel_probe
        if probe is None:
            return False
        try:
            return bool(probe())
        except Exception:
            # A misbehaving probe must never break the request path.
            return False

    def _wait_or_cancel(self, seconds: float) -> None:
        """Sleeps, aborting early if a cancel probe fires.

        With no probe installed (batch and benchmark runs) this is a plain
        ``time.sleep`` so the retry path keeps its original cost and timing.
        Only interactive sessions slice the wait, which is what makes a cancel
        during a backoff responsive.

        This governs time spent *waiting* between attempts. It cannot interrupt a
        socket already blocked in ``urlopen``; that limit is documented in
        SPEC.md section 5.
        """
        if self._cancel_probe is None:
            if seconds > 0:
                time.sleep(seconds)
            return
        if seconds <= 0:
            if self._cancelled():
                raise ModelCancelledError("Cancelled by user while waiting to retry.")
            return
        deadline_at = time.monotonic() + seconds
        while True:
            if self._cancelled():
                raise ModelCancelledError("Cancelled by user while waiting to retry.")
            remaining = deadline_at - time.monotonic()
            if remaining <= 0:
                return
            time.sleep(min(0.1, remaining))

    def set_reasoning_effort(self, effort: str | None) -> str | None:
        """Dynamically switches reasoning effort level ('low', 'medium', 'high', numeric budget, or 'off'/None)."""
        self.reasoning_effort = normalize_reasoning_effort(effort)
        return self.reasoning_effort

    def get_reasoning_effort(self) -> str | None:
        """Returns the current reasoning effort level or None."""
        return self.reasoning_effort

    def _build_headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if self.auth_scheme == "bearer" and self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        if self.cache_ttl:
            if self.cache_strategy == "anthropic_block":
                if self.cache_ttl == "1h":
                    headers.setdefault("anthropic-beta", "extended-cache-ttl-2025-04-11")
            if self.reasoning_format == "openrouter":
                if self.cache_ttl == "1h":
                    headers.setdefault("X-OpenRouter-Cache-TTL", "3600")
                elif self.cache_ttl == "5m":
                    headers.setdefault("X-OpenRouter-Cache-TTL", "300")
        return headers

    def _build_payload(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> dict[str, Any]:
        formatted_messages: list[dict[str, Any]] = []
        system_cached = False
        supports_cache_ctrl = self.cache_strategy == "anthropic_block"
        for msg in messages:
            msg_dict = msg.to_dict()
            if self.cache_ttl and supports_cache_ctrl and msg.role == "system" and not system_cached:
                cache_ctrl: dict[str, Any] = {"type": "ephemeral"}
                if self.cache_ttl == "1h":
                    cache_ctrl["ttl"] = "1h"
                msg_dict["content"] = [
                    {
                        "type": "text",
                        "text": msg.content or "",
                        "cache_control": cache_ctrl,
                    }
                ]
                system_cached = True
            formatted_messages.append(msg_dict)

        payload: dict[str, Any] = {
            "model": self.model,
            "messages": formatted_messages,
        }
        if self.reasoning_effort and self.reasoning_format != "none":
            effort = self.reasoning_effort.lower()
            if self.reasoning_format == "openrouter":
                if effort in ("low", "medium", "high", "xhigh", "max"):
                    payload["reasoning_effort"] = effort
                    payload["reasoning"] = {"effort": effort}
                elif effort.isdigit():
                    payload["reasoning"] = {"max_tokens": int(effort)}
            else:
                standard_effort = "high" if effort in ("xhigh", "max") else effort
                if standard_effort in ("low", "medium", "high"):
                    payload["reasoning_effort"] = standard_effort

        if tools:
            if self.cache_ttl and supports_cache_ctrl:
                cache_ctrl = {"type": "ephemeral"}
                if self.cache_ttl == "1h":
                    cache_ctrl["ttl"] = "1h"
                tools_payload = [dict(t) for t in tools]
                tools_payload[-1] = {**tools_payload[-1], "cache_control": cache_ctrl}
                payload["tools"] = tools_payload
            else:
                payload["tools"] = tools
            payload["tool_choice"] = "auto"
        return payload

    def _secrets_for_attempt(self) -> dict[str, Any]:
        """Merges stored auth secrets with the legacy api_key, without provider branches."""
        secrets = dict(self.auth_secrets)
        if "api_key" not in secrets and self.api_key:
            secrets["api_key"] = self.api_key
        return secrets

    def _send_request(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        """Posts the payload, retrying transient failures under :attr:`retry`.

        Each failure class gets its own handler; a handler either schedules the
        next attempt (and returns) or raises a terminal error. Headers are
        rebuilt and re-signed on every attempt so timestamped schemes stay
        fresh across retries.
        """
        body_bytes = json.dumps(payload).encode("utf-8")
        for attempt in range(self.retry.max_attempts):
            # Honour a pending cancel before spending another request on it.
            self._raise_if_cancelled("before retry attempt")
            base_headers = self._build_headers()
            if headers:
                base_headers.update(headers)
            secrets = self._secrets_for_attempt()
            if self.auth_scheme == "bearer" and not secrets.get("api_key"):
                signed_headers = base_headers
            else:
                signed_headers = get_signer(self.auth_scheme).sign(
                    method="POST",
                    url=self.endpoint,
                    headers=base_headers,
                    body=body_bytes,
                    options=self.auth_options,
                    secrets=secrets,
                )
            request = urllib.request.Request(
                self.endpoint,
                data=body_bytes,
                headers=signed_headers,
                method="POST",
            )
            try:
                with urllib.request.urlopen(request, timeout=self._request_timeout(attempt)) as resp:
                    return json.loads(resp.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                self._handle_http_error(exc, attempt)
            except TimeoutError:
                self._handle_timeout(attempt)
            except urllib.error.URLError as exc:
                self._handle_url_error(exc, attempt)
        raise RuntimeError(f"Model request to {self.endpoint} exhausted all attempts.")

    def _request_timeout(self, attempt: int) -> float:
        """Bounds the socket timeout by whatever is left of the task deadline."""
        if self.deadline is None:
            return self.timeout
        remaining = self.deadline - time.time()
        if remaining <= 0:
            raise TimeoutError(
                f"Task timeout reached before model request attempt {attempt + 1}"
            )
        return min(self.timeout, max(remaining, 0.1))

    def _handle_http_error(self, exc: urllib.error.HTTPError, attempt: int) -> None:
        """Routes an HTTP error: throttling first, then transient server errors."""
        body = exc.read().decode("utf-8", errors="replace")
        if exc.code in (RATE_LIMIT_STATUS, PAYMENT_REQUIRED_STATUS):
            self._wait_out_rate_limit(exc, body, attempt)
            return
        if exc.code in TRANSIENT_STATUS and self.retry.can_retry(attempt):
            self._sleep_before_retry(self.retry.backoff(attempt), f" HTTP {exc.code}")
            return
        raise RuntimeError(
            f"Model API returned HTTP {exc.code} from {self.endpoint}:\n{body}"
        ) from exc

    def _wait_out_rate_limit(
        self, exc: urllib.error.HTTPError, body: str, attempt: int
    ) -> None:
        """Waits out a 429, or fails immediately on 402 (no credit, no point retrying)."""
        delay = extract_retry_delay(exc.headers)
        if exc.code == PAYMENT_REQUIRED_STATUS:
            raise RateLimitError(
                f"Model API rejected the request for payment (HTTP 402) from {self.endpoint}:\n{body}",
                retry_after=delay,
            ) from exc

        if delay is None:
            delay = self.retry.backoff(attempt)
        # Time spent honouring a server-requested pause is not charged to the task budget.
        self.total_rate_limit_wait_s += delay
        if self.deadline is not None:
            self.deadline += delay

        if not self.retry.can_retry(attempt):
            raise RateLimitError(
                f"Model API rate limit (HTTP 429) exceeded after "
                f"{self.retry.max_attempts} retries from {self.endpoint}:\n{body}",
                retry_after=delay,
            ) from exc
        print(
            f"\n⚠️ [Rate Limit 429]: Waiting {delay:.1f}s before retry "
            f"({attempt + 1}/{self.retry.max_attempts})..."
        )
        self._wait_or_cancel(delay)

    def _handle_timeout(self, attempt: int) -> None:
        """Retries a timed-out request unless the task deadline has already passed."""
        if self.deadline is not None and time.time() >= self.deadline:
            raise TimeoutError("Task timeout reached during model request")
        if not self.retry.can_retry(attempt):
            raise TimeoutError("Model request timed out")
        self._sleep_before_retry(self.retry.backoff(attempt), " timed-out request")

    def _handle_url_error(self, exc: urllib.error.URLError, attempt: int) -> None:
        """Retries connection failures, surfacing a friendlier error once attempts run out."""
        reason = exc.reason
        timed_out = isinstance(reason, TimeoutError) or str(reason).lower().startswith("timed out")
        if timed_out and self.deadline is not None and time.time() >= self.deadline:
            raise TimeoutError("Task timeout reached during model request") from exc
        if self.retry.can_retry(attempt):
            self._sleep_before_retry(
                self.retry.backoff(attempt), f" connection error: {reason}"
            )
            return
        raise ConnectionError(
            f"Could not connect to model endpoint at {self.endpoint}.\n"
            f"Is the server running and reachable? (Detail: {reason})"
        ) from exc

    def _sleep_before_retry(self, seconds: float, label: str) -> None:
        """Waits before the next attempt, refusing to wait past the task deadline."""
        if self.deadline is not None and (time.time() + seconds) >= self.deadline:
            raise TimeoutError(f"Task timeout reached while waiting to retry{label}")
        self._wait_or_cancel(seconds)

    def _raise_if_cancelled(self, context: str) -> None:
        if self._cancelled():
            raise ModelCancelledError(f"Cancelled by user {context}.")

    def _parse_tool_calls(self, raw_tool_calls: list[dict[str, Any]] | None) -> list[ToolCall]:
        if not raw_tool_calls:
            return []
        tool_calls: list[ToolCall] = []
        for raw_tc in raw_tool_calls:
            func_data = raw_tc["function"]
            args_raw = func_data.get("arguments", "{}")
            if isinstance(args_raw, str):
                try:
                    parsed_args = json.loads(args_raw)
                except json.JSONDecodeError:
                    parsed_args = {"_raw_arguments": args_raw}
            else:
                parsed_args = args_raw or {}

            extra_content = raw_tc.get("extra_content")
            tool_calls.append(
                ToolCall(
                    id=raw_tc["id"],
                    name=func_data["name"],
                    arguments=parsed_args,
                    extra_content=extra_content if isinstance(extra_content, dict) else None,
                )
            )
        return tool_calls

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]:
        headers = self._build_headers()
        payload = self._build_payload(messages, tools)
        res_body = self._send_request(payload, headers)

        self._update_usage(res_body.get("usage", {}))

        choice = res_body["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls = self._parse_tool_calls(choice.get("tool_calls"))

        return content, tool_calls


class MockModel(UsageTelemetry):
    """Deterministic model for offline testing and educational walkthroughs."""

    def __init__(
        self,
        script: list[tuple[str, list[ToolCall]]],
        cache_ttl: str | None = None,
        reasoning_effort: str | None = None,
    ):
        self.model = "mock"
        self.provider = "mock"
        self.script = list(script)
        self.step_idx = 0
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self._init_cache_telemetry(cache_ttl)

    def set_reasoning_effort(self, effort: str | None) -> str | None:
        """Dynamically switches reasoning effort level on MockModel."""
        self.reasoning_effort = normalize_reasoning_effort(effort)
        return self.reasoning_effort

    def get_reasoning_effort(self) -> str | None:
        """Returns current reasoning effort level on MockModel."""
        return self.reasoning_effort

    def reset(self) -> None:
        """Resets the script execution index for subsequent task runs."""
        self.step_idx = 0

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]:
        if self.step_idx < len(self.script):
            result = self.script[self.step_idx]
            self.step_idx += 1
            return result
        return "I have completed the task and verified all results.", []
