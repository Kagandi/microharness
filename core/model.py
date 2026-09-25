from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from typing import Any, Protocol

from core.types import Message, ToolCall


class Model(Protocol):
    """Protocol for pluggable LLM callers.

    Any class or adapter with a `generate` method can be plugged directly
    into the Harness without modifying any harness code.
    """

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]: ...


def normalize_cache_ttl(ttl: str | None) -> str | None:
    """Normalizes cache TTL setting ('5m', '1h', or None for disabled).

    Supported representations:
    - '5m', '5min', '5mins', '300', '300s' -> '5m' (standard ephemeral cache)
    - '1h', '1hour', '1hr', '60m', '3600', '3600s' -> '1h' (extended beta cache)
    - 'off', 'none', 'disable', 'disabled', 'false', '0', '', None -> None
    """
    if ttl is None:
        return None
    val = str(ttl).strip().lower()
    if val in ("off", "none", "disable", "disabled", "false", "0", ""):
        return None
    if val in ("5m", "5min", "5mins", "300", "300s"):
        return "5m"
    if val in ("1h", "1hour", "1hr", "60m", "3600", "3600s"):
        return "1h"
    raise ValueError(
        f"Unsupported cache TTL: '{ttl}'. Valid options: '5m' (5 minutes), '1h' (1 hour), or 'off'."
    )


class HttpChatModel:
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
    ):
        clean_url = base_url.rstrip("/")
        # Automatically handle both base URL and full /chat/completions URL
        if not clean_url.endswith("/chat/completions"):
            self.endpoint = f"{clean_url}/chat/completions"
        else:
            self.endpoint = clean_url

        self.model = model
        self.api_key = api_key
        self.extra_headers = headers or {}
        # OpenRouter recommends HTTP-Referer and X-Title
        if "openrouter.ai" in self.endpoint:
            if "HTTP-Referer" not in self.extra_headers:
                self.extra_headers["HTTP-Referer"] = "https://github.com/microharness"
            if "X-Title" not in self.extra_headers:
                self.extra_headers["X-Title"] = "microharness"
        self.timeout = timeout
        self.cache_ttl = normalize_cache_ttl(cache_ttl)

        # Telemetry & prompt caching metrics
        self.last_usage: dict[str, Any] = {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0

    def set_cache_ttl(self, ttl: str | None) -> str | None:
        """Dynamically switches prompt caching duration on the fly ('5m', '1h', or 'off'/None)."""
        self.cache_ttl = normalize_cache_ttl(ttl)
        return self.cache_ttl

    def get_cache_stats(self) -> dict[str, Any]:
        """Returns prompt cache TTL and cumulative token utilization metrics."""
        hit_rate = (
            (self.total_cached_tokens / self.total_prompt_tokens * 100)
            if self.total_prompt_tokens > 0
            else 0.0
        )
        return {
            "cache_ttl": self.cache_ttl,
            "last_usage": dict(self.last_usage),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "cache_hit_rate_pct": round(hit_rate, 1),
        }

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]:
        req_headers = {
            "Content-Type": "application/json",
            **self.extra_headers,
        }
        if self.api_key:
            req_headers["Authorization"] = f"Bearer {self.api_key}"

        # Inject provider-specific cache headers when prompt caching is active
        if self.cache_ttl == "1h":
            req_headers["anthropic-beta"] = "extended-cache-ttl-2025-04-11"
            req_headers["X-OpenRouter-Cache-TTL"] = "3600"
        elif self.cache_ttl == "5m":
            req_headers["X-OpenRouter-Cache-TTL"] = "300"

        # Format messages with cache breakpoint on system prompt if caching is active
        formatted_messages: list[dict[str, Any]] = []
        system_cached = False
        for msg in messages:
            msg_dict = msg.to_dict()
            if self.cache_ttl and msg.role == "system" and not system_cached:
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
        if tools:
            if self.cache_ttl:
                cache_ctrl = {"type": "ephemeral"}
                if self.cache_ttl == "1h":
                    cache_ctrl["ttl"] = "1h"
                # Attach cache_control to the last tool definition for Anthropic / OpenRouter
                tools_payload = [dict(t) for t in tools]
                tools_payload[-1] = {**tools_payload[-1], "cache_control": cache_ctrl}
                payload["tools"] = tools_payload
            else:
                payload["tools"] = tools
            payload["tool_choice"] = "auto"

        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=req_headers,
            method="POST",
        )

        max_retries = 4
        base_delay = 2.0
        res_body: dict[str, Any] = {}

        for attempt in range(max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as response:
                    res_body = json.loads(response.read().decode("utf-8"))
                    break
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")
                # Retry on rate limits (429) and transient server errors (500, 502, 503, 504)
                if exc.code in (429, 500, 502, 503, 504) and attempt < max_retries - 1:
                    sleep_time = base_delay * (2**attempt)
                    time.sleep(sleep_time)
                    continue
                raise RuntimeError(
                    f"Model API returned HTTP {exc.code} from {self.endpoint}:\n{err_body}"
                ) from exc
            except urllib.error.URLError as exc:
                if attempt < max_retries - 1:
                    sleep_time = base_delay * (2**attempt)
                    time.sleep(sleep_time)
                    continue
                raise ConnectionError(
                    f"Could not connect to model endpoint at {self.endpoint}.\n"
                    f"Is the server running and reachable? (Detail: {exc.reason})"
                ) from exc

        # Extract usage and track cumulative prompt caching metrics
        usage = res_body.get("usage", {})
        self.last_usage = usage
        prompt_toks = usage.get("prompt_tokens", 0) or 0
        comp_toks = usage.get("completion_tokens", 0) or 0

        cached_toks = 0
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached_toks = prompt_details.get("cached_tokens", 0) or 0
        if not cached_toks:
            cached_toks = (
                usage.get("cache_read_input_tokens", 0) or usage.get("cached_tokens", 0) or 0
            )

        self.total_prompt_tokens += prompt_toks
        self.total_completion_tokens += comp_toks
        self.total_cached_tokens += cached_toks

        choice = res_body["choices"][0]["message"]
        content = choice.get("content") or ""
        tool_calls: list[ToolCall] = []

        if choice.get("tool_calls"):
            for raw_tc in choice["tool_calls"]:
                func_data = raw_tc["function"]
                args_raw = func_data.get("arguments", "{}")
                if isinstance(args_raw, str):
                    try:
                        parsed_args = json.loads(args_raw)
                    except json.JSONDecodeError:
                        parsed_args = {"_raw_arguments": args_raw}
                else:
                    parsed_args = args_raw or {}

                tool_calls.append(
                    ToolCall(
                        id=raw_tc["id"],
                        name=func_data["name"],
                        arguments=parsed_args,
                    )
                )

        return content, tool_calls


class MockModel:
    """Deterministic model for offline testing and educational walkthroughs."""

    def __init__(
        self,
        script: list[tuple[str, list[ToolCall]]],
        cache_ttl: str | None = None,
    ):
        self.script = list(script)
        self.step_idx = 0
        self.cache_ttl = normalize_cache_ttl(cache_ttl)
        self.last_usage: dict[str, Any] = {}
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.total_cached_tokens = 0

    def set_cache_ttl(self, ttl: str | None) -> str | None:
        """Dynamically switches prompt caching duration on the fly."""
        self.cache_ttl = normalize_cache_ttl(ttl)
        return self.cache_ttl

    def get_cache_stats(self) -> dict[str, Any]:
        """Returns prompt cache TTL and cumulative token metrics."""
        hit_rate = (
            (self.total_cached_tokens / self.total_prompt_tokens * 100)
            if self.total_prompt_tokens > 0
            else 0.0
        )
        return {
            "cache_ttl": self.cache_ttl,
            "last_usage": dict(self.last_usage),
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_cached_tokens": self.total_cached_tokens,
            "cache_hit_rate_pct": round(hit_rate, 1),
        }

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]:
        if self.step_idx < len(self.script):
            result = self.script[self.step_idx]
            self.step_idx += 1
            return result
        return "I have completed the task and verified all results.", []
