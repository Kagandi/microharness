from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from core.types import Message, ToolCall


class Model(Protocol):
    """Protocol for pluggable LLM callers.

    Any class or adapter with a `generate` method can be plugged directly
    into the Harness without modifying any harness code.
    """

    def generate(
        self, messages: list[Message], tools: list[dict[str, Any]]
    ) -> tuple[str, list[ToolCall]]: ...


class RateLimitError(RuntimeError):
    """Raised when an LLM provider returns HTTP 429 Too Many Requests or quota exhaustion."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


MAX_RETRY_SLEEP_S: float = 120.0


def _is_unrecoverable_quota_error(status_code: int, err_body: str) -> bool:
    """Returns True if the error indicates a prolonged or unrecoverable daily quota or credit limit."""
    if status_code == 402:
        return True
    body_clean = re.sub(r"[\s_\-]+", "", err_body.lower())
    unrecoverable_tokens = (
        "perday",
        "dailylimit",
        "dailyrequestlimit",
        "requestsperday",
        "limitreachedfortoday",
        "quotaexceededfortoday",
        "exceededyourdaily",
        "freetierlimitexceededfortoday",
        "insufficientcredits",
        "creditlimitreached",
        "paymentrequired",
        "insufficientquota",
        "creditbalanceistoolow",
        "outofcredits",
        "insufficientbalance",
        "exceededyourcurrentquota",
        "currentquota",
    )
    return any(token in body_clean for token in unrecoverable_tokens)


def extract_retry_delay(
    err_body: str,
    retry_header: str | None = None,
    reset_header: str | None = None,
) -> float | None:
    """Extracts rate limit retry delay in seconds from HTTP headers or response body."""
    # 1. Standard HTTP header (Retry-After in seconds)
    if retry_header:
        try:
            val = float(retry_header)
            if val >= 0:
                return val
        except ValueError:
            pass

    # 2. X-RateLimit-Reset header (epoch seconds or epoch milliseconds)
    if reset_header:
        try:
            val = float(reset_header)
            now = time.time()
            reset_s = val / 1000.0 if val > 1e11 else val
            diff = reset_s - now
            if diff >= 0:
                return diff
        except ValueError:
            pass

    if not err_body:
        return None

    # 3. JSON structured error response (e.g. Google RPC RetryInfo, OpenRouter metadata headers)
    try:
        data = json.loads(err_body)
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict):
            err_dict = data.get("error", data) if isinstance(data, dict) else {}
            # Check metadata headers if embedded in JSON response (e.g. OpenRouter root or error dict)
            meta = (data.get("metadata") if isinstance(data, dict) else None) or (
                err_dict.get("metadata") if isinstance(err_dict, dict) else None
            )
            if isinstance(meta, dict):
                meta_hdrs = meta.get("headers", {})
                if isinstance(meta_hdrs, dict):
                    reset_val = meta_hdrs.get("X-RateLimit-Reset") or meta_hdrs.get("x-ratelimit-reset")
                    if reset_val:
                        try:
                            val = float(reset_val)
                            reset_s = val / 1000.0 if val > 1e11 else val
                            diff = reset_s - time.time()
                            if diff >= 0:
                                return diff
                        except ValueError:
                            pass
            if isinstance(err_dict, dict):
                for detail in err_dict.get("details", []):
                    if isinstance(detail, dict) and "RetryInfo" in str(detail.get("@type", "")):
                        raw_delay = str(detail.get("retryDelay", "")).strip().rstrip("s")
                        try:
                            return float(raw_delay)
                        except ValueError:
                            pass
                if "retry_after" in err_dict:
                    try:
                        return float(err_dict["retry_after"])
                    except ValueError:
                        pass
    except Exception:
        pass

    # 4. Regex fallback in error message text
    # Matches patterns like "Please retry in 55.116226147s.", "retry after 30s", or "Resets in 5 seconds."
    match = re.search(
        r"(?:retry|try|wait|reset)s?\s+(?:again\s+)?(?:in|after)\s+([0-9.]+)\s*s(?:ec(?:ond)?s?)?",
        err_body,
        re.IGNORECASE,
    )
    if match:
        try:
            return float(match.group(1))
        except ValueError:
            pass

    return None


@runtime_checkable
class ResettableModel(Model, Protocol):
    """Protocol for models that can be reset between benchmark tasks or test runs."""

    def reset(self) -> None: ...


@runtime_checkable
class CacheAwareModel(Model, Protocol):
    """Protocol for models that support dynamic prompt cache TTL and telemetry."""

    def set_cache_ttl(self, ttl: str | None) -> str | None: ...

    def get_cache_stats(self) -> dict[str, Any]: ...


@runtime_checkable
class ReasoningAwareModel(Model, Protocol):
    """Protocol for models that support dynamic reasoning effort level control."""

    def set_reasoning_effort(self, effort: str | None) -> str | None: ...

    def get_reasoning_effort(self) -> str | None: ...


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


def normalize_reasoning_effort(effort: str | None) -> str | None:
    """Normalizes reasoning/thinking effort level ('low', 'medium', 'high', numeric budget, or None)."""
    if effort is None:
        return None
    val = str(effort).strip().lower()
    if val in ("off", "none", "disable", "disabled", "false", "0", ""):
        return None
    if val in ("low", "medium", "high"):
        return val
    if val.isdigit():
        return val
    raise ValueError(
        f"Unsupported reasoning effort: '{effort}'. Valid options: 'low', 'medium', 'high', numeric token budget, or 'off'."
    )


PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google",
    "google-ai-studio": "google",
    "open-router": "openrouter",
}


def get_builtin_providers_path() -> Path:
    """Returns the path to the repository's built-in providers.json file."""
    return Path(__file__).parent / "providers.json"


def strip_json_comments(text: str) -> str:
    """Strips # and // comments from JSON-like text while preserving string literals."""
    out: list[str] = []
    i = 0
    n = len(text)
    in_string = False
    escape = False

    while i < n:
        c = text[i]
        if in_string:
            out.append(c)
            if escape:
                escape = False
            elif c == "\\":
                escape = True
            elif c == '"':
                in_string = False
            i += 1
        else:
            if c == '"':
                in_string = True
                escape = False
                out.append(c)
                i += 1
            elif c == "#":
                while i < n and text[i] != "\n":
                    i += 1
            elif c == "/" and i + 1 < n and text[i + 1] == "/":
                while i < n and text[i] != "\n":
                    i += 1
            else:
                out.append(c)
                i += 1

    return "".join(out)


def _read_json_with_comments(file_path: Path | None) -> dict[str, Any]:
    if not file_path or not file_path.is_file():
        return {}
    try:
        raw = file_path.read_text(encoding="utf-8")
        clean_json = strip_json_comments(raw)
        data = json.loads(clean_json)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def load_builtin_providers() -> dict[str, dict[str, Any]]:
    """Loads declarative provider definitions from the committed core/providers.json file."""
    path = get_builtin_providers_path()
    data = _read_json_with_comments(path)
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def seed_user_providers(global_dir: Path | None = None) -> Path | None:
    """Seeds ~/.microharness/providers.json if global_dir exists and providers.json is missing."""
    if not global_dir:
        return None
    try:
        global_path = Path(global_dir)
        if not global_path.exists():
            global_path.mkdir(parents=True, exist_ok=True)
        dest = global_path / "providers.json"
        if not dest.exists():
            builtin_path = get_builtin_providers_path()
            if builtin_path.is_file():
                dest.write_text(builtin_path.read_text(encoding="utf-8"), encoding="utf-8")
                return dest
    except Exception:
        pass
    return None


def load_provider_registry(
    global_dir: Path | None = None,
    project_dir: Path | None = None,
    inline_providers: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Loads and deep-merges providers across built-in, user global, project, and inline sources.

    Resolution order (later layers override earlier ones):
    1. Built-in: core/providers.json
    2. User Global: ~/.microharness/providers.json
    3. Project Local: <project_dir>/.microharness/providers.json
    4. Inline Config: config.json 'providers' dictionary
    """
    registry: dict[str, dict[str, Any]] = load_builtin_providers()

    # 2. User Global
    if global_dir:
        user_file = Path(global_dir) / "providers.json"
        if user_file.is_file():
            user_data = _read_json_with_comments(user_file)
            for k, v in user_data.items():
                if isinstance(v, dict):
                    registry[k] = {**registry.get(k, {}), **v}

    # 3. Project Local
    if project_dir:
        proj_dir = Path(project_dir)
        proj_file = proj_dir / ".microharness" / "providers.json"
        if not proj_file.is_file():
            proj_file = proj_dir / "providers.json"
        proj_data = _read_json_with_comments(proj_file)
        for k, v in proj_data.items():
            if isinstance(v, dict):
                registry[k] = {**registry.get(k, {}), **v}

    # 4. Inline Config
    if inline_providers and isinstance(inline_providers, dict):
        for k, v in inline_providers.items():
            if isinstance(v, dict):
                registry[k] = {**registry.get(k, {}), **v}

    return registry


def resolve_provider(
    registry: dict[str, dict[str, Any]],
    provider_name: str | None,
) -> tuple[str | None, dict[str, Any]]:
    """Resolves provider profile from registry considering case-insensitivity and aliases."""
    if not provider_name:
        return None, {}
    clean_name = str(provider_name).strip().lower()
    canonical = PROVIDER_ALIASES.get(clean_name, clean_name)
    if canonical in registry:
        return canonical, dict(registry[canonical])
    for k, v in registry.items():
        if k.lower() == canonical:
            return k, dict(v)
    return canonical, {}


def find_provider_by_endpoint(
    registry: dict[str, dict[str, Any]],
    endpoint: str | None,
) -> tuple[str | None, dict[str, Any]]:
    """Finds provider profile by matching an endpoint URL prefix."""
    if not endpoint:
        return None, {}
    clean_ep = endpoint.rstrip("/").lower()
    for name, prof in registry.items():
        prof_ep = (prof.get("endpoint") or "").rstrip("/").lower()
        if prof_ep and (clean_ep == prof_ep or clean_ep.startswith(prof_ep)):
            return name, dict(prof)
    return None, {}


class _CacheTelemetryMixin:
    """Provides common prompt cache telemetry and token usage accounting."""

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
        """Returns normalized token counts and cache hit percentage for the most recent generation step."""
        usage = self.last_usage or {}
        p_tok = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("promptTokenCount")
            or 0
        )
        c_tok = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("candidatesTokenCount")
            or 0
        )
        prompt_details = usage.get("prompt_tokens_details")
        cached_tok = 0
        if isinstance(prompt_details, dict):
            cached_tok = int(prompt_details.get("cached_tokens") or 0)
        if not cached_tok:
            cached_tok = int(
                usage.get("cache_read_input_tokens")
                or usage.get("cached_tokens")
                or usage.get("cachedContentTokenCount")
                or usage.get("native_tokens_prompt_cache_read")
                or 0
            )
        hit_pct = round((cached_tok / p_tok * 100), 1) if p_tok > 0 else 0.0
        return {
            "prompt_tokens": p_tok,
            "completion_tokens": c_tok,
            "cached_tokens": cached_tok,
            "cache_hit_rate_pct": hit_pct,
        }

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

    def _update_usage(self, usage: dict[str, Any]) -> None:
        self.last_usage = usage
        prompt_toks = int(
            usage.get("prompt_tokens")
            or usage.get("input_tokens")
            or usage.get("promptTokenCount")
            or 0
        )
        comp_toks = int(
            usage.get("completion_tokens")
            or usage.get("output_tokens")
            or usage.get("candidatesTokenCount")
            or 0
        )

        cached_toks = 0
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict):
            cached_toks = int(prompt_details.get("cached_tokens") or 0)
        if not cached_toks:
            cached_toks = int(
                usage.get("cache_read_input_tokens")
                or usage.get("cached_tokens")
                or usage.get("cachedContentTokenCount")
                or usage.get("native_tokens_prompt_cache_read")
                or 0
            )

        self.total_prompt_tokens += prompt_toks
        self.total_completion_tokens += comp_toks
        self.total_cached_tokens += cached_toks


class HttpChatModel(_CacheTelemetryMixin):
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

        self.model = model
        self.api_key = api_key
        # Merge declarative profile headers with explicitly passed headers (explicit wins)
        self.extra_headers = {**default_headers, **(headers or {})}
        self.timeout = timeout
        self.deadline = deadline
        self.total_rate_limit_wait_s: float = 0.0
        self.reasoning_effort = normalize_reasoning_effort(reasoning_effort)
        self.reasoning_format = (reasoning_format or default_reasoning_format).lower()
        self.cache_strategy = (cache_strategy or default_cache_strategy).lower()
        self._init_cache_telemetry(cache_ttl)

    def set_deadline(self, deadline: float | None) -> None:
        """Sets the task deadline timestamp to strictly bound in-flight HTTP requests."""
        self.deadline = deadline

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
        if self.api_key:
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
            if effort in ("low", "medium", "high"):
                payload["reasoning_effort"] = effort
                if self.reasoning_format == "openrouter":
                    payload["reasoning"] = {"effort": effort}
            elif effort.isdigit() and self.reasoning_format == "openrouter":
                payload["reasoning"] = {"max_tokens": int(effort)}

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

    def _send_request(self, payload: dict[str, Any], headers: dict[str, str]) -> dict[str, Any]:
        req = urllib.request.Request(
            self.endpoint,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        max_retries = 6
        base_delay = 2.0
        for attempt in range(max_retries):
            # Check deadline before attempting request
            req_timeout = self.timeout
            if self.deadline is not None:
                remaining = self.deadline - time.time()
                if remaining <= 0:
                    raise TimeoutError(
                        f"Task timeout reached before model request attempt {attempt + 1}"
                    )
                req_timeout = min(self.timeout, max(remaining, 0.1))

            try:
                with urllib.request.urlopen(req, timeout=req_timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                err_body = exc.read().decode("utf-8", errors="replace")
                if exc.code in (429, 402):
                    retry_header = exc.headers.get("Retry-After") if exc.headers else None
                    reset_header = (
                        (exc.headers.get("X-RateLimit-Reset") or exc.headers.get("x-ratelimit-reset"))
                        if exc.headers
                        else None
                    )
                    delay = extract_retry_delay(err_body, retry_header=retry_header, reset_header=reset_header)

                    # 1. Daily Quota / Credit Exhaustion without a short recoverable reset hint
                    if _is_unrecoverable_quota_error(exc.code, err_body):
                        if delay is None or delay > MAX_RETRY_SLEEP_S:
                            raise RateLimitError(
                                f"Model API daily quota or credit limit exhausted (HTTP {exc.code}) from {self.endpoint}:\n{err_body}",
                                retry_after=delay,
                            ) from exc

                    # 2. Prolonged Limit: Delay exceeds maximum wait threshold (120s)
                    if delay is not None and delay > MAX_RETRY_SLEEP_S:
                        raise RateLimitError(
                            f"Model API rate limit retry delay ({delay:.1f}s) exceeds maximum allowed sleep "
                            f"({MAX_RETRY_SLEEP_S:.0f}s). Daily quota or prolonged limit reached from {self.endpoint}:\n{err_body}",
                            retry_after=delay,
                        ) from exc

                    # 3. Transient Rate Limit: Use extracted delay, or fall back to exponential backoff
                    if delay is not None and delay >= 0:
                        if not retry_header and not reset_header:
                            delay = delay + 1.0
                    else:
                        delay = min(base_delay * (2**attempt), 60.0)

                    self.total_rate_limit_wait_s += delay
                    if self.deadline is not None:
                        self.deadline += delay

                    if attempt < max_retries - 1:
                        print(
                            f"\n⚠️ [Rate Limit {exc.code}]: Waiting {delay:.1f}s before retry ({attempt + 1}/{max_retries})..."
                        )
                        time.sleep(delay)
                        continue
                    raise RateLimitError(
                        f"Model API rate limit (HTTP {exc.code}) exceeded after {max_retries} retries from {self.endpoint}:\n{err_body}",
                        retry_after=delay,
                    ) from exc

                if exc.code in (500, 502, 503, 504) and attempt < max_retries - 1:
                    sleep_s = min(base_delay * (2**attempt), 60.0)
                    if self.deadline is not None and (time.time() + sleep_s) >= self.deadline:
                        raise TimeoutError(
                            f"Task timeout reached while waiting to retry HTTP {exc.code}"
                        ) from exc
                    time.sleep(sleep_s)
                    continue
                raise RuntimeError(
                    f"Model API returned HTTP {exc.code} from {self.endpoint}:\n{err_body}"
                ) from exc
            except TimeoutError as exc:
                if self.deadline is not None and time.time() >= self.deadline:
                    raise TimeoutError("Task timeout reached during model request") from exc
                if attempt < max_retries - 1:
                    sleep_s = min(base_delay * (2**attempt), 60.0)
                    if self.deadline is not None and (time.time() + sleep_s) >= self.deadline:
                        raise TimeoutError(
                            "Task timeout reached while waiting to retry timed-out request"
                        ) from exc
                    time.sleep(sleep_s)
                    continue
                raise
            except urllib.error.URLError as exc:
                if isinstance(exc.reason, TimeoutError) or str(exc.reason).lower().startswith("timed out"):
                    if self.deadline is not None and time.time() >= self.deadline:
                        raise TimeoutError("Task timeout reached during model request") from exc
                if attempt < max_retries - 1:
                    sleep_s = min(base_delay * (2**attempt), 60.0)
                    if self.deadline is not None and (time.time() + sleep_s) >= self.deadline:
                        raise TimeoutError(
                            f"Task timeout reached while waiting to retry connection error: {exc.reason}"
                        ) from exc
                    time.sleep(sleep_s)
                    continue
                raise ConnectionError(
                    f"Could not connect to model endpoint at {self.endpoint}.\n"
                    f"Is the server running and reachable? (Detail: {exc.reason})"
                ) from exc
        return {}

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


class MockModel(_CacheTelemetryMixin):
    """Deterministic model for offline testing and educational walkthroughs."""

    def __init__(
        self,
        script: list[tuple[str, list[ToolCall]]],
        cache_ttl: str | None = None,
        reasoning_effort: str | None = None,
    ):
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
