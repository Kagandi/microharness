"""Rate-limit interpretation and the retry policy shared by HTTP model clients.

Providers signal "come back later" in wildly different shapes: standard headers,
epoch reset stamps, Google RPC ``RetryInfo`` details, OpenRouter metadata blocks,
or a sentence in the error text. Everything that turns those shapes into a
single number of seconds lives here, so a new provider quirk is a new strategy
function instead of new branching inside the request loop.
"""

from __future__ import annotations

import json
import re
import time
from collections.abc import Callable
from dataclasses import dataclass


class RateLimitError(RuntimeError):
    """Raised when an LLM provider returns HTTP 429 Too Many Requests or quota exhaustion."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class ModelCancelledError(RuntimeError):
    """Raised when a cancel probe reports that the user aborted the request.

    Only raised while the model is *waiting* (retry backoff). An in-flight HTTP
    read cannot be interrupted this way; see SPEC.md section 5.
    """


# 402 needs no detection pass: it always means "pay before we serve you".
RATE_LIMIT_STATUS_CODES: frozenset[int] = frozenset({402, 429})
TRANSIENT_STATUS_CODES: frozenset[int] = frozenset({500, 502, 503, 504})

UNRECOVERABLE_QUOTA_TOKENS: tuple[str, ...] = (
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

# "Please retry in 55.116226147s.", "retry after 30s", "Resets in 5 seconds."
_RETRY_IN_TEXT = re.compile(
    r"(?:retry|try|wait|reset)s?\s+(?:again\s+)?(?:in|after)\s+([0-9.]+)\s*s(?:ec(?:ond)?s?)?",
    re.IGNORECASE,
)


def is_unrecoverable_quota_error(status_code: int, err_body: str) -> bool:
    """Returns True if the error indicates a prolonged or unrecoverable daily quota or credit limit."""
    if status_code == 402:
        return True
    body_clean = re.sub(r"[\s_\-]+", "", err_body.lower())
    return any(token in body_clean for token in UNRECOVERABLE_QUOTA_TOKENS)


def format_delay(delay: float) -> str:
    """Formats a retry delay in seconds as a concise, human-friendly duration."""
    if delay < 3600.0:
        return f"{delay:.1f}s"
    hours = delay / 3600.0
    if hours >= 1_000_000.0:
        return f"~{hours / 1_000_000.0:.1f}M hours"
    if hours >= 1_000.0:
        return f"~{hours / 1_000.0:.1f}k hours"
    if hours >= 24.0:
        return f"{hours / 24.0:.1f}d ({hours:.1f} hours)"
    return f"{hours:.1f} hours"


@dataclass(frozen=True)
class RateLimitHints:
    """The raw signals one rate-limited response offers, in provider-specific shapes."""

    body: str
    retry_after: str | None = None
    reset: str | None = None


def _as_float(value: object) -> float | None:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None


def _as_epoch_seconds(value: object) -> float | None:
    """Normalizes a reset stamp given as either epoch seconds or epoch milliseconds."""
    val = _as_float(value)
    if val is None:
        return None
    return val / 1000.0 if val > 1e11 else val


def _seconds_until(stamp: object) -> float | None:
    epoch_s = _as_epoch_seconds(stamp)
    if epoch_s is None:
        return None
    diff = epoch_s - time.time()
    return diff if diff >= 0 else None


def _delay_from_retry_after(hints: RateLimitHints) -> float | None:
    """Strategy 1: the standard ``Retry-After`` header, in seconds."""
    val = _as_float(hints.retry_after)
    return val if val is not None and val >= 0 else None


def _delay_from_reset_header(hints: RateLimitHints) -> float | None:
    """Strategy 2: the ``X-RateLimit-Reset`` header, as an absolute epoch stamp."""
    return _seconds_until(hints.reset) if hints.reset else None


def _delay_from_metadata(node: object) -> float | None:
    """Strategy 3a: gateways such as OpenRouter echo retry headers in a metadata block."""
    if not isinstance(node, dict):
        return None
    meta = node.get("metadata")
    if not isinstance(meta, dict) or not isinstance(meta.get("headers"), dict):
        return None
    headers = meta["headers"]
    return _seconds_until(headers.get("X-RateLimit-Reset") or headers.get("x-ratelimit-reset"))


def _delay_from_error_dict(err: dict) -> float | None:
    """Strategy 3b: structured provider payloads (Google RPC RetryInfo, ``retry_after``)."""
    for detail in err.get("details") or []:
        if isinstance(detail, dict) and "RetryInfo" in str(detail.get("@type", "")):
            delay = _as_float(str(detail.get("retryDelay", "")).strip().rstrip("s"))
            if delay is not None:
                return delay
    return _as_float(err.get("retry_after"))


def _delay_from_json_body(hints: RateLimitHints) -> float | None:
    """Strategy 3: JSON error payloads, at the root or nested under ``error``."""
    if not hints.body:
        return None
    try:
        data = json.loads(hints.body)
    except Exception:
        return None
    if isinstance(data, list) and data:
        data = data[0]
    if not isinstance(data, dict):
        return None
    err = data.get("error") if isinstance(data.get("error"), dict) else data
    return _delay_from_metadata(data) or _delay_from_metadata(err) or _delay_from_error_dict(err)


def _delay_from_message_text(hints: RateLimitHints) -> float | None:
    """Strategy 4: a human-readable sentence in the error body."""
    match = _RETRY_IN_TEXT.search(hints.body)
    return _as_float(match.group(1)) if match else None


DELAY_STRATEGIES: tuple[Callable[[RateLimitHints], float | None], ...] = (
    _delay_from_retry_after,
    _delay_from_reset_header,
    _delay_from_json_body,
    _delay_from_message_text,
)


def extract_retry_delay(
    err_body: str,
    retry_header: str | None = None,
    reset_header: str | None = None,
) -> float | None:
    """Extracts a retry delay in seconds from HTTP headers or the response body.

    Strategies are tried most to least authoritative; the first one that yields a
    non-negative delay wins.
    """
    hints = RateLimitHints(body=err_body or "", retry_after=retry_header, reset=reset_header)
    for strategy in DELAY_STRATEGIES:
        delay = strategy(hints)
        if delay is not None:
            return delay
    return None


def resolve_rate_limit_delay(
    hints: RateLimitHints,
    attempt: int,
    policy: "RetryPolicy",
) -> float:
    """Picks how long to wait after a rate-limited attempt.

    A parsed delay is trusted as-is when the server sent an explicit header, and
    gets a 1s safety buffer when it was inferred from the body (to avoid racing
    the provider's own clock). With no usable signal, exponential backoff applies.
    """
    parsed = extract_retry_delay(
        hints.body, retry_header=hints.retry_after, reset_header=hints.reset
    )
    if parsed is None or parsed < 0:
        return policy.backoff(attempt)
    if not hints.retry_after and not hints.reset:
        return parsed + 1.0
    return parsed


@dataclass(frozen=True)
class RetryPolicy:
    """How many attempts to make and how long to wait between them."""

    max_attempts: int = 6
    base_delay: float = 2.0
    max_delay: float = 60.0
    max_retry_sleep_s: float = 120.0

    def backoff(self, attempt: int) -> float:
        """Exponential backoff for ``attempt`` (0-based), capped at ``max_delay``."""
        return min(self.base_delay * (2**attempt), self.max_delay)

    def can_retry(self, attempt: int) -> bool:
        return attempt < self.max_attempts - 1