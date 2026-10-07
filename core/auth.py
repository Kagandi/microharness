"""Pluggable request-signing auth schemes.

Wave 1 ships the ``RequestSigner`` protocol, the scheme registry, and the
Tier 1 built-ins (``none``/``bearer``/``header``). A signer is pure: it takes
(method, url, headers, body, options, secrets) and returns a NEW headers
dict, never mutating its inputs.

Only the standard library is used.
"""

from __future__ import annotations

import os
from typing import Protocol, runtime_checkable


@runtime_checkable
class RequestSigner(Protocol):
    """Signs an outgoing request by returning updated headers."""

    def sign(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        options: dict,
        secrets: dict,
    ) -> dict[str, str]: ...


class _NoneSigner:
    """Passes headers through unchanged (as a copy)."""

    def sign(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        options: dict,
        secrets: dict,
    ) -> dict[str, str]:
        return dict(headers)


class _BearerSigner:
    """Sets ``Authorization: Bearer <secrets['api_key']>``."""

    def sign(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        options: dict,
        secrets: dict,
    ) -> dict[str, str]:
        api_key = secrets.get("api_key") if isinstance(secrets, dict) else None
        if not api_key:
            raise ValueError("bearer auth requires secrets['api_key']")
        new_headers = dict(headers)
        new_headers["Authorization"] = f"Bearer {api_key}"
        return new_headers


class _HeaderSigner:
    """Sets ``options['name']`` to a value resolved from ``value_source``.

    Sources: ``env:VAR`` (process environment), ``config:field`` (secrets
    dict), ``literal:...`` (verbatim rest, may contain colons).
    """

    def sign(
        self,
        *,
        method: str,
        url: str,
        headers: dict[str, str],
        body: bytes,
        options: dict,
        secrets: dict,
    ) -> dict[str, str]:
        name = options.get("name") if isinstance(options, dict) else None
        if not name:
            raise ValueError("header auth requires options['name']")
        source = options.get("value_source") if isinstance(options, dict) else None
        if not isinstance(source, str) or ":" not in source:
            raise ValueError(
                "header auth requires options['value_source'] "
                "like 'env:VAR', 'config:field', or 'literal:...'"
            )
        prefix, _, rest = source.partition(":")
        if prefix == "env":
            value = os.environ.get(rest)
            if value is None:
                raise ValueError(f"header auth: environment variable {rest!r} is not set")
        elif prefix == "config":
            value = secrets.get(rest) if isinstance(secrets, dict) else None
            if value is None:
                raise ValueError(f"header auth: secrets field {rest!r} is missing")
        elif prefix == "literal":
            value = rest
        else:
            raise ValueError(f"header auth: unknown value source prefix {prefix!r}")
        new_headers = dict(headers)
        new_headers[name] = value
        return new_headers


_BUILTIN_NAMES = frozenset({"none", "bearer", "header"})

_REGISTRY: dict[str, RequestSigner] = {
    "none": _NoneSigner(),
    "bearer": _BearerSigner(),
    "header": _HeaderSigner(),
}


def _normalize(name: str) -> str:
    if not isinstance(name, str) or not name:
        raise ValueError("scheme name must be a non-empty string")
    return name.lower()


def register_scheme(name: str, signer: RequestSigner) -> None:
    """Registers a signer under ``name`` (case-insensitive)."""
    key = _normalize(name)
    if not hasattr(signer, "sign") or not callable(signer.sign):
        raise TypeError("signer must provide a callable sign() method")
    _REGISTRY[key] = signer


def get_signer(name: str) -> RequestSigner:
    """Returns the signer for ``name``; raises KeyError when unknown."""
    try:
        return _REGISTRY[_normalize(name)]
    except KeyError:
        raise KeyError(f"unknown auth scheme: {name!r}") from None


def builtin_schemes() -> list[str]:
    """Lists the built-in Tier 1 scheme names."""
    return sorted(_BUILTIN_NAMES)


def resolve_auth(profile: dict) -> tuple[str, dict]:
    """Splits a provider profile's ``auth`` into (scheme, options).

    Shapes: missing/None -> ("bearer", {}); a bare string names the scheme
    with empty options; a dict carries {"scheme": name, **options}. Unknown
    scheme names pass through here and fail later at ``get_signer`` time.
    """
    auth = profile.get("auth")
    if auth is None:
        return ("bearer", {})
    if isinstance(auth, str):
        if not auth:
            raise ValueError("auth scheme name must be a non-empty string")
        return (auth.lower(), {})
    if isinstance(auth, dict):
        scheme = auth.get("scheme")
        if not isinstance(scheme, str) or not scheme:
            raise ValueError("auth dict requires a non-empty 'scheme' name")
        return (scheme.lower(), {k: v for k, v in auth.items() if k != "scheme"})
    raise ValueError(f"auth must be a scheme name or dict, got {type(auth).__name__}")
