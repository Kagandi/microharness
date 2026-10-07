"""Pluggable request-signing auth schemes.

The ``RequestSigner`` protocol, the scheme registry, and the Tier 1 built-ins
(``none``/``bearer``/``header`` plus the ``sigv4`` reference implementation in
``core.auth_sigv4``). A signer is pure: it takes
(method, url, headers, body, options, secrets) and returns a NEW headers
dict, never mutating its inputs.

Only the standard library is used.
"""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Protocol, runtime_checkable

from core.auth_sigv4 import SigV4Signer


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


_BUILTIN_NAMES = frozenset({"none", "bearer", "header", "sigv4"})

_REGISTRY: dict[str, RequestSigner] = {
    "none": _NoneSigner(),
    "bearer": _BearerSigner(),
    "header": _HeaderSigner(),
    "sigv4": SigV4Signer(),
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


class _FunctionSigner:
    """Adapts a module-level ``sign()`` function to the RequestSigner protocol."""

    def __init__(self, func):
        self._func = func

    def sign(self, *, method, url, headers, body, options, secrets):
        return self._func(
            method=method,
            url=url,
            headers=headers,
            body=body,
            options=options,
            secrets=secrets,
        )


def load_auth_plugins(global_dir: Path | str | None) -> dict[str, str]:
    """Loads Tier 2 auth-scheme plugins from the trusted global dir.

    Only ``global_dir`` is accepted by construction: project-local dirs are
    never searched because auth code sees secrets. Callers must pass the
    trusted global dir only (e.g. ``~/.microharness/auth``) and call once at
    startup. No sandboxing beyond this scoping; a plugin runs with full
    process privileges.

    Contract per file: ``SCHEME_NAME: str`` (non-empty) plus EITHER a
    module-level ``SIGNER`` object with a callable ``sign()`` OR a
    module-level ``sign()`` function (wrapped in an adapter). Failures
    (import error, bad contract, name collision) print a stderr warning
    with the full path and skip the file (fail closed). Built-ins and
    earlier files always win collisions.

    Returns a mapping of loaded (normalized) scheme name -> source file path.
    """
    if global_dir is None:
        return {}
    base = Path(global_dir)
    if not base.is_dir():
        return {}
    try:
        entries = sorted(
            (
                p
                for p in base.iterdir()
                if p.is_file()
                and p.suffix == ".py"
                and p.name != "__init__.py"
                and not p.name.startswith("_")
            ),
            key=lambda p: p.name,
        )
    except OSError as exc:
        print(f"Warning: cannot list auth plugins in '{base}': {exc}", file=sys.stderr)
        return {}
    loaded: dict[str, str] = {}
    for path in entries:
        try:
            spec = importlib.util.spec_from_file_location(
                f"_mh_auth_{path.stem}_{abs(hash(str(path)))}", path
            )
            if spec is None or spec.loader is None:
                raise ImportError(f"cannot create module spec for '{path}'")
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
        except Exception as exc:
            print(
                f"Warning: Skipping auth plugin '{path}': import failed: {exc}",
                file=sys.stderr,
            )
            continue
        try:
            scheme = getattr(module, "SCHEME_NAME", None)
            if not isinstance(scheme, str) or not scheme:
                raise TypeError("SCHEME_NAME must be a non-empty string")
            name = _normalize(scheme)
            if name in _REGISTRY:
                print(
                    f"Warning: Skipping auth plugin '{path}': "
                    f"scheme {name!r} already registered",
                    file=sys.stderr,
                )
                continue
            signer = getattr(module, "SIGNER", None)
            if signer is not None and hasattr(signer, "sign"):
                if not callable(signer.sign):
                    raise TypeError("SIGNER.sign must be callable")
            elif callable(getattr(module, "sign", None)):
                signer = _FunctionSigner(module.sign)
            else:
                raise TypeError("plugin must define SIGNER.sign or sign()")
            register_scheme(name, signer)
            loaded[name] = str(path)
        except Exception as exc:
            print(
                f"Warning: Skipping auth plugin '{path}': {exc}",
                file=sys.stderr,
            )
            continue
    return loaded


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
