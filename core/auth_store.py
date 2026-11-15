"""Global per-provider credential store (``~/.microharness/auth.json``).

Stores API credentials per canonical provider (never per model) in an
owner-only JSON file inside ``global_dir``. Only Python's standard library is
used, and writes are performed atomically via ``os.open(..., 0o600)`` +
``os.replace(...)`` so secrets are never briefly world-readable.
"""

from __future__ import annotations

import json
import os
import secrets
from collections.abc import Mapping
from pathlib import Path
from typing import Any

AUTH_FILENAME = "auth.json"
AUTH_TYPE_API_KEY = "api_key"


def get_auth_store_path(global_dir: Path | str | None) -> Path | None:
    """Returns ``<global_dir>/auth.json`` or ``None`` when ``global_dir`` is unset."""
    if not global_dir:
        return None
    return Path(global_dir) / AUTH_FILENAME


def mask_secret(secret: str | None) -> str:
    """Returns a redacted preview of a secret suitable for terminal/UI output."""
    if not secret:
        return ""
    clean = secret.strip()
    if not clean:
        return ""
    if len(clean) < 12:
        return "****"
    return f"{clean[:4]}...{clean[-4:]}"


def _default_aliases() -> Mapping[str, str]:
    from core.providers import PROVIDER_ALIASES

    return PROVIDER_ALIASES


def canonicalize_provider(
    provider: str | None,
    aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Normalizes a provider name through lowercase/strip and alias mapping."""
    if not provider or not isinstance(provider, str):
        return None
    clean = provider.strip().lower()
    if not clean:
        return None
    alias_map = _default_aliases() if aliases is None else aliases
    return alias_map.get(clean, clean)


def _read_raw_auth_object(path: Path | None) -> dict[str, Any]:
    """Reads the raw JSON object from ``auth.json``, failing safely on any error."""
    if not path or not path.is_file():
        return {}
    try:
        raw_text = path.read_text(encoding="utf-8")
        data = json.loads(raw_text)
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def load_auth_store(
    global_dir: Path | str | None,
    *,
    aliases: Mapping[str, str] | None = None,
) -> dict[str, dict[str, Any]]:
    """Loads valid provider credential records keyed by canonical provider name.

    Corrupt, unreadable, or non-object files return ``{}`` without crashing or
    leaking file contents. Exact canonical keys take precedence over alias keys.
    """
    path = get_auth_store_path(global_dir)
    raw = _read_raw_auth_object(path)
    if not raw:
        return {}

    alias_map = _default_aliases() if aliases is None else aliases
    store: dict[str, dict[str, Any]] = {}
    exact_hits: set[str] = set()

    for key, entry in raw.items():
        if not isinstance(key, str) or not isinstance(entry, dict):
            continue
        clean_key = key.strip().lower()
        canonical = canonicalize_provider(clean_key, alias_map)
        if not canonical:
            continue
        is_exact = clean_key == canonical
        if canonical in exact_hits and not is_exact:
            continue
        store[canonical] = dict(entry)
        if is_exact:
            exact_hits.add(canonical)

    return store


def get_auth_credential(
    global_dir: Path | str | None,
    provider: str | None,
    *,
    aliases: Mapping[str, str] | None = None,
) -> str | None:
    """Returns the stored API key for ``provider`` (canonicalized), or ``None``."""
    canonical = canonicalize_provider(provider, aliases)
    if not canonical:
        return None
    store = load_auth_store(global_dir, aliases=aliases)
    entry = store.get(canonical)
    if not isinstance(entry, dict):
        return None
    entry_type = entry.get("type", AUTH_TYPE_API_KEY)
    if entry_type != AUTH_TYPE_API_KEY:
        return None
    key = entry.get("key")
    if not isinstance(key, str):
        return None
    stripped = key.strip()
    return stripped or None


def _write_auth_store(dest: Path, data: dict[str, Any]) -> Path:
    """Atomically writes ``auth.json`` with ``0o600`` permissions."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(data, indent=2, sort_keys=True) + "\n"
    tmp_path = dest.parent / f".auth.{os.getpid()}.{secrets.token_hex(6)}.tmp"
    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW

    fd = os.open(tmp_path, flags, 0o600)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(payload)
            fh.flush()
            os.fsync(fh.fileno())
        if os.name == "posix":
            os.chmod(tmp_path, 0o600)
        os.replace(tmp_path, dest)
    except BaseException:
        try:
            if tmp_path.exists():
                tmp_path.unlink()
        except OSError:
            pass
        raise
    return dest


def save_auth_credential(
    global_dir: Path | str | None,
    provider: str,
    api_key: str,
    *,
    aliases: Mapping[str, str] | None = None,
) -> Path:
    """Saves or updates an API key for ``provider`` in ``<global_dir>/auth.json``."""
    dest = get_auth_store_path(global_dir)
    if dest is None:
        raise ValueError("global_dir is required to save credentials.")
    canonical = canonicalize_provider(provider, aliases)
    if not canonical:
        raise ValueError("Provider name cannot be empty.")
    if not isinstance(api_key, str) or not api_key.strip():
        raise ValueError("API key cannot be empty.")

    store = load_auth_store(global_dir, aliases=aliases)
    store[canonical] = {
        "type": AUTH_TYPE_API_KEY,
        "key": api_key.strip(),
    }
    return _write_auth_store(dest, store)


def remove_auth_credential(
    global_dir: Path | str | None,
    provider: str,
    *,
    aliases: Mapping[str, str] | None = None,
) -> bool:
    """Removes stored credentials for ``provider`` from ``<global_dir>/auth.json``.

    Returns ``True`` if an entry was removed, ``False`` if none existed.
    """
    dest = get_auth_store_path(global_dir)
    if dest is None:
        return False
    canonical = canonicalize_provider(provider, aliases)
    if not canonical:
        raise ValueError("Provider name cannot be empty.")
    if not dest.is_file():
        return False

    store = load_auth_store(global_dir, aliases=aliases)
    if canonical not in store:
        return False
    del store[canonical]
    _write_auth_store(dest, store)
    return True
