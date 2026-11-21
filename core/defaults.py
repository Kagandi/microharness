"""Declarative operational defaults loaded from ``core/defaults.json``."""

from __future__ import annotations

import functools
import json
from pathlib import Path
from typing import Any

DEFAULTS_PATH = Path(__file__).parent / "defaults.json"


@functools.lru_cache(maxsize=1)
def load_defaults() -> dict[str, Any]:
    """Loads built-in operational defaults from ``core/defaults.json``."""
    data = json.loads(DEFAULTS_PATH.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError(f"Expected JSON object in {DEFAULTS_PATH}")
    return data


def get_default(key: str) -> Any:
    """Returns a built-in default by key from ``core/defaults.json``."""
    return load_defaults()[key]
