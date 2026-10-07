"""The declarative provider registry and model-spec resolution.

Provider knowledge lives in ``core/providers.json`` (endpoint, default model,
API-key environment variables, reasoning/cache dialects) and is layered with
user-global, project-local, and inline config. This module resolves a model
specification — a bare id, a ``provider/model`` pair, a config alias, or a
mapping — against those layers and instantiates the client.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.config import read_jsonc
from core.protocols import Model

PROVIDER_ALIASES: dict[str, str] = {
    "gemini": "google",
    "google-ai-studio": "google",
    "open-router": "openrouter",
    "lm-studio": "lmstudio",
    "llama-cpp": "llamacpp",
    "llama.cpp": "llamacpp",
}

# Last-resort API key lookup when neither the spec nor the provider profile names one.
FALLBACK_ENV_KEYS: tuple[str, ...] = (
    "GEMINI_API_KEY",
    "GOOGLE_API_KEY",
    "OPENROUTER_API_KEY",
    "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY",
    "DEEPSEEK_API_KEY",
    "GROQ_API_KEY",
)

# Used only when no profile and no config supply a model at all.
LAST_RESORT_MODEL = "anthropic/claude-3.7-sonnet"


def get_builtin_providers_path() -> Path:
    """Returns the path to the repository's built-in providers.json file."""
    return Path(__file__).parent / "providers.json"


def load_builtin_providers() -> dict[str, dict[str, Any]]:
    """Loads declarative provider definitions from the committed core/providers.json file."""
    data = read_jsonc(get_builtin_providers_path())
    return {k: v for k, v in data.items() if isinstance(v, dict)}


def seed_user_providers(global_dir: Path | None = None) -> Path | None:
    """Seeds ``~/.microharness/providers.json`` from the built-ins when it is missing.

    Returns the written path, or ``None`` when there was nothing to seed.
    """
    if not global_dir:
        return None
    try:
        global_path = Path(global_dir)
        global_path.mkdir(parents=True, exist_ok=True)
        dest = global_path / "providers.json"
        if dest.exists():
            return None
        builtin_path = get_builtin_providers_path()
        if not builtin_path.is_file():
            return None
        dest.write_text(builtin_path.read_text(encoding="utf-8"), encoding="utf-8")
        return dest
    except Exception:
        return None


def _merge_profiles(
    registry: dict[str, dict[str, Any]], data: dict[str, Any]
) -> dict[str, dict[str, Any]]:
    """Shallow-merges profile dictionaries, letting later layers override earlier keys."""
    for key, profile in data.items():
        if isinstance(profile, dict):
            registry[key] = {**registry.get(key, {}), **profile}
    return registry


def load_provider_registry(
    global_dir: Path | None = None,
    project_dir: Path | None = None,
    inline_providers: dict[str, Any] | None = None,
) -> dict[str, dict[str, Any]]:
    """Loads and deep-merges providers across built-in, user global, project, and inline sources.

    Resolution order (later layers override earlier ones):
    1. Built-in: ``core/providers.json``
    2. User Global: ``~/.microharness/providers.json``
    3. Project Local: ``<project_dir>/.microharness/providers.json`` (or ``providers.json``)
    4. Inline Config: ``config.json`` ``providers`` dictionary
    """
    registry = load_builtin_providers()

    if global_dir:
        _merge_profiles(registry, read_jsonc(Path(global_dir) / "providers.json"))

    if project_dir:
        project = Path(project_dir)
        project_file = project / ".microharness" / "providers.json"
        if not project_file.is_file():
            project_file = project / "providers.json"
        _merge_profiles(registry, read_jsonc(project_file))

    if isinstance(inline_providers, dict):
        _merge_profiles(registry, inline_providers)

    return registry


def resolve_provider(
    registry: dict[str, dict[str, Any]],
    provider_name: str | None,
) -> tuple[str | None, dict[str, Any]]:
    """Resolves a provider profile from the registry, honouring aliases and casing."""
    if not provider_name:
        return None, {}
    clean_name = str(provider_name).strip().lower()
    canonical = PROVIDER_ALIASES.get(clean_name, clean_name)
    if canonical in registry:
        return canonical, dict(registry[canonical])
    for key, profile in registry.items():
        if key.lower() == canonical:
            return key, dict(profile)
    return canonical, {}


def find_provider_by_endpoint(
    registry: dict[str, dict[str, Any]],
    endpoint: str | None,
) -> tuple[str | None, dict[str, Any]]:
    """Finds the provider profile whose endpoint URL prefixes ``endpoint``."""
    if not endpoint:
        return None, {}
    clean_ep = endpoint.rstrip("/").lower()
    for name, profile in registry.items():
        profile_ep = (profile.get("endpoint") or "").rstrip("/").lower()
        if profile_ep and (clean_ep == profile_ep or clean_ep.startswith(profile_ep)):
            return name, dict(profile)
    return None, {}


def _provider_prefix(model: str) -> str | None:
    """Returns the ``provider`` part of a ``provider/model`` string, if present."""
    provider, separator, _ = model.partition("/")
    return provider if separator else None


def _entry_from_string(name: str, model: str, registry: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Builds a listing entry for a bare model id, described by its provider profile."""
    provider_name, profile = resolve_provider(registry, _provider_prefix(model))
    return {
        "name": name,
        "model": model,
        "provider": provider_name or "",
        "description": profile.get("description", f"Model {model}"),
    }


def _entry_from_mapping(name: str, spec: dict[str, Any]) -> dict[str, Any]:
    """Builds a listing entry for a fully-specified mapping (endpoint/keys included)."""
    model = str(spec.get("model") or name).strip()
    return {
        "name": name,
        "model": model,
        "provider": spec.get("provider") or _provider_prefix(model) or "",
        "endpoint": spec.get("endpoint"),
        "api_key": spec.get("api_key"),
        "description": spec.get("description", f"Model {model}"),
    }


def _configured_models(
    config: dict[str, Any], registry: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Normalizes ``config['models']`` given either as a list or as an alias mapping."""
    cfg_models = config.get("models")
    if isinstance(cfg_models, list):
        pairs: list[tuple[str | None, Any]] = [(None, item) for item in cfg_models]
    elif isinstance(cfg_models, dict):
        pairs = [(str(alias), item) for alias, item in cfg_models.items()]
    else:
        return []

    entries: list[dict[str, Any]] = []
    for alias, item in pairs:
        if isinstance(item, str) and item.strip():
            model = item.strip()
            entries.append(_entry_from_string(alias or model, model, registry))
        elif isinstance(item, dict) and (item.get("model") or item.get("endpoint")):
            name = alias or item.get("name") or item.get("alias") or str(item["model"])
            entries.append(_entry_from_mapping(str(name), item))
    return entries


def _discovered_models(
    config: dict[str, Any], registry: dict[str, dict[str, Any]]
) -> list[dict[str, Any]]:
    """Falls back to the configured default model plus each provider's default model."""
    entries: list[dict[str, Any]] = []
    current = config.get("default_model") or config.get("defaultModel") or config.get("model")
    if current and str(current).strip():
        model = str(current).strip()
        entries.append(
            {
                "name": model,
                "model": model,
                "provider": config.get("provider") or _provider_prefix(model) or "",
                "description": f"Configured default model ({model})",
            }
        )

    for provider, profile in sorted(registry.items()):
        default_model = profile.get("default_model")
        if default_model:
            entries.append(
                {
                    "name": f"{provider}/{default_model}",
                    "model": default_model,
                    "provider": provider,
                    "description": f"{provider.title()} default ({default_model})",
                }
            )
    return entries


def get_available_models_from_config(
    config: dict[str, Any],
    registry: dict[str, dict[str, Any]] | None = None,
) -> list[dict[str, Any]]:
    """Extracts a structured list of available models from config and the provider registry.

    Supports:
    1. ``config['models']`` as a list: ``["gpt-4o", "claude-3-5-sonnet", ...]``
    2. ``config['models']`` as a dict: ``{"fast": "gemini-2.5-flash", "smart": {"model": "gpt-4o"}}``
    3. Fallback: ``config['model']`` plus each provider's ``default_model``.
    """
    resolved = load_builtin_providers() if registry is None else registry
    return _configured_models(config, resolved) or _discovered_models(config, resolved)


@dataclass(frozen=True)
class ModelRequest:
    """A model specification after alias resolution, before provider defaults are applied."""

    model: str | None = None
    provider: str | None = None
    endpoint: str | None = None
    api_key: str | None = None
    headers: dict[str, Any] = field(default_factory=dict)


def _follow_config_alias(spec: str | dict[str, Any], config: dict[str, Any]) -> Any:
    """Replaces a spec that names an alias in ``config['models']`` with its definition."""
    cfg_models = config.get("models")
    if isinstance(spec, str) and isinstance(cfg_models, dict) and spec in cfg_models:
        return cfg_models[spec]
    return spec


def parse_model_spec(
    spec: str | dict[str, Any],
    config: dict[str, Any],
    registry: dict[str, dict[str, Any]],
) -> ModelRequest:
    """Parses a bare id, a ``provider/model`` pair, a config alias, or a mapping."""
    resolved = _follow_config_alias(spec, config)

    if isinstance(resolved, dict):
        headers = resolved.get("headers")
        return ModelRequest(
            model=resolved.get("model"),
            provider=resolved.get("provider"),
            endpoint=resolved.get("endpoint"),
            api_key=resolved.get("api_key"),
            headers=dict(headers) if isinstance(headers, dict) else {},
        )

    raw = str(resolved).strip()
    provider = _provider_prefix(raw)
    if provider and provider.lower() in registry:
        return ModelRequest(model=raw[len(provider) + 1 :], provider=provider.lower())
    return ModelRequest(model=raw)


def resolve_api_key(
    request: ModelRequest,
    profile: dict[str, Any],
    config: dict[str, Any],
) -> str | None:
    """Finds an API key: explicit spec, then provider env vars, then well-known env vars."""
    if request.api_key:
        return request.api_key
    for env_key in [*(profile.get("env_keys") or []), *FALLBACK_ENV_KEYS]:
        if value := os.getenv(env_key):
            return value
    return config.get("api_key")


@dataclass(frozen=True)
class Connection:
    """Everything needed to construct a client, after the registry has been applied."""

    model: str
    endpoint: str | None = None
    provider: str | None = None
    api_key: str | None = None
    headers: dict[str, Any] = field(default_factory=dict)
    reasoning_format: str = "standard"
    cache_strategy: str = "implicit"

    @property
    def is_offline(self) -> bool:
        """True when no endpoint could be resolved, so no HTTP client is possible."""
        return not self.endpoint


def resolve_connection(
    spec: str | dict[str, Any],
    config: dict[str, Any] | None = None,
    registry: dict[str, dict[str, Any]] | None = None,
) -> Connection:
    """Resolves a model specification into concrete connection settings.

    ``endpoint`` may be ``None``, which means the caller has no way to reach a
    provider and should fall back to an offline model. Callers that want to show
    what was selected (the CLI, for instance) can inspect the result before
    handing it to :func:`build_model_from_spec`.
    """
    cfg = dict(config or {})
    resolved_registry = load_builtin_providers() if registry is None else registry
    request = parse_model_spec(spec, cfg, resolved_registry)

    provider_name, profile = resolve_provider(
        resolved_registry, request.provider or cfg.get("provider")
    )

    endpoint = request.endpoint
    if not endpoint and profile.get("endpoint"):
        endpoint = profile.get("endpoint")
    if not endpoint:
        endpoint = cfg.get("endpoint") or os.getenv("MODEL_ENDPOINT")

    if not profile and endpoint:
        detected_name, detected_profile = find_provider_by_endpoint(resolved_registry, endpoint)
        if detected_profile:
            provider_name, profile = detected_name, detected_profile

    headers = {**(profile.get("headers") or {}), **(cfg.get("headers") or {})}
    headers.update(request.headers)

    return Connection(
        model=(
            request.model
            or cfg.get("default_model")
            or cfg.get("defaultModel")
            or cfg.get("model")
            or os.getenv("MODEL_NAME")
            or profile.get("default_model", LAST_RESORT_MODEL)
        ),
        endpoint=endpoint,
        provider=provider_name or profile.get("name"),
        api_key=resolve_api_key(request, profile, cfg),
        headers=headers,
        reasoning_format=profile.get("reasoning_format", "standard"),
        cache_strategy=profile.get("cache_strategy", "implicit"),
    )


def build_model_from_spec(
    spec: str | dict[str, Any],
    config: dict[str, Any] | None = None,
    registry: dict[str, dict[str, Any]] | None = None,
    cache_ttl: str | None = None,
    reasoning_effort: str | None = None,
) -> Model:
    """Instantiates a :class:`~core.protocols.Model` from a name, alias, or mapping specification."""
    # Imported here because core.model re-exports this module's helpers; a module-level
    # import would close the cycle.
    from core.model import HttpChatModel, MockModel

    if not spec:
        raise ValueError("Model specification cannot be empty.")
    if not isinstance(spec, (str, dict)):
        raise ValueError(f"Expected model spec as str or dict, got {type(spec).__name__}")

    connection = resolve_connection(spec, config, registry)
    if connection.is_offline:
        return MockModel([("Mock response (offline mode)", [])])

    return HttpChatModel(
        base_url=connection.endpoint,
        model=connection.model,
        api_key=connection.api_key,
        headers=connection.headers,
        cache_ttl=cache_ttl,
        reasoning_effort=reasoning_effort,
        reasoning_format=connection.reasoning_format,
        cache_strategy=connection.cache_strategy,
        provider=connection.provider,
    )


def resolve_model_spec(
    spec: str,
    available: list[dict[str, Any]],
    registry: dict[str, dict[str, Any]] | None = None,
) -> tuple[bool, str, str]:
    """Resolves a user-supplied model name to a canonical spec.

    Accepts a configured alias, the underlying model id of one, or a provider
    name (``openai`` or ``openai/gpt-4o``) present in the provider registry.

    Returns:
        ``(ok, canonical_spec, error)``
    """
    clean = spec.strip()
    if not clean:
        return False, "", "Model specification cannot be empty."

    registry = registry or {}
    by_alias = {m["name"].lower(): m["name"] for m in available}
    by_model_id = {m.get("model", "").lower(): m["name"] for m in available if m.get("model")}
    lowered = clean.lower()

    if lowered in by_alias:
        return True, by_alias[lowered], ""
    if lowered in by_model_id:
        return True, by_model_id[lowered], ""
    if "/" in clean and clean.split("/", 1)[0].lower() in registry:
        return True, clean, ""
    if lowered in registry:
        return True, clean, ""

    valid = ", ".join(m["name"] for m in available) or "none configured"
    return False, "", f"Unknown model '{spec}'. Available models: {valid}"