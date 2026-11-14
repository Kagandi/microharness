"""The declarative provider registry and model-spec resolution.

Provider knowledge lives in ``core/providers.json`` (endpoint, default model,
API-key environment variables, reasoning/cache dialects) and is layered with
user-global, project-local, and inline config. This module resolves a model
specification — a bare id, a ``provider/model`` pair, a config alias, or a
mapping — against those layers and instantiates the client.
"""

from __future__ import annotations

import copy
import fnmatch
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from core.auth import get_signer, resolve_auth
from core.auth_sigv4 import resolve_aws_credentials, resolve_aws_region
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
LAST_RESORT_MODEL = "cohere/north-mini-code:free"


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
    builtin_defaults = {name: dict(profile) for name, profile in registry.items()}

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

    # Migration backfill: a user copy predating a new built-in key (e.g. the
    # bedrock ``auth`` block) gains the ABSENT keys. User-set keys always win.
    # Values are deep-copied so mutating the returned registry (e.g. a nested
    # ``auth`` dict) can never alias the built-in defaults.
    for name, builtin_profile in builtin_defaults.items():
        merged = registry.get(name)
        if isinstance(merged, dict):
            for key, value in builtin_profile.items():
                merged.setdefault(key, copy.deepcopy(value))

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
    """Finds the provider profile whose endpoint URL prefixes ``endpoint``.

    A ``{region}`` placeholder in the profile endpoint (Bedrock) matches any
    concrete region host, so a regional URL is attributed back to the profile.
    """
    if not endpoint:
        return None, {}
    clean_ep = endpoint.rstrip("/").lower()
    for name, profile in registry.items():
        profile_ep = (profile.get("endpoint") or "").rstrip("/").lower()
        if not profile_ep:
            continue
        if "{region}" in profile_ep:
            if fnmatch.fnmatch(clean_ep, profile_ep.replace("{region}", "*")):
                return name, dict(profile)
            continue
        if clean_ep == profile_ep or clean_ep.startswith(profile_ep):
            return name, dict(profile)
    return None, {}


def resolve_region(config: dict[str, Any] | None = None) -> str | None:
    """Resolves the AWS region: config ``region``, the AWS environment, then the shared file."""
    return resolve_aws_region(config)


def _aws_profile_name(config: dict[str, Any] | None = None) -> str:
    """Names the AWS profile for shared-file/SSO lookups (config, then environment)."""
    cfg = config or {}
    if cfg.get("aws_profile"):
        return str(cfg["aws_profile"])
    return os.getenv("AWS_PROFILE") or "default"


def _apply_region(endpoint: str | None, config: dict[str, Any] | None = None) -> str | None:
    """Substitutes ``{region}`` in a regional endpoint template.

    Returns ``None`` (offline) when the template cannot be filled, so no caller
    ever receives a literal ``{region}`` URL.
    """
    if not endpoint or "{region}" not in endpoint:
        return endpoint
    region = resolve_region(config)
    if not region:
        return None
    return endpoint.replace("{region}", region)


def _provider_prefix(model: str) -> str | None:
    """Returns the ``provider`` part of a ``provider/model`` string, if present."""
    provider, separator, _ = model.partition("/")
    return provider if separator else None


def _entry_from_string(
    name: str, model: str, registry: dict[str, dict[str, Any]]
) -> dict[str, Any]:
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


def _normalize_endpoint(value: Any) -> str:
    """Normalizes an endpoint URL for comparison (trailing slash and case ignored)."""
    if not isinstance(value, str):
        return ""
    return value.rstrip("/").lower()


def _canonical_provider_name(name: str | None) -> str | None:
    """Maps a provider name through aliases and casing to its canonical form."""
    if not name:
        return None
    clean = str(name).strip().lower()
    return PROVIDER_ALIASES.get(clean, clean)


def _is_local_endpoint(value: Any) -> bool:
    """Mirrors Connection.is_local for endpoint strings (loopback hosts)."""
    if not isinstance(value, str):
        return False
    lowered = value.lower()
    return "localhost" in lowered or "127.0.0.1" in lowered or "::1" in lowered


def _merge_alias_into_dict_spec(
    spec: str | dict[str, Any], config: dict[str, Any]
) -> str | dict[str, Any]:
    """Merges a config alias entry underneath a dict spec that names the alias.

    Only dict specs whose ``model`` is a non-empty string matching a key in
    ``config['models']`` are merged: a dict entry is laid underneath with the
    spec's non-None values winning (headers merge entry-under-spec), while a
    string entry becomes the spec's model id. Anything else passes through
    untouched, so :func:`_follow_config_alias` string behavior is unchanged.
    """
    if not isinstance(spec, dict):
        return spec
    model_ref = spec.get("model")
    if not isinstance(model_ref, str) or not model_ref:
        return spec
    cfg_models = config.get("models")
    if not isinstance(cfg_models, dict):
        return spec
    entry = cfg_models.get(model_ref)
    if isinstance(entry, str):
        if not entry.strip():
            return spec
        merged_str = dict(spec)
        merged_str["model"] = entry.strip()
        return merged_str
    if not isinstance(entry, dict):
        return spec
    merged: dict[str, Any] = dict(entry)
    for key, value in spec.items():
        if key == "model" or value is None:
            continue
        if key == "headers" and isinstance(value, dict):
            base = merged.get("headers")
            merged["headers"] = {**(base if isinstance(base, dict) else {}), **value}
        else:
            merged[key] = value
    if not isinstance(merged.get("model"), str) or not merged["model"]:
        merged["model"] = model_ref
    return merged


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
    auth_scheme: str = "bearer"
    auth_options: dict = field(default_factory=dict)
    auth_error: str | None = None

    @property
    def is_offline(self) -> bool:
        """True when no endpoint could be resolved, so no HTTP client is possible."""
        return not self.endpoint

    @property
    def is_local(self) -> bool:
        """True when the endpoint targets a local development server without API auth."""
        if not self.endpoint:
            return False
        ep = self.endpoint.lower()
        return "localhost" in ep or "127.0.0.1" in ep or "::1" in ep

    @property
    def missing_credentials(self) -> bool:
        """True when the connection's auth scheme has no usable credentials."""
        if self.is_offline or self.is_local:
            return False
        if self.auth_scheme == "none":
            return False
        if self.auth_scheme == "sigv4":
            try:
                profile = None
                if isinstance(self.auth_options, dict):
                    profile = self.auth_options.get("aws_profile") or self.auth_options.get(
                        "profile"
                    )
                return resolve_aws_credentials(profile) is None
            except Exception:
                return True
        if self.auth_scheme == "header":
            source = None
            if isinstance(self.auth_options, dict):
                source = self.auth_options.get("value_source")
            if isinstance(source, str) and source.startswith("env:"):
                return not bool(os.getenv(source[len("env:") :]))
            return False
        if self.auth_scheme != "bearer":
            return True
        return not bool(self.api_key)


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
    merged_spec = _merge_alias_into_dict_spec(spec, cfg)
    request = parse_model_spec(merged_spec, cfg, resolved_registry)

    provider_name, profile = resolve_provider(
        resolved_registry, request.provider or cfg.get("provider")
    )

    # Endpoint precedence: an explicit request endpoint (flag/env, alias entry,
    # or direct) always wins. Otherwise the profile endpoint and the global
    # config endpoint compete: when both are present but different, the config
    # endpoint is attributed to a provider — it wins when unattributable or
    # when it belongs to the selected provider, else the profile wins.
    profile_endpoint = profile.get("endpoint")
    cfg_endpoint = cfg.get("endpoint") or os.getenv("MODEL_ENDPOINT")
    if request.endpoint:
        endpoint = request.endpoint
    else:
        norm_profile = _normalize_endpoint(profile_endpoint)
        norm_cfg = _normalize_endpoint(cfg_endpoint)
        if norm_profile and norm_cfg and norm_profile != norm_cfg:
            profile_local = _is_local_endpoint(profile_endpoint)
            cfg_local = _is_local_endpoint(cfg_endpoint)
            if cfg_local and not profile_local:
                # A loopback global default never hijacks a cloud provider.
                endpoint = profile_endpoint
            elif profile_local and not cfg_local:
                endpoint = cfg_endpoint
            else:
                attributed, _ = find_provider_by_endpoint(resolved_registry, cfg_endpoint)
                if attributed is None or _canonical_provider_name(
                    attributed
                ) == _canonical_provider_name(provider_name):
                    endpoint = cfg_endpoint
                else:
                    endpoint = profile_endpoint
        elif norm_profile:
            endpoint = profile_endpoint
        else:
            endpoint = cfg_endpoint or None

    if not profile and endpoint:
        detected_name, detected_profile = find_provider_by_endpoint(resolved_registry, endpoint)
        if detected_profile:
            provider_name, profile = detected_name, detected_profile

    headers = {**(profile.get("headers") or {}), **(cfg.get("headers") or {})}
    headers.update(request.headers)

    # Auth precedence: spec-dict auth > config auth > profile auth. The merged
    # spec carries alias-entry auth for dict specs; a string spec naming an
    # alias is followed to its definition (a raw dict spec is unchanged).
    resolved_spec = _follow_config_alias(merged_spec, cfg)
    spec_auth = resolved_spec.get("auth") if isinstance(resolved_spec, dict) else None
    if spec_auth is not None:
        auth_scheme, auth_options = resolve_auth({"auth": spec_auth})
    elif cfg.get("auth") is not None:
        auth_scheme, auth_options = resolve_auth({"auth": cfg.get("auth")})
    else:
        auth_scheme, auth_options = resolve_auth(profile)

    auth_error: str | None = None
    if auth_scheme == "sigv4":
        aws_profile = _aws_profile_name(cfg)
        auth_options = {**auth_options, "aws_profile": aws_profile}
        aws_region = resolve_aws_region(cfg)
        print(
            f"SigV4 auth: AWS profile '{aws_profile}', region '{aws_region or 'unset'}'",
            file=sys.stderr,
        )
        if resolve_aws_credentials(aws_profile) is None:
            auth_error = (
                f"missing AWS credentials for SigV4 auth (provider "
                f"'{provider_name or profile.get('name') or 'unknown'}'): no environment, "
                f"shared-file, or SSO credentials found for profile '{aws_profile}'. "
                f"Run `aws sso login` (or set AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY)."
            )

    return Connection(
        model=(
            request.model
            or cfg.get("default_model")
            or cfg.get("defaultModel")
            or cfg.get("model")
            or os.getenv("MODEL_NAME")
            or profile.get("default_model", LAST_RESORT_MODEL)
        ),
        endpoint=_apply_region(endpoint, cfg),
        provider=provider_name or profile.get("name"),
        api_key=resolve_api_key(request, profile, cfg),
        headers=headers,
        reasoning_format=profile.get("reasoning_format", "standard"),
        cache_strategy=profile.get("cache_strategy", "implicit"),
        auth_scheme=auth_scheme,
        auth_options=auth_options,
        auth_error=auth_error,
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
        message = "Mock response (offline mode)"
        profile = resolve_provider(registry or load_builtin_providers(), connection.provider)[1]
        if "{region}" in str(profile.get("endpoint") or ""):
            message += ": a Bedrock endpoint needs a region (set AWS_REGION or 'region' in config)"
        return MockModel([(message, [])])
    effective_scheme = connection.auth_scheme
    auth_error = connection.auth_error
    # Bearer override only when the profile's own token env var is set (e.g.
    # BEDROCK_API_KEY). A resolved api_key alone is not proof: global configs
    # commonly carry api_key "none" for keyless local endpoints, which must
    # never flip SigV4 to bearer.
    _, override_profile = resolve_provider(
        registry or load_builtin_providers(), connection.provider
    )
    bearer_env_set = any(os.getenv(key) for key in (override_profile.get("env_keys") or []))
    if connection.auth_scheme == "sigv4" and connection.api_key and bearer_env_set:
        print(
            "SigV4 auth: profile token env var present, using bearer auth override",
            file=sys.stderr,
        )
        effective_scheme = "bearer"
        auth_error = None
    try:
        get_signer(effective_scheme)
    except KeyError:
        raise ValueError(f"unknown auth scheme: {effective_scheme!r}") from None
    if auth_error:
        return MockModel([(auth_error, [])])

    if effective_scheme == "sigv4":
        aws_profile = connection.auth_options.get("aws_profile") or _aws_profile_name(config or {})
        auth_secrets: dict[str, Any] = resolve_aws_credentials(aws_profile) or {}
        auth_secrets["region"] = resolve_aws_region(config or {})
    elif connection.api_key:
        auth_secrets = {"api_key": connection.api_key}
    else:
        auth_secrets = {}

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
        auth_scheme=effective_scheme,
        auth_options=connection.auth_options,
        auth_secrets=auth_secrets,
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


def _strict_context_tokens(value: Any) -> int | None:
    """Returns ``value`` only when it is a genuine positive int (never coerced)."""
    if type(value) is int and value > 0:
        return value
    return None


def resolve_model_context_tokens(
    spec: str | dict[str, Any] | None,
    cfg: dict[str, Any] | None = None,
    registry: dict[str, dict[str, Any]] | None = None,
) -> int | None:
    """Resolves the per-model context cap without ever naming a provider.

    Precedence: direct-dict ``max_context_tokens`` > config-alias dict entry
    ``max_context_tokens`` > provider-profile ``context_tokens`` > the global
    triple (``max_context_tokens`` / ``max_tokens`` / ``context_tokens``).
    Each level is strict (``type(x) is int and x > 0``); anything else skips
    the level. Returns ``None`` when nothing is declared.
    """
    config = cfg if isinstance(cfg, dict) else {}
    resolved_registry = load_builtin_providers() if registry is None else registry

    if isinstance(spec, dict):
        hit = _strict_context_tokens(spec.get("max_context_tokens"))
        if hit is not None:
            return hit
        # Startup dicts carry the alias/model id under "model": fall through
        # to the alias entry so --model <alias> honors per-model caps.
        model_ref = spec.get("model")
        if isinstance(model_ref, str):
            models = config.get("models")
            if isinstance(models, dict):
                entry = models.get(model_ref)
                if isinstance(entry, dict):
                    hit = _strict_context_tokens(entry.get("max_context_tokens"))
                    if hit is not None:
                        return hit

    if isinstance(spec, str):
        models = config.get("models")
        if isinstance(models, dict):
            entry = models.get(spec)
            if isinstance(entry, dict):
                hit = _strict_context_tokens(entry.get("max_context_tokens"))
                if hit is not None:
                    return hit

    if isinstance(spec, (str, dict)):
        try:
            request = parse_model_spec(spec, config, resolved_registry)
        except Exception:
            request = None
        if request is not None:
            provider_name = request.provider or config.get("provider")
            if provider_name is None and isinstance(spec, str):
                provider_name = spec.strip() or None
            _, profile = resolve_provider(resolved_registry, provider_name)
            hit = _strict_context_tokens(profile.get("context_tokens"))
            if hit is not None:
                return hit

    for key in ("max_context_tokens", "max_tokens", "context_tokens"):
        hit = _strict_context_tokens(config.get(key))
        if hit is not None:
            return hit
    return None
