"""AWS Signature Version 4 signing plus stdlib AWS credential/region resolution.

The pure ``sigv4_headers`` works against any endpoint/service (Bedrock today,
S3 clones tomorrow). The ``resolve_aws_*`` helpers are AWS-specific: env vars,
then the shared ``~/.aws`` files, then SSO cache + one ``GetRoleCredentials``
call. Everything here uses the standard library only, and the resolvers never
raise outward: any failure resolves to ``None``.

This module imports nothing from ``core.auth`` or ``core.providers`` so that
``core.auth`` can import it for scheme registration without a cycle.
"""

from __future__ import annotations

import configparser
import hashlib
import hmac
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

AWS_REGION_ENV_KEYS: tuple[str, ...] = ("AWS_REGION", "AWS_DEFAULT_REGION")

_SSO_CACHE_GLOB = "*.json"
_SSO_TIMEOUT_S = 15


def _sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _hmac(key: bytes, data: str) -> bytes:
    return hmac.new(key, data.encode("utf-8"), hashlib.sha256).digest()


def _signing_key(secret_key: str, date_stamp: str, region: str, service: str) -> bytes:
    key = ("AWS4" + secret_key).encode("utf-8")
    for value in (date_stamp, region, service, "aws4_request"):
        key = hmac.new(key, value.encode("utf-8"), hashlib.sha256).digest()
    return key


def _quote(value: str) -> str:
    return urllib.parse.quote(value, safe="-_.~")


def _canonical_query_string(query: str) -> str:
    if not query:
        return ""
    pairs = urllib.parse.parse_qsl(query, keep_blank_values=True)
    return "&".join(f"{_quote(k)}={_quote(v)}" for k, v in sorted(pairs))


def sigv4_headers(
    *,
    method: str,
    url: str,
    headers: dict,
    body: bytes,
    access_key: str,
    secret_key: str,
    session_token: str | None,
    region: str,
    service: str,
    amz_date: str,
) -> dict:
    """Returns a NEW headers dict with the AWS SigV4 ``Authorization`` stamp.

    ``headers`` and ``body`` are never mutated. ``host`` (from the URL) and
    ``x-amz-date`` (from ``amz_date``) are added when absent, and
    ``x-amz-security-token`` when a session token is given; every header in
    the returned dict is part of the signature.
    """
    if not access_key or not secret_key:
        raise ValueError("sigv4 signing requires an access key and a secret key")
    if not region or not service:
        raise ValueError("sigv4 signing requires a region and a service")
    if not amz_date:
        raise ValueError("sigv4 signing requires an amz date stamp")

    parts = urllib.parse.urlsplit(url)
    host = parts.hostname or ""
    if parts.port and parts.port not in (80, 443):
        host = f"{host}:{parts.port}"
    canonical_uri = urllib.parse.quote(parts.path or "/", safe="/-_.~")
    canonical_query = _canonical_query_string(parts.query)

    # Reserved stamp keys are dropped case-insensitively first, so a stale
    # "Host"/"X-Amz-Date"/"X-Amz-Security-Token" (any casing) can neither
    # duplicate the stamped headers nor leak into the signature.
    reserved = ("host", "x-amz-date", "x-amz-security-token")
    clean = {k: v for k, v in dict(headers).items() if str(k).lower() not in reserved}
    signed = {str(k).lower(): str(v).strip() for k, v in clean.items()}
    signed.setdefault("host", host)
    signed["x-amz-date"] = amz_date
    if session_token:
        signed["x-amz-security-token"] = session_token
    signed_names = sorted(signed)
    canonical_headers = "".join(f"{name}:{signed[name]}\n" for name in signed_names)

    payload_hash = _sha256_hex(body or b"")
    canonical_request = "\n".join(
        [
            method.upper(),
            canonical_uri,
            canonical_query,
            canonical_headers,
            ";".join(signed_names),
            payload_hash,
        ]
    )
    date_stamp = amz_date[:8]
    scope = f"{date_stamp}/{region}/{service}/aws4_request"
    string_to_sign = "\n".join(
        ["AWS4-HMAC-SHA256", amz_date, scope, _sha256_hex(canonical_request.encode())]
    )
    signature = hmac.new(
        _signing_key(secret_key, date_stamp, region, service),
        string_to_sign.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()

    return {
        **clean,
        "host": signed["host"],
        "x-amz-date": amz_date,
        **({"x-amz-security-token": session_token} if session_token else {}),
        "Authorization": (
            f"AWS4-HMAC-SHA256 Credential={access_key}/{scope}, "
            f"SignedHeaders={';'.join(signed_names)}, Signature={signature}"
        ),
    }


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _amz_timestamp(now: datetime | None = None) -> str:
    return (_utcnow() if now is None else now).strftime("%Y%m%dT%H%M%SZ")


class SigV4Signer:
    """A :class:`RequestSigner` for AWS SigV4 (default service ``bedrock``).

    Options: ``service`` (default ``"bedrock"``), ``region`` (fallback when
    secrets carry none). Secrets: ``access_key``, ``secret_key``,
    ``session_token`` (optional), ``region`` (optional). The timestamp is
    minted per ``sign()`` call, so retries re-sign with a fresh date.
    """

    def sign(
        self,
        *,
        method: str,
        url: str,
        headers: dict,
        body: bytes,
        options: dict,
        secrets: dict,
    ) -> dict:
        opts = options if isinstance(options, dict) else {}
        creds = secrets if isinstance(secrets, dict) else {}
        service = str(opts.get("service") or "bedrock")
        region = creds.get("region") or opts.get("region")
        if not region:
            raise ValueError("sigv4 auth requires a region in secrets or options")
        return sigv4_headers(
            method=method,
            url=url,
            headers=headers,
            body=body,
            access_key=creds.get("access_key") or "",
            secret_key=creds.get("secret_key") or "",
            session_token=creds.get("session_token"),
            region=str(region),
            service=service,
            amz_date=_amz_timestamp(),
        )


def _profile_name(cfg: dict | None, explicit: str | None = None) -> str:
    if explicit:
        return explicit
    if isinstance(cfg, dict) and cfg.get("aws_profile"):
        return str(cfg["aws_profile"])
    return os.getenv("AWS_PROFILE") or "default"


def _read_ini_file(path: Path) -> configparser.ConfigParser | None:
    try:
        if not path.is_file():
            return None
        parser = configparser.ConfigParser()
        parser.read(path, encoding="utf-8")
        return parser
    except Exception:
        return None


def _config_section(profile: str) -> str:
    return "default" if profile == "default" else f"profile {profile}"


def _aws_config_value(profile: str, key: str) -> str | None:
    parser = _read_ini_file(Path.home() / ".aws" / "config")
    if parser is None:
        return None
    try:
        if not parser.has_section(_config_section(profile)):
            return None
        value = parser.get(_config_section(profile), key, fallback=None)
    except Exception:
        return None
    return value.strip() or None if isinstance(value, str) else None


def resolve_aws_region(cfg: dict | None = None) -> str | None:
    """Resolves the AWS region: config, then env, then the shared AWS config file."""
    try:
        config = cfg or {}
        region = config.get("region")
        if isinstance(region, str) and region.strip():
            return region.strip()
        for env_key in AWS_REGION_ENV_KEYS:
            value = os.getenv(env_key)
            if value and value.strip():
                return value.strip()
        return _aws_config_value(_profile_name(config), "region")
    except Exception:
        return None


def _credentials_from_env() -> dict | None:
    access_key = os.getenv("AWS_ACCESS_KEY_ID")
    secret_key = os.getenv("AWS_SECRET_ACCESS_KEY")
    if not access_key or not secret_key:
        return None
    return {
        "access_key": access_key,
        "secret_key": secret_key,
        "session_token": os.getenv("AWS_SESSION_TOKEN") or None,
    }


def _credentials_from_shared_file(profile: str) -> dict | None:
    parser = _read_ini_file(Path.home() / ".aws" / "credentials")
    if parser is None or not parser.has_section(profile):
        return None
    try:
        access_key = parser.get(profile, "aws_access_key_id", fallback=None)
        secret_key = parser.get(profile, "aws_secret_access_key", fallback=None)
    except Exception:
        return None
    if not access_key or not secret_key:
        return None
    try:
        session_token = parser.get(profile, "aws_session_token", fallback=None)
    except Exception:
        session_token = None
    return {
        "access_key": access_key.strip(),
        "secret_key": secret_key.strip(),
        "session_token": session_token.strip() if session_token else None,
    }


def _parse_expiry(raw: object) -> datetime | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed


def _sso_tokens(start_url: str) -> list[str]:
    cache_dir = Path.home() / ".aws" / "sso" / "cache"
    try:
        files = sorted(cache_dir.glob(_SSO_CACHE_GLOB))
    except Exception:
        return []
    tokens: list[str] = []
    for path in files:
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if not isinstance(payload, dict) or payload.get("startUrl") != start_url:
            continue
        token = payload.get("accessToken")
        expiry = _parse_expiry(payload.get("expiresAt"))
        if not token or expiry is None or expiry <= _utcnow():
            continue
        tokens.append(token)
    return tokens


def _credentials_from_sso(profile: str) -> dict | None:
    start_url = _aws_config_value(profile, "sso_start_url")
    sso_region = _aws_config_value(profile, "sso_region")
    account_id = _aws_config_value(profile, "sso_account_id")
    role_name = _aws_config_value(profile, "sso_role_name")
    if not start_url or not sso_region or not account_id or not role_name:
        return None
    tokens = _sso_tokens(start_url)
    if not tokens:
        return None
    query = urllib.parse.urlencode({"account_id": account_id, "role_name": role_name})
    endpoint = f"https://portal.sso.{sso_region}.amazonaws.com/federation/credentials?{query}"
    for token in tokens:
        try:
            request = urllib.request.Request(
                endpoint,
                data=b"",
                headers={"x-amz-sso_bearer_token": token},
                method="POST",
            )
            with urllib.request.urlopen(request, timeout=_SSO_TIMEOUT_S) as resp:
                payload = json.loads(resp.read().decode("utf-8"))
        except Exception:
            continue
        try:
            role = payload["roleCredentials"]
            return {
                "access_key": role["accessKeyId"],
                "secret_key": role["secretAccessKey"],
                "session_token": role.get("sessionToken"),
            }
        except (KeyError, TypeError):
            continue
    return None


def resolve_aws_credentials(profile: str | None = None) -> dict | None:
    """Resolves AWS credentials: env, then shared file, then SSO. Never raises."""
    try:
        name = _profile_name(None, profile)
        return (
            _credentials_from_env()
            or _credentials_from_shared_file(name)
            or _credentials_from_sso(name)
        )
    except Exception:
        return None
