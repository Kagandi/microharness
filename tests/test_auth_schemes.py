import os
import unittest
from unittest.mock import patch

from core.auth import (
    RequestSigner,
    builtin_schemes,
    get_signer,
    register_scheme,
    resolve_auth,
)


class DummySigner:
    def sign(self, *, method, url, headers, body, options, secrets):
        new_headers = dict(headers)
        new_headers["X-Dummy"] = "1"
        return new_headers


class TestSchemeRegistry(unittest.TestCase):
    def test_builtin_signers_satisfy_the_protocol(self):
        """Invariant: every built-in scheme returns a RequestSigner."""
        for name in ("none", "bearer", "header"):
            self.assertIsInstance(get_signer(name), RequestSigner)

    def test_builtin_schemes_lists_exactly_tier_1(self):
        """Invariant: Wave 1 ships exactly none/bearer/header, Wave 3 adds sigv4."""
        self.assertEqual(sorted(builtin_schemes()), ["bearer", "header", "none", "sigv4"])

    def test_sigv4_signer_satisfies_the_protocol(self):
        """Invariant: the sigv4 scheme returns a RequestSigner."""
        from core.auth import get_signer as _get

        self.assertIsInstance(_get("sigv4"), RequestSigner)

    def test_get_unknown_scheme_raises_keyerror(self):
        """Invariant: unknown schemes fail closed with KeyError at lookup time."""
        with self.assertRaises(KeyError):
            get_signer("definitely-not-a-scheme-xyz")

    def test_register_and_get_roundtrip(self):
        """Invariant: a registered signer is returned back by name."""
        dummy = DummySigner()
        register_scheme("test-dummy-unique", dummy)
        self.assertIs(get_signer("test-dummy-unique"), dummy)

    def test_scheme_names_are_case_insensitive(self):
        """Invariant: scheme names are normalized to lowercase."""
        dummy = DummySigner()
        register_scheme("Test-Case-Unique", dummy)
        self.assertIs(get_signer("test-case-unique"), dummy)
        self.assertIs(get_signer("TEST-CASE-UNIQUE"), dummy)
        self.assertIsInstance(get_signer("BEARER"), RequestSigner)


class TestRegisterSchemeForce(unittest.TestCase):
    def test_register_builtin_without_force_raises(self):
        """Invariant: Tier 1 names cannot be replaced without explicit force."""
        for name in ("bearer", "BEARER", "none", "header", "sigv4"):
            with self.assertRaises(ValueError):
                register_scheme(name, DummySigner())

    def test_register_builtin_with_force_replaces_and_restores(self):
        """Invariant: force=True swaps a builtin and the original can be put back."""
        original = get_signer("none")
        dummy = DummySigner()
        try:
            register_scheme("none", dummy, force=True)
            self.assertIs(get_signer("none"), dummy)
        finally:
            register_scheme("none", original, force=True)
        self.assertIs(get_signer("none"), original)


class TestNoneSigner(unittest.TestCase):
    def test_returns_headers_unchanged_as_a_copy(self):
        """Invariant: none returns the headers unchanged but never the same object."""
        signer = get_signer("none")
        headers = {"Content-Type": "application/json"}
        out = signer.sign(
            method="GET",
            url="https://example.com",
            headers=headers,
            body=b"",
            options={},
            secrets={},
        )
        self.assertEqual(out, {"Content-Type": "application/json"})
        self.assertIsNot(out, headers)
        self.assertEqual(headers, {"Content-Type": "application/json"})


class TestBearerSigner(unittest.TestCase):
    def test_sets_authorization_header(self):
        """Invariant: bearer sets exactly Authorization: Bearer <api_key>."""
        signer = get_signer("bearer")
        headers = {"Content-Type": "application/json"}
        secrets = {"api_key": "sk-123"}
        out = signer.sign(
            method="POST",
            url="https://example.com",
            headers=headers,
            body=b"{}",
            options={},
            secrets=secrets,
        )
        self.assertEqual(
            out,
            {
                "Content-Type": "application/json",
                "Authorization": "Bearer sk-123",
            },
        )
        self.assertEqual(headers, {"Content-Type": "application/json"})
        self.assertEqual(secrets, {"api_key": "sk-123"})

    def test_overwrites_stale_authorization(self):
        """Invariant: bearer replaces any pre-existing Authorization value."""
        signer = get_signer("bearer")
        out = signer.sign(
            method="GET",
            url="https://example.com",
            headers={"Authorization": "Bearer old"},
            body=b"",
            options={},
            secrets={"api_key": "new"},
        )
        self.assertEqual(out, {"Authorization": "Bearer new"})

    def test_missing_api_key_raises_valueerror(self):
        """Invariant: bearer without secrets['api_key'] fails instead of sending nothing."""
        signer = get_signer("bearer")
        with self.assertRaises(ValueError):
            signer.sign(
                method="GET",
                url="https://example.com",
                headers={},
                body=b"",
                options={},
                secrets={},
            )

    def test_empty_api_key_raises_valueerror(self):
        """Invariant: an empty api_key is rejected like a missing one."""
        signer = get_signer("bearer")
        with self.assertRaises(ValueError):
            signer.sign(
                method="GET",
                url="https://example.com",
                headers={},
                body=b"",
                options={},
                secrets={"api_key": ""},
            )


class TestHeaderSigner(unittest.TestCase):
    def test_literal_value_source(self):
        """Invariant: literal: uses the rest verbatim, including extra colons."""
        signer = get_signer("header")
        options = {"name": "X-Api-Key", "value_source": "literal:abc:def"}
        out = signer.sign(
            method="GET",
            url="https://example.com",
            headers={},
            body=b"",
            options=options,
            secrets={},
        )
        self.assertEqual(out, {"X-Api-Key": "abc:def"})
        self.assertEqual(
            options, {"name": "X-Api-Key", "value_source": "literal:abc:def"}
        )

    def test_env_value_source(self):
        """Invariant: env:VAR reads the process environment."""
        signer = get_signer("header")
        with patch.dict(os.environ, {"TEST_MH_AUTH_KEY": "env-secret"}):
            out = signer.sign(
                method="GET",
                url="https://example.com",
                headers={},
                body=b"",
                options={"name": "X-Api-Key", "value_source": "env:TEST_MH_AUTH_KEY"},
                secrets={},
            )
        self.assertEqual(out, {"X-Api-Key": "env-secret"})

    def test_config_value_source(self):
        """Invariant: config:field reads the secrets dict."""
        signer = get_signer("header")
        headers = {"Content-Type": "application/json"}
        secrets = {"token": "cfg-secret"}
        out = signer.sign(
            method="GET",
            url="https://example.com",
            headers=headers,
            body=b"",
            options={"name": "X-Api-Key", "value_source": "config:token"},
            secrets=secrets,
        )
        self.assertEqual(
            out, {"Content-Type": "application/json", "X-Api-Key": "cfg-secret"}
        )
        self.assertEqual(headers, {"Content-Type": "application/json"})
        self.assertEqual(secrets, {"token": "cfg-secret"})

    def test_missing_name_raises_valueerror(self):
        """Invariant: header without options['name'] is rejected."""
        signer = get_signer("header")
        with self.assertRaises(ValueError):
            signer.sign(
                method="GET",
                url="https://example.com",
                headers={},
                body=b"",
                options={"value_source": "literal:x"},
                secrets={},
            )

    def test_missing_value_source_raises_valueerror(self):
        """Invariant: header without a value_source is rejected."""
        signer = get_signer("header")
        with self.assertRaises(ValueError):
            signer.sign(
                method="GET",
                url="https://example.com",
                headers={},
                body=b"",
                options={"name": "X-Api-Key"},
                secrets={},
            )

    def test_missing_env_var_raises_valueerror(self):
        """Invariant: env:VAR with VAR unset fails instead of sending an empty header."""
        signer = get_signer("header")
        with patch.dict(os.environ):
            os.environ.pop("TEST_MH_AUTH_MISSING_XYZ", None)
            with self.assertRaises(ValueError):
                signer.sign(
                    method="GET",
                    url="https://example.com",
                    headers={},
                    body=b"",
                    options={
                        "name": "X-Api-Key",
                        "value_source": "env:TEST_MH_AUTH_MISSING_XYZ",
                    },
                    secrets={},
                )

    def test_missing_config_field_raises_valueerror(self):
        """Invariant: config:field absent from secrets fails instead of sending nothing."""
        signer = get_signer("header")
        with self.assertRaises(ValueError):
            signer.sign(
                method="GET",
                url="https://example.com",
                headers={},
                body=b"",
                options={"name": "X-Api-Key", "value_source": "config:absent"},
                secrets={},
            )

    def test_unknown_source_prefix_raises_valueerror(self):
        """Invariant: an unrecognized value_source prefix is rejected."""
        signer = get_signer("header")
        with self.assertRaises(ValueError):
            signer.sign(
                method="GET",
                url="https://example.com",
                headers={},
                body=b"",
                options={"name": "X-Api-Key", "value_source": "vault:foo"},
                secrets={},
            )

    def test_prefixless_value_source_raises_valueerror(self):
        """Invariant: a value_source without a prefix is rejected."""
        signer = get_signer("header")
        with self.assertRaises(ValueError):
            signer.sign(
                method="GET",
                url="https://example.com",
                headers={},
                body=b"",
                options={"name": "X-Api-Key", "value_source": "justastring"},
                secrets={},
            )


class TestResolveAuth(unittest.TestCase):
    def test_empty_profile_defaults_to_bearer(self):
        """Invariant: no auth key means the legacy bearer behavior."""
        self.assertEqual(resolve_auth({}), ("bearer", {}))

    def test_none_auth_defaults_to_bearer(self):
        """Invariant: explicit null auth means the legacy bearer behavior."""
        self.assertEqual(resolve_auth({"auth": None}), ("bearer", {}))

    def test_string_auth_selects_scheme_with_empty_options(self):
        """Invariant: a bare scheme name carries no options."""
        self.assertEqual(resolve_auth({"auth": "none"}), ("none", {}))

    def test_string_auth_is_case_insensitive(self):
        """Invariant: scheme names normalize to lowercase."""
        self.assertEqual(resolve_auth({"auth": "Bearer"}), ("bearer", {}))

    def test_dict_auth_splits_scheme_from_options(self):
        """Invariant: rest keys after scheme become the signer options."""
        profile = {
            "auth": {
                "scheme": "header",
                "name": "X-Api-Key",
                "value_source": "literal:v",
            }
        }
        self.assertEqual(
            resolve_auth(profile),
            ("header", {"name": "X-Api-Key", "value_source": "literal:v"}),
        )
        self.assertEqual(
            profile,
            {
                "auth": {
                    "scheme": "header",
                    "name": "X-Api-Key",
                    "value_source": "literal:v",
                }
            },
        )

    def test_unknown_scheme_does_not_raise_here(self):
        """Invariant: resolve_auth defers unknown-scheme failure to get_signer."""
        self.assertEqual(resolve_auth({"auth": "future-scheme"}), ("future-scheme", {}))
        with self.assertRaises(KeyError):
            get_signer("future-scheme")

    def test_dict_auth_missing_scheme_raises_valueerror(self):
        """Invariant: an auth dict without a scheme name is malformed."""
        with self.assertRaises(ValueError):
            resolve_auth({"auth": {"name": "X-Api-Key"}})

    def test_non_string_non_dict_auth_raises_valueerror(self):
        """Invariant: a non-string, non-dict auth value is malformed."""
        with self.assertRaises(ValueError):
            resolve_auth({"auth": 42})


if __name__ == "__main__":
    unittest.main()
