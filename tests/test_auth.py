import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from core.auth import (
    RequestSigner,
    builtin_schemes,
    get_signer,
    load_auth_plugins,
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
        """Invariant: Wave 1 ships exactly none/bearer/header."""
        self.assertEqual(sorted(builtin_schemes()), ["bearer", "header", "none"])

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


def _write_plugin(tmpdir: str, filename: str, content: str) -> str:
    path = Path(tmpdir) / filename
    path.write_text(content, encoding="utf-8")
    return str(path)


def _load_with_stderr(global_dir) -> tuple[dict, str]:
    buf = io.StringIO()
    with contextlib.redirect_stderr(buf):
        result = load_auth_plugins(global_dir)
    return result, buf.getvalue()


class TestLoadAuthPlugins(unittest.TestCase):
    def test_none_and_missing_dir_return_empty(self):
        """Invariant: None or a missing dir loads nothing without error."""
        self.assertEqual(_load_with_stderr(None)[0], {})
        self.assertEqual(
            _load_with_stderr(os.path.join(tempfile.gettempdir(), "mh-no-such-dir-xyz"))[
                0
            ],
            {},
        )

    def test_file_path_as_dir_returns_empty(self):
        """Invariant: a global_dir pointing at a file (not a dir) loads nothing."""
        with tempfile.NamedTemporaryFile(suffix=".py") as tmp:
            result, _ = _load_with_stderr(tmp.name)
            self.assertEqual(result, {})

    def test_happy_path_sign_function(self):
        """Invariant: SCHEME_NAME + sign() plugin signs via get_signer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = _write_plugin(
                tmpdir,
                "myplug.py",
                "SCHEME_NAME = 'test-wave2-happy'\n"
                "def sign(*, method, url, headers, body, options, secrets):\n"
                "    out = dict(headers)\n"
                "    out['X-Custom'] = 'yes:' + secrets.get('token', '')\n"
                "    return out\n",
            )
            result, stderr = _load_with_stderr(tmpdir)
            self.assertEqual(result, {"test-wave2-happy": plugin_path})
            self.assertEqual(stderr, "")
            signer = get_signer("test-wave2-happy")
            self.assertIsInstance(signer, RequestSigner)
            headers = {"Content-Type": "application/json"}
            out = signer.sign(
                method="GET",
                url="https://example.com",
                headers=headers,
                body=b"",
                options={},
                secrets={"token": "abc"},
            )
            self.assertEqual(
                out, {"Content-Type": "application/json", "X-Custom": "yes:abc"}
            )
            self.assertEqual(headers, {"Content-Type": "application/json"})

    def test_happy_path_signer_object(self):
        """Invariant: SCHEME_NAME + SIGNER object plugin signs via get_signer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = _write_plugin(
                tmpdir,
                "objplug.py",
                "SCHEME_NAME = 'test-wave2-signer-obj'\n"
                "class _S:\n"
                "    def sign(self, *, method, url, headers, body, options, secrets):\n"
                "        out = dict(headers)\n"
                "        out['X-Obj'] = 'obj'\n"
                "        return out\n"
                "SIGNER = _S()\n",
            )
            result, _ = _load_with_stderr(tmpdir)
            self.assertEqual(result, {"test-wave2-signer-obj": plugin_path})
            out = get_signer("test-wave2-signer-obj").sign(
                method="GET",
                url="https://example.com",
                headers={},
                body=b"",
                options={},
                secrets={},
            )
            self.assertEqual(out, {"X-Obj": "obj"})

    def test_collision_with_builtin_keeps_builtin(self):
        """Invariant: a plugin naming a builtin never overrides it."""
        original = get_signer("bearer")
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = _write_plugin(
                tmpdir,
                "evil.py",
                "SCHEME_NAME = 'bearer'\n"
                "def sign(*, method, url, headers, body, options, secrets):\n"
                "    return {'X-Evil': '1'}\n",
            )
            result, stderr = _load_with_stderr(tmpdir)
            self.assertEqual(result, {})
            self.assertIn(plugin_path, stderr)
            self.assertIs(get_signer("bearer"), original)

    def test_malformed_plugin_skipped_with_warning(self):
        """Invariant: a plugin raising at import is skipped, loader keeps going."""
        with tempfile.TemporaryDirectory() as tmpdir:
            bad_path = _write_plugin(tmpdir, "bad.py", "raise RuntimeError('boom')\n")
            good_path = _write_plugin(
                tmpdir,
                "good.py",
                "SCHEME_NAME = 'test-wave2-after-bad'\n"
                "def sign(*, method, url, headers, body, options, secrets):\n"
                "    return dict(headers)\n",
            )
            result, stderr = _load_with_stderr(tmpdir)
            self.assertEqual(result, {"test-wave2-after-bad": good_path})
            self.assertIn(bad_path, stderr)

    def test_missing_scheme_name_skipped(self):
        """Invariant: a plugin without SCHEME_NAME is skipped with a warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = _write_plugin(
                tmpdir,
                "noscheme.py",
                "def sign(*, method, url, headers, body, options, secrets):\n"
                "    return dict(headers)\n",
            )
            result, stderr = _load_with_stderr(tmpdir)
            self.assertEqual(result, {})
            self.assertIn(plugin_path, stderr)

    def test_empty_scheme_name_skipped(self):
        """Invariant: an empty SCHEME_NAME is rejected like a missing one."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = _write_plugin(
                tmpdir,
                "empty.py",
                "SCHEME_NAME = ''\n"
                "def sign(*, method, url, headers, body, options, secrets):\n"
                "    return dict(headers)\n",
            )
            result, stderr = _load_with_stderr(tmpdir)
            self.assertEqual(result, {})
            self.assertIn(plugin_path, stderr)

    def test_missing_sign_callable_skipped(self):
        """Invariant: SCHEME_NAME without sign()/SIGNER is skipped with warning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            plugin_path = _write_plugin(
                tmpdir, "nosign.py", "SCHEME_NAME = 'test-wave2-nosign'\n"
            )
            result, stderr = _load_with_stderr(tmpdir)
            self.assertEqual(result, {})
            self.assertIn(plugin_path, stderr)

    def test_non_py_files_ignored(self):
        """Invariant: non-.py files never load and never warn."""
        with tempfile.TemporaryDirectory() as tmpdir:
            txt_path = _write_plugin(tmpdir, "notes.txt", "SCHEME_NAME = 'x'\n")
            result, stderr = _load_with_stderr(tmpdir)
            self.assertEqual(result, {})
            self.assertEqual(stderr, "")
            self.assertTrue(Path(txt_path).exists())

    def test_underscore_and_init_files_skipped_silently(self):
        """Invariant: __init__.py and _-prefixed files are not plugins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_plugin(
                tmpdir,
                "__init__.py",
                "SCHEME_NAME = 'test-wave2-init'\n"
                "def sign(*, method, url, headers, body, options, secrets):\n"
                "    return dict(headers)\n",
            )
            _write_plugin(
                tmpdir,
                "_private.py",
                "SCHEME_NAME = 'test-wave2-private'\n"
                "def sign(*, method, url, headers, body, options, secrets):\n"
                "    return dict(headers)\n",
            )
            result, stderr = _load_with_stderr(tmpdir)
            self.assertEqual(result, {})
            self.assertEqual(stderr, "")

    def test_first_sorted_file_wins_collision(self):
        """Invariant: two files claiming one scheme -> sorted-first wins."""
        with tempfile.TemporaryDirectory() as tmpdir:
            a_path = _write_plugin(
                tmpdir,
                "a_first.py",
                "SCHEME_NAME = 'test-wave2-dup'\n"
                "def sign(*, method, url, headers, body, options, secrets):\n"
                "    out = dict(headers)\n"
                "    out['X-Winner'] = 'a'\n"
                "    return out\n",
            )
            b_path = _write_plugin(
                tmpdir,
                "b_second.py",
                "SCHEME_NAME = 'test-wave2-dup'\n"
                "def sign(*, method, url, headers, body, options, secrets):\n"
                "    out = dict(headers)\n"
                "    out['X-Winner'] = 'b'\n"
                "    return out\n",
            )
            result, stderr = _load_with_stderr(tmpdir)
            self.assertEqual(result, {"test-wave2-dup": a_path})
            self.assertIn(b_path, stderr)
            out = get_signer("test-wave2-dup").sign(
                method="GET",
                url="https://example.com",
                headers={},
                body=b"",
                options={},
                secrets={},
            )
            self.assertEqual(out, {"X-Winner": "a"})


if __name__ == "__main__":
    unittest.main()
