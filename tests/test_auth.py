import contextlib
import io
import os
import tempfile
import unittest
import unittest.mock
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


class TestSigV4Headers(unittest.TestCase):
    def test_aws_docs_known_answer_vector(self):
        """Invariant: the official get-vanilla case yields its exact Authorization value."""
        from core.auth_sigv4 import sigv4_headers

        out = sigv4_headers(
            method="GET",
            url="https://example.amazonaws.com/",
            headers={},
            body=b"",
            access_key="AKIDEXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token=None,
            region="us-east-1",
            service="service",
            amz_date="20150830T123600Z",
        )
        self.assertEqual(
            out["Authorization"],
            "AWS4-HMAC-SHA256 "
            "Credential=AKIDEXAMPLE/20150830/us-east-1/service/aws4_request, "
            "SignedHeaders=host;x-amz-date, "
            "Signature=5fa00fa31553b73ebf1942676e86291e8372ff2a2260956d9b8aae1d763fbf31",
        )

    def test_vector_carries_the_amz_date_header(self):
        """Invariant: signing stamps the x-amz-date header it signed."""
        from core.auth_sigv4 import sigv4_headers

        out = sigv4_headers(
            method="GET",
            url="https://iam.amazonaws.com/",
            headers={},
            body=b"",
            access_key="AKIDEXAMPLE",
            secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY",
            session_token=None,
            region="us-east-1",
            service="iam",
            amz_date="20150830T123600Z",
        )
        self.assertEqual(out["x-amz-date"], "20150830T123600Z")

    def test_inputs_are_never_mutated(self):
        """Invariant: sigv4_headers returns a new dict and leaves inputs alone."""
        from core.auth_sigv4 import sigv4_headers

        headers = {"Content-Type": "application/json"}
        out = sigv4_headers(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1/chat/completions",
            headers=headers,
            body=b"{}",
            access_key="AKID",
            secret_key="SECRET",
            session_token=None,
            region="us-east-1",
            service="bedrock",
            amz_date="20260101T000000Z",
        )
        self.assertIsNot(out, headers)
        self.assertEqual(headers, {"Content-Type": "application/json"})
        self.assertIn("Authorization", out)
        self.assertIn("Content-Type", out)

    def test_session_token_is_signed_and_sent(self):
        """Invariant: a session token adds the security-token header to the signature."""
        from core.auth_sigv4 import sigv4_headers

        out = sigv4_headers(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
            headers={},
            body=b"{}",
            access_key="AKID",
            secret_key="SECRET",
            session_token="TOKEN",
            region="us-east-1",
            service="bedrock",
            amz_date="20260101T000000Z",
        )
        self.assertEqual(out["x-amz-security-token"], "TOKEN")
        credential_scope = out["Authorization"].split("Credential=")[1]
        self.assertTrue(credential_scope.startswith("AKID/20260101/us-east-1/bedrock/aws4_request"))

    def test_wrong_secret_changes_the_signature(self):
        """Invariant: the signature actually depends on the secret key."""
        from core.auth_sigv4 import sigv4_headers

        kwargs = dict(
            method="GET",
            url="https://example.amazonaws.com/",
            headers={},
            body=b"",
            access_key="AKIDEXAMPLE",
            session_token=None,
            region="us-east-1",
            service="service",
            amz_date="20150830T123600Z",
        )
        good = sigv4_headers(secret_key="wJalrXUtnFEMI/K7MDENG+bPxRfiCYEXAMPLEKEY", **kwargs)
        bad = sigv4_headers(secret_key="wrong", **kwargs)
        self.assertNotEqual(good["Authorization"], bad["Authorization"])

    def test_missing_credential_raises_valueerror(self):
        """Invariant: signing without a key pair fails instead of sending garbage."""
        from core.auth_sigv4 import sigv4_headers

        with self.assertRaises(ValueError):
            sigv4_headers(
                method="GET",
                url="https://example.com/",
                headers={},
                body=b"",
                access_key="",
                secret_key="",
                session_token=None,
                region="us-east-1",
                service="iam",
                amz_date="20150830T123600Z",
            )


class TestSigV4Signer(unittest.TestCase):
    def test_sign_uses_service_and_region_from_options_and_secrets(self):
        """Invariant: SigV4Signer defaults to bedrock, region from secrets."""
        from core.auth_sigv4 import SigV4Signer

        signer = SigV4Signer()
        out = signer.sign(
            method="POST",
            url="https://bedrock-runtime.us-east-1.amazonaws.com/openai/v1",
            headers={},
            body=b"{}",
            options={},
            secrets={"access_key": "AKID", "secret_key": "SECRET", "region": "us-east-1"},
        )
        self.assertTrue(out["Authorization"].startswith("AWS4-HMAC-SHA256 "))
        self.assertIn("/us-east-1/bedrock/aws4_request", out["Authorization"])

    def test_region_may_come_from_options(self):
        """Invariant: options region fills in when secrets carry no region."""
        from core.auth_sigv4 import SigV4Signer

        signer = SigV4Signer()
        out = signer.sign(
            method="POST",
            url="https://example.com/",
            headers={},
            body=b"{}",
            options={"region": "eu-west-1", "service": "execute-api"},
            secrets={"access_key": "AKID", "secret_key": "SECRET"},
        )
        self.assertIn("/eu-west-1/execute-api/aws4_request", out["Authorization"])

    def test_missing_secrets_raise_valueerror(self):
        """Invariant: SigV4Signer without credentials fails closed."""
        from core.auth_sigv4 import SigV4Signer

        signer = SigV4Signer()
        with self.assertRaises(ValueError):
            signer.sign(
                method="GET",
                url="https://example.com/",
                headers={},
                body=b"",
                options={},
                secrets={},
            )

    def test_missing_region_raises_valueerror(self):
        """Invariant: SigV4Signer without a region fails closed."""
        from core.auth_sigv4 import SigV4Signer

        signer = SigV4Signer()
        with self.assertRaises(ValueError):
            signer.sign(
                method="GET",
                url="https://example.com/",
                headers={},
                body=b"",
                options={},
                secrets={"access_key": "AKID", "secret_key": "SECRET"},
            )


def _clear_aws_env(home: str) -> dict:
    return {
        "HOME": home,
        "AWS_ACCESS_KEY_ID": "",
        "AWS_SECRET_ACCESS_KEY": "",
        "AWS_SESSION_TOKEN": "",
        "AWS_REGION": "",
        "AWS_DEFAULT_REGION": "",
        "AWS_PROFILE": "",
    }


class TestResolveAwsRegion(unittest.TestCase):
    def test_config_region_wins_over_everything(self):
        """Invariant: an explicit config region outranks env and shared files."""
        from core.auth_sigv4 import resolve_aws_region

        with tempfile.TemporaryDirectory() as home:
            env = _clear_aws_env(home)
            env["AWS_REGION"] = "eu-west-1"
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(resolve_aws_region({"region": "ap-south-1"}), "ap-south-1")

    def test_aws_region_env_beats_default_region_env(self):
        """Invariant: AWS_REGION outranks AWS_DEFAULT_REGION."""
        from core.auth_sigv4 import resolve_aws_region

        with tempfile.TemporaryDirectory() as home:
            env = _clear_aws_env(home)
            env.update({"AWS_REGION": "eu-west-1", "AWS_DEFAULT_REGION": "us-east-1"})
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(resolve_aws_region({}), "eu-west-1")

    def test_default_region_env_is_the_last_env_fallback(self):
        """Invariant: AWS_DEFAULT_REGION is honoured when AWS_REGION is unset."""
        from core.auth_sigv4 import resolve_aws_region

        with tempfile.TemporaryDirectory() as home:
            env = _clear_aws_env(home)
            env["AWS_DEFAULT_REGION"] = "us-east-2"
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(resolve_aws_region({}), "us-east-2")

    def test_shared_config_default_profile_region(self):
        """Invariant: ~/.aws/config [default] region fills the chain."""
        from core.auth_sigv4 import resolve_aws_region

        with tempfile.TemporaryDirectory() as home:
            aws_dir = Path(home) / ".aws"
            aws_dir.mkdir()
            (aws_dir / "config").write_text("[default]\nregion = us-west-2\n", encoding="utf-8")
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                self.assertEqual(resolve_aws_region({}), "us-west-2")

    def test_shared_config_named_profile_region(self):
        """Invariant: aws_profile selects the [profile X] section region."""
        from core.auth_sigv4 import resolve_aws_region

        with tempfile.TemporaryDirectory() as home:
            aws_dir = Path(home) / ".aws"
            aws_dir.mkdir()
            (aws_dir / "config").write_text(
                "[default]\nregion = us-west-2\n[profile myprof]\nregion = eu-central-1\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                self.assertEqual(
                    resolve_aws_region({"aws_profile": "myprof"}), "eu-central-1"
                )

    def test_aws_profile_env_selects_the_section(self):
        """Invariant: AWS_PROFILE names the config section when cfg has no aws_profile."""
        from core.auth_sigv4 import resolve_aws_region

        with tempfile.TemporaryDirectory() as home:
            aws_dir = Path(home) / ".aws"
            aws_dir.mkdir()
            (aws_dir / "config").write_text(
                "[profile myprof]\nregion = eu-central-1\n", encoding="utf-8"
            )
            env = _clear_aws_env(home)
            env["AWS_PROFILE"] = "myprof"
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(resolve_aws_region({}), "eu-central-1")

    def test_missing_config_file_yields_none(self):
        """Invariant: no shared config anywhere means no region, not an error."""
        from core.auth_sigv4 import resolve_aws_region

        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                self.assertIsNone(resolve_aws_region({}))


def _sso_config_text() -> str:
    return (
        "[profile sso-prof]\n"
        "sso_start_url = https://my.awsapps.com/start\n"
        "sso_region = us-east-1\n"
        "sso_account_id = 123456789012\n"
        "sso_role_name = MyRole\n"
        "region = us-east-1\n"
    )


def _write_sso_fixture(home: str, tokens: dict) -> None:
    aws_dir = Path(home) / ".aws"
    (aws_dir / "sso" / "cache").mkdir(parents=True, exist_ok=True)
    (aws_dir / "config").write_text(_sso_config_text(), encoding="utf-8")
    for name, payload in tokens.items():
        (aws_dir / "sso" / "cache" / name).write_text(payload, encoding="utf-8")


def _role_response(access="AKID-SSO", secret="SECRET-SSO", token="TOKEN-SSO"):
    resp = unittest.mock.MagicMock()
    resp.read.return_value = (
        '{"roleCredentials": {"accessKeyId": "%s", "secretAccessKey": "%s", "sessionToken": "%s"}}'
        % (access, secret, token)
    ).encode("utf-8")
    resp.__enter__.return_value = resp
    return resp


class TestResolveAwsCredentials(unittest.TestCase):
    def test_env_credentials_win(self):
        """Invariant: AWS_ACCESS_KEY_ID/SECRET resolve without touching files."""
        from core.auth_sigv4 import resolve_aws_credentials

        with tempfile.TemporaryDirectory() as home:
            env = _clear_aws_env(home)
            env.update(
                {
                    "AWS_ACCESS_KEY_ID": "AKID-ENV",
                    "AWS_SECRET_ACCESS_KEY": "SECRET-ENV",
                    "AWS_SESSION_TOKEN": "TOKEN-ENV",
                }
            )
            with patch.dict(os.environ, env, clear=True):
                self.assertEqual(
                    resolve_aws_credentials(),
                    {
                        "access_key": "AKID-ENV",
                        "secret_key": "SECRET-ENV",
                        "session_token": "TOKEN-ENV",
                    },
                )

    def test_env_without_session_token_yields_none_token(self):
        """Invariant: a missing session token resolves to None, not an empty string."""
        from core.auth_sigv4 import resolve_aws_credentials

        with tempfile.TemporaryDirectory() as home:
            env = _clear_aws_env(home)
            env.update(
                {"AWS_ACCESS_KEY_ID": "AKID", "AWS_SECRET_ACCESS_KEY": "SECRET"}
            )
            with patch.dict(os.environ, env, clear=True):
                creds = resolve_aws_credentials()
                self.assertIsNotNone(creds)
                assert creds is not None
                self.assertIsNone(creds["session_token"])

    def test_shared_credentials_file(self):
        """Invariant: ~/.aws/credentials supplies the key pair for a named profile."""
        from core.auth_sigv4 import resolve_aws_credentials

        with tempfile.TemporaryDirectory() as home:
            aws_dir = Path(home) / ".aws"
            aws_dir.mkdir()
            (aws_dir / "credentials").write_text(
                "[myprof]\naws_access_key_id = AKID-FILE\naws_secret_access_key = SECRET-FILE\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                self.assertEqual(
                    resolve_aws_credentials("myprof"),
                    {
                        "access_key": "AKID-FILE",
                        "secret_key": "SECRET-FILE",
                        "session_token": None,
                    },
                )

    def test_nothing_anywhere_yields_none(self):
        """Invariant: no env, no files, no SSO means None, never an exception."""
        from core.auth_sigv4 import resolve_aws_credentials

        with tempfile.TemporaryDirectory() as home:
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                self.assertIsNone(resolve_aws_credentials())

    def test_sso_valid_token_fetches_role_credentials(self):
        """Invariant: an unexpired SSO cache token drives one GetRoleCredentials call."""
        from core.auth_sigv4 import resolve_aws_credentials

        with tempfile.TemporaryDirectory() as home:
            _write_sso_fixture(
                home,
                {
                    "valid.json": '{"startUrl": "https://my.awsapps.com/start", '
                    '"accessToken": "TOKEN-VALID", "expiresAt": "2030-01-01T00:00:00Z"}'
                },
            )
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                with patch(
                    "urllib.request.urlopen", return_value=_role_response()
                ) as mock_urlopen:
                    creds = resolve_aws_credentials("sso-prof")
        self.assertEqual(
            creds,
            {
                "access_key": "AKID-SSO",
                "secret_key": "SECRET-SSO",
                "session_token": "TOKEN-SSO",
            },
        )
        (request,), _ = mock_urlopen.call_args
        self.assertIn("account_id=123456789012", request.full_url)
        self.assertIn("role_name=MyRole", request.full_url)
        self.assertEqual(request.get_header("X-amz-sso_bearer_token"), "TOKEN-VALID")

    def test_sso_expired_token_is_skipped_for_a_valid_one(self):
        """Invariant: an expired cache entry never authenticates the role call."""
        from core.auth_sigv4 import resolve_aws_credentials

        with tempfile.TemporaryDirectory() as home:
            _write_sso_fixture(
                home,
                {
                    "a-expired.json": '{"startUrl": "https://my.awsapps.com/start", '
                    '"accessToken": "TOKEN-OLD", "expiresAt": "2000-01-01T00:00:00Z"}',
                    "b-valid.json": '{"startUrl": "https://my.awsapps.com/start", '
                    '"accessToken": "TOKEN-VALID", "expiresAt": "2030-01-01T00:00:00Z"}',
                },
            )
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                with patch(
                    "urllib.request.urlopen", return_value=_role_response()
                ) as mock_urlopen:
                    creds = resolve_aws_credentials("sso-prof")
        self.assertIsNotNone(creds)
        (request,), _ = mock_urlopen.call_args
        self.assertEqual(request.get_header("X-amz-sso_bearer_token"), "TOKEN-VALID")

    def test_sso_malformed_cache_entry_is_skipped(self):
        """Invariant: a corrupt cache file cannot take down credential resolution."""
        from core.auth_sigv4 import resolve_aws_credentials

        with tempfile.TemporaryDirectory() as home:
            _write_sso_fixture(
                home,
                {
                    "broken.json": "{ not json",
                    "ok.json": '{"startUrl": "https://my.awsapps.com/start", '
                    '"accessToken": "TOKEN-VALID", "expiresAt": "2030-06-01T00:00:00.000Z"}',
                },
            )
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                with patch("urllib.request.urlopen", return_value=_role_response()):
                    creds = resolve_aws_credentials("sso-prof")
        assert creds is not None
        self.assertEqual(creds["access_key"], "AKID-SSO")

    def test_sso_only_expired_tokens_yield_none(self):
        """Invariant: all-expired SSO cache means None without any network call."""
        from core.auth_sigv4 import resolve_aws_credentials

        with tempfile.TemporaryDirectory() as home:
            _write_sso_fixture(
                home,
                {
                    "old.json": '{"startUrl": "https://my.awsapps.com/start", '
                    '"accessToken": "TOKEN-OLD", "expiresAt": "2000-01-01T00:00:00Z"}'
                },
            )
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                with patch("urllib.request.urlopen") as mock_urlopen:
                    self.assertIsNone(resolve_aws_credentials("sso-prof"))
                    mock_urlopen.assert_not_called()

    def test_sso_network_error_yields_none(self):
        """Invariant: a failed GetRoleCredentials call resolves to None, not a raise."""
        import urllib.error

        from core.auth_sigv4 import resolve_aws_credentials

        with tempfile.TemporaryDirectory() as home:
            _write_sso_fixture(
                home,
                {
                    "valid.json": '{"startUrl": "https://my.awsapps.com/start", '
                    '"accessToken": "TOKEN-VALID", "expiresAt": "2030-01-01T00:00:00Z"}'
                },
            )
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                with patch(
                    "urllib.request.urlopen",
                    side_effect=urllib.error.URLError("down"),
                ):
                    self.assertIsNone(resolve_aws_credentials("sso-prof"))


if __name__ == "__main__":
    unittest.main()
