import os
import tempfile
import unittest
import unittest.mock
from pathlib import Path
from unittest.mock import patch


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

    def test_sso_first_token_network_error_falls_through_to_next(self):
        """Invariant: a failed role call on one token tries the next token."""
        import urllib.error

        from core.auth_sigv4 import resolve_aws_credentials

        with tempfile.TemporaryDirectory() as home:
            _write_sso_fixture(
                home,
                {
                    "a-first.json": '{"startUrl": "https://my.awsapps.com/start", '
                    '"accessToken": "TOKEN-FIRST", "expiresAt": "2030-01-01T00:00:00Z"}',
                    "b-second.json": '{"startUrl": "https://my.awsapps.com/start", '
                    '"accessToken": "TOKEN-SECOND", "expiresAt": "2030-01-01T00:00:00Z"}',
                },
            )
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                with patch(
                    "urllib.request.urlopen",
                    side_effect=[
                        urllib.error.URLError("down"),
                        _role_response(),
                    ],
                ) as mock_urlopen:
                    creds = resolve_aws_credentials("sso-prof")
        self.assertIsNotNone(creds)
        assert creds is not None
        self.assertEqual(creds["access_key"], "AKID-SSO")
        self.assertEqual(mock_urlopen.call_count, 2)
        (second_request,), _ = mock_urlopen.call_args
        self.assertEqual(
            second_request.get_header("X-amz-sso_bearer_token"), "TOKEN-SECOND"
        )

    def test_sso_first_payload_malformed_falls_through_to_next(self):
        """Invariant: a role response without roleCredentials tries the next token."""
        from core.auth_sigv4 import resolve_aws_credentials

        bad = unittest.mock.MagicMock()
        bad.read.return_value = b'{"noRoleCredentials": true}'
        bad.__enter__.return_value = bad
        with tempfile.TemporaryDirectory() as home:
            _write_sso_fixture(
                home,
                {
                    "a-first.json": '{"startUrl": "https://my.awsapps.com/start", '
                    '"accessToken": "TOKEN-FIRST", "expiresAt": "2030-01-01T00:00:00Z"}',
                    "b-second.json": '{"startUrl": "https://my.awsapps.com/start", '
                    '"accessToken": "TOKEN-SECOND", "expiresAt": "2030-01-01T00:00:00Z"}',
                },
            )
            with patch.dict(os.environ, _clear_aws_env(home), clear=True):
                with patch(
                    "urllib.request.urlopen",
                    side_effect=[bad, _role_response()],
                ) as mock_urlopen:
                    creds = resolve_aws_credentials("sso-prof")
        self.assertIsNotNone(creds)
        assert creds is not None
        self.assertEqual(creds["access_key"], "AKID-SSO")
        self.assertEqual(mock_urlopen.call_count, 2)
        (second_request,), _ = mock_urlopen.call_args
        self.assertEqual(
            second_request.get_header("X-amz-sso_bearer_token"), "TOKEN-SECOND"
        )


if __name__ == "__main__":
    unittest.main()
