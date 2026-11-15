import io
import json
import os
import stat
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from unittest.mock import patch

import main as cli_main
from core.auth_store import (
    AUTH_FILENAME,
    AUTH_TYPE_API_KEY,
    canonicalize_provider,
    get_auth_credential,
    get_auth_store_path,
    load_auth_store,
    mask_secret,
    remove_auth_credential,
    save_auth_credential,
)
from core.providers import PROVIDER_ALIASES


class TestAuthStorePrimitives(unittest.TestCase):
    def test_get_auth_store_path_and_canonicalize(self):
        self.assertIsNone(get_auth_store_path(None))
        self.assertIsNone(get_auth_store_path(""))
        self.assertEqual(get_auth_store_path("/tmp/mh"), Path("/tmp/mh") / AUTH_FILENAME)

        self.assertEqual(
            canonicalize_provider("  Vercel-AI-Gateway ", PROVIDER_ALIASES),
            "vercel",
        )
        self.assertEqual(canonicalize_provider("OpenRouter", PROVIDER_ALIASES), "openrouter")
        self.assertEqual(canonicalize_provider("google", PROVIDER_ALIASES), "google")
        self.assertEqual(canonicalize_provider("gemini", PROVIDER_ALIASES), "gemini")
        self.assertEqual(
            canonicalize_provider("Google AI Studio", PROVIDER_ALIASES),
            "google-ai-studio",
        )
        self.assertEqual(
            canonicalize_provider("google_ai_studio", PROVIDER_ALIASES),
            "google-ai-studio",
        )
        self.assertIsNone(canonicalize_provider("   ", PROVIDER_ALIASES))
        self.assertIsNone(canonicalize_provider(None, PROVIDER_ALIASES))

    def test_mask_secret_redacts_short_and_long_keys(self):
        self.assertEqual(mask_secret(None), "")
        self.assertEqual(mask_secret("   "), "")
        self.assertEqual(mask_secret("short-key"), "****")
        self.assertEqual(mask_secret("sk-or-v1-1234567890abcdef"), "sk-o...cdef")
        self.assertNotIn("1234567890", mask_secret("sk-or-v1-1234567890abcdef"))

    def test_save_creates_owner_only_atomic_file_and_canonicalizes_aliases(self):
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp) / "nested-global"
            old_umask = os.umask(0o000)
            try:
                saved_path = save_auth_credential(
                    global_dir,
                    "vercel-ai-gateway",
                    "  vck-secret-value-1234  ",
                    aliases=PROVIDER_ALIASES,
                )
            finally:
                os.umask(old_umask)

            expected_path = global_dir / AUTH_FILENAME
            self.assertEqual(saved_path, expected_path)
            self.assertTrue(expected_path.is_file())

            if os.name == "posix":
                mode = stat.S_IMODE(expected_path.stat().st_mode)
                self.assertEqual(mode, 0o600)

            raw = json.loads(expected_path.read_text(encoding="utf-8"))
            self.assertEqual(
                raw,
                {"vercel": {"type": AUTH_TYPE_API_KEY, "key": "vck-secret-value-1234"}},
            )
            self.assertEqual(
                get_auth_credential(global_dir, "vercel", aliases=PROVIDER_ALIASES),
                "vck-secret-value-1234",
            )
            self.assertEqual(
                get_auth_credential(global_dir, "vercel-ai-gateway", aliases=PROVIDER_ALIASES),
                "vck-secret-value-1234",
            )

    def test_save_preserves_other_providers_and_future_auth_types(self):
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp)
            auth_path = global_dir / AUTH_FILENAME
            auth_path.write_text(
                json.dumps(
                    {
                        "anthropic": {"type": "oauth", "token": "oauth-tok"},
                        "openai": {"type": "api_key", "key": "sk-openai-12345678"},
                    }
                ),
                encoding="utf-8",
            )

            save_auth_credential(
                global_dir, "openrouter", "sk-or-secret-9999", aliases=PROVIDER_ALIASES
            )

            raw = json.loads(auth_path.read_text(encoding="utf-8"))
            self.assertEqual(raw["anthropic"], {"type": "oauth", "token": "oauth-tok"})
            self.assertEqual(raw["openai"], {"type": "api_key", "key": "sk-openai-12345678"})
            self.assertEqual(raw["openrouter"], {"type": "api_key", "key": "sk-or-secret-9999"})
            # Non-api_key records are ignored by get_auth_credential rather than misread.
            self.assertIsNone(
                get_auth_credential(global_dir, "anthropic", aliases=PROVIDER_ALIASES)
            )

    def test_load_prefers_exact_canonical_key_over_alias_key(self):
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp)
            (global_dir / AUTH_FILENAME).write_text(
                json.dumps(
                    {
                        "vercel-ai-gateway": {"type": "api_key", "key": "alias-key"},
                        "vercel": {"type": "api_key", "key": "canonical-key"},
                    }
                ),
                encoding="utf-8",
            )
            self.assertEqual(
                get_auth_credential(global_dir, "vercel-ai-gateway", aliases=PROVIDER_ALIASES),
                "canonical-key",
            )

    def test_remove_auth_credential_handles_present_and_absent_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp)
            self.assertFalse(
                remove_auth_credential(global_dir, "openrouter", aliases=PROVIDER_ALIASES)
            )
            self.assertFalse((global_dir / AUTH_FILENAME).exists())

            save_auth_credential(global_dir, "openrouter", "sk-or-1", aliases=PROVIDER_ALIASES)
            save_auth_credential(global_dir, "vercel", "vck-1", aliases=PROVIDER_ALIASES)

            self.assertTrue(
                remove_auth_credential(global_dir, "vercel-ai-gateway", aliases=PROVIDER_ALIASES)
            )
            self.assertIsNone(get_auth_credential(global_dir, "vercel", aliases=PROVIDER_ALIASES))
            self.assertEqual(
                get_auth_credential(global_dir, "openrouter", aliases=PROVIDER_ALIASES), "sk-or-1"
            )
            self.assertFalse(remove_auth_credential(global_dir, "vercel", aliases=PROVIDER_ALIASES))

    def test_validation_rejects_empty_inputs(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                save_auth_credential(None, "openrouter", "key")
            with self.assertRaises(ValueError):
                save_auth_credential(tmp, "   ", "key")
            with self.assertRaises(ValueError):
                save_auth_credential(tmp, "openrouter", "   ")
            with self.assertRaises(ValueError):
                remove_auth_credential(tmp, "   ")

    def test_corrupt_or_malformed_auth_json_fails_safely(self):
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp)
            auth_path = global_dir / AUTH_FILENAME

            # 1. Invalid JSON syntax
            auth_path.write_text("{broken-json", encoding="utf-8")
            self.assertEqual(load_auth_store(global_dir, aliases=PROVIDER_ALIASES), {})
            self.assertIsNone(
                get_auth_credential(global_dir, "openrouter", aliases=PROVIDER_ALIASES)
            )

            # 2. Valid JSON, non-dict root
            for payload in ("[]", '"hello"', "123", "null"):
                auth_path.write_text(payload, encoding="utf-8")
                self.assertEqual(load_auth_store(global_dir, aliases=PROVIDER_ALIASES), {})
                self.assertIsNone(
                    get_auth_credential(global_dir, "openrouter", aliases=PROVIDER_ALIASES)
                )

            # 3. Non-dict provider entry / missing or empty key
            auth_path.write_text(
                json.dumps(
                    {
                        "openrouter": "not-a-dict",
                        "google": {"type": "api_key", "key": 12345},
                        "openai": {"type": "api_key", "key": "   "},
                    }
                ),
                encoding="utf-8",
            )
            self.assertIsNone(
                get_auth_credential(global_dir, "openrouter", aliases=PROVIDER_ALIASES)
            )
            self.assertIsNone(get_auth_credential(global_dir, "google", aliases=PROVIDER_ALIASES))
            self.assertIsNone(get_auth_credential(global_dir, "openai", aliases=PROVIDER_ALIASES))

            # 4. Path is a directory instead of a file
            auth_path.unlink()
            auth_path.mkdir()
            self.assertEqual(load_auth_store(global_dir, aliases=PROVIDER_ALIASES), {})


class TestCliAuthCommands(unittest.TestCase):
    def test_cli_auth_login_list_logout_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp) / "global"
            proj_dir = Path(tmp) / "proj"
            proj_dir.mkdir()
            secret = "vck-super-secret-token-9876"

            with patch.dict(os.environ, {}, clear=True):
                # 1. Login with alias canonicalizes to 'vercel'
                stdout = io.StringIO()
                with (
                    patch("getpass.getpass", return_value=secret),
                    redirect_stdout(stdout),
                ):
                    cli_main.main(
                        [
                            "auth",
                            "login",
                            "vercel-ai-gateway",
                            "--global-dir",
                            str(global_dir),
                            "-d",
                            str(proj_dir),
                        ]
                    )
                login_out = stdout.getvalue()
                self.assertIn("Saved credentials for 'vercel'", login_out)
                self.assertNotIn(secret, login_out)
                self.assertEqual(
                    get_auth_credential(global_dir, "vercel", aliases=PROVIDER_ALIASES),
                    secret,
                )

                # 2. List displays auth.json with masked preview and never raw secret
                list_out = io.StringIO()
                with redirect_stdout(list_out):
                    cli_main.main(
                        [
                            "auth",
                            "list",
                            "--global-dir",
                            str(global_dir),
                            "-d",
                            str(proj_dir),
                        ]
                    )
                list_text = list_out.getvalue()
                self.assertIn("vercel             auth.json (vck-...9876)", list_text)
                self.assertNotIn(secret, list_text)

                # 3. Logout removes stored entry
                logout_out = io.StringIO()
                with redirect_stdout(logout_out):
                    cli_main.main(
                        [
                            "auth",
                            "logout",
                            "vercel",
                            "--global-dir",
                            str(global_dir),
                            "-d",
                            str(proj_dir),
                        ]
                    )
                self.assertIn("Removed stored credentials for 'vercel'", logout_out.getvalue())
                self.assertIsNone(
                    get_auth_credential(global_dir, "vercel", aliases=PROVIDER_ALIASES)
                )

    def test_cli_auth_login_interactive_provider_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp) / "global"
            proj_dir = Path(tmp) / "proj"
            proj_dir.mkdir()

            with patch.dict(os.environ, {}, clear=True):
                stdout = io.StringIO()
                with (
                    patch("builtins.input", return_value="openrouter"),
                    patch("getpass.getpass", return_value="sk-or-interactive-1234"),
                    redirect_stdout(stdout),
                ):
                    cli_main.main(
                        [
                            "auth",
                            "login",
                            "--global-dir",
                            str(global_dir),
                            "-d",
                            str(proj_dir),
                        ]
                    )
                self.assertIn("Saved credentials for 'openrouter'", stdout.getvalue())
                self.assertEqual(
                    get_auth_credential(global_dir, "openrouter", aliases=PROVIDER_ALIASES),
                    "sk-or-interactive-1234",
                )

    def test_cli_auth_login_negative_cases(self):
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp) / "global"
            proj_dir = Path(tmp) / "proj"
            proj_dir.mkdir()

            with patch.dict(os.environ, {}, clear=True):
                # Unknown provider exits non-zero
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as cm:
                    cli_main.main(
                        [
                            "auth",
                            "login",
                            "not-a-provider",
                            "--global-dir",
                            str(global_dir),
                            "-d",
                            str(proj_dir),
                        ]
                    )
                self.assertEqual(cm.exception.code, 1)
                self.assertIn("Unknown provider 'not-a-provider'", stderr.getvalue())

                # Empty API key exits non-zero
                stderr = io.StringIO()
                with (
                    patch("getpass.getpass", return_value="   "),
                    redirect_stderr(stderr),
                    self.assertRaises(SystemExit) as cm,
                ):
                    cli_main.main(
                        [
                            "auth",
                            "login",
                            "openrouter",
                            "--global-dir",
                            str(global_dir),
                            "-d",
                            str(proj_dir),
                        ]
                    )
                self.assertEqual(cm.exception.code, 1)
                self.assertIn("API key cannot be empty", stderr.getvalue())

    def test_missing_credentials_message_points_to_auth_login_and_slash_login(self):
        with tempfile.TemporaryDirectory() as tmp:
            global_dir = Path(tmp) / "global"
            proj_dir = Path(tmp) / "proj"
            global_dir.mkdir()
            proj_dir.mkdir()
            stderr = io.StringIO()
            with (
                patch.dict(os.environ, {}, clear=True),
                patch("sys.stdin.isatty", return_value=True),
                patch("main.get_default_global_dir", return_value=global_dir),
                redirect_stderr(stderr),
                self.assertRaises(SystemExit) as cm,
            ):
                cli_main.main(["-d", str(proj_dir), "--provider", "openrouter"])
            self.assertEqual(cm.exception.code, 1)
            msg = stderr.getvalue()
            self.assertIn("microharness auth login [provider]", msg)
            self.assertIn("/login", msg)


if __name__ == "__main__":
    unittest.main()
