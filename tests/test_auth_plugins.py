import contextlib
import io
import os
import tempfile
import unittest
from pathlib import Path

from core.auth import RequestSigner, get_signer, load_auth_plugins


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
            _load_with_stderr(os.path.join(tempfile.gettempdir(), "mh-no-such-dir-xyz"))[0],
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
            self.assertEqual(out, {"Content-Type": "application/json", "X-Custom": "yes:abc"})
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
            plugin_path = _write_plugin(tmpdir, "nosign.py", "SCHEME_NAME = 'test-wave2-nosign'\n")
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
