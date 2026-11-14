from __future__ import annotations

import shutil
import subprocess
import tempfile
import unittest
import unittest.mock
from pathlib import Path

from core.discovery import (
    DiscoveredResource,
    discover_resources,
    discover_workspace_files,
    expand_tagged_files,
    resolve_search_directories,
)


class TestDiscovery(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()
        self.base = Path(self.temp_dir)

    def test_resolve_search_directories_precedence_and_filtering(self):
        """Invariant: resolve_search_directories respects global-then-project order and skips non-existent directories."""
        global_dir = self.base / "global"
        workspace_dir = self.base / "workspace"

        # General resource (skills): searches global then project candidates
        (global_dir / "skills").mkdir(parents=True)
        (workspace_dir / ".skills").mkdir(parents=True)
        (workspace_dir / ".microharness" / "skills").mkdir(parents=True)

        dirs = resolve_search_directories(
            "skills", workspace_dir=workspace_dir, global_dir=global_dir
        )

        # Invariant: 3 existing directories found in strict global -> project order
        self.assertEqual(len(dirs), 3)
        self.assertEqual(dirs[0], ((global_dir / "skills").resolve(), "global"))
        self.assertEqual(dirs[1], ((workspace_dir / ".skills").resolve(), "project"))
        self.assertEqual(
            dirs[2], ((workspace_dir / ".microharness" / "skills").resolve(), "project")
        )

        # For "tools", project search is strictly restricted to .microharness/tools
        (global_dir / "tools").mkdir(parents=True)
        (workspace_dir / "tools").mkdir(parents=True)
        (workspace_dir / ".tools").mkdir(parents=True)
        (workspace_dir / ".microharness" / "tools").mkdir(parents=True)

        tool_dirs = resolve_search_directories(
            "tools", workspace_dir=workspace_dir, global_dir=global_dir
        )
        self.assertEqual(len(tool_dirs), 2)
        self.assertEqual(tool_dirs[0], ((global_dir / "tools").resolve(), "global"))
        self.assertEqual(
            tool_dirs[1], ((workspace_dir / ".microharness" / "tools").resolve(), "project")
        )

    def test_resolve_search_directories_fallback_when_both_none(self):
        """Invariant: If both workspace_dir and global_dir are None, fallback resolves ~/.microharness."""
        dirs = resolve_search_directories("skills", workspace_dir=None, global_dir=None)
        # Even if ~/.microharness does or doesn't have skills, every entry in dirs must have scope 'global'
        for _, scope in dirs:
            self.assertEqual(scope, "global")

    def test_discover_resources_with_pattern_filtering_and_shadowing(self):
        """Invariant: discover_resources filters by pattern and project-level items shadow global items."""
        global_dir = self.base / "global"
        workspace_dir = self.base / "workspace"
        g_cmds = global_dir / "commands"
        w_cmds = workspace_dir / ".microharness" / "commands"
        g_cmds.mkdir(parents=True)
        w_cmds.mkdir(parents=True)

        # Global commands: review, status
        (g_cmds / "review.md").write_text("Global review", encoding="utf-8")
        (g_cmds / "status.md").write_text("Global status", encoding="utf-8")
        # Irrelevant files that must be ignored
        (g_cmds / "README.md").write_text("Ignore me", encoding="utf-8")
        (g_cmds / "_hidden.md").write_text("Ignore me", encoding="utf-8")
        (g_cmds / "ignored.txt").write_text("Not markdown", encoding="utf-8")

        # Project commands: review (shadows global), deploy
        (w_cmds / "review.md").write_text("Project specialized review", encoding="utf-8")
        (w_cmds / "deploy.md").write_text("Project deploy", encoding="utf-8")

        def loader(path: Path, scope: str):
            content = path.read_text(encoding="utf-8")
            return path.stem, (content, scope)

        results = discover_resources(
            resource_name="commands",
            loader=loader,
            workspace_dir=workspace_dir,
            global_dir=global_dir,
            file_pattern="*.md",
        )

        self.assertEqual(len(results), 3)
        self.assertIn("review", results)
        self.assertIn("status", results)
        self.assertIn("deploy", results)

        # Invariant 1: Review is shadowed by project
        review_res = results["review"]
        self.assertIsInstance(review_res, DiscoveredResource)
        self.assertEqual(review_res.scope, "project")
        self.assertEqual(review_res.item[0], "Project specialized review")
        self.assertEqual(review_res.source_path, (w_cmds / "review.md").resolve())

        # Invariant 2: Status is retained from global
        status_res = results["status"]
        self.assertEqual(status_res.scope, "global")
        self.assertEqual(status_res.item[0], "Global status")

        # Invariant 3: Deploy is added from project
        deploy_res = results["deploy"]
        self.assertEqual(deploy_res.scope, "project")
        self.assertEqual(deploy_res.item[0], "Project deploy")

    def test_discover_resources_generator_loader_multiple_items_per_file(self):
        """Invariant: A loader returning a generator of multiple items per file registers all items."""
        workspace_dir = self.base / "workspace"
        tools_dir = workspace_dir / ".microharness" / "tools"
        tools_dir.mkdir(parents=True)

        (tools_dir / "multi_plugin.py").write_text("# multi-tool plugin", encoding="utf-8")

        def multi_loader(path: Path, scope: str):
            yield f"{path.stem}_a", 100
            yield f"{path.stem}_b", 200

        results = discover_resources(
            resource_name="tools",
            loader=multi_loader,
            workspace_dir=workspace_dir,
            global_dir=None,
            file_pattern="*.py",
        )

        self.assertEqual(len(results), 2)
        self.assertEqual(results["multi_plugin_a"].item, 100)
        self.assertEqual(results["multi_plugin_b"].item, 200)

    def test_discover_resources_negative_cases(self):
        """Invariant: discover_resources gracefully skips invalid items, failing loaders, and None returns."""
        workspace_dir = self.base / "workspace"
        res_dir = workspace_dir / "resources"
        res_dir.mkdir(parents=True)

        (res_dir / "valid.dat").write_text("valid", encoding="utf-8")
        (res_dir / "fail.dat").write_text("fail", encoding="utf-8")
        (res_dir / "skip.dat").write_text("skip", encoding="utf-8")
        (res_dir / ".hidden.dat").write_text("hidden", encoding="utf-8")

        def buggy_loader(path: Path, scope: str):
            if path.name == "fail.dat":
                raise RuntimeError("Simulated read/parse error")
            if path.name == "skip.dat":
                return None
            return path.stem, path.read_text(encoding="utf-8")

        results = discover_resources(
            resource_name="resources",
            loader=buggy_loader,
            workspace_dir=workspace_dir,
            global_dir=None,
            file_pattern="*.dat",
        )

        # Invariant: Only 'valid' survived; fail.dat, skip.dat, and .hidden.dat were safely skipped
        self.assertEqual(len(results), 1)
        self.assertIn("valid", results)
        self.assertEqual(results["valid"].item, "valid")

    def test_discover_workspace_files_filtering_and_sorting(self):
        """Invariant: discover_workspace_files finds project files, ignores binary/vcs/caches, and sorts results."""
        workspace = self.base / "ws_files"
        (workspace / "src").mkdir(parents=True)
        (workspace / "tests").mkdir(parents=True)
        (workspace / ".git" / "objects").mkdir(parents=True)
        (workspace / "__pycache__").mkdir(parents=True)
        (workspace / ".venv" / "lib").mkdir(parents=True)

        # Valid source files
        (workspace / "README.md").write_text("# Readme", encoding="utf-8")
        (workspace / ".gitignore").write_text("*.pyc", encoding="utf-8")
        (workspace / "src" / "app.py").write_text("print('app')", encoding="utf-8")
        (workspace / "tests" / "test_app.py").write_text("def test(): pass", encoding="utf-8")

        # Files that MUST be ignored
        (workspace / ".git" / "HEAD").write_text("ref: refs/heads/main", encoding="utf-8")
        (workspace / "__pycache__" / "app.cpython-311.pyc").write_bytes(b"\x00\x01")
        (workspace / ".venv" / "lib" / "pip.py").write_text("# venv pip", encoding="utf-8")
        (workspace / "src" / "compiled.so").write_bytes(b"\x7fELF")
        (workspace / "logo.png").write_bytes(b"\x89PNG")
        (workspace / ".secret").write_text("hidden", encoding="utf-8")

        files = discover_workspace_files(workspace)

        # Invariant: Only valid non-ignored files are returned, with normalized relative paths
        expected = [".gitignore", "README.md", "src/app.py", "tests/test_app.py"]
        self.assertEqual(files, expected)

        # Invariant: Non-existent directory returns empty list
        self.assertEqual(discover_workspace_files(self.base / "nonexistent_dir"), [])

    def test_expand_tagged_files_context_injection_and_edge_cases(self):
        """Invariant: expand_tagged_files attaches file contents with markdown blocks and handles edge cases."""
        workspace = self.base / "ws_expand"
        (workspace / "core").mkdir(parents=True)

        tui_content = "class CursesTUI:\n    pass\n"
        readme_content = "# Project Info\nLine 2\n"
        (workspace / "core" / "tui.py").write_text(tui_content, encoding="utf-8")
        (workspace / "README.md").write_text(readme_content, encoding="utf-8")

        # 1. Single tagged file
        text1 = "Please review @core/tui.py today"
        prompt1, attached1 = expand_tagged_files(text1, workspace)
        self.assertEqual(attached1, ["core/tui.py"])
        self.assertIn("Please review @core/tui.py today", prompt1)
        self.assertIn("[Attached File: @core/tui.py]", prompt1)
        self.assertIn("```\nclass CursesTUI:\n    pass\n\n```", prompt1)

        # 2. Multiple tagged files with trailing punctuation
        text2 = "Compare @core/tui.py, with @README.md."
        prompt2, attached2 = expand_tagged_files(text2, workspace)
        self.assertEqual(attached2, ["core/tui.py", "README.md"])
        self.assertIn("[Attached File: @core/tui.py]", prompt2)
        self.assertIn("[Attached File: @README.md]", prompt2)

        # 3. Duplicate tags attached only once
        text3 = "Check @core/tui.py and re-check @core/tui.py"
        prompt3, attached3 = expand_tagged_files(text3, workspace)
        self.assertEqual(attached3, ["core/tui.py"])
        self.assertEqual(prompt3.count("[Attached File: @core/tui.py]"), 1)

        # 4. Non-existent file is safely ignored
        text4 = "Look at @nonexistent.py"
        prompt4, attached4 = expand_tagged_files(text4, workspace)
        self.assertEqual(attached4, [])
        self.assertEqual(prompt4, text4)

        # 5. Email addresses not treated as file tags
        text5 = "Contact developer@example.com for info"
        prompt5, attached5 = expand_tagged_files(text5, workspace)
        self.assertEqual(attached5, [])
        self.assertEqual(prompt5, text5)

        # 6. File truncation when exceeding limit
        long_content = "x" * 150
        (workspace / "long.txt").write_text(long_content, encoding="utf-8")
        prompt6, attached6 = expand_tagged_files("@long.txt", workspace, max_file_chars=50)
        self.assertEqual(attached6, ["long.txt"])
        self.assertIn("[... Truncated: File exceeds 50 characters ...]", prompt6)

    def _make_ws(self, root: Path) -> Path:
        for sub in ("src", "build", "dist", "node_modules/pkg", "sub/generated"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        (root / "src" / "app.py").write_text("x = 1", encoding="utf-8")
        (root / "build" / "out.py").write_text("# build output", encoding="utf-8")
        (root / "dist" / "bundle.js").write_text("bundled", encoding="utf-8")
        (root / "node_modules" / "pkg" / "i.js").write_text("dep", encoding="utf-8")
        (root / "sub" / "generated" / "g.ts").write_text("gen", encoding="utf-8")
        (root / "app.log").write_text("noise", encoding="utf-8")
        (root / "important.log").write_text("keep me", encoding="utf-8")
        (root / "secrets.txt").write_text("TOKEN=abc", encoding="utf-8")
        (root / ".gitignore").write_text(
            "build/\ndist/\n*.log\nsecrets.txt\n!important.log\n", encoding="utf-8"
        )
        return root

    @unittest.skipUnless(shutil.which("git"), "git is required for check-ignore delegation")
    def test_discover_workspace_files_defers_to_repository_ignore_rules(self):
        """Invariant: in a git repo the repo's own ignore rules decide, including negation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ws = self._make_ws(Path(tmp_dir) / "repo")
            subprocess.run(["git", "init", "-q", "."], cwd=ws, check=True)

            found = discover_workspace_files(ws)

            # Excluded by the repository's rules.
            self.assertNotIn("build/out.py", found)
            self.assertNotIn("dist/bundle.js", found)
            self.assertNotIn("node_modules/pkg/i.js", found)
            self.assertNotIn("app.log", found)
            self.assertNotIn("secrets.txt", found)
            # Negation (!important.log) must win over the earlier *.log rule.
            self.assertIn("important.log", found)
            # Not mentioned by any rule.
            self.assertIn("src/app.py", found)
            self.assertIn("sub/generated/g.ts", found)
            self.assertIn(".gitignore", found)

    def test_discover_workspace_files_falls_back_when_not_a_repository(self):
        """Invariant: without git the built-in fallback applies and never drops the whole list."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ws = self._make_ws(Path(tmp_dir) / "plain")
            # Deliberately not a git repository.

            found = discover_workspace_files(ws)

            self.assertIn("src/app.py", found)
            # Fallback still excludes the universally-noisy entries.
            self.assertNotIn("node_modules/pkg/i.js", found)
            # And still excludes binary extensions.
            (ws / "logo.png").write_bytes(b"\x89PNG")
            self.assertNotIn("logo.png", discover_workspace_files(ws))

    def test_discover_workspace_files_keeps_tracked_files_matching_ignore_rules(self):
        """Invariant: a committed file stays visible even when it matches a broad ignore rule."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ws = self._make_ws(Path(tmp_dir) / "repo3")
            subprocess.run(["git", "init", "-q", "."], cwd=ws, check=True)
            # *.log is ignored, but this one is force-added and therefore tracked.
            subprocess.run(["git", "add", "-f", "app.log"], cwd=ws, check=True)
            subprocess.run(
                ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "x"],
                cwd=ws,
                check=True,
            )

            found = discover_workspace_files(ws)
            self.assertIn("app.log", found)
            # Untracked files matching the same rule stay hidden.
            self.assertNotIn("secrets.txt", found)

    def test_git_ignored_passes_a_timeout(self):
        """Invariant: the subprocess is bounded so a wedged git cannot hang TUI startup."""
        from core.discovery import _git_ignored

        with tempfile.TemporaryDirectory() as tmp_dir:
            with unittest.mock.patch.object(subprocess, "run") as run:
                run.return_value = subprocess.CompletedProcess(
                    args=[], returncode=1, stdout="", stderr=""
                )
                _git_ignored(Path(tmp_dir), ["a.py"])
            self.assertIsNotNone(run.call_args.kwargs.get("timeout"))

    def test_git_ignored_helper_degrades_quietly(self):
        """Invariant: _git_ignored returns an empty set rather than raising when git cannot answer."""
        from core.discovery import _git_ignored

        with tempfile.TemporaryDirectory() as tmp_dir:
            ws = Path(tmp_dir)  # not a repository
            self.assertEqual(_git_ignored(ws, ["a.py", "b.py"]), set())
            self.assertEqual(_git_ignored(ws, []), set())

            # A missing git binary must also degrade, not raise.
            with unittest.mock.patch.object(subprocess, "run", side_effect=FileNotFoundError):
                self.assertEqual(_git_ignored(ws, ["a.py"]), set())
            with unittest.mock.patch.object(
                subprocess, "run", side_effect=subprocess.TimeoutExpired("git", 10)
            ):
                self.assertEqual(_git_ignored(ws, ["a.py"]), set())

    def test_explicitly_tagged_file_is_readable_despite_ignore_rules(self):
        """Invariant: the ignore list only drives autocomplete; @path still attaches anything."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            ws = self._make_ws(Path(tmp_dir) / "repo2")
            subprocess.run(["git", "init", "-q", "."], cwd=ws, check=True)

            self.assertNotIn("secrets.txt", discover_workspace_files(ws))
            _, attached = expand_tagged_files("look at @secrets.txt", ws)
            self.assertEqual(attached, ["secrets.txt"])


if __name__ == "__main__":
    unittest.main()
