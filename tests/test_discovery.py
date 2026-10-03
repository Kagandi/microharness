from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from core.discovery import (
    DiscoveredResource,
    discover_resources,
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

        # Create only specific candidate directories
        (global_dir / "tools").mkdir(parents=True)
        (workspace_dir / ".tools").mkdir(parents=True)
        (workspace_dir / ".microharness" / "tools").mkdir(parents=True)

        dirs = resolve_search_directories("tools", workspace_dir=workspace_dir, global_dir=global_dir)

        # Invariant: 3 existing directories found in strict global -> project order
        self.assertEqual(len(dirs), 3)
        self.assertEqual(dirs[0], ((global_dir / "tools").resolve(), "global"))
        self.assertEqual(dirs[1], ((workspace_dir / ".tools").resolve(), "project"))
        self.assertEqual(dirs[2], ((workspace_dir / ".microharness" / "tools").resolve(), "project"))

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
        tools_dir = workspace_dir / "tools"
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


if __name__ == "__main__":
    unittest.main()
