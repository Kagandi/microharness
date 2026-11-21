import subprocess
import tempfile
import unittest
from pathlib import Path

from core.checkpoints import CheckpointManager
from core.memory import Memory


def _git(cwd: Path, *args: str) -> str:
    res = subprocess.run(
        ["git", *args],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=True,
    )
    return res.stdout


class TestCheckpointManagerGit(unittest.TestCase):
    def setUp(self):
        self.repo = Path(tempfile.mkdtemp(prefix="mh_cp_git_")).resolve()
        _git(self.repo, "init")
        _git(self.repo, "config", "user.name", "Test User")
        _git(self.repo, "config", "user.email", "test@example.com")
        (self.repo / "app.py").write_text("x = 1\n", encoding="utf-8")
        (self.repo / "keep.txt").write_text("keep\n", encoding="utf-8")
        _git(self.repo, "add", "app.py", "keep.txt")
        _git(self.repo, "commit", "-m", "initial")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.repo, ignore_errors=True)

    def test_git_checkpoint_restores_modified_deleted_and_removes_created_files(self):
        """Invariant: undo restores edited/deleted files and removes newly created files without touching HEAD."""
        head_before = _git(self.repo, "rev-parse", "HEAD").strip()
        mgr = CheckpointManager(self.repo)
        cp = mgr.create(label="turn 1")
        self.assertIsNotNone(cp)
        assert cp is not None
        self.assertEqual(cp.backend, "git")
        self.assertFalse(cp.dirty_before)
        self.assertEqual(mgr.count, 1)

        # Simulate agent mutations during the turn
        (self.repo / "app.py").write_text("x = 999\n", encoding="utf-8")
        (self.repo / "keep.txt").unlink()
        (self.repo / "sub" / "added.py").parent.mkdir(parents=True)
        (self.repo / "sub" / "added.py").write_text("new = True\n", encoding="utf-8")

        result = mgr.undo()
        self.assertTrue(result.ok)
        self.assertEqual(result.checkpoint_id, cp.id)
        self.assertEqual(result.removed_files, ("sub/added.py",))
        self.assertEqual(result.restored_files, ("app.py", "keep.txt"))
        self.assertEqual((self.repo / "app.py").read_text(encoding="utf-8"), "x = 1\n")
        self.assertEqual((self.repo / "keep.txt").read_text(encoding="utf-8"), "keep\n")
        self.assertFalse((self.repo / "sub" / "added.py").exists())
        self.assertFalse((self.repo / "sub").exists())
        self.assertEqual(_git(self.repo, "rev-parse", "HEAD").strip(), head_before)
        self.assertEqual(mgr.count, 0)

    def test_git_checkpoint_preserves_user_staged_and_preexisting_unstaged_state(self):
        """Invariant: pre-turn staged index entries and pre-turn unstaged edits survive checkpoint and undo."""
        # User stages a new file and makes an unstaged edit before the turn
        (self.repo / "staged.txt").write_text("staged v1\n", encoding="utf-8")
        _git(self.repo, "add", "staged.txt")
        (self.repo / "app.py").write_text("x = 2  # user wip\n", encoding="utf-8")
        (self.repo / "untracked_user.txt").write_text("notes\n", encoding="utf-8")

        mgr = CheckpointManager(self.repo)
        cp = mgr.create(label="wip turn")
        self.assertIsNotNone(cp)
        assert cp is not None
        self.assertTrue(cp.dirty_before)

        # Agent overwrites app.py and untracked_user.txt and creates extra.py
        (self.repo / "app.py").write_text("x = 999\n", encoding="utf-8")
        (self.repo / "untracked_user.txt").write_text("clobbered\n", encoding="utf-8")
        (self.repo / "extra.py").write_text("junk\n", encoding="utf-8")

        result = mgr.undo()
        self.assertTrue(result.ok)
        self.assertEqual((self.repo / "app.py").read_text(encoding="utf-8"), "x = 2  # user wip\n")
        self.assertEqual((self.repo / "untracked_user.txt").read_text(encoding="utf-8"), "notes\n")
        self.assertFalse((self.repo / "extra.py").exists())

        # User's staged index entry is still staged cleanly
        status_lines = _git(self.repo, "status", "--porcelain").splitlines()
        self.assertIn("A  staged.txt", status_lines)
        self.assertIn(" M app.py", status_lines)
        self.assertIn("?? untracked_user.txt", status_lines)

    def test_git_checkpoint_lifo_stack_and_memory_rollback(self):
        """Invariant: multiple checkpoints restore in LIFO order and roll back memory messages/modified_files."""
        memory = Memory(system_prompt="sys", user_prompt="")
        memory.add_user("first")
        memory.add_assistant("done1")
        memory.record_modified_file("prev.py")

        mgr = CheckpointManager(self.repo)
        # Turn 2 begins
        pre_len_2 = len(memory.messages)
        cp1 = mgr.create(
            label="turn 2",
            memory_len=pre_len_2,
            modified_files=frozenset(memory.modified_files),
        )
        self.assertIsNotNone(cp1)
        memory.add_user("second")
        memory.add_assistant("editing app.py")
        memory.record_modified_file("app.py")
        (self.repo / "app.py").write_text("x = 2\n", encoding="utf-8")

        # Turn 3 begins
        pre_len_3 = len(memory.messages)
        cp2 = mgr.create(
            label="turn 3",
            memory_len=pre_len_3,
            modified_files=frozenset(memory.modified_files),
        )
        self.assertIsNotNone(cp2)
        memory.add_user("third")
        (self.repo / "app.py").write_text("x = 3\n", encoding="utf-8")

        # First undo reverts turn 3 back to x = 2
        res1 = mgr.undo(memory=memory)
        self.assertTrue(res1.ok)
        self.assertEqual((self.repo / "app.py").read_text(encoding="utf-8"), "x = 2\n")
        self.assertEqual(len(memory.messages), pre_len_3)
        self.assertEqual(memory.modified_files, {"prev.py", "app.py"})

        # Second undo reverts turn 2 back to x = 1
        res2 = mgr.undo(memory=memory)
        self.assertTrue(res2.ok)
        self.assertEqual((self.repo / "app.py").read_text(encoding="utf-8"), "x = 1\n")
        self.assertEqual(len(memory.messages), pre_len_2)
        self.assertEqual(memory.modified_files, {"prev.py"})

        # Third undo has nothing left
        res3 = mgr.undo(memory=memory)
        self.assertFalse(res3.ok)
        self.assertIsNone(res3.checkpoint_id)
        self.assertIn("Nothing to undo", res3.message)

    def test_git_checkpoint_noop_turn_reports_unchanged(self):
        """Invariant: undoing a turn with no file edits reports workspace unchanged."""
        mgr = CheckpointManager(self.repo)
        mgr.create(label="read only turn")
        res = mgr.undo()
        self.assertTrue(res.ok)
        self.assertEqual(res.restored_files, ())
        self.assertEqual(res.removed_files, ())
        self.assertIn("already unchanged", res.message)

    def test_git_checkpoint_scoped_to_subdirectory_workspace(self):
        """Invariant: checkpoints inside a repo subdirectory never alter sibling directories."""
        sub_ws = self.repo / "packages" / "app"
        sibling = self.repo / "packages" / "other"
        sub_ws.mkdir(parents=True)
        sibling.mkdir(parents=True)
        (sub_ws / "main.py").write_text("v = 1\n", encoding="utf-8")
        (sibling / "other.py").write_text("other = 1\n", encoding="utf-8")
        _git(self.repo, "add", ".")
        _git(self.repo, "commit", "-m", "add packages")

        mgr = CheckpointManager(sub_ws)
        mgr.create(label="sub turn")
        (sub_ws / "main.py").write_text("v = 2\n", encoding="utf-8")
        (sub_ws / "extra.py").write_text("x = 1\n", encoding="utf-8")
        (sibling / "other.py").write_text("other = 999\n", encoding="utf-8")

        res = mgr.undo()
        self.assertTrue(res.ok)
        self.assertEqual(res.restored_files, ("main.py",))
        self.assertEqual(res.removed_files, ("extra.py",))
        self.assertEqual((sub_ws / "main.py").read_text(encoding="utf-8"), "v = 1\n")
        self.assertFalse((sub_ws / "extra.py").exists())
        # Sibling outside sub_ws must remain untouched
        self.assertEqual((sibling / "other.py").read_text(encoding="utf-8"), "other = 999\n")


class TestCheckpointManagerFilesFallback(unittest.TestCase):
    def setUp(self):
        self.ws = Path(tempfile.mkdtemp(prefix="mh_cp_files_")).resolve()
        self.storage = Path(tempfile.mkdtemp(prefix="mh_cp_store_")).resolve()
        (self.ws / "src").mkdir()
        (self.ws / "src" / "mod.py").write_text("def f(): return 1\n", encoding="utf-8")
        (self.ws / "notes.txt").write_text("hello\n", encoding="utf-8")

    def tearDown(self):
        import shutil

        shutil.rmtree(self.ws, ignore_errors=True)
        shutil.rmtree(self.storage, ignore_errors=True)

    def test_file_backup_checkpoint_restores_and_removes_created_files(self):
        """Invariant: non-git workspaces snapshot and restore files cleanly without touching skipped dirs."""
        (self.ws / ".logs").mkdir()
        (self.ws / ".logs" / "trace.log").write_text("keep log\n", encoding="utf-8")

        mgr = CheckpointManager(self.ws, storage_dir=self.storage)
        cp = mgr.create(label="non-git turn")
        self.assertIsNotNone(cp)
        assert cp is not None
        self.assertEqual(cp.backend, "files")
        self.assertEqual(cp.manifest, ("notes.txt", "src/mod.py"))

        (self.ws / "src" / "mod.py").write_text("broken\n", encoding="utf-8")
        (self.ws / "notes.txt").unlink()
        (self.ws / "new_dir" / "new.txt").parent.mkdir()
        (self.ws / "new_dir" / "new.txt").write_text("extra\n", encoding="utf-8")

        res = mgr.undo()
        self.assertTrue(res.ok)
        self.assertEqual(res.removed_files, ("new_dir/new.txt",))
        self.assertEqual(res.restored_files, ("notes.txt", "src/mod.py"))
        self.assertEqual(
            (self.ws / "src" / "mod.py").read_text(encoding="utf-8"), "def f(): return 1\n"
        )
        self.assertEqual((self.ws / "notes.txt").read_text(encoding="utf-8"), "hello\n")
        self.assertFalse((self.ws / "new_dir").exists())
        self.assertEqual(
            (self.ws / ".logs" / "trace.log").read_text(encoding="utf-8"), "keep log\n"
        )

    def test_file_backup_skips_when_workspace_exceeds_safety_limits(self):
        """Invariant: oversized non-git workspaces skip snapshotting safely and set last_warning."""
        mgr = CheckpointManager(self.ws, storage_dir=self.storage, max_files=1)
        cp = mgr.create(label="too big")
        self.assertIsNone(cp)
        self.assertIsNotNone(mgr.last_warning)
        assert mgr.last_warning is not None
        self.assertIn("exceeds safety limit", mgr.last_warning)
        self.assertEqual(mgr.count, 0)

    def test_missing_workspace_and_missing_snapshot_dir_negative_cases(self):
        """Invariant: missing workspace skips create and missing backup directory reports failure cleanly."""
        missing_mgr = CheckpointManager(self.ws / "does_not_exist")
        self.assertIsNone(missing_mgr.create())

        mgr = CheckpointManager(self.ws, storage_dir=self.storage)
        cp = mgr.create()
        assert cp is not None and cp.snapshot_dir is not None
        import shutil

        shutil.rmtree(cp.snapshot_dir)
        res = mgr.undo()
        self.assertFalse(res.ok)
        self.assertIn("missing", res.message)

    def test_missing_git_binary_falls_back_to_file_checkpoints(self):
        """Invariant: if git is not installed, CheckpointManager falls back to file backups cleanly."""
        from unittest.mock import patch

        # Even if a .git directory exists, missing git executable must not raise FileNotFoundError
        (self.ws / ".git").mkdir()
        mgr = CheckpointManager(self.ws, storage_dir=self.storage)

        with patch("core.checkpoints.shutil.which", return_value=None):
            cp = mgr.create(label="no git binary")
            self.assertIsNotNone(cp)
            assert cp is not None
            self.assertEqual(cp.backend, "files")

            (self.ws / "notes.txt").write_text("changed\n", encoding="utf-8")
            (self.ws / "added.txt").write_text("extra\n", encoding="utf-8")

            res = mgr.undo()
            self.assertTrue(res.ok)
            self.assertEqual((self.ws / "notes.txt").read_text(encoding="utf-8"), "hello\n")
            self.assertFalse((self.ws / "added.txt").exists())


if __name__ == "__main__":
    unittest.main()
