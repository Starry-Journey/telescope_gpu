"""TaskRunner lifecycle tests: finish/cancel/cleanup/error paths.

These need PySide6; they are skipped on the core CI job and run in the
desktop-smoke job which installs PySide6 with QT_QPA_PLATFORM=offscreen.
"""
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from PySide6.QtCore import QCoreApplication
    from desktop.tasks import TaskRunner

OK_WORKER = """
import json, sys
args = sys.argv[1:]
res = args[args.index("--result") + 1]
with open(res, "w", encoding="utf-8") as fh:
    json.dump({"ok": True, "echo": 7}, fh)
"""

NO_RESULT_WORKER = "import sys\nsys.exit(0)\n"

SLOW_WORKER = "import time\ntime.sleep(30)\n"


def _write_script(td, name, body):
    p = Path(td) / name
    p.write_text(body, encoding="utf-8")
    return str(p)


def _builder(script):
    def build(task, job_path, result_path, frozen=False, root=ROOT):
        return sys.executable, [script, "--job", str(job_path), "--result", str(result_path)]
    return build


@unittest.skipUnless(HAS_QT, "PySide6 not installed; covered by desktop-smoke CI")
class TaskRunnerTest(unittest.TestCase):
    def setUp(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        self.app = QCoreApplication.instance() or QCoreApplication([])
        self.td = tempfile.TemporaryDirectory()
        self.addCleanup(self.td.cleanup)
        self.root = Path(self.td.name) / "tasks"

    def _pump(self, cond, timeout=10.0):
        t0 = time.monotonic()
        while time.monotonic() - t0 < timeout:
            self.app.processEvents()
            if cond():
                return True
            time.sleep(0.02)
        return False

    def _runner(self, script=None, builder=None):
        if builder is None:
            builder = _builder(_write_script(self.td.name, "w.py", script))
        return TaskRunner(self.root, command_builder=builder)

    def test_finish_cleanup(self):
        r = self._runner(OK_WORKER)
        got = {}
        r.finished.connect(lambda d: got.setdefault("data", d))
        r.start("ccd", {"x": 1})
        self.assertTrue(self._pump(lambda: "data" in got), "finished not emitted")
        self.assertEqual(got["data"]["echo"], 7)
        self.assertFalse(r.busy())
        self.assertEqual(list(self.root.glob("task_*")), [])
        self.assertGreater(r.last_elapsed, 0)

    def test_no_result_is_failure(self):
        r = self._runner(NO_RESULT_WORKER)
        errs = []
        r.failed.connect(errs.append)
        r.start("s", {})
        self.assertTrue(self._pump(lambda: errs), "failed not emitted")
        self.assertIn("worker produced no result", errs[0])
        self.assertEqual(list(self.root.glob("task_*")), [])

    def test_start_failure(self):
        def bad_builder(task, job_path, result_path, frozen=False, root=ROOT):
            return str(Path(self.td.name) / "no-such-program-xyz"), ["--worker", task]
        r = self._runner(builder=bad_builder)
        errs = []
        r.failed.connect(errs.append)
        r.start("s", {})
        self.assertTrue(self._pump(lambda: errs or not r.busy()), "error not emitted")
        self.assertTrue(errs, "expected failed signal")
        self.assertEqual(list(self.root.glob("task_*")), [])

    def test_cancel_cleanup_and_restart(self):
        script = _write_script(self.td.name, "slow.py", SLOW_WORKER)
        r = self._runner(builder=_builder(script))
        cancels = []
        r.cancelled.connect(lambda: cancels.append(True))
        r.start("ccd", {"x": 1})
        self.assertTrue(self._pump(lambda: r.busy()))
        self.assertTrue(r.cancel())
        self._pump(lambda: cancels, timeout=5.0)
        self.assertTrue(cancels, "cancelled not emitted")
        self.assertFalse(r.busy())
        self.assertEqual(list(self.root.glob("task_*")), [])
        # immediately start another task on the same runner
        r2_script = _write_script(self.td.name, "ok.py", OK_WORKER)
        r._command_builder = _builder(r2_script)
        got = {}
        r.finished.connect(lambda d: got.setdefault("data", d))
        r.start("s", {})
        self.assertTrue(self._pump(lambda: "data" in got), "second task did not finish")
        self.assertEqual(got["data"]["ok"], True)

    def test_job_write_failure_cleans_dir(self):
        r = self._runner(OK_WORKER)

        class Bad:
            def __str__(self):
                raise TypeError("cannot serialize")

        with self.assertRaises(TypeError):
            r.start("ccd", {"bad": Bad()})
        self.assertFalse(r.busy())
        self.assertEqual(list(self.root.glob("task_*")), [])

    def test_stale_cleanup_on_init(self):
        stale = self.root / "task_stale"
        stale.mkdir(parents=True)
        (stale / "job.json").write_text("{}", encoding="utf-8")
        self._runner(OK_WORKER)
        self.assertEqual(list(self.root.glob("task_*")), [])


if __name__ == "__main__":
    unittest.main()

