import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HAS_QT = importlib.util.find_spec("PySide6") is not None
if HAS_QT:
    from desktop.tasks import build_worker_command


@unittest.skipUnless(HAS_QT, "PySide6 not installed; covered by desktop-smoke CI")
class WorkerCommandTest(unittest.TestCase):
    def test_source_points_to_entrypoint(self):
        prog, args = build_worker_command("ccd", "job.json", "res.json", frozen=False,
                                          root=ROOT)
        self.assertTrue(args[0].endswith("desktop/qt_app.py"))
        self.assertIn("--worker", args)

    def test_frozen_uses_exe(self):
        prog, args = build_worker_command("s", "j", "r", frozen=True, root=ROOT)
        self.assertEqual(args[0], "--worker")


if __name__ == "__main__":
    unittest.main()
