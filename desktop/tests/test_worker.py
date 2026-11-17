import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class WorkerDispatchTest(unittest.TestCase):
    def _run_worker(self, task, payload):
        with tempfile.TemporaryDirectory() as td:
            job = Path(td) / "job.json"
            result = Path(td) / "result.json"
            job.write_text(json.dumps(payload), encoding="utf-8")
            subprocess.run([sys.executable, str(ROOT / "desktop" / "qt_app.py"),
                            "--worker", task, "--job", str(job), "--result", str(result)],
                           check=True, cwd=str(ROOT), timeout=120)
            return json.loads(result.read_text(encoding="utf-8"))

    def test_no_qt_in_worker(self):
        code = ("import sys; sys.path.insert(0, r'%s'); "
                "import desktop.worker; "
                "assert 'PySide6' not in sys.modules; print('ok')" % ROOT)
        out = subprocess.check_output([sys.executable, "-c", code], cwd=str(ROOT), text=True)
        self.assertIn("ok", out)

    def test_s_task(self):
        d = self._run_worker("s", {})
        self.assertEqual(len(d["values"]), 3)
        self.assertEqual(d["rank"], 3)

    def test_ccd_task(self):
        d = self._run_worker("ccd", {"c4": 0.3, "c8": 0.1, "c9": -0.05,
                                     "noise": 2, "stars": 8, "defocus": [-1, 0, 1]})
        self.assertEqual(len(d["psf"]), 3)
        self.assertEqual(len(d["psf"][0]), 64)


if __name__ == "__main__":
    unittest.main()
