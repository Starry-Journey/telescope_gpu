import base64
import json
import struct
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]


class WorkerDispatchTest(unittest.TestCase):
    def _run_worker(self, task, payload):
        from desktop.protocol import make_job
        with tempfile.TemporaryDirectory() as td:
            job = Path(td) / "job.json"
            result = Path(td) / "result.json"
            job.write_text(json.dumps(make_job(task, payload, "test-task")), encoding="utf-8")
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

    def test_fea_viz_task(self):
        u = np.linspace(-400.0, 400.0, 12)
        xx, yy = np.meshgrid(u, u)
        zz = 120.0 * np.exp(-(xx ** 2 + yy ** 2) / 9.0e4) + 0.02 * xx
        payload = {"x": xx.ravel().tolist(), "y": yy.ravel().tolist(),
                   "z": zz.ravel().tolist(), "label": "synthetic_12x12",
                   "azim": -60, "elev": 25}
        d = self._run_worker("fea_viz", payload)
        self.assertGreater(len(d["png3d_b64"]), 2000)
        stats = d["stats"]
        for key in ("n", "pv_nm", "rms_nm", "obscuration", "masked_obscuration"):
            self.assertIn(key, stats)
        self.assertEqual(stats["n"], 144)
        self.assertEqual(stats["masked_obscuration"], 0)
        self.assertEqual(d["label"], "synthetic_12x12")
        g = d["grid"]
        self.assertEqual(len(g["x"]), len(g["y"]))
        self.assertEqual(len(g["z"]), len(g["y"]))
        self.assertEqual(len(g["z"][0]), len(g["x"]))
        # compact PNG contract: <= 400x300 px and < 300 KB
        raw = base64.b64decode(d["png3d_b64"])
        self.assertLess(len(raw), 300 * 1024)
        w, h = struct.unpack(">II", raw[16:24])
        self.assertLessEqual(w, 400)
        self.assertLessEqual(h, 300)

    def test_sens_measure_task(self):
        # one-dof minimal bracketed series, frames from the single forward
        try:
            import nlopt  # noqa: F401
        except Exception:
            self.skipTest("nlopt not available")
        from products.optics.forward import forward_stack_numpy
        gain = 4.04  # Z4 waves per mm on the proxy m2_focus axis
        frames = {
            "ref": forward_stack_numpy([0.02, 0.0, 0.0], n=32).tolist(),
            "pos": forward_stack_numpy([0.02 + gain * 0.05, 0.0, 0.0], n=32).tolist(),
            "neg": forward_stack_numpy([0.02 - gain * 0.05, 0.0, 0.0], n=32).tolist(),
        }
        payload = {
            "poses": [
                {"dof": None, "level": 0.0, "frames": frames["ref"], "timestamp": 0},
                {"dof": "m2_focus", "level": 0.05, "frames": frames["pos"], "timestamp": 1},
                {"dof": None, "level": 0.0, "frames": frames["ref"], "timestamp": 2},
                {"dof": "m2_focus", "level": -0.05, "frames": frames["neg"], "timestamp": 3},
                {"dof": None, "level": 0.0, "frames": frames["ref"], "timestamp": 4},
            ],
            "maxeval": 150, "n_starts": 2,
        }
        d = self._run_worker("sens_measure", payload)
        self.assertEqual(d["terms"], ["Z4", "Z8", "Z9"])
        self.assertEqual(d["dofs"], ["m2_focus"])
        self.assertAlmostEqual(d["S"][0][0], gain, delta=0.25)


if __name__ == "__main__":
    unittest.main()
