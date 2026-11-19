import json
import unittest
from pathlib import Path

import numpy as np

from products.control import service as fw

ROOT = Path(__file__).resolve().parents[2]
REAL = ROOT / "data" / "zemax" / "sensitivity_zemax.json"


class FrameworkServiceTest(unittest.TestCase):
    def test_metadata(self):
        m = fw.metadata()
        self.assertEqual(m["target_names"], ["Z4", "Z8", "Z9"])
        self.assertEqual(len(m["control_names"]), 5)
        self.assertEqual(m["control_names"][0], "m2_focus")
        self.assertEqual(len(m["control_meta"]), 5)
        self.assertEqual(len(m["target"]), 3)

    def test_proxy_sensitivity(self):
        s = fw.sensitivity(1e-3, source="proxy")
        self.assertEqual(s["rank"], 3)
        self.assertEqual(s["nullspace"], 2)
        self.assertAlmostEqual(s["condition"], 1.0897071937446394, places=6)
        self.assertEqual(len(s["values"]), 3)
        self.assertEqual(len(s["values"][0]), 5)
        self.assertAlmostEqual(s["values"][0][0], 4.04, places=6)
        self.assertEqual(s["source"], "synthetic-proxy")

    def test_control_residual_proxy(self):
        m = fw.metadata()
        c = fw.control(m["target"], lam=1e-8, use_bounds=True, source="proxy")
        self.assertEqual(c["rank"], 3)
        self.assertEqual(c["nullspace"], 2)
        self.assertLess(max(abs(v) for v in c["residual"]), 1e-8)
        self.assertEqual(len(c["u"]), 5)

    def test_bad_target(self):
        with self.assertRaises(ValueError):
            fw.control([0.0, 1.0], lam=0.0)

    def test_bad_source(self):
        with self.assertRaises(ValueError):
            fw.sensitivity(source="whatever")

    @unittest.skipUnless(REAL.exists(), "real Zemax matrix not present")
    def test_real_sensitivity_loaded(self):
        doc = json.loads(REAL.read_text(encoding="utf-8"))
        s = fw.sensitivity(source="zemax")
        self.assertEqual(s["source"], "zemax-zos-api")
        self.assertEqual(s["label"], "EXTERNAL-CHECK")
        S = np.asarray(s["values"], dtype=float)
        self.assertEqual(S.shape, (3, 5))
        np.testing.assert_allclose(S, np.asarray(doc["matrix"]), atol=1e-12)
        self.assertEqual(s["rank"], 3)
        self.assertGreater(s["condition"], 1.0)

    @unittest.skipUnless(REAL.exists(), "real Zemax matrix not present")
    def test_real_control_reaches_target(self):
        doc = json.loads(REAL.read_text(encoding="utf-8"))
        S = np.asarray(doc["matrix"], dtype=float)
        target = (S @ np.array([0.02, 0.01, -0.01, 0.005, -0.005])).tolist()
        c = fw.control(target, lam=0.0, use_bounds=False, source="zemax")
        self.assertLess(max(abs(v) for v in c["residual"]), 1e-9)
        self.assertEqual(c["source"], "zemax-zos-api")

    @unittest.skipUnless(REAL.exists(), "real Zemax matrix not present")
    def test_auto_prefers_real(self):
        self.assertEqual(fw.sensitivity()["source"], "zemax-zos-api")


if __name__ == "__main__":
    unittest.main()
