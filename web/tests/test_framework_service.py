import unittest

import numpy as np

from web import framework_service as fw


class FrameworkServiceTest(unittest.TestCase):
    def test_metadata(self):
        m = fw.metadata()
        self.assertEqual(m["target_names"], ["Z4", "Z8", "Z9"])
        self.assertEqual(len(m["control_names"]), 5)
        self.assertEqual(m["control_names"][0], "m2_focus")
        self.assertEqual(len(m["control_meta"]), 5)
        self.assertEqual(len(m["target"]), 3)

    def test_sensitivity(self):
        s = fw.sensitivity(1e-3)
        self.assertEqual(s["rank"], 3)
        self.assertEqual(s["nullspace"], 2)
        self.assertAlmostEqual(s["condition"], 1.0897071937446394, places=6)
        self.assertEqual(len(s["values"]), 3)
        self.assertEqual(len(s["values"][0]), 5)
        self.assertAlmostEqual(s["values"][0][0], 4.04, places=6)

    def test_control_residual(self):
        m = fw.metadata()
        c = fw.control(m["target"], lam=1e-8, use_bounds=True)
        self.assertEqual(c["rank"], 3)
        self.assertEqual(c["nullspace"], 2)
        self.assertLess(max(abs(v) for v in c["residual"]), 1e-8)
        self.assertEqual(len(c["u"]), 5)

    def test_bad_target(self):
        with self.assertRaises(ValueError):
            fw.control([0.0, 1.0], lam=0.0)


if __name__ == "__main__":
    unittest.main()
