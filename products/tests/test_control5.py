"""口径 v2 tests: stacked [S_Z; P] 5x5 control model + S_F interpretation."""
import json
import unittest
from pathlib import Path

import numpy as np

from products.control import service as f

ROOT = Path(__file__).resolve().parents[2]
FULL = ROOT / "data" / "zemax" / "sensitivity_zemax_full.json"


@unittest.skipUnless(FULL.exists(), "full [S_Z; P] measurement not present")
class StackedControlTest(unittest.TestCase):
    def test_full_json_shapes_and_astigmatism_evidence(self):
        doc = json.loads(FULL.read_text(encoding="utf-8"))
        self.assertEqual(np.asarray(doc["S_Z"]["matrix"]).shape, (3, 5))
        self.assertEqual(np.asarray(doc["P"]["matrix"]).shape, (2, 5))
        ast = np.asarray(doc["S_astig"]["matrix"], dtype=float)
        # 口径 v2 point 7 evidence: M2 produces no astigmatism
        self.assertLess(float(np.max(np.abs(ast))), 1e-9)

    def test_stacked_is_square_full_rank(self):
        d = f.sensitivity5()
        A = np.asarray(d["matrix"], dtype=float)
        self.assertEqual(A.shape, (5, 5))
        self.assertEqual(d["rank"], 5)
        self.assertEqual(d["nullspace"], 0)
        self.assertTrue(d["unique_solution"])
        self.assertEqual(d["source"], "zemax-zos-api")

    def test_control5_unique_solution_recovery(self):
        d = f.sensitivity5()
        A = np.asarray(d["matrix"], dtype=float)
        u_true = np.array([0.02, 0.01, -0.01, 0.005, -0.005])
        c = f.control5((A @ u_true).tolist(), use_bounds=False)
        self.assertTrue(c["unique_solution"])
        np.testing.assert_allclose(c["u"], u_true, atol=1e-12)
        self.assertLess(c["wavefront_residual_waves"], 1e-12)
        self.assertLess(c["pointing_residual_arcsec"], 1e-12)

    def test_control5_bounded_keeps_in_box(self):
        d = f.sensitivity5()
        A = np.asarray(d["matrix"], dtype=float)
        big = (A @ np.array([0.4, 0.3, -0.3, 0.2, -0.2])).tolist()
        c = f.control5(big, use_bounds=True)
        u = np.asarray(c["u"], dtype=float)
        self.assertTrue(np.all(u >= f.LOWER - 1e-12))
        self.assertTrue(np.all(u <= f.UPPER + 1e-12))

    def test_control5_rejects_wrong_target_size(self):
        with self.assertRaises(ValueError):
            f.control5([0.0, 0.0, 0.0])

    def test_pointing_row_is_nonzero(self):
        """The pointing rows must be non-degenerate (they break the ambiguity)."""
        d = f.sensitivity5()
        A = np.asarray(d["matrix"], dtype=float)
        self.assertGreater(float(np.max(np.abs(A[3:, :]))), 0.1)


class SemicolonInterpretationTest(unittest.TestCase):
    def test_environment_sensitivity_carries_interpretation(self):
        from products.sensitivity import service as sens
        d = sens.environment_sensitivity(source="abaqus")
        self.assertIn("interpretation", d)
        self.assertIn("background", d["interpretation"])
        self.assertIn("Z8/Z9", d["interpretation"])


if __name__ == "__main__":
    unittest.main()
