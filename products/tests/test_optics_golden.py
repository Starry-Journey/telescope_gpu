"""Golden vectors for the single-source Zernike/forward implementation.

Expected values are analytic values of the normalized Zernike polynomials at
rho=0.5.  Any change to term order or normalization must update this table
deliberately.
"""
import unittest

import numpy as np

from products.optics import forward as fwd
from products.optics import zernike as z


class ZernikeGoldenTest(unittest.TestCase):
    RHO = np.array([0.5])
    TH0 = np.array([0.0])
    TH90 = np.array([np.pi / 2])

    def val(self, n, m, theta):
        return float(z.zernike(n, m, self.RHO, theta)[0])

    def test_known_values_theta0(self):
        cases = {
            (0, 0): 1.0,
            (1, 1): 1.0,
            (1, -1): 0.0,
            (2, 0): -0.86602540,
            (2, -2): 0.0,
            (2, 2): 0.61237244,
            (3, -1): 0.0,
            (3, 1): -1.76776695,
            (3, -3): 0.0,
            (3, 3): 0.35355339,
            (4, 0): -0.27950850,
        }
        for (n, m), expected in cases.items():
            self.assertAlmostEqual(self.val(n, m, self.TH0), expected, places=6,
                                   msg=f"Z({n},{m}) at theta=0")

    def test_known_values_theta90(self):
        cases = {
            (1, -1): 1.0,
            (3, -1): -1.76776695,
            (3, -3): -0.35355339,
        }
        for (n, m), expected in cases.items():
            self.assertAlmostEqual(self.val(n, m, self.TH90), expected, places=6,
                                   msg=f"Z({n},{m}) at theta=90")

    def test_term_table(self):
        self.assertEqual(len(z.TERMS), 66)
        self.assertEqual(z.TERMS[3], (2, 0))   # Z4 defocus
        self.assertEqual(z.TERMS[6], (3, -1))  # Z9 project coma y
        self.assertEqual(z.TERMS[7], (3, 1))   # Z8 project coma x
        self.assertEqual(z.TERMS[8], (3, -3))

    def test_forward_single_source(self):
        a = fwd.forward_psf_numpy([0.1, 0.02, -0.01], 0.5, n=32)
        b = fwd.forward_psf_numpy([0.1, 0.02, -0.01], 0.5, n=32)
        np.testing.assert_allclose(a, b)
        self.assertAlmostEqual(float(a.sum()), 1.0, places=9)

    def test_ml_compat_reexport(self):
        from ml.traditional import forward_psf_numpy as legacy
        a = fwd.forward_psf_numpy([0.1, 0.02, -0.01], 0.5, n=16)
        b = legacy([0.1, 0.02, -0.01], 0.5, n=16)
        np.testing.assert_allclose(a, b, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
