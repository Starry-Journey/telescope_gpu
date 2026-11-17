import unittest

from products.ccd.service import run_ccd_pipeline
from products.pd.service import invert_psf
from products.sensitivity import matrix_service as sm


class ProductModulesTest(unittest.TestCase):
    def test_ccd_pipeline_and_pd(self):
        r = run_ccd_pipeline(c4=0.3, c8=0.1, c9=-0.05, noise=2,
                             stars=8, defocus=[-1, 0, 1], seed=5)
        self.assertEqual(r["metrics"]["n_frames"], 3)
        self.assertEqual(len(r["psf"]), 3)
        self.assertEqual(len(r["psf"][0]), 64)
        pd = invert_psf(r["psf"], r["defocus"], r["truth"], maxeval=150)
        self.assertEqual(pd["nfev"] > 0, True)
        self.assertIn("Z4", pd["zernike"])

    def test_custom_matrix(self):
        S = [[4.04, 0.083, -0.083, 0.976, -0.976],
             [0, -4.633, 0, 0, 0],
             [0, 0, -4.633, 0, 0]]
        a = sm.analyze(S)
        self.assertEqual(a["shape"], [3, 5])
        self.assertEqual(a["rank"], 3)
        self.assertEqual(a["nullspace"], 2)
        c = sm.control(S, [0.23377, -0.09266, 0.04633], lam=1e-8, use_bounds=False)
        self.assertLess(max(abs(v) for v in c["residual"]), 1e-8)


if __name__ == "__main__":
    unittest.main()
