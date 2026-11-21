"""Phase-diversity N-term state tests (3-term compatibility + 6-term round trip).

Every existing Web/desktop/batch caller uses the historical 3-term state
(Z4/Z8/Z9), so its forward output is pinned against a stored golden vector and
against the explicit low_order_grid() reference.  The 6-term engineering state
(Z4/Z5/Z6/Z8/Z9/Z11) adds astigmatism and spherical and is validated by a
noiseless forward -> invert round trip.
"""
import unittest

import numpy as np

from products.optics import forward as fwd
from products.optics import zernike as z

try:
    import nlopt  # noqa: F401
    HAVE_NLOPT = True
except Exception:
    HAVE_NLOPT = False

SIX_TERMS = ("Z4", "Z5", "Z6", "Z8", "Z9", "Z11")
DEFOCUS = (-2.0, -1.0, 0.0, 1.0, 2.0)


def _legacy_three_term_psf(coeff, defocus, n=16):
    """Frozen copy of the pre-N-term forward model (low_order_grid only).

    Kept inline on purpose: it is the stored expectation the generalized
    forward_psf_numpy() default must still reproduce.
    """
    fft_n = n * 2
    mask, z4, z8, z9 = z.low_order_grid(fft_n)
    phase = 2 * np.pi * (coeff[0] * z4 + coeff[1] * z8 + coeff[2] * z9
                         + defocus * z4)
    pupil = mask * np.exp(1j * phase)
    field = np.fft.fftshift(np.fft.fft2(pupil, norm="ortho"))
    psf = np.abs(field) ** 2
    psf /= max(psf.sum(), 1e-12)
    psf = psf.reshape(n, 2, n, 2).mean(axis=(1, 3))
    psf /= max(psf.sum(), 1e-12)
    return psf


class ZernikeTermMapTest(unittest.TestCase):
    def test_term_map_follows_project_label(self):
        # Project/UI convention: Z8 = x coma (3,1), Z9 = y coma (3,-1).
        self.assertEqual(z.PROJECT_LABEL[8], "Z8")
        self.assertEqual(z.PROJECT_LABEL[7], "Z9")
        self.assertEqual(z.PROJECT_TERM_MAP["Z4"], z.TERMS[3])    # defocus
        self.assertEqual(z.PROJECT_TERM_MAP["Z5"], z.TERMS[4])
        self.assertEqual(z.PROJECT_TERM_MAP["Z6"], z.TERMS[5])
        self.assertEqual(z.PROJECT_TERM_MAP["Z8"], z.TERMS[7])
        self.assertEqual(z.PROJECT_TERM_MAP["Z9"], z.TERMS[6])
        # Z11 has no PROJECT_LABEL override: default f"Z{i}" slot (4,0).
        self.assertEqual(z.PROJECT_TERM_MAP["Z11"], z.TERMS[10])

    def test_default_terms_unchanged(self):
        self.assertEqual(z.DEFAULT_TERMS, ("Z4", "Z8", "Z9"))
        self.assertEqual(z.resolve_terms(None), ("Z4", "Z8", "Z9"))
        self.assertEqual(z.resolve_terms(SIX_TERMS), SIX_TERMS)

    def test_default_grids_match_low_order_grid(self):
        for n in (16, 32):
            mask, z4, z8, z9 = z.low_order_grid(n, 0.45, 0.28)
            g = z.term_grids(n, ("Z4", "Z8", "Z9"), 0.45, 0.28)
            self.assertEqual(tuple(g), ("Z4", "Z8", "Z9"))
            np.testing.assert_allclose(g["Z4"], z4, rtol=0, atol=1e-12)
            np.testing.assert_allclose(g["Z8"], z8, rtol=0, atol=1e-12)
            np.testing.assert_allclose(g["Z9"], z9, rtol=0, atol=1e-12)


class ThreeTermForwardCompatTest(unittest.TestCase):
    COEFF = (0.1, 0.02, -0.01)
    DEFOCUS = 0.5

    def test_forward_matches_stored_golden(self):
        psf = fwd.forward_psf_numpy(self.COEFF, self.DEFOCUS, n=16)
        self.assertEqual(psf.shape, (16, 16))
        self.assertAlmostEqual(float(psf.sum()), 1.0, places=9)
        pixels = ((8, 8), (8, 9), (7, 8), (6, 6), (10, 3))
        # Values stored when the fixed 3-term model was the only model.
        expected = np.array([0.002055258913138504, 0.004588964209617494,
                             0.002949694710605296, 0.0023625094714022714,
                             0.008159953791612261])
        got = np.array([float(psf[i, j]) for i, j in pixels])
        np.testing.assert_allclose(got, expected, rtol=0, atol=1e-12)

    def test_forward_matches_legacy_three_term_formula(self):
        cases = (((0.1, 0.02, -0.01), 0.5, 16),
                 ((0.0, 0.0, 0.0), 0.0, 16),
                 ((0.2, 0.05, -0.03), -1.0, 24))
        for coeff, defocus, n in cases:
            got = fwd.forward_psf_numpy(coeff, defocus, n=n)
            ref = _legacy_three_term_psf(coeff, defocus, n=n)
            np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)

    def test_stack_default_unchanged(self):
        stack = fwd.forward_stack_numpy(self.COEFF, n=16, defocus=DEFOCUS)
        ref = np.stack([_legacy_three_term_psf(self.COEFF, d, n=16)
                        for d in DEFOCUS])
        np.testing.assert_allclose(stack, ref, rtol=0, atol=1e-12)


class SixTermForwardTest(unittest.TestCase):
    def test_extra_terms_enter_the_model(self):
        # Zero new-term coefficients must reduce to the 3-term PSF exactly;
        # nonzero Z5/Z11 must actually change it.
        base3 = (0.1, 0.02, -0.01)
        six_zero = (0.1, 0.0, 0.0, 0.02, -0.01, 0.0)   # order: Z4 Z5 Z6 Z8 Z9 Z11
        a = fwd.forward_psf_numpy(six_zero, 0.5, n=16, terms=SIX_TERMS)
        b = fwd.forward_psf_numpy(base3, 0.5, n=16)
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-12)
        c = fwd.forward_psf_numpy((0.1, 0.03, -0.02, 0.02, -0.01, 0.02),
                                  0.5, n=16, terms=SIX_TERMS)
        self.assertGreater(float(np.max(np.abs(c - b))), 1e-6)


class UnknownTermTest(unittest.TestCase):
    def test_unknown_labels_raise(self):
        for bad in (("Z4", "Z99"), ("Z7",), ("z4",), ("Z4", "Z4")):
            with self.assertRaises(ValueError, msg=f"term_grids {bad}"):
                z.term_grids(8, bad)
            with self.assertRaises(ValueError, msg=f"forward {bad}"):
                fwd.forward_psf_numpy([0.0] * len(bad), 0.0, n=8, terms=bad)

    @unittest.skipUnless(HAVE_NLOPT, "nlopt not available")
    def test_invert_unknown_labels_raise(self):
        from products.pd.service import invert_psf
        stack = fwd.forward_stack_numpy([0.0, 0.0, 0.0], n=8, defocus=(-1.0, 1.0))
        for bad in (("Z4", "Z99"), ("Z7",), ("z4",), ("Z4", "Z4")):
            with self.assertRaises(ValueError, msg=f"invert {bad}"):
                invert_psf(stack, (-1.0, 1.0), terms=bad)


@unittest.skipUnless(HAVE_NLOPT, "nlopt not available")
class SixTermRoundTripTest(unittest.TestCase):
    N = 32
    TRUTH = (0.2, 0.05, -0.03, 0.04, -0.03, 0.02)

    def test_round_trip_recovers_6_term_state(self):
        from products.pd.service import invert_psf
        stack = fwd.forward_stack_numpy(self.TRUTH, n=self.N, defocus=DEFOCUS,
                                        terms=SIX_TERMS)
        self.assertEqual(stack.shape, (len(DEFOCUS), self.N, self.N))
        result = invert_psf(stack, DEFOCUS, self.TRUTH, maxeval=400,
                            n_starts=2, seed=7, terms=SIX_TERMS)
        self.assertEqual(sorted(result["zernike"]), sorted(SIX_TERMS))
        self.assertEqual(result["terms"], list(SIX_TERMS))
        self.assertEqual(sorted(result["truth"]), sorted(SIX_TERMS))
        for label, expected in zip(SIX_TERMS, self.TRUTH):
            self.assertAlmostEqual(result["zernike"][label], expected, delta=0.05,
                                   msg=f"6-term recovery {label}")
        # Noiseless stack, so the recovered state must reach the global minimum.
        self.assertLess(result["cost"], 1e-12)


if __name__ == "__main__":
    unittest.main()
