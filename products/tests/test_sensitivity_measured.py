"""Measured-sensitivity estimator tests (SYNTHETIC-VALIDATED evidence).

The estimator tests drive measure_sensitivity with a scripted invert() so the
pose/drift/repeat math is checked exactly; the closed-loop test runs the real
PD chain on synthetic frames generated from a known ground-truth S.
"""
import json
import unittest
from pathlib import Path

import numpy as np

from products.sensitivity.measured import (Pose, measure_sensitivity,
                                           synthetic_poses)

try:
    import nlopt  # noqa: F401
    HAVE_NLOPT = True
except Exception:
    HAVE_NLOPT = False

S3 = np.array([
    [4.040, 0.083, -0.083],
    [0.000, -4.633, 0.000],
    [0.000, 0.000, -4.633],
])
DOFS3 = ["m2_focus", "m2_decenter_x", "m2_decenter_y"]
STATIC = np.array([0.02, -0.01, 0.005])
DUMMY_FRAMES = np.zeros((5, 8, 8))


def series_from_states(states, dofs, levels, repeats=1, noise=0.0, seed=0):
    """Build poses + matching z list so a scripted invert() replays them.

    levels: positive amplitudes; each is visited as +A then -A, bracketed
    by reference poses (u = 0) on both sides.
    """
    zs, poses = [], []
    t = 0.0
    for rep in range(repeats):
        zs.append(np.array(states(0.0, np.zeros(len(dofs)))))
        poses.append(Pose(None, 0.0, DUMMY_FRAMES, timestamp=t, repeat=rep))
        t += 1.0
        for j, dof in enumerate(dofs):
            for amp in levels:
                for sign in (+1.0, -1.0):
                    u = np.zeros(len(dofs))
                    u[j] = sign * float(amp)
                    zs.append(np.array(states(t, u)))
                    poses.append(Pose(dof, float(u[j]), DUMMY_FRAMES,
                                      timestamp=t, repeat=rep))
                    t += 1.0
                    zs.append(np.array(states(t, np.zeros(len(dofs)))))
                    poses.append(Pose(None, 0.0, DUMMY_FRAMES, timestamp=t,
                                      repeat=rep))
                    t += 1.0
    if noise > 0:
        rng = np.random.default_rng(seed)
        zs = [z + rng.normal(scale=noise, size=3) for z in zs]
    return poses, zs


def scripted_invert(zs):
    it = iter(zs)

    def inv(frames):
        z = next(it)
        return {"zernike": {"Z4": float(z[0]), "Z8": float(z[1]),
                            "Z9": float(z[2])},
                "cost": 0.0, "residual_rms": 0.0}
    return inv


class MeasuredSensitivityTest(unittest.TestCase):
    def test_exact_recovery(self):
        poses, zs = series_from_states(
            lambda t, u: STATIC + S3 @ u, DOFS3, [0.05])
        r = measure_sensitivity(poses, invert=scripted_invert(zs))
        np.testing.assert_allclose(r["S"], S3, atol=1e-10)
        self.assertEqual(r["terms"], ["Z4", "Z8", "Z9"])
        self.assertEqual(r["dofs"], DOFS3)
        self.assertEqual(r["rank"], 3)
        self.assertEqual(r["nullspace_dim"], 0)
        sv = np.linalg.svd(S3, compute_uv=False)
        self.assertAlmostEqual(r["condition"], float(sv[0] / sv[-1]), places=9)

    def test_linear_drift_removed(self):
        drift = np.array([2e-4, -1e-4, 3e-4])
        poses, zs = series_from_states(
            lambda t, u: STATIC + drift * t + S3 @ u, DOFS3, [0.05])
        r = measure_sensitivity(poses, invert=scripted_invert(zs))
        np.testing.assert_allclose(r["S"], S3, atol=1e-10)
        self.assertLess(r["drift_rms"], 1e-12)

    def test_repeats_and_std(self):
        poses, zs = series_from_states(
            lambda t, u: STATIC + S3 @ u, DOFS3, [0.05],
            repeats=3, noise=2e-4, seed=3)
        r = measure_sensitivity(poses, invert=scripted_invert(zs))
        err = np.abs(np.asarray(r["S"]) - S3)
        std = np.asarray(r["S_std"])
        self.assertTrue(np.all(std > 0))
        # noise 2e-4 waves through +-0.05 differencing -> std ~ 2.8e-3
        self.assertLess(float(np.max(err / std)), 8.0)

    def test_nullspace_reported(self):
        S4 = np.zeros((3, 4))
        S4[:, :3] = S3
        dofs4 = DOFS3 + ["dead_axis"]
        poses, zs = series_from_states(
            lambda t, u: STATIC + S4 @ u, dofs4, [0.05])
        r = measure_sensitivity(poses, invert=scripted_invert(zs))
        self.assertEqual(r["rank"], 3)
        self.assertEqual(r["nullspace_dim"], 1)
        v = np.asarray(r["nullspace_vectors"])[0]
        self.assertAlmostEqual(abs(v[-1]), 1.0, places=9)
        np.testing.assert_allclose(v[:3], 0.0, atol=1e-9)

    def test_asymmetric_bracket_p2_1(self):
        """P2-1: bracket -a1/+a2 (a1 != a2) uses the commanded span as denominator."""
        a_minus, a_plus = 0.03, 0.05
        poses, zs = [], []
        t = 0.0
        plan = [(None, 0.0), ("d0", a_plus), (None, 0.0),
                ("d0", -a_minus), (None, 0.0)]
        for dof, lv in plan:
            u = np.zeros(3)
            if dof:
                u[0] = lv
            zs.append(STATIC + S3 @ u)
            poses.append(Pose(dof, lv, DUMMY_FRAMES, timestamp=t))
            t += 1.0
        it = iter(zs)

        def inv(frames):
            z = next(it)
            return {"zernike": {"Z4": float(z[0]), "Z8": float(z[1]),
                                "Z9": float(z[2])},
                    "cost": 0.0, "residual_rms": 0.0}

        r = measure_sensitivity(poses, invert=inv)
        # exact slope over the asymmetric span
        np.testing.assert_allclose(np.asarray(r["S"])[:, 0], S3[:, 0],
                                   atol=1e-10)

    def test_cubic_nonlinearity_flagged(self):
        a = 0.05

        def states(t, u):
            z = STATIC + S3 @ u
            z[0] += 3.0 * u[0] ** 3          # cubic response on dof 0
            return z

        poses, zs = series_from_states(
            states, DOFS3, [0.5 * a, a])     # two amplitudes -> slope fit
        r = measure_sensitivity(poses, invert=scripted_invert(zs))
        self.assertGreater(r["nonlinearity"]["m2_focus"], 1e-9)
        # slope fit still lands within ~a^2 of the true linear gain
        self.assertAlmostEqual(r["S"][0][0], S3[0][0], delta=3.0 * a ** 2)

    def test_unbracketed_warning(self):
        # excitation before the first reference -> extrapolated drift model
        poses = [Pose("d0", 0.05, DUMMY_FRAMES, timestamp=0.0),
                 Pose(None, 0.0, DUMMY_FRAMES, timestamp=1.0),
                 Pose("d0", -0.05, DUMMY_FRAMES, timestamp=2.0),
                 Pose(None, 0.0, DUMMY_FRAMES, timestamp=3.0)]
        zs = [STATIC + S3[:, 0] * 0.05, STATIC,
              STATIC - S3[:, 0] * 0.05, STATIC]
        r = measure_sensitivity(poses, invert=scripted_invert(zs))
        # only the first excitation lacks a bracketing reference on one side
        self.assertEqual(r["unbracketed_poses"], 1)
        self.assertTrue(r["warning"])
        np.testing.assert_allclose(np.asarray(r["S"])[:, 0], S3[:, 0],
                                   atol=1e-10)


class ZemaxCrossCheckTest(unittest.TestCase):
    """measured S vs the real ZOS-API matrix (EXTERNAL-CHECK column)."""

    REAL = (Path(__file__).resolve().parents[2] / "data" / "zemax"
            / "sensitivity_zemax.json")

    @unittest.skipUnless(REAL.exists(), "real Zemax matrix not present")
    def test_identical_matrix_has_zero_difference(self):
        from products.sensitivity.measured import compare_with_zemax
        doc = json.loads(self.REAL.read_text(encoding="utf-8"))
        S = np.asarray(doc["matrix"], dtype=float)
        result = {"S": S.tolist(), "terms": doc["targets"],
                  "dofs": doc["controls"]}
        cmp = compare_with_zemax(result, measured_wavelength_nm=700.0)
        self.assertTrue(cmp["available"])
        self.assertTrue(cmp["comparable"])
        self.assertEqual(cmp["label"], "EXTERNAL-CHECK")
        self.assertAlmostEqual(cmp["max_abs_rel"], 0.0, places=12)

    @unittest.skipUnless(REAL.exists(), "real Zemax matrix not present")
    def test_scaled_matrix_reports_relative_difference(self):
        from products.sensitivity.measured import compare_with_zemax
        doc = json.loads(self.REAL.read_text(encoding="utf-8"))
        S = np.asarray(doc["matrix"], dtype=float) * 1.5
        result = {"S": S.tolist(), "terms": doc["targets"],
                  "dofs": doc["controls"]}
        cmp = compare_with_zemax(result, measured_wavelength_nm=700.0)
        self.assertAlmostEqual(cmp["max_abs_rel"], 0.5, places=9)
        focus = cmp["per_column"]["m2_focus"]
        self.assertAlmostEqual(focus["relative_norm_diff"], 0.5, places=9)

    def test_partial_dof_overlap(self):
        from products.sensitivity.measured import compare_with_zemax
        try:
            import json as _json
            doc = _json.loads(self.REAL.read_text(encoding="utf-8"))
        except FileNotFoundError:
            self.skipTest("real Zemax matrix not present")
        S = np.asarray(doc["matrix"], dtype=float)
        keep = [0, 1]                       # Z4 row, Z8 row; focus + decenter_x
        result = {"S": S[keep][:, :2].tolist(), "terms": ["Z4", "Z8"],
                  "dofs": ["m2_focus", "m2_decenter_x"]}
        cmp = compare_with_zemax(result, measured_wavelength_nm=700.0)
        self.assertEqual(cmp["terms"], ["Z4", "Z8"])
        self.assertEqual(cmp["dofs"], ["m2_focus", "m2_decenter_x"])
        self.assertAlmostEqual(cmp["max_abs_rel"], 0.0, places=12)

    def test_missing_file_is_reported(self):
        from products.sensitivity.measured import compare_with_zemax
        cmp = compare_with_zemax({"S": [[1.0]], "terms": ["Z4"],
                                  "dofs": ["m2_focus"]},
                                 path="C:/does/not/exist.json")
        self.assertFalse(cmp["available"])
        self.assertIn("note", cmp)


    def test_wavelength_alignment_p1_2(self):
        """P1-2: measured@632.8 vs zemax@700 aligns to ~0 relative difference."""
        doc = json.loads(self.REAL.read_text(encoding="utf-8"))
        S = np.asarray(doc["matrix"], dtype=float)
        lam = 700.0 / 632.8
        result = {"S": (S * lam).tolist(), "terms": doc["targets"],
                  "dofs": doc["controls"]}
        from products.sensitivity.measured import compare_with_zemax
        cmp = compare_with_zemax(result)   # default measured_wavelength_nm=632.8
        self.assertLess(cmp["max_abs_rel"], 1e-9)


class ClosedLoopPDTest(unittest.TestCase):
    @unittest.skipUnless(HAVE_NLOPT, "nlopt not available")
    def test_real_pd_chain_recovers_S(self):
        from products.pd.service import invert_psf
        from products.control.service import A, CONTROL_NAMES
        S_true = A[:, :2]
        dofs = CONTROL_NAMES[:2]
        # PD accuracy scan (see docs/实测灵敏度标定设计.md): n=32 with
        # per-pixel noise <=3e-4 and maxeval>=150 recovers z to ~0.01 waves.
        poses = synthetic_poses(S_true, dofs, amplitude=0.05, n=32,
                                noise=3e-4, seed=5)
        r = measure_sensitivity(
            poses,
            invert=lambda fr: invert_psf(fr, (-2.0, -1.0, 0.0, 1.0, 2.0),
                                         None, maxeval=150, n_starts=2,
                                         seed=5),
            label="SYNTHETIC-VALIDATED")
        S_meas = np.asarray(r["S"])
        err = float(np.sqrt(np.mean((S_meas - S_true) ** 2)))
        corr = float(np.corrcoef(S_meas.ravel(), S_true.ravel())[0, 1])
        self.assertGreater(corr, 0.95)
        self.assertLess(err, 0.25)
        self.assertEqual(r["label"], "SYNTHETIC-VALIDATED")


if __name__ == "__main__":
    unittest.main()
