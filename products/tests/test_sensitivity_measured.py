"""Measured-sensitivity estimator tests (SYNTHETIC-VALIDATED evidence).

The estimator tests drive measure_sensitivity with a scripted invert() so the
pose/drift/repeat math is checked exactly; the closed-loop test runs the real
PD chain on synthetic frames generated from a known ground-truth S.
"""
import unittest

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
