"""Tests for the upgraded synthetic CCD pipeline (products/ccd).

Covers backward compatibility with the historical positional signature,
determinism, the physical detector-chain sanity checks (exposure, read noise,
saturation, star density, bad pixels) and the machine-readable metadata
contract the GUI / export layer relies on.
"""
import json
import unittest

import numpy as np

from products.ccd.model import CcdModelConfig, simulate
from products.ccd.service import run_ccd_pipeline


class BackwardCompatibilityTest(unittest.TestCase):
    def test_legacy_positional_call(self):
        # Original signature: c4, c8, c9, noise, stars, defocus, seed.
        r = run_ccd_pipeline(0.3, 0.1, -0.05, 2.0, 8, [-1.0, 0.0, 1.0], 5)

        self.assertEqual(r["metrics"]["n_frames"], 3)
        self.assertEqual(len(r["psf"]), 3)
        psf = np.asarray(r["psf"], dtype=float)
        self.assertEqual(psf.shape, (3, 64, 64))
        self.assertTrue(np.all(np.isfinite(psf)))
        self.assertAlmostEqual(float(psf[0].sum()), 1.0, places=6)
        self.assertGreater(float(psf[0].max()), 0.0)
        cy, cx = np.unravel_index(int(np.argmax(psf[0])), psf[0].shape)
        self.assertLessEqual(abs(int(cx) - 32), 4)
        self.assertLessEqual(abs(int(cy) - 32), 4)

        self.assertEqual(len(r["frames"]), 3)
        for frame in r["frames"]:
            self.assertTrue(frame["raw"].startswith("data:image/png;base64,"))
            self.assertTrue(frame["cal"].startswith("data:image/png;base64,"))
            self.assertIn("defocus", frame)
        self.assertEqual(len(r["psf_png"]), 3)
        self.assertEqual(r["defocus"], [-1.0, 0.0, 1.0])
        self.assertEqual(r["truth"], [0.3, 0.1, -0.05])

        # Historical result keys + metric keys are all still there.
        for key in ("frames", "psf", "psf_png", "defocus", "truth", "star",
                    "metrics", "note", "manifest"):
            self.assertIn(key, r)
        for key in ("n_frames", "shape", "read_noise", "background_median",
                    "psf_peak"):
            self.assertIn(key, r["metrics"])
        self.assertEqual(r["metrics"]["shape"], [192, 192])
        for key in ("x", "y", "flux"):
            self.assertIn(key, r["star"])
        self.assertIsInstance(r["manifest"], dict)

        # The legacy `noise` positional still sets the read noise when the new
        # read_noise_e keyword is left at its default.
        self.assertEqual(r["metrics"]["read_noise"], 2.0)
        self.assertEqual(r["meta"]["read_noise_e"], 2.0)

    def test_return_frames_false(self):
        r = run_ccd_pipeline(seed=3, return_frames=False)
        self.assertEqual(r["frames"], [])
        self.assertEqual(len(r["psf"]), 5)
        self.assertEqual(np.asarray(r["psf"]).shape, (5, 64, 64))
        self.assertEqual(len(r["psf_png"]), 5)

    def test_default_defocus_fallback(self):
        r = run_ccd_pipeline(defocus=[0.0], return_frames=False)
        self.assertGreaterEqual(len(r["defocus"]), 2)
        self.assertEqual(len(r["psf"]), len(r["defocus"]))


class DeterminismTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.a = run_ccd_pipeline(seed=11, return_frames=False)
        cls.b = run_ccd_pipeline(seed=11, return_frames=False)
        cls.c = run_ccd_pipeline(seed=12, return_frames=False)

    def test_same_seed_identical(self):
        self.assertTrue(np.array_equal(np.asarray(self.a["psf"]),
                                       np.asarray(self.b["psf"])))
        self.assertEqual(self.a["metrics"], self.b["metrics"])
        self.assertEqual(self.a["meta"], self.b["meta"])
        self.assertEqual(self.a["manifest"], self.b["manifest"])
        self.assertEqual(self.a["star"], self.b["star"])

    def test_different_seed_differs(self):
        self.assertFalse(np.array_equal(np.asarray(self.a["psf"]),
                                        np.asarray(self.c["psf"])))
        self.assertNotEqual(self.a["manifest"]["catalog"],
                            self.c["manifest"]["catalog"])


class PhysicalSanityTest(unittest.TestCase):
    def test_exposure_increases_total_electrons(self):
        e1 = run_ccd_pipeline(exposure_s=1.0, return_frames=False)
        e3 = run_ccd_pipeline(exposure_s=3.0, return_frames=False)
        self.assertGreater(e3["metrics"]["total_electrons"],
                           e1["metrics"]["total_electrons"])
        self.assertGreater(e3["metrics"]["background_e_per_pix"],
                           e1["metrics"]["background_e_per_pix"])
        self.assertAlmostEqual(
            e3["metrics"]["total_electrons"] / e1["metrics"]["total_electrons"],
            3.0, places=6)

    def test_read_noise_increases_background_rms(self):
        low = run_ccd_pipeline(read_noise_e=1.0, return_frames=False)
        high = run_ccd_pipeline(read_noise_e=30.0, return_frames=False)
        self.assertLess(low["metrics"]["background_rms_adu"], 5.0)
        self.assertGreater(high["metrics"]["background_rms_adu"],
                           low["metrics"]["background_rms_adu"])
        self.assertGreater(high["metrics"]["background_rms_adu"], 20.0)

    def test_saturation_at_bright_mag_zero(self):
        normal = run_ccd_pipeline(return_frames=False)
        bright = run_ccd_pipeline(mag_zero=0.0, return_frames=False)
        self.assertGreater(bright["meta"]["saturation_count"], 0)
        self.assertGreater(bright["metrics"]["saturation_count"],
                           normal["metrics"]["saturation_count"])
        self.assertEqual(bright["meta"]["saturation_count"],
                         bright["metrics"]["saturation_count"])

    def test_more_stars_at_higher_density(self):
        sparse = run_ccd_pipeline(star_density_per_arcmin2=5.0,
                                  return_frames=False)
        dense = run_ccd_pipeline(star_density_per_arcmin2=20.0,
                                 return_frames=False)
        self.assertEqual(sparse["meta"]["n_stars_rendered"], 25)
        self.assertGreater(dense["meta"]["n_stars_rendered"],
                           sparse["meta"]["n_stars_rendered"])
        self.assertEqual(dense["meta"]["n_stars_rendered"], 100)
        self.assertEqual(len(dense["manifest"]["catalog"]),
                         dense["meta"]["n_stars_rendered"])

    def test_bad_pixel_fraction(self):
        clean = run_ccd_pipeline(bad_pixel_fraction=0.0, return_frames=False)
        dirty = run_ccd_pipeline(bad_pixel_fraction=0.01, return_frames=False)
        self.assertEqual(clean["meta"]["bad_pixel_count"], 0)
        self.assertGreater(dirty["meta"]["bad_pixel_count"], 0)
        self.assertEqual(dirty["meta"]["bad_pixel_count"],
                         int(round(0.01 * 192 * 192)))
        self.assertEqual(dirty["meta"]["bad_pixel_count"],
                         dirty["metrics"]["bad_pixel_count"])

    def test_cosmic_ray_rate(self):
        none = run_ccd_pipeline(cosmic_ray_rate=0.0, return_frames=False)
        many = run_ccd_pipeline(cosmic_ray_rate=1e-3, return_frames=False)
        self.assertEqual(none["meta"]["cosmic_ray_count"], 0)
        self.assertGreater(many["meta"]["cosmic_ray_count"], 0)
        self.assertEqual(many["meta"]["cosmic_ray_count"],
                         many["metrics"]["cosmic_ray_count"])

    def test_detector_clipping_and_quantisation(self):
        # Exercise the detector chain directly: 12-bit ADC, bright field.
        cfg = CcdModelConfig(defocus=(0.0,), n_stars=5, mag_zero=0.0,
                             bit_depth=12, saturation_adu=65535, seed=3)
        sim = simulate(cfg)
        raw = np.asarray(sim["raw"][0], dtype=float)
        self.assertEqual(raw.shape, (192, 192))
        self.assertLessEqual(float(raw.max()), 4095.0)
        self.assertGreaterEqual(float(raw.min()), 0.0)
        # Digitised to whole ADU.
        self.assertTrue(np.allclose(raw, np.rint(raw)))
        self.assertGreater(sim["counts"]["saturation_count"], 0)
        self.assertEqual(sim["adu"]["max_adu"], 4095)


class MetadataContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.r = run_ccd_pipeline(seed=7, return_frames=False)

    def test_required_meta_keys(self):
        meta = self.r["meta"]
        for key in ("params", "n_stars_rendered", "fov_arcmin",
                    "plate_scale_arcsec_per_pix", "seeing_arcsec",
                    "exposure_s", "saturation_count", "bad_pixel_count",
                    "cosmic_ray_count", "background_e_per_pix",
                    "read_noise_e", "seed", "units", "model", "label"):
            self.assertIn(key, meta)
        self.assertEqual(meta["label"], "SYNTHETIC-VALIDATED")
        self.assertEqual(meta["mode"], "synthetic")
        self.assertIn("full-psf", meta["model"])
        self.assertIn("seeing", meta["model"].lower())
        self.assertIn("psf", meta["units"])
        self.assertIn("frames.raw", meta["units"])
        self.assertEqual(meta["seed"], 7)

    def test_params_used_are_recorded(self):
        params = self.r["meta"]["params"]
        for key in ("c4", "c8", "c9", "defocus", "exposure_s",
                    "adu_gain_e_per_adu", "read_noise_e",
                    "dark_current_e_per_s", "sky_background_e_per_pix_s",
                    "plate_scale_arcsec_per_pix", "seeing_arcsec",
                    "star_density_per_arcmin2", "mag_zero", "mag_limit",
                    "saturation_adu", "bit_depth", "readout_bias_adu",
                    "bad_pixel_fraction", "cosmic_ray_rate", "jitter_arcsec",
                    "aperture_m", "obscuration", "fov_arcmin", "seed"):
            self.assertIn(key, params)
        self.assertEqual(params["seeing_arcsec"], 0.7)
        self.assertEqual(params["plate_scale_arcsec_per_pix"], 0.05)
        self.assertEqual(params["fov_arcmin"], 2.56)
        self.assertEqual(params["mag_zero"], 10.0)
        self.assertEqual(params["mag_limit"], 16.0)

    def test_meta_and_manifest_are_json_serialisable(self):
        json.dumps(self.r["meta"])
        json.dumps(self.r["manifest"])
        self.assertEqual(self.r["manifest"]["schema_version"], 2)
        self.assertEqual(len(self.r["manifest"]["catalog"]),
                         self.r["meta"]["n_stars_rendered"])

    def test_fast_path(self):
        fast = run_ccd_pipeline(fast=True, return_frames=False)
        self.assertTrue(fast["meta"]["fast_mode"])
        self.assertTrue(fast["meta"]["model"].startswith("fast-gaussian"))
        self.assertEqual(np.asarray(fast["psf"]).shape, (5, 64, 64))
        self.assertAlmostEqual(float(np.asarray(fast["psf"])[0].sum()), 1.0,
                               places=6)


if __name__ == "__main__":
    unittest.main()
