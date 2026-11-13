"""Unit tests for the CCD pipeline (stdlib unittest).

Run from the repository root:

    python3 -m unittest discover -s pipeline/tests -v
"""
from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pipeline.astrometry import DistortionModel, match_sky
from pipeline.config import DetectConfig, PipelineConfig, SimConfig
from pipeline.detect import detect_moving, detect_stars, hough_lines, phase_cross_correlation
from pipeline.simulate import distort_pixels, make_wcs, simulate_dataset, undistort_pixels


class TestGeometry(unittest.TestCase):
    def test_wcs_roundtrip(self):
        cfg = SimConfig(width=128, height=128, crpix=(64.5, 64.5))
        w = make_wcs(cfg)
        x = np.array([40.0, 80.0, 64.5])
        y = np.array([30.0, 90.0, 64.5])
        sky = w.pixel_to_world(x, y)
        x2, y2 = w.world_to_pixel(sky)
        np.testing.assert_allclose(x2, x, atol=1e-6)
        np.testing.assert_allclose(y2, y, atol=1e-6)

    def test_distortion_inverse(self):
        cfg = SimConfig(width=256, height=256)
        x = np.array([50.0, 100.0, 200.0])
        y = np.array([40.0, 120.0, 210.0])
        xd, yd = distort_pixels(x, y, cfg)
        xi, yi = undistort_pixels(xd, yd, cfg)
        np.testing.assert_allclose(xi, x, atol=1e-4)
        np.testing.assert_allclose(yi, y, atol=1e-4)

    def test_distortion_fit(self):
        cfg = SimConfig(width=256, height=256)
        rng = np.random.default_rng(1)
        x = rng.uniform(20, 230, 30)
        y = rng.uniform(20, 230, 30)
        xi, yi = undistort_pixels(x, y, cfg)
        # fit detector -> ideal
        model = DistortionModel(cx=128, cy=128, scale=256, degree=3)
        model.fit(x, y, xi, yi)
        xf, yf = model.apply(x, y)
        self.assertLess(float(np.max(np.abs(xf - xi))), 0.005)
        self.assertLess(float(np.max(np.abs(yf - yi))), 0.005)


class TestDetection(unittest.TestCase):
    def test_phase_correlation(self):
        rng = np.random.default_rng(2)
        a = np.zeros((96, 96))
        for _ in range(12):
            x, y = rng.uniform(10, 85, 2)
            yy, xx = np.mgrid[0:96, 0:96]
            a += np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.2 ** 2))
        b = np.roll(np.roll(a, 2, axis=0), -3, axis=1)
        dy, dx = phase_cross_correlation(a, b)
        self.assertLess(abs(dy + 2), 0.3)
        self.assertLess(abs(dx - 3), 0.3)

    def test_star_centroid(self):
        h = w = 64
        yy, xx = np.mgrid[0:h, 0:w]
        x0, y0 = 31.4, 28.7
        sigma = 1.6
        img = 1200.0 * np.exp(-((xx - x0) ** 2 + (yy - y0) ** 2) / (2 * sigma ** 2))
        img += np.random.default_rng(3).normal(0, 2.0, img.shape)
        stars = detect_stars(img, DetectConfig(star_threshold_sigma=5.0))
        self.assertTrue(stars)
        best = min(stars, key=lambda s: (s["x"] - x0) ** 2 + (s["y"] - y0) ** 2)
        self.assertLess(math.hypot(best["x"] - x0, best["y"] - y0), 0.35)

    def test_hough_line(self):
        img = np.zeros((100, 100), dtype=int)
        for t in np.linspace(0, 1, 60):
            x = int(20 + 50 * t)
            y = int(80 - 30 * t)
            img[max(0, y - 1):y + 2, x] = 1
        lines = hough_lines(img > 0, theta_bins=180, rho_bin=1.0, threshold=20)
        self.assertTrue(lines)
        # Hough theta is the normal angle; the line angle is theta + 90 deg.
        target = (math.pi - math.atan2(30.0, 50.0)) % math.pi
        def circ_diff(a, b):
            d = abs((a - b) % math.pi)
            return min(d, math.pi - d)
        self.assertLess(min(circ_diff((l["theta"] + math.pi / 2) % math.pi, target)
                            for l in lines), 0.2)

    def test_moving_track_synthetic(self):
        """A deterministic moving point must form a track after frame differencing."""
        h = w = 96
        yy, xx = np.mgrid[0:h, 0:w]
        rng = np.random.default_rng(5)
        fixed = [(20, 20, 5000), (72, 28, 4000), (28, 72, 4500), (76, 76, 3500)]
        frames = []
        for f in range(10):
            img = np.full((h, w), 10.0)
            for x, y, flux in fixed:
                img += flux * np.exp(-((xx - x) ** 2 + (yy - y) ** 2) / (2 * 1.3 ** 2))
            tx, ty = 18 + 2.5 * f, 70 - 1.8 * f
            img += 3500 * np.exp(-((xx - tx) ** 2 + (yy - ty) ** 2) / (2 * 1.3 ** 2))
            img += rng.normal(0, 1.0, img.shape)
            frames.append(img)
        cfg = DetectConfig(moving_threshold_sigma=3.0, track_min_length=3,
                           track_max_step=6.0)
        moving = detect_moving(frames, cfg)
        self.assertGreaterEqual(len(moving["tracks"]), 1)
        self.assertGreaterEqual(max(len(t["detections"]) for t in moving["tracks"]), 5)

    def test_small_simulation_products(self):
        with tempfile.TemporaryDirectory() as td:
            cfg = SimConfig(width=128, height=128, n_frames=6, n_stars=8,
                            target_flux=2500.0, cosmic_rays=0, jitter=0.2,
                            target_start=(45.0, 50.0), target_velocity=(1.2, 0.6))
            simulate_dataset(td, cfg)
            self.assertTrue((Path(td) / "catalog.csv").exists())
            self.assertTrue((Path(td) / "raw" / "frame_000.fits").exists())


if __name__ == "__main__":
    unittest.main()
