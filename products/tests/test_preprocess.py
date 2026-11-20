"""预处理单元测试：主帧/定标恒等、坏像元、宇宙线、背景、星点、质心、配准。

全部使用合成 numpy 数据；不需要相机/网络/GUI。重点是数学正确性与
"向量化实现"的行为契约，不依赖任何真实观测。
"""
import tempfile
import unittest

import numpy as np

from products.preprocess import (MasterFrames, bad_pixel_map,
                                 build_master_bias, build_master_dark,
                                 build_master_flat, calibrate,
                                 centroid_subpixel, clean_psf, cutout,
                                 detect_stars, estimate_background,
                                 register_shift, remove_cosmic_rays,
                                 repair_bad_pixels, saturation_mask)


def _gaussian(frame, x, y, amp, sigma):
    yy, xx = np.indices(frame.shape, dtype=float)
    return frame + amp * np.exp(-((xx - x) ** 2 + (yy - y) ** 2)
                                / (2.0 * sigma ** 2))


def _fourier_shift(image, dy, dx):
    """精确(亚像元)傅里叶平移：img(y,x) = image(y-dy, x-dx)。"""
    fy = np.fft.fftfreq(image.shape[0])[:, None]
    fx = np.fft.fftfreq(image.shape[1])[None, :]
    spec = np.fft.fft2(image) * np.exp(-2j * np.pi * (fy * dy + fx * dx))
    return np.real(np.fft.ifft2(spec))


class MasterFrameTest(unittest.TestCase):
    def test_builders_and_calibration_identity(self):
        rng = np.random.default_rng(0)
        bias = np.full((12, 12), 500.0)
        dark = np.full((12, 12), 2.0)
        flat = 1.0 + 0.1 * np.linspace(-1, 1, 12)[None, :] * np.ones((12, 1))
        flat = flat / np.median(flat)
        signal = rng.normal(20.0, 1.0, size=(3, 12, 12))

        # 造与 calibrate 定义一致的 raw：signal*flat + bias + dark*exp_scale
        raw = signal * flat[None, :, :] + bias[None, :, :] + dark[None, :, :] * 0.5
        master = MasterFrames(bias=bias, dark=dark, flat=flat,
                              dark_exposure_s=1.0)
        calibrated = calibrate(raw, master, exposure_s=0.5)
        np.testing.assert_allclose(calibrated, signal, atol=1e-9)

        # 主帧构造：中值 + 曝光元数据
        bias_stack = np.stack([bias + rng.normal(0, 2, bias.shape)
                               for _ in range(5)])
        dark_stack = np.stack([bias + dark + rng.normal(0, 2, bias.shape)
                               for _ in range(5)])
        master_bias = build_master_bias(bias_stack)
        self.assertLess(float(np.mean(np.abs(master_bias - bias))), 2.0)
        master_dark = build_master_dark(dark_stack, 1.0, bias=master_bias)
        self.assertLess(float(np.mean(np.abs(master_dark - dark))), 2.0)
        master_flat = build_master_flat(np.stack([flat * 1000.0] * 3))
        self.assertAlmostEqual(float(np.median(master_flat)), 1.0, places=12)
        np.testing.assert_allclose(master_flat, flat, atol=1e-12)

    def test_bad_pixel_map_and_repair(self):
        flat = np.ones((32, 32))
        flat[3, 4] = 1.8          # hot
        flat[10, 20] = 0.05       # dead
        mask = bad_pixel_map(flat, sigma=5.0)
        self.assertTrue(bool(mask[3, 4]))
        self.assertTrue(bool(mask[10, 20]))
        self.assertLessEqual(int(mask.sum()), 4)
        self.assertEqual(mask.shape, flat.shape)
        repaired = repair_bad_pixels(np.stack([flat, flat]), mask)
        self.assertAlmostEqual(float(repaired[0, 3, 4]), 1.0, places=6)
        self.assertAlmostEqual(float(repaired[0, 10, 20]), 1.0, places=6)

    def test_master_frames_save_load_roundtrip(self):
        master = MasterFrames(
            bias=np.full((8, 8), 500.0), dark=np.full((8, 8), 1.5),
            flat=np.ones((8, 8)), bias_exposure_s=0.0, dark_exposure_s=2.0,
            n_bias=3, n_dark=2, n_flat=1)
        with tempfile.TemporaryDirectory() as tmp:
            path = master.save(tmp)
            self.assertTrue(path.exists())
            loaded = MasterFrames.load(tmp)
            np.testing.assert_array_equal(loaded.bias, master.bias)
            np.testing.assert_array_equal(loaded.dark, master.dark)
            self.assertAlmostEqual(loaded.dark_exposure_s, 2.0)
            self.assertEqual(loaded.n_bias, 3)
            self.assertEqual(loaded.shape, (8, 8))


class SingleFrameTest(unittest.TestCase):
    def test_estimate_background_recovers_level_and_noise(self):
        rng = np.random.default_rng(1)
        frame = 50.0 + rng.normal(0.0, 2.0, size=(64, 64))
        frame = _gaussian(frame, 20.0, 20.0, 800.0, 1.5)
        frame = _gaussian(frame, 45.0, 40.0, 400.0, 1.2)
        value, noise = estimate_background(frame, sigma_clip=3.0)
        self.assertAlmostEqual(value, 50.0, delta=0.5)
        self.assertAlmostEqual(noise, 2.0, delta=0.3)

    def test_remove_cosmic_rays(self):
        rng = np.random.default_rng(2)
        frame = 100.0 + rng.normal(0.0, 0.5, size=(48, 48))
        spikes = [(5, 5), (20, 30), (40, 10)]
        spiky = frame.copy()
        for y, x in spikes:
            spiky[y, x] = 5000.0
        cleaned = remove_cosmic_rays(spiky, sigma=5.0)
        for y, x in spikes:
            self.assertLess(float(cleaned[y, x]), 300.0)
        untouched = np.ones((48, 48), dtype=bool)
        for y, x in spikes:
            untouched[y, x] = False
        np.testing.assert_allclose(cleaned[untouched], frame[untouched],
                                   atol=1e-9)

    def test_saturation_mask(self):
        frame = np.zeros((4, 4))
        frame[0, 0] = 100.0
        frame[1, 1] = np.nan
        mask = saturation_mask(frame, adu_max=100.0)
        self.assertTrue(bool(mask[0, 0]))
        self.assertTrue(bool(mask[1, 1]))
        self.assertEqual(int(mask.sum()), 2)

    def test_detect_stars_on_synthetic_frame(self):
        rng = np.random.default_rng(3)
        frame = 10.0 + rng.normal(0.0, 1.0, size=(64, 64))
        truth = [(15.3, 20.7, 800.0), (45.1, 40.2, 500.0), (30.5, 50.4, 300.0)]
        for x, y, amp in truth:
            frame = _gaussian(frame, x, y, amp, 1.8)
        stars = detect_stars(frame, threshold_sigma=5.0, min_pixels=3)
        self.assertGreaterEqual(len(stars), 3)
        for x, y, _ in truth:
            best = min(stars, key=lambda s: (s["x"] - x) ** 2 + (s["y"] - y) ** 2)
            self.assertLess(float(np.hypot(best["x"] - x, best["y"] - y)), 0.5)
            self.assertGreater(best["flux"], 0.0)
            self.assertGreater(best["peak"], 0.0)

    def test_centroid_subpixel_error_below_0p1_px(self):
        cut = np.zeros((64, 64))
        cut = _gaussian(cut, 20.3, 15.7, 1000.0, 4.0)
        x, y = centroid_subpixel(cut)
        self.assertLess(abs(x - 20.3), 0.1)
        self.assertLess(abs(y - 15.7), 0.1)

    def test_register_shift_known_subpixel_shift(self):
        yy, xx = np.mgrid[0:64, 0:64]
        ref = np.exp(-((xx - 31.5) ** 2 + (yy - 30.5) ** 2) / (2 * 3.0 ** 2))
        img = _fourier_shift(ref, 2.4, -1.7)
        dy, dx, err = register_shift(ref, img)
        self.assertLess(abs(dy - 2.4), 0.1)
        self.assertLess(abs(dx + 1.7), 0.1)
        self.assertLess(err, 0.1)
        dy0, dx0, err0 = register_shift(ref, ref)
        self.assertLess(abs(dy0), 1e-6)
        self.assertLess(abs(dx0), 1e-6)
        self.assertLess(err0, 1e-9)

    def test_cutout_shapes_and_clamping(self):
        frame = np.arange(64 * 64, dtype=float).reshape(64, 64)
        cut = cutout(frame, 31.0, 31.0, 16)
        self.assertEqual(cut.shape, (16, 16))
        self.assertEqual(float(cut[0, 0]), float(frame[23, 23]))
        # 64 窗口在中央取整时会压到边界；clamp=True 必须仍返回完整窗口
        clipped = cutout(frame, 32.4, 32.4, 64, clamp=True)
        self.assertEqual(clipped.shape, (64, 64))
        with self.assertRaises(ValueError):
            cutout(frame, 33.0, 32.0, 64)

    def test_clean_psf_is_unit_sum_and_non_negative(self):
        cut = np.zeros((32, 32))
        cut = _gaussian(cut, 16.0, 16.0, 100.0, 2.0)
        cut += 3.0 + 0.1 * np.indices((32, 32))[0]     # 背景平面
        cleaned = clean_psf(cut, border=4)
        self.assertAlmostEqual(float(cleaned.sum()), 1.0, places=10)
        self.assertGreaterEqual(float(cleaned.min()), 0.0)


if __name__ == "__main__":
    unittest.main()
