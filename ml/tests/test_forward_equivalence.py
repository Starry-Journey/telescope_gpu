"""torch ↔ numpy 前向模型金标一致性测试。

对照 ml.phase_torch.forward_stack（可微 torch 实现）与
products.optics.forward.forward_stack_numpy（工程侧单一来源 numpy 实现），
用固定 seed 的 Zernike 系数和同一离焦列表逐元素对比 PSF。

口径对齐结论（通读双方源码后确认，测试按此对齐输入）：

* 网格：两者同为 linspace(-1, 1, n*oversample) + meshgrid(indexing="ij")，
  Z4/Z8/Z9 公式逐项一致（sqrt(3)*(2ρ²-1)、sqrt(8)*(3ρ³-2ρ)*cos/sin θ），
  mask 同为 r <= pupil_radius（默认 0.45）——网格口径天然一致，无需改写。
* 槽位：torch coeff = [c4, c8, c9] 与 numpy terms=("Z4","Z8","Z9") 顺序一致，
  已知 defocus 在两侧都只作用于 Z4。
* 归一化 / binning：两侧均为"归一 → oversample² 均值 binning → 再归一"；
  torch 的 F.avg_pool2d(kernel=oversample) 与 numpy 的
  reshape(n, os, n, os).mean(axis=(1, 3)) 数学等价。
* 唯一实质差异是 dtype：ml.phase_torch.zernike_basis 的网格跟随 torch 全局
  默认 dtype（默认 float32，实测 PSF 最大偏差 ~1.4e-8，超出 1e-8 容差），
  故本测试在 setUp 临时切到 float64 并在 tearDown/addCleanup 恢复；
  numpy 侧本身是 float64。对齐后两实现逐元素一致到机器精度（~1e-16）。
"""
import unittest

import numpy as np

try:
    import torch
except ImportError:  # pragma: no cover - torch 为可选依赖
    torch = None

from ml.phase_torch import ForwardConfig, forward_stack
from products.optics.forward import forward_stack_numpy

N = 32                    # 小画幅：binning 后 32x32
DEFOCUS = (-1.5, 0.75)    # 2 帧，避开默认五帧的大网格
SEED = 42                 # 固定 seed 的 Zernike 系数


def _golden_coeffs() -> np.ndarray:
    rng = np.random.default_rng(SEED)
    return rng.uniform(-0.8, 0.8, size=(2, 3))


@unittest.skipUnless(torch is not None, "torch 不可用")
class TestForwardEquivalence(unittest.TestCase):
    def setUp(self):
        # ml.phase_torch 的网格 dtype 跟随全局默认值，这里临时切 float64
        # （numpy 金标侧本身 float64），测试结束恢复，避免影响同进程其他测试。
        self._saved_dtype = torch.get_default_dtype()
        torch.set_default_dtype(torch.float64)
        self.addCleanup(torch.set_default_dtype, self._saved_dtype)

    def _torch_stack(self, coeffs: np.ndarray) -> np.ndarray:
        cfg = ForwardConfig(n=N, defocus=DEFOCUS, noise_sigma=0.0)
        return forward_stack(torch.tensor(coeffs), cfg).numpy()

    def _numpy_stack(self, coeff: np.ndarray) -> np.ndarray:
        return forward_stack_numpy(coeff, n=N, defocus=DEFOCUS)

    def test_single_coeff_frames_match(self):
        for coeff in _golden_coeffs():
            with self.subTest(coeff=[round(float(c), 6) for c in coeff]):
                t = self._torch_stack(coeff)
                g = self._numpy_stack(coeff)
                self.assertEqual(t.shape, g.shape)
                for j in range(len(DEFOCUS)):
                    self.assertTrue(
                        np.allclose(t[j], g[j], rtol=1e-8, atol=1e-8),
                        "frame %d: max|torch-numpy| = %.3e"
                        % (j, np.abs(t[j] - g[j]).max()))

    def test_batched_coeff_matches(self):
        coeffs = _golden_coeffs()
        t = self._torch_stack(coeffs)
        self.assertEqual(t.shape, (len(coeffs), len(DEFOCUS), N, N))
        for i, coeff in enumerate(coeffs):
            g = self._numpy_stack(coeff)
            self.assertTrue(
                np.allclose(t[i], g, rtol=1e-8, atol=1e-8),
                "coeff %d: max|torch-numpy| = %.3e"
                % (i, np.abs(t[i] - g).max()))

    def test_normalisation_and_energy(self):
        """双方 PSF 都应单位能量归一（防止全零/退化比较造成的假阳性）。"""
        for coeff in _golden_coeffs():
            with self.subTest(coeff=[round(float(c), 6) for c in coeff]):
                t = self._torch_stack(coeff)
                g = self._numpy_stack(coeff)
                np.testing.assert_allclose(t.sum(axis=(-2, -1)), 1.0, atol=1e-12)
                np.testing.assert_allclose(g.sum(axis=(-2, -1)), 1.0, atol=1e-12)
                # 峰值位置也应一致（同一格点，容差 1e-8）
                self.assertEqual(int(np.argmax(t[0])), int(np.argmax(g[0])))


if __name__ == "__main__":
    unittest.main()
