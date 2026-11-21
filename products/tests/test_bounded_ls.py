"""products.lsq.bounded_least_squares 的契约测试。

覆盖四类场景（对应"假·有界最小二乘 -> 真有界最小二乘"的替换验收）:
1. 宽界退化：解在界内时应与 np.linalg.lstsq（含最小范数解、Tikhonov 增广）
   一致；
2. 触界情形：有界解残差 <= clip(lstsq) 残差，且构造 clip 明显次优的用例；
3. 2D 多目标：逐列独立求解，与逐列一维调用一致；
4. NaN/Inf 防护：与调用点（matrix_service._as_matrix 等）一致地快速失败。
"""
import unittest

import numpy as np

from products.lsq import bounded_least_squares

# 3x5、秩 3 的欠定系统（与 products.control.service 的代理矩阵同构）
PROXY_S = np.array([
    [4.040, 0.083, -0.083, 0.976, -0.976],
    [0.000, -4.633, 0.000, 0.000, 0.000],
    [0.000, 0.000, -4.633, 0.000, 0.000],
])
# 列强耦合的 2x2 满秩系统：对 target=[10, 0]，clip(lstsq)=[1,-1] 残差 100.01，
# 真有界最优 [1, 1] 残差 68.41 —— clip 明显次优。
COUPLED_S = np.array([
    [1.0, 1.0],
    [1.0, 1.1],
])


class BoundedLeastSquaresTest(unittest.TestCase):
    def test_wide_bounds_match_lstsq_full_rank(self):
        rng = np.random.RandomState(2026)
        S = rng.randn(6, 4)
        t = rng.randn(6)
        lo, hi = np.full(4, -1e9), np.full(4, 1e9)
        u, info = bounded_least_squares(S, t, lo, hi)
        u_ref, *_ = np.linalg.lstsq(S, t, rcond=None)
        np.testing.assert_allclose(u, u_ref, rtol=1e-9, atol=1e-9)
        self.assertEqual(info["n_active"], 0)
        self.assertTrue(info["success"])

    def test_wide_bounds_match_lstsq_rank_deficient(self):
        # 欠定（3x5，秩 3）系统的最小范数解也应保留
        z = PROXY_S @ np.array([0.05, 0.02, -0.01, 0.01, -0.02])
        lo, hi = np.full(5, -1e9), np.full(5, 1e9)
        u, info = bounded_least_squares(PROXY_S, z, lo, hi)
        u_ref, *_ = np.linalg.lstsq(PROXY_S, z, rcond=None)
        np.testing.assert_allclose(u, u_ref, rtol=1e-9, atol=1e-9)
        self.assertEqual(info["n_active"], 0)

    def test_tikhonov_interior_matches_lstsq_augmented(self):
        # lam>0 时与原调用点的增广 lstsq 一致（解在界内的情形）
        z = PROXY_S @ np.array([0.05, 0.02, -0.01, 0.01, -0.02])
        lam = 1e-8
        lo, hi = np.full(5, -0.5), np.full(5, 0.5)
        u, info = bounded_least_squares(PROXY_S, z, lo, hi, lam=lam)
        Aaug = np.vstack([PROXY_S, np.sqrt(lam) * np.eye(5)])
        baug = np.concatenate([z, np.zeros(5)])
        u_ref, *_ = np.linalg.lstsq(Aaug, baug, rcond=None)
        np.testing.assert_allclose(u, u_ref, rtol=1e-8, atol=1e-9)
        self.assertEqual(info["n_active"], 0)

    def test_touch_bound_beats_clip(self):
        t = np.array([10.0, 0.0])
        lo, hi = np.full(2, -1.0), np.full(2, 1.0)
        u, info = bounded_least_squares(COUPLED_S, t, lo, hi)
        # 解严格在界内，且两个分量都压在上界
        self.assertTrue(np.all(u >= lo))
        self.assertTrue(np.all(u <= hi))
        np.testing.assert_allclose(u, [1.0, 1.0], atol=1e-6)
        self.assertEqual(info["n_active"], 2)
        self.assertTrue(bool(np.all(info["active_bounds"])))
        # 有界解残差 <= clip(lstsq) 残差（本用例差距明显：68.41 vs 100.01）
        u_ls, *_ = np.linalg.lstsq(COUPLED_S, t, rcond=None)
        u_clip = np.clip(u_ls, lo, hi)
        res_bounded = float(np.sum((COUPLED_S @ u - t) ** 2))
        res_clip = float(np.sum((COUPLED_S @ u_clip - t) ** 2))
        self.assertLessEqual(res_bounded, res_clip)
        self.assertLess(res_bounded, res_clip - 1.0)

    def test_touch_bound_with_tikhonov_beats_clip(self):
        t = np.array([10.0, 0.0])
        lam = 1e-6
        lo, hi = np.full(2, -1.0), np.full(2, 1.0)
        u, _ = bounded_least_squares(COUPLED_S, t, lo, hi, lam=lam)
        Aaug = np.vstack([COUPLED_S, np.sqrt(lam) * np.eye(2)])
        baug = np.concatenate([t, np.zeros(2)])
        u_ls, *_ = np.linalg.lstsq(Aaug, baug, rcond=None)
        u_clip = np.clip(u_ls, lo, hi)

        def cost(v):
            return float(np.sum((COUPLED_S @ v - t) ** 2) + lam * np.sum(v ** 2))

        self.assertTrue(np.all(u >= lo))
        self.assertTrue(np.all(u <= hi))
        self.assertLessEqual(cost(u), cost(u_clip))

    def test_compensation_convention_minus_target(self):
        # orchestrator 约定：传 -target => S u ≈ -target => target + S u ≈ 0
        target = np.array([0.2, -0.1, 0.05])
        lo, hi = np.full(5, -0.5), np.full(5, 0.5)
        u, info = bounded_least_squares(PROXY_S, -target, lo, hi)
        residual = target + PROXY_S @ u
        self.assertLess(float(np.max(np.abs(residual))), 1e-8)
        self.assertEqual(info["n_active"], 0)

    def test_multi_target_2d(self):
        # 第 1 列触界（最优 [1,1]），第 2 列界内（精确解 [0.3, 0]）
        T = np.array([[10.0, 0.3],
                      [0.0, 0.3]])
        lo, hi = np.full(2, -1.0), np.full(2, 1.0)
        u, info = bounded_least_squares(COUPLED_S, T, lo, hi)
        self.assertEqual(u.shape, (2, 2))
        self.assertEqual(info["active_bounds"].shape, (2, 2))
        self.assertTrue(info["success"])
        # 逐列与一维调用一致
        u_c0, _ = bounded_least_squares(COUPLED_S, T[:, 0], lo, hi)
        u_c1, _ = bounded_least_squares(COUPLED_S, T[:, 1], lo, hi)
        np.testing.assert_allclose(u[:, 0], u_c0, atol=1e-9)
        np.testing.assert_allclose(u[:, 1], u_c1, atol=1e-9)
        # 第 1 列两个分量触上界，第 2 列全部界内
        self.assertTrue(bool(np.all(info["active_bounds"][:, 0])))
        self.assertFalse(bool(np.any(info["active_bounds"][:, 1])))
        np.testing.assert_allclose(u[:, 1], [0.3, 0.0], atol=1e-9)
        self.assertEqual(
            info["n_active"], int(np.count_nonzero(info["active_bounds"])))

    def test_return_contract(self):
        lo, hi = np.full(2, -1.0), np.full(2, 1.0)
        u, info = bounded_least_squares(COUPLED_S, [10.0, 0.0], lo, hi)
        self.assertIsInstance(u, np.ndarray)
        self.assertEqual(u.dtype, np.dtype(float))
        self.assertEqual(u.shape, (2,))
        for key in ("success", "status", "message",
                    "active_bounds", "n_active"):
            self.assertIn(key, info)
        self.assertTrue(info["success"])
        self.assertIsInstance(info["active_bounds"], np.ndarray)
        self.assertEqual(info["active_bounds"].dtype, np.dtype(bool))
        self.assertIsInstance(info["n_active"], int)

    def test_input_guards_match_call_sites(self):
        lo, hi = np.full(2, -1.0), np.full(2, 1.0)
        ok_S, ok_t = COUPLED_S, [1.0, 0.0]
        # S 含 NaN/Inf -> 快速失败（调用点不应再拿到静默污染的解）
        with self.assertRaises(ValueError):
            bounded_least_squares(
                np.array([[np.nan, 1.0], [1.0, 1.0]]), ok_t, lo, hi)
        with self.assertRaises(ValueError):
            bounded_least_squares(
                np.array([[np.inf, 1.0], [1.0, 1.0]]), ok_t, lo, hi)
        # target / 界含非有限值
        with self.assertRaises(ValueError):
            bounded_least_squares(ok_S, [np.inf, 0.0], lo, hi)
        with self.assertRaises(ValueError):
            bounded_least_squares(ok_S, ok_t, [np.nan, -1.0], hi)
        # 相等界（scipy 要求严格 lo < hi）与 lam<0、形状不匹配
        with self.assertRaises(ValueError):
            bounded_least_squares(ok_S, ok_t, [1.0, -1.0], [1.0, 1.0])
        with self.assertRaises(ValueError):
            bounded_least_squares(ok_S, ok_t, lo, hi, lam=-1e-3)
        with self.assertRaises(ValueError):
            bounded_least_squares(ok_S, [1.0, 0.0, 2.0], lo, hi)
        # 与调用点的既有输入防护一致：matrix_service 同样拒绝 NaN 矩阵
        from products.sensitivity import matrix_service as sm
        with self.assertRaises(ValueError):
            sm.control([[np.nan, 1.0], [1.0, 1.0]], [1.0, 2.0],
                       use_bounds=True)


if __name__ == "__main__":
    unittest.main()
