"""mock 硬件单元测试：机构量化/回差/限位、望远镜 slew/遥测、相机 raw ADU。

这些测试锁定 mock 的行为契约（mode/label/simulated、确定性、单位），也
锁定"外部边界是显式 mock"这一事实：没有真实望远镜/相机/机构 SDK。
"""
import unittest

import numpy as np

from products.acquisition.mock import (MOCK_CAMERA_TERMS, MockCamera,
                                       MockMechanism, MockTelescopeTerminal,
                                       build_mock_rig)


class MockMechanismTest(unittest.TestCase):
    def test_dofs_schema_and_units(self):
        mech = MockMechanism(n_dofs=2)
        dofs = mech.dofs()
        self.assertEqual([d["name"] for d in dofs],
                         ["m2_focus", "m2_decenter_x"])
        self.assertEqual(dofs[0]["unit"], "mm")
        for dof in dofs:
            for key in ("name", "unit", "low", "high", "step"):
                self.assertIn(key, dof)
            self.assertLess(dof["low"], dof["high"])
            self.assertGreater(dof["step"], 0.0)
        self.assertTrue(mech.simulated)
        self.assertEqual(mech.mode, "mock")
        self.assertEqual(mech.label, "SYNTHETIC-VALIDATED")

    def test_move_is_quantised_and_bounded(self):
        mech = MockMechanism(n_dofs=1)
        step = float(mech.dofs()[0]["step"])
        mech.move("m2_focus", 0.0512)
        value = mech.readback()["m2_focus"]
        self.assertAlmostEqual(value / step, round(value / step), places=9)
        self.assertAlmostEqual(value, 0.05, places=12)
        with self.assertRaises(ValueError):
            mech.move("m2_focus", 0.6)
        with self.assertRaises(ValueError):
            mech.move("m2_focus", -0.6)
        with self.assertRaises(KeyError):
            mech.move("not_a_dof", 0.01)
        mech.home()
        self.assertAlmostEqual(mech.readback()["m2_focus"], 0.0)

    def test_hysteresis_on_direction_reversal(self):
        mech = MockMechanism(n_dofs=1, hysteresis={"m2_focus": 0.02})
        mech.move("m2_focus", 0.05)
        self.assertAlmostEqual(mech.readback()["m2_focus"], 0.05)
        mech.move("m2_focus", -0.05)          # 换向：背隙吃掉一部分
        reversed_value = mech.readback()["m2_focus"]
        self.assertGreater(reversed_value, -0.05)
        self.assertAlmostEqual(reversed_value, -0.03, places=12)
        mech.move("m2_focus", -0.05)          # 同向重复：背隙已消除
        self.assertAlmostEqual(mech.readback()["m2_focus"], -0.05)

    def test_truth_coupling_is_deterministic_and_labelled(self):
        mech = MockMechanism(n_dofs=2, base_state={"Z4": 0.1})
        mech.move("m2_focus", 0.05)
        state = mech.wavefront_state()
        self.assertEqual(set(state), set(MOCK_CAMERA_TERMS))
        # 口径 v2 / P1-1: truth is the real Zemax S_Z (focus -> Z4 = -23.284
        # waves/mm), so a +0.05 mm axial move DRIVES Z4 negative from the
        # +0.1 base state.
        self.assertLess(state["Z4"], -1.0)
        self.assertGreater(state["Z8"], -0.01)   # decoupled at this level
        self.assertIn("truth_source", mech.__dict__)
        kin = mech.kinematics()
        self.assertIn("truth_S", kin)
        self.assertEqual(kin["label"], "SYNTHETIC-VALIDATED")


class MockTelescopeTest(unittest.TestCase):
    def test_goto_pointing_and_operating_condition(self):
        tel = MockTelescopeTerminal(seed=3, elevation_deg=45.0)
        status = tel.goto_elevation(15.0, timeout_s=10.0)
        self.assertEqual(status["mode"], "mock")
        self.assertEqual(status["label"], "SYNTHETIC-VALIDATED")
        self.assertTrue(status["simulated"])
        self.assertAlmostEqual(status["elevation_deg"], 15.0)
        self.assertGreater(status["last_slew_s"], 0.0)
        self.assertTrue(status["settled"])
        pointing = tel.pointing()
        for key in ("elevation_deg", "azimuth_deg", "ra_deg", "dec_deg",
                    "epoch", "timestamp", "units"):
            self.assertIn(key, pointing)
        self.assertEqual(pointing["epoch"], "J2000")
        cond = tel.operating_condition()
        self.assertEqual(cond["source"], "NO_REAL_OPERATING_CONDITION_DATA")
        self.assertEqual(cond["mode"], "mock")
        self.assertIn("aberration_waves", cond)
        # 仰角越低，mock 重力/失焦代理越大
        tel.goto_elevation(75.0)
        high = tel.condition_aberration()["Z4"]
        tel.goto_elevation(15.0)
        low = tel.condition_aberration()["Z4"]
        self.assertGreater(low, high)

    def test_slew_timeout_is_reported(self):
        tel = MockTelescopeTerminal(elevation_deg=15.0)
        with self.assertRaises(TimeoutError):
            tel.goto_elevation(80.0, timeout_s=0.01)
        self.assertAlmostEqual(tel.pointing()["elevation_deg"], 15.0)


class MockCameraTest(unittest.TestCase):
    def test_capture_returns_raw_adu_and_is_deterministic(self):
        def provider():
            return {"Z4": 0.15, "Z8": 0.06, "Z9": 0.04}

        cam_a = MockCamera(state_provider=provider, seed=11)
        cam_b = MockCamera(state_provider=provider, seed=11)
        defocus = [-2.0, -1.0, 1.0, 2.0]
        raw_a = cam_a.capture(defocus, 0.5)
        raw_b = cam_b.capture(defocus, 0.5)
        self.assertEqual(raw_a.shape, (4, 64, 64))
        np.testing.assert_array_equal(raw_a, raw_b)   # 固定 seed 确定性
        np.testing.assert_array_equal(raw_a, np.rint(raw_a))
        meta = cam_a.metadata()
        self.assertEqual(meta["mode"], "mock")
        self.assertEqual(meta["label"], "SYNTHETIC-VALIDATED")
        self.assertIn("ADU", meta["raw_units"])
        self.assertEqual(meta["terms"], list(MOCK_CAMERA_TERMS))
        self.assertLessEqual(float(raw_a.max()), float(meta["adu_max"]))
        self.assertEqual(len(meta["flat_field"]), 64)
        self.assertEqual(len(meta["defect_map"]), 64)

    def test_capture_rejects_unsupported_terms(self):
        with self.assertRaises(ValueError):
            MockCamera(state_provider=None, terms=("Z4", "Z5"))

    def test_bias_exposure_produces_bias_level(self):
        cam = MockCamera(state_provider=None, seed=5)
        frames = cam.capture([0.0, 0.0, 0.0], 0.0)
        median = float(np.median(frames))
        self.assertAlmostEqual(median, cam.readout_bias_adu, delta=5.0)


class MockRigTest(unittest.TestCase):
    def test_rig_state_combines_elevation_and_mechanism(self):
        rig = build_mock_rig(n_dofs=2, seed=7)
        rig.telescope.goto_elevation(15.0)
        low = rig.state()["Z4"]
        rig.telescope.goto_elevation(75.0)
        high = rig.state()["Z4"]
        self.assertGreater(low, high)
        rig.mechanism.move("m2_focus", 0.05)
        moved = rig.state()["Z4"]
        # real S_Z: focus drives Z4 NEGATIVE (was proxy +4.04 sign, 口径 v2)
        self.assertLess(moved, -1.0)
        summary = rig.summary()
        self.assertEqual(summary["label"], "SYNTHETIC-VALIDATED")
        self.assertEqual(summary["mode"], "mock")


if __name__ == "__main__":
    unittest.main()
