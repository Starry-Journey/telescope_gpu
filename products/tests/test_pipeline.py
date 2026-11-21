"""端到端管线测试：2 个俯仰角 / 2 个 DOF / Z4,Z8,Z9 的完整工作流。

覆盖：compensation_table、fea_request、keep_frames 内存约束、确定性、
calib_dir 加载定标帧、运行时间 < 60 s。全部跑在显式 mock 台架上
（label=SYNTHETIC-VALIDATED），不连接任何硬件。
"""
import json
import tempfile
import time
import unittest

import numpy as np

from products.acquisition.mock import build_mock_rig
from products.pipeline import PipelineConfig, WavefrontPipeline
from products.preprocess import MasterFrames

try:
    import nlopt  # noqa: F401
    HAVE_NLOPT = True
except Exception:                                  # pragma: no cover
    HAVE_NLOPT = False


def _contains_key(obj, key):
    """递归检查 dict/list 中是否出现某个 key（内存约束测试用）。"""
    if isinstance(obj, dict):
        return key in obj or any(_contains_key(v, key) for v in obj.values())
    if isinstance(obj, (list, tuple)):
        return any(_contains_key(v, key) for v in obj)
    return False


class PipelineEndToEndTest(unittest.TestCase):
    """2 elevations x 2 DOFs x Z4/Z8/Z9（要求 < 60 s）。"""

    @classmethod
    def setUpClass(cls):
        if not HAVE_NLOPT:
            raise unittest.SkipTest("nlopt not available")
        cls.rig = build_mock_rig(n_dofs=2, seed=7)
        cls.config = PipelineConfig(
            elevations_deg=[15.0, 75.0],
            defocus=[-2.0, -1.0, 1.0, 2.0],
            n_frames=3,
            exposure_s=0.5,
            terms=["Z4", "Z8", "Z9"],
            pose_amplitude=0.05,
            repeats=1,
            seed=7,
            rel_threshold=0.2,
            pd_maxeval=40,
            pd_n_starts=2,
            keep_frames=False,
        )
        start = time.perf_counter()
        cls.result = WavefrontPipeline(cls.rig.telescope, cls.rig.camera,
                                       cls.rig.mechanism, cls.config).run()
        cls.elapsed = time.perf_counter() - start

    def test_runtime_under_60_seconds(self):
        self.assertLess(self.elapsed, 60.0)

    def test_result_structure_and_provenance(self):
        result = self.result
        for key in ("label", "mode", "stages", "per_elevation",
                    "compensation_table", "fea_request", "warnings",
                    "manifest"):
            self.assertIn(key, result)
        self.assertEqual(result["mode"], "mock")
        self.assertEqual(result["label"], "SYNTHETIC-VALIDATED")
        self.assertEqual(result["status"], "ok")
        self.assertEqual(result["calibration"], "synthetic")
        self.assertEqual(len(result["per_elevation"]), 2)
        self.assertIn("invert", result["stages"])
        self.assertGreater(result["stages"]["invert"], 0.0)
        self.assertEqual(result["manifest"]["schema_version"], 1)
        self.assertEqual(result["manifest"]["mode"], "mock")
        for rec in result["per_elevation"]:
            for key in ("elevation_deg", "target_aberration_waves",
                        "measured_sensitivity", "compensation",
                        "observability", "stages"):
                self.assertIn(key, rec)

    def test_compensation_table(self):
        rows = self.result["compensation_table"]
        self.assertEqual(len(rows), 4)           # 2 elevations x 2 DOFs
        elevations = {row["elevation_deg"] for row in rows}
        self.assertEqual(elevations, {15.0, 75.0})
        for row in rows:
            self.assertIn(row["dof"], ("m2_focus", "m2_decenter_x"))
            self.assertGreaterEqual(row["command"], row["lower"] - 1e-12)
            self.assertLessEqual(row["command"], row["upper"] + 1e-12)
            self.assertIsInstance(row["at_bound"], bool)
            self.assertIn("residual_rms_waves", row)
            self.assertIn("target_Z4_waves", row)
            self.assertIn("residual_Z9_waves", row)
        # 有界最小二乘应当把可补偿项压下去（Z9 是故意不可补偿项）
        for rec in self.result["per_elevation"]:
            target_rms = float(np.sqrt(np.mean(
                np.square(list(rec["target_aberration_waves"].values())))))
            self.assertLessEqual(
                rec["compensation"]["residual_rms_waves"], target_rms)

    def test_fea_request_flags_uncompensatable_terms(self):
        fea = self.result["fea_request"]
        self.assertEqual(fea["label"], "FEA-REQUEST")
        self.assertTrue(fea["no_fea_performed"])
        self.assertTrue(fea["un_compensatable_terms"])
        # 2-DOF focus+decenter_x 对 Z9（y 彗差）没有权限 -> 应进入 FEA 请求
        self.assertIn("Z9", fea["un_compensatable_terms"])
        self.assertEqual(set(fea["un_compensatable_terms"]),
                         set(fea["by_term"]))
        for record in fea["terms"]:
            for key in ("term", "elevation_deg", "magnitude_waves",
                        "magnitude_nm_wavefront", "magnitude_nm_surface",
                        "suggested_action"):
                self.assertIn(key, record)
            self.assertGreaterEqual(record["magnitude_nm_wavefront"], 0.0)
        self.assertIn("boundary", fea)

    def test_memory_bound_and_json_safety(self):
        # keep_frames=False：结果里不应出现任何 "frames" 键
        self.assertFalse(_contains_key(self.result, "frames"))
        text = json.dumps(self.result, ensure_ascii=False)
        self.assertGreater(len(text), 1000)
        # 确定性字段可以 JSON 往返
        roundtrip = json.loads(text)
        self.assertEqual(roundtrip["compensation_table"],
                         self.result["compensation_table"])


class PipelineAuxTest(unittest.TestCase):
    """确定性 + calib_dir 加载（都用便宜的 1 elevation / 1 DOF 配置）。"""

    @classmethod
    def setUpClass(cls):
        if not HAVE_NLOPT:
            raise unittest.SkipTest("nlopt not available")

    def _cheap_config(self, **kwargs):
        params = dict(elevations_deg=[45.0], defocus=[-2.0, 2.0],
                      n_frames=2, exposure_s=0.5, terms=["Z4", "Z8", "Z9"],
                      pose_amplitude=0.05, repeats=1, seed=7,
                      pd_maxeval=30, pd_n_starts=2)
        params.update(kwargs)
        return PipelineConfig(**params)

    def _run(self, config, n_dofs=1):
        rig = build_mock_rig(n_dofs=n_dofs, seed=7)
        return WavefrontPipeline(rig.telescope, rig.camera, rig.mechanism,
                                 config).run()

    def test_determinism(self):
        config = self._cheap_config()
        first = self._run(config)
        second = self._run(config)
        self.assertEqual(first["compensation_table"],
                         second["compensation_table"])
        self.assertEqual(first["fea_request"]["un_compensatable_terms"],
                         second["fea_request"]["un_compensatable_terms"])
        for a, b in zip(first["per_elevation"], second["per_elevation"]):
            self.assertEqual(a["target_aberration_waves"],
                             b["target_aberration_waves"])
            np.testing.assert_allclose(a["measured_sensitivity"]["S"],
                                       b["measured_sensitivity"]["S"],
                                       atol=1e-12)

    def test_calib_dir_master_frames_are_loaded(self):
        with tempfile.TemporaryDirectory() as tmp:
            MasterFrames(
                bias=np.full((64, 64), 500.0),
                dark=np.zeros((64, 64)),
                flat=np.ones((64, 64)),
                bias_exposure_s=0.0, dark_exposure_s=0.5, n_bias=3,
            ).save(tmp)
            result = self._run(self._cheap_config(calib_dir=tmp))
            calibration = result["manifest"]["calibration"]
            self.assertEqual(calibration["mode"], "loaded")
            self.assertTrue(calibration["path"])
            self.assertFalse(any("calibration='synthetic'" in w
                                 for w in result["warnings"]))


if __name__ == "__main__":
    unittest.main()
