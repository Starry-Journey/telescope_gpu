"""端到端数据管线编排：采集 -> 预处理 -> PD 反演 -> 灵敏度 -> 俯仰扫描补偿 -> FEA 请求。

本模块把用户给的下一代工作流串起来，并且**只通过抽象接口**
（:mod:`products.acquisition.interfaces`）访问硬件边界。今天没有真实
望远镜/相机/机构 API，也没有工况（operating condition）数据，所以默认
用 :mod:`products.acquisition.mock` 的显式 mock 跑通；真实 SDK 到位后，
换 adapter 即可，管线代码不变。

流程（每步都记录在 ``stages`` 计时里）::

    calibration   可选定标（calib_dir 加载实测 master；否则用相机前几帧
                  的 bias 曝光 + 相机 metadata 的 dark/flat 合成 master，
                  记为 calibration="synthetic"）
    for elevation in elevations_deg:
        telescope.goto_elevation(elevation)
        mechanism.home()
        逐 DOF 施加 +A / -A 激励，每个位姿 capture -> preprocess -> PD 反演
        measure_sensitivity(...) 得到实测 S
        有界最小二乘求补偿量（products.lsq 的 scipy lsq_linear，不 import GUI）
        analyze_control_space 找不可补偿项 -> fea_request

单俯仰角的实现（采集/预处理/反演/补偿/可观测性/配准 QC/报告装配）按数据
流拆分在 :mod:`products.pipeline._elevation`（纯行为保持重构，对外契约不变）。

输入:
    - 三个接口实现（mock 或 adapter）+ :class:`PipelineConfig`。
输出:
    - JSON 安全 dict：``label``/``mode``/``stages``/``per_elevation``/
      ``compensation_table``/``fea_request``/``warnings``/``manifest``。
边界:
    - 不 import 任何 GUI/Qt/网络；不 import products.control 的 GUI 服务
      （补偿用 products.lsq / scipy lsq_linear 解有界最小二乘；
      control.service 只可能被 mock 用作真值）；
    - 一次只处理一个俯仰角，``keep_frames=False``（默认）时不在结果里保留
      任何原始帧，内存有界；
    - 没有运行 FEA、没有真实工况数据；``fea_request`` 是**前处理请求**，
      不是 FEA 结果。
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from products.acquisition.interfaces import (CameraApi, MechanismApi,
                                             TelescopeTerminalApi)
from products.optics.zernike import resolve_terms
from products.pipeline._elevation import run_elevation
from products.preprocess import (MasterFrames, bad_pixel_map,
                                 build_master_bias, build_master_flat)
from products.sensitivity.observability import WAVELENGTH_NM

#: 结果 manifest 的 schema 版本。
MANIFEST_SCHEMA_VERSION = 1


def _jsonable(value: Any) -> Any:
    """递归转成 JSON 安全对象（numpy -> python，未知类型 -> str）。"""
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


# ===========================================================================
# 配置
# ===========================================================================

@dataclass
class PipelineConfig:
    """端到端管线配置（所有长度/角度/曝光单位见字段注释）。

    ``n_frames`` 是**每个主定标帧**使用的相机帧数（bias/dark 合成），
    science 每个位姿的帧数由 ``len(defocus)`` 决定。

    额外字段（``rel_threshold``/``pd_maxeval``/``pd_n_starts`` 等）不改变
    工作流语义，只是把不可补偿判据和 PD 求解预算显式化，便于真实数据上
    调参和测试提速。
    """

    elevations_deg: List[float] = field(
        default_factory=lambda: [15.0, 45.0, 75.0])
    # 默认离焦集故意不含 in-focus(0.0)：PD 只需要已知 diversity 帧；且 in-focus
    # 帧对探测器像素积分（mock 的 bilinear splat）与 PD 前向采样约定最敏感，
    # 会放大 mock 与真实相机之间的模型误差。真实数据可用 --defocus 覆盖。
    defocus: List[float] = field(
        default_factory=lambda: [-2.0, -1.0, 1.0, 2.0])
    n_frames: int = 3
    exposure_s: float = 0.5
    terms: List[str] = field(default_factory=lambda: ["Z4", "Z8", "Z9"])
    pose_amplitude: float = 0.05
    repeats: int = 1
    fov_arcmin: float = 2.56
    plate_scale_arcsec_per_pix: float = 0.05
    seed: int = 7
    calib_dir: Optional[str] = None
    stop_on_error: bool = True
    keep_frames: bool = False
    # --- 判据/预算（可选调参） ---
    rel_threshold: float = 0.2
    pd_maxeval: int = 80
    pd_n_starts: int = 2
    cr_sigma: float = 5.0
    bad_pixel_sigma: float = 5.0
    register_tolerance_px: float = 1.0
    slew_timeout_s: Optional[float] = None
    psf_window: Optional[int] = None

    def __post_init__(self) -> None:
        elevations = [float(e) for e in self.elevations_deg]
        if not elevations:
            raise ValueError("elevations_deg must not be empty")
        if any(not np.isfinite(e) or not (0.0 <= e <= 90.0) for e in elevations):
            raise ValueError("elevations_deg must be finite and in [0, 90]")
        self.elevations_deg = elevations
        self.defocus = [float(d) for d in self.defocus]
        if len(self.defocus) < 2:
            raise ValueError("defocus needs >= 2 frames for phase diversity")
        self.terms = list(resolve_terms(self.terms))
        if self.exposure_s < 0 or not np.isfinite(self.exposure_s):
            raise ValueError("exposure_s must be finite and >= 0")
        if self.pose_amplitude <= 0 or not np.isfinite(self.pose_amplitude):
            raise ValueError("pose_amplitude must be finite and > 0")
        if int(self.repeats) < 1:
            raise ValueError("repeats must be >= 1")
        if int(self.n_frames) < 1:
            raise ValueError("n_frames must be >= 1")
        if not (0.0 < float(self.rel_threshold) <= 1.0):
            raise ValueError("rel_threshold must be in (0, 1]")
        if int(self.pd_maxeval) < 10:
            raise ValueError("pd_maxeval must be >= 10")
        if int(self.pd_n_starts) < 1:
            raise ValueError("pd_n_starts must be >= 1")
        if self.psf_window is not None and int(self.psf_window) < 16:
            raise ValueError("psf_window must be >= 16")


# ===========================================================================
# 管线
# ===========================================================================

class WavefrontPipeline:
    """端到端波前数据管线（硬件边界由传入的三个接口实现提供）。"""

    def __init__(self, telescope: TelescopeTerminalApi, camera: CameraApi,
                 mechanism: MechanismApi,
                 config: Optional[PipelineConfig] = None) -> None:
        self.telescope = telescope
        self.camera = camera
        self.mechanism = mechanism
        self.config = config or PipelineConfig()
        self._camera_meta: Dict[str, Any] = {}
        self._psf_window = 64

    # ------------------------------------------------------------------ run
    def run(self) -> Dict[str, Any]:
        """执行完整工作流，返回 JSON 安全结果字典。"""
        config = self.config
        warnings: List[str] = []
        stages: Dict[str, float] = {}

        t_start = time.perf_counter()
        self._camera_meta = dict(self.camera.metadata())
        self._validate_capabilities(warnings)
        mode, label = self._provenance()

        # a) 定标 ---------------------------------------------------------
        t0 = time.perf_counter()
        master, calib_meta = self._obtain_calibration(warnings)
        stages["calibration"] = time.perf_counter() - t0

        per_elevation: List[Dict[str, Any]] = []
        table_rows: List[Dict[str, Any]] = []
        fea_records: List[Dict[str, Any]] = []
        t_loop = time.perf_counter()
        for elevation in config.elevations_deg:
            try:
                rec, rows, fea_terms, elev_stages = self._run_elevation(
                    float(elevation), master, calib_meta, label, warnings)
            except Exception as exc:                    # noqa: BLE001
                if config.stop_on_error:
                    raise
                msg = f"elevation {elevation}: {type(exc).__name__}: {exc}"
                warnings.append(msg + " (stop_on_error=False, skipped)")
                continue
            per_elevation.append(rec)
            table_rows.extend(rows)
            fea_records.extend(fea_terms)
            for key, value in elev_stages.items():
                stages[key] = stages.get(key, 0.0) + float(value)
        stages["elevation_loop"] = time.perf_counter() - t_loop
        if not per_elevation:
            raise RuntimeError("no elevation completed; see warnings")

        fea_request = self._build_fea_request(fea_records, per_elevation,
                                              mode, label)
        stages["total"] = time.perf_counter() - t_start

        result: Dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "label": label,
            "mode": mode,
            "status": "ok",
            "calibration": str(calib_meta.get("mode", "unknown")),
            "config": asdict(config),
            "summary": {
                "n_elevations": len(per_elevation),
                "elevations_deg": [float(e) for e in config.elevations_deg],
                "terms": list(config.terms),
                "n_dofs": len(self.mechanism.dofs()),
                "n_compensation_rows": len(table_rows),
                "n_un_compensatable_terms": len(
                    fea_request["un_compensatable_terms"]),
                "max_residual_rms_waves": max(
                    (rec["compensation"]["residual_rms_waves"]
                     for rec in per_elevation), default=0.0),
            },
            "stages": {k: float(v) for k, v in stages.items()},
            "per_elevation": per_elevation,
            "compensation_table": table_rows,
            "fea_request": fea_request,
            "warnings": warnings,
            "manifest": self._manifest(mode, label, config, calib_meta,
                                       stages),
        }
        return _jsonable(result)

    # ------------------------------------------------------- provenance/units
    def _provenance(self) -> Tuple[str, str]:
        modes = [str(getattr(obj, "mode", "hardware"))
                 for obj in (self.telescope, self.camera, self.mechanism)]
        if all(m == "mock" for m in modes):
            return "mock", "SYNTHETIC-VALIDATED"
        if any(m == "mock" for m in modes):
            return "mixed", "UNVERIFIED"
        return "hardware", "UNVERIFIED"

    def _validate_capabilities(self, warnings: List[str]) -> None:
        config = self.config
        meta = self._camera_meta
        supported = meta.get("supported_terms") or meta.get("terms")
        if supported:
            missing = [t for t in config.terms if t not in supported]
            if missing:
                raise ValueError(
                    f"camera does not support term(s) {missing}; supported="
                    f"{list(supported)}. An extended PD state needs a real "
                    "camera/forward-model adapter.")
        shape = meta.get("shape")
        w = None
        if isinstance(shape, (list, tuple)) and len(shape) == 2:
            w = int(min(int(shape[0]), int(shape[1])))
        if config.psf_window is not None:
            w = int(config.psf_window)
        if w is None:
            w = 64
            warnings.append(
                "camera metadata has no 'shape'; defaulting PSF window to 64")
        if w < 16:
            raise ValueError("PSF window must be >= 16 pixels")
        self._psf_window = int(w)
        dofs = self.mechanism.dofs()
        if not dofs:
            raise ValueError("mechanism.dofs() must contain at least one DOF")
        for spec in dofs:
            for key in ("name", "unit", "low", "high"):
                if key not in spec:
                    raise ValueError(f"mechanism DOF missing {key!r}: {spec}")
            if not np.isfinite(float(spec["low"])) or \
                    not np.isfinite(float(spec["high"])) or \
                    float(spec["low"]) >= float(spec["high"]):
                raise ValueError(f"mechanism DOF {spec['name']}: bad bounds")
        if len(dofs) < 1:
            raise ValueError("need at least one mechanism DOF")
        # camera data contract
        for key in ("adu_max", "adu_gain_e_per_adu"):
            if key not in meta:
                warnings.append(f"camera metadata missing {key!r}; using default")

    # -------------------------------------------------------------- manifest
    def _manifest(self, mode: str, label: str, config: PipelineConfig,
                  calib_meta: Dict[str, Any],
                  stages: Dict[str, float]) -> Dict[str, Any]:
        manifest: Dict[str, Any] = {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "generated_by": ("products/pipeline/orchestrator.py:"
                             "WavefrontPipeline.run"),
            "label": label,
            "mode": mode,
            "interfaces": {
                "telescope": type(self.telescope).__name__,
                "camera": type(self.camera).__name__,
                "mechanism": type(self.mechanism).__name__,
            },
            "config": asdict(config),
            "calibration": calib_meta,
            "stage_keys": sorted(stages),
            "outputs": ["pipeline_result.json", "compensation_table.csv",
                        "fea_request.json"],
            "units": {
                "zernike": "waves @632.8 nm wavefront",
                "S": "waves per mechanism DOF unit (as declared by dofs())",
                "elevation_deg": "deg",
                "exposure_s": "s",
                "adu": "raw ADU before calibration",
                "wavelength_nm": float(WAVELENGTH_NM),
            },
            "boundary": (
                "没有真实望远镜控制 API、没有相机/机构 SDK、也没有真实工况"
                "（operating condition）数据；外部边界全部由显式 mock 或未验证"
                " adapter 提供。mock 数据的 label=SYNTHETIC-VALIDATED，"
                "不能当作实测证据。"),
        }
        for obj_name, obj in (("telescope", self.telescope),
                              ("camera", self.camera),
                              ("mechanism", self.mechanism)):
            manifest[obj_name] = {
                "class": type(obj).__name__,
                "mode": str(getattr(obj, "mode", "hardware")),
                "label": str(getattr(obj, "label", "UNVERIFIED")),
            }
        return manifest

    # ---------------------------------------------------------- calibration
    def _obtain_calibration(self, warnings: List[str]
                            ) -> Tuple[MasterFrames, Dict[str, Any]]:
        config = self.config
        meta = self._camera_meta
        if config.calib_dir:
            master = MasterFrames.load(config.calib_dir)
            shape = master.shape
            cam_shape = meta.get("shape")
            if shape is not None and isinstance(cam_shape, (list, tuple)):
                if list(shape) != [int(cam_shape[0]), int(cam_shape[1])]:
                    raise ValueError(
                        f"master frame shape {shape} does not match camera "
                        f"frame shape {cam_shape}")
            bad = (bad_pixel_map(master.flat, sigma=config.bad_pixel_sigma)
                   if master.flat is not None
                   else np.zeros(shape or (1, 1), dtype=bool))
            return master, {
                "mode": "loaded",
                "path": str(config.calib_dir),
                "master": master.to_dict(),
                "bad_pixel_count": int(np.sum(bad)),
                "bad_pixel_mask": bad,
                "defect_mask": np.zeros(bad.shape, dtype=bool),
            }

        # --- synthetic：用相机前几帧的 bias 曝光 + 相机 metadata 合成 ---
        n = max(1, int(config.n_frames))
        bias_frames = self.camera.capture([0.0] * n, 0.0)
        bias = build_master_bias(bias_frames)
        gain = float(meta.get("adu_gain_e_per_adu", 1.0) or 1.0)
        gain = gain if gain > 0 else 1.0
        dark_rate = float(meta.get("dark_current_e_per_s", 0.0) or 0.0)
        dark = np.full_like(bias, dark_rate * float(config.exposure_s) / gain)
        flat_src = meta.get("flat_field")
        if flat_src is not None:
            flat = build_master_flat([np.asarray(flat_src, dtype=float)])
            flat_source = "camera metadata flat_field"
        else:
            flat = np.ones_like(bias)
            flat_source = "uniform (no flat data)"
            warnings.append(
                "calibration='synthetic': no calib_dir and camera has no "
                "flat_field; a uniform flat is used (未定标 mock)")
        bad = bad_pixel_map(flat, sigma=config.bad_pixel_sigma)
        defect = meta.get("defect_map")
        defect_mask = (np.asarray(defect, dtype=bool) if defect is not None
                       else np.zeros_like(bad))
        if defect_mask.shape != bad.shape:
            raise ValueError("camera defect_map shape does not match flat")
        bad_all = bad | defect_mask
        master = MasterFrames(
            bias=bias, dark=dark, flat=flat,
            bias_exposure_s=0.0, dark_exposure_s=float(config.exposure_s),
            flat_exposure_s=float(config.exposure_s),
            n_bias=n, n_dark=0, n_flat=1, label=str(
                getattr(self.camera, "label", "SYNTHETIC-VALIDATED")))
        warnings.append(
            "calibration='synthetic': master bias from camera bias exposures, "
            "dark/flat from camera metadata; provide a calib_dir with real "
            "master frames for an actual measurement")
        return master, {
            "mode": "synthetic",
            "source": ("first captures (bias exposures) + camera metadata; "
                       "no real calibration data available"),
            "master": master.to_dict(),
            "flat_source": flat_source,
            "bad_pixel_count": int(np.sum(bad_all)),
            "bad_pixel_mask": bad_all,
            "defect_mask": defect_mask,
        }

    # ------------------------------------------------------------- elevation
    def _run_elevation(self, elevation: float, master: MasterFrames,
                       calib_meta: Dict[str, Any], label: str,
                       warnings: List[str]
                       ) -> Tuple[Dict[str, Any], List[Dict[str, Any]],
                                  List[Dict[str, Any]], Dict[str, float]]:
        """单俯仰角处理（采集 -> 反演 -> 补偿 -> 可观测性 -> QC -> 报告装配）。

        实现按数据流拆分在 :mod:`products.pipeline._elevation`（私有步骤
        函数 + ``_ElevationRun`` 状态对象传递中间数据）；本方法仅保留类上
        的稳定入口，签名、调用时机与返回值 ``(rec, rows, fea_terms,
        stages)`` 的语义与拆分前完全一致。
        """
        return run_elevation(self, elevation, master, calib_meta, label,
                             warnings)

    def _build_fea_request(self, records: Sequence[Dict[str, Any]],
                           per_elevation: Sequence[Dict[str, Any]],
                           mode: str, label: str) -> Dict[str, Any]:
        by_term: Dict[str, List[Dict[str, Any]]] = {}
        for rec in records:
            by_term.setdefault(str(rec["term"]), []).append(dict(rec))
        un_comp = sorted(by_term)
        actions = [{"term": term,
                    "suggested_action": by_term[term][0]["suggested_action"]}
                   for term in un_comp]
        summary: Dict[str, Any] = {}
        for term in un_comp:
            items = by_term[term]
            mags = [float(it["magnitude_waves"]) for it in items]
            summary[term] = {
                "elevations_deg": [float(it["elevation_deg"]) for it in items],
                "max_magnitude_waves": max(mags),
                "max_magnitude_nm_wavefront": max(
                    float(it["magnitude_nm_wavefront"]) for it in items),
                "max_magnitude_nm_surface": max(
                    float(it["magnitude_nm_surface"]) for it in items),
                "max_dof_gain_waves_per_unit": max(
                    float(it["max_dof_gain_waves_per_unit"]) for it in items),
                "suggested_action": items[0]["suggested_action"],
            }
        rel = float(self.config.rel_threshold)
        return {
            "schema_version": MANIFEST_SCHEMA_VERSION,
            "label": "FEA-REQUEST",
            "mode": mode,
            "source_label": label,
            "generated_by": ("products/pipeline/orchestrator.py:"
                             "WavefrontPipeline._build_fea_request"),
            "no_fea_performed": True,
            "source": "products.sensitivity.observability.analyze_control_space",
            "rel_threshold": rel,
            "wavelength_nm": float(WAVELENGTH_NM),
            "units": {
                "target_waves": "waves @632.8 nm wavefront",
                "magnitude_nm_wavefront": "nm wavefront",
                "magnitude_nm_surface": "nm surface (wavefront/2, reflection)",
                "max_dof_gain_waves_per_unit":
                    "waves per mechanism DOF unit (as declared by dofs())",
            },
            "un_compensatable_terms": un_comp,
            "terms": list(records),
            "by_term": summary,
            "actions": actions,
            "inputs": {
                "elevations_deg": [float(e) for e in
                                   self.config.elevations_deg],
                "terms": list(self.config.terms),
                "dofs": [str(d["name"]) for d in self.mechanism.dofs()],
                "pose_amplitude": float(self.config.pose_amplitude),
                "rel_threshold": rel,
                "compensation_rows_per_elevation": {
                    str(rec["elevation_deg"]):
                        rec["compensation"]["commands"]
                    for rec in per_elevation},
            },
            "boundary": (
                "今天没有真实工况（operating condition）数据，也没有运行"
                "有限元求解器；本块是给 FEA/装调前处理阶段的**请求**，"
                "不是 FEA 结果，也不能作为合规/装调结论。"),
            "note": (
                "不可补偿项判据：S 中该项所在行的最大机构响应 < "
                f"rel_threshold({rel:g}) x 全局最大 |S|；真实数据需在"
                "bench 上复测 S 并确认判据。"),
        }


# ===========================================================================
# 便捷入口
# ===========================================================================

def run_pipeline(telescope: TelescopeTerminalApi, camera: CameraApi,
                 mechanism: MechanismApi,
                 config: Optional[PipelineConfig] = None) -> Dict[str, Any]:
    """一次性构造并运行 :class:`WavefrontPipeline`。"""
    return WavefrontPipeline(telescope, camera, mechanism, config).run()
