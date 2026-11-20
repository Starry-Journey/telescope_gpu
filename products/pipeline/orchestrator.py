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
        有界最小二乘求补偿量（numpy.linalg.lstsq + clip，不 import GUI）
        analyze_control_space 找不可补偿项 -> fea_request

输入:
    - 三个接口实现（mock 或 adapter）+ :class:`PipelineConfig`。
输出:
    - JSON 安全 dict：``label``/``mode``/``stages``/``per_elevation``/
      ``compensation_table``/``fea_request``/``warnings``/``manifest``。
边界:
    - 不 import 任何 GUI/Qt/网络；不 import products.control 的 GUI 服务
      （只用 numpy 解有界最小二乘；control.service 只可能被 mock 用作真值）；
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
from products.preprocess import (MasterFrames, bad_pixel_map,
                                 build_master_bias, build_master_flat,
                                 calibrate, centroid_subpixel, clean_psf,
                                 cutout, detect_stars, estimate_background,
                                 register_shift, remove_cosmic_rays,
                                 repair_bad_pixels, saturation_mask)
from products.sensitivity.measured import Pose, measure_sensitivity
from products.sensitivity.observability import (WAVELENGTH_NM,
                                                analyze_control_space)

#: 结果 manifest 的 schema 版本。
MANIFEST_SCHEMA_VERSION = 1

#: 无法补偿的项 -> 建议的有限元/装调前处理动作。
_FEA_ACTIONS: Dict[str, str] = {
    "Z4": ("离焦项：检查 M2 调焦行程/镜间距是否耗尽；若行程足够，先补一次"
           "调焦模型标定；否则请求 FEA 复核重力/温变下的 despace 与焦面位置"),
    "Z5": ("像散项：平移机构通常无法有效产生 45° 像散；请求 M1/M2 支撑结构"
           "FEA（重力/装调应力）并评估四点轴向像散校正器的位置与行程"),
    "Z6": ("像散项：平移机构通常无法有效产生 0° 像散；请求 M1/M2 支撑结构"
           "FEA（重力/装调应力）并评估四点轴向像散校正器的位置与行程"),
    "Z8": ("x 彗差：检查 M2 偏心/倾斜权限与指向预算；请求副镜支撑结构 FEA "
           "与装调偏心量复核"),
    "Z9": ("y 彗差：检查 M2 偏心/倾斜权限与指向预算；请求副镜支撑结构 FEA "
           "与装调偏心量复核"),
    "Z11": ("球差项：平移机构无有效自由度；请求主镜面形/热梯度 FEA 与"
            "光学装调（间隔/面形）复核"),
}


def _suggested_action(term: str) -> str:
    return _FEA_ACTIONS.get(
        term, f"{term}：该系数在实测灵敏度矩阵中低于相对阈值，平移机构无法"
              "有效补偿；请求 FEA/装调前处理复核该像差的来源与专用校正器")


def _rms(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    return float(np.sqrt(np.mean(arr ** 2))) if arr.size else 0.0


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
        config = self.config
        elevation_warnings: List[str] = []
        stages = {"capture": 0.0, "preprocess": 0.0, "invert": 0.0,
                  "sensitivity": 0.0, "compensation": 0.0,
                  "observability": 0.0}

        tel_status = self.telescope.goto_elevation(
            float(elevation), timeout_s=config.slew_timeout_s)
        self.mechanism.home()
        pointing = self.telescope.pointing()
        condition = None
        if hasattr(self.telescope, "operating_condition"):
            condition = self.telescope.operating_condition()

        dof_specs = [dict(spec) for spec in self.mechanism.dofs()]
        dof_names = [spec["name"] for spec in dof_specs]
        units = {spec["name"]: spec["unit"] for spec in dof_specs}
        lows = {spec["name"]: float(spec["low"]) for spec in dof_specs}
        highs = {spec["name"]: float(spec["high"]) for spec in dof_specs}
        terms = list(config.terms)
        defocus = list(config.defocus)
        amplitude = float(config.pose_amplitude)

        poses: List[Pose] = []
        ref_indices: List[int] = []
        captures: List[Dict[str, Any]] = []
        kept_frames: List[np.ndarray] = []
        timestamp = 0.0
        n_captures = 0

        def capture_pose(dof: Optional[str], level: float, repeat: int) -> None:
            nonlocal timestamp, n_captures
            t0 = time.perf_counter()
            raw = self.camera.capture(defocus, config.exposure_s)
            stages["capture"] += time.perf_counter() - t0
            n_captures += 1
            t1 = time.perf_counter()
            psf, info = self._preprocess_capture(
                raw, master, calib_meta, elevation_warnings)
            stages["preprocess"] += time.perf_counter() - t1
            if config.keep_frames:
                kept_frames.append(np.asarray(raw, dtype=float))
            poses.append(Pose(dof, float(level), psf,
                              timestamp=timestamp, repeat=repeat,
                              meta=info))
            captures.append({"dof": dof, "level": float(level),
                             "repeat": int(repeat),
                             "star": info["star"],
                             "background_adu": info["background_adu"],
                             "noise_adu": info["noise_adu"],
                             "n_saturated_px": info["n_saturated_px"]})
            timestamp += 1.0

        for repeat in range(int(config.repeats)):
            capture_pose(None, 0.0, repeat)
            ref_indices.append(len(poses) - 1)
            for dof in dof_names:
                for sign in (1.0, -1.0):
                    self.mechanism.move(dof, sign * amplitude)
                    capture_pose(dof, sign * amplitude, repeat)
                    self.mechanism.move(dof, 0.0)
                    capture_pose(None, 0.0, repeat)
                    ref_indices.append(len(poses) - 1)
        readback = self.mechanism.readback()

        # --- 实测灵敏度（复用 products.sensitivity.measured，不重写估计器）---
        z_log: List[Dict[str, float]] = []
        invert_time = [0.0]

        def invert_hook(frames: Any) -> Dict[str, Any]:
            # 延迟 import：没有 nlopt 时管线模块仍可 import（测试可跳过）。
            from products.pd.service import invert_psf
            t0 = time.perf_counter()
            res = invert_psf(np.asarray(frames, dtype=float), defocus,
                             terms=terms, maxeval=config.pd_maxeval,
                             n_starts=config.pd_n_starts, seed=config.seed)
            invert_time[0] += time.perf_counter() - t0
            z_log.append({t: float(res["zernike"][t]) for t in terms})
            return res

        t0 = time.perf_counter()
        sens = measure_sensitivity(poses, invert=invert_hook, terms=terms,
                                   defocus=defocus, label=label)
        stages["invert"] = invert_time[0]
        stages["sensitivity"] = max(
            0.0, time.perf_counter() - t0 - invert_time[0])

        target = np.mean(
            np.asarray([[z_log[i][t] for t in terms] for i in ref_indices],
                       dtype=float), axis=0)

        # --- 有界最小二乘补偿（numpy，不 import GUI）-----------------------
        t0 = time.perf_counter()
        S = np.asarray(sens["S"], dtype=float)
        dofs_used = list(sens["dofs"])
        lo = np.asarray([lows[d] for d in dofs_used], dtype=float)
        hi = np.asarray([highs[d] for d in dofs_used], dtype=float)
        # 机构命令 u 产生的波前增量是 S u；目标是抵消 target => S u = -target
        u_raw, *_ = np.linalg.lstsq(S, -target, rcond=None)
        u = np.nan_to_num(np.asarray(u_raw, dtype=float), nan=0.0,
                          posinf=0.0, neginf=0.0)
        u = np.clip(u, lo, hi)
        predicted = S @ u
        residual = target + predicted
        active_bounds = [
            dofs_used[j] for j in range(len(dofs_used))
            if bool(np.isclose(u[j], lo[j]) or np.isclose(u[j], hi[j]))]
        stages["compensation"] = time.perf_counter() - t0

        # --- 可观测性 / 不可补偿项 ----------------------------------------
        t0 = time.perf_counter()
        obs = analyze_control_space(S, terms, dofs_used,
                                    rel_threshold=config.rel_threshold)
        stages["observability"] = time.perf_counter() - t0

        fea_terms: List[Dict[str, Any]] = []
        for term in obs["dropped_rows"]:
            i = terms.index(term)
            amp = float(abs(target[i]))
            gain = float(obs["per_target"][term]["max_abs"])
            fea_terms.append({
                "term": term,
                "elevation_deg": float(elevation),
                "target_waves": float(target[i]),
                "magnitude_waves": amp,
                "magnitude_nm_wavefront": amp * WAVELENGTH_NM,
                "magnitude_nm_surface": amp * WAVELENGTH_NM / 2.0,
                "max_dof_gain_waves_per_unit": gain,
                "relative_to_global": (
                    float(gain / obs["global_max"])
                    if obs["global_max"] > 0 else 0.0),
                "threshold_waves": float(obs["threshold_abs"]),
                "rel_threshold": float(config.rel_threshold),
                "reason": (
                    "所有机构 DOF 在该项上的最大响应低于 "
                    f"{config.rel_threshold:g} x 全局最大 |S| "
                    f"({obs['threshold_abs']:.6g} waves/unit)"),
                "suggested_action": _suggested_action(term),
            })

        # --- 帧间配准 QC（参考位姿漂移）-----------------------------------
        # 不同离焦帧的 PSF 形状本来就不同，不能互相配准；这里固定用 |defocus|
        # 最小的一帧，比较各次 reference pose（u=0）的漂移。离焦环状 PSF 的
        # 相位相关可能出现对称歧义，所以用质心漂移做判据，同时记录相位相关
        # 结果作为诊断信息。
        qc_frame = int(np.argmin(np.abs(np.asarray(defocus, dtype=float))))
        ref_frames = poses[ref_indices[0]].frames
        ref_cx, ref_cy = centroid_subpixel(ref_frames[qc_frame])
        registrations: List[Dict[str, Any]] = []
        for i in ref_indices:
            pose = poses[i]
            cx, cy = centroid_subpixel(pose.frames[qc_frame])
            shift = float(np.hypot(cy - ref_cy, cx - ref_cx))
            pdy, pdx, perr = register_shift(ref_frames[qc_frame],
                                            pose.frames[qc_frame])
            registrations.append({
                "dof": pose.dof, "level": float(pose.level),
                "repeat": int(pose.repeat),
                "centroid_drift_px": shift,
                "phase_dy_px": float(pdy), "phase_dx_px": float(pdx),
                "phase_error": float(perr),
            })
        max_shift = max((r["centroid_drift_px"] for r in registrations),
                        default=0.0)
        if max_shift > float(config.register_tolerance_px):
            elevation_warnings.append(
                f"registration: max reference-pose centroid drift "
                f"{max_shift:.3f} px > {config.register_tolerance_px} px "
                "tolerance")

        if active_bounds:
            elevation_warnings.append(
                f"compensation hit mechanism bounds for {active_bounds}; "
                "residual may be limited by travel, not by the solve")
        if int(obs["rank"]) < len(dofs_used):
            elevation_warnings.append(
                f"measured S rank {obs['rank']} < n_dofs {len(dofs_used)}; "
                "some DOF directions are degenerate")
        if not fea_terms:
            elevation_warnings.append(
                "every target term is observable above rel_threshold; no "
                "FEA/alignment request needed at this elevation")
        warnings.extend(f"elevation {elevation:g}: {w}" for w in
                        elevation_warnings)

        rows = self._compensation_rows(
            elevation, dofs_used, units, u, lo, hi, target, predicted,
            residual, terms, active_bounds)

        rec: Dict[str, Any] = {
            "elevation_deg": float(elevation),
            "telescope": {"status": tel_status, "pointing": pointing,
                          "operating_condition": condition},
            "dofs": dof_specs,
            "terms": terms,
            "defocus": defocus,
            "n_poses": len(poses),
            "n_captures": int(n_captures),
            "n_inversions": len(z_log),
            "reference_poses": len(ref_indices),
            "mechanism_readback": readback,
            "target_aberration_waves": {t: float(target[i])
                                        for i, t in enumerate(terms)},
            "target_aberration_nm_wavefront": {
                t: float(target[i]) * WAVELENGTH_NM
                for i, t in enumerate(terms)},
            "measured_sensitivity": {
                "terms": list(sens["terms"]),
                "dofs": dofs_used,
                "S": sens["S"],
                "S_std": sens.get("S_std"),
                "rank": int(sens["rank"]),
                "condition": float(sens["condition"]),
                "singular_values": sens.get("singular_values"),
                "hysteresis": sens.get("hysteresis"),
                "nonlinearity": sens.get("nonlinearity"),
                "drift_rms": float(sens.get("drift_rms", 0.0)),
                "unbracketed_poses": int(sens.get("unbracketed_poses", 0)),
                "label": sens.get("label", label),
                "method": sens.get("method"),
            },
            "compensation": {
                "commands": {d: float(u[j]) for j, d in enumerate(dofs_used)},
                "units": {d: units[d] for d in dofs_used},
                "bounds": {d: [lows[d], highs[d]] for d in dofs_used},
                "predicted_correction_waves": {
                    t: float(predicted[i]) for i, t in enumerate(terms)},
                "residual_waves": {t: float(residual[i])
                                   for i, t in enumerate(terms)},
                "residual_rms_waves": _rms(residual),
                "max_abs_residual_waves": float(np.max(np.abs(residual))),
                "active_bounds": active_bounds,
            },
            "observability": {
                "rel_threshold": float(config.rel_threshold),
                "threshold_abs": float(obs["threshold_abs"]),
                "global_max": float(obs["global_max"]),
                "rank": int(obs["rank"]),
                "condition": float(obs["condition"]),
                "dropped_rows": list(obs["dropped_rows"]),
                "dropped_cols": list(obs["dropped_cols"]),
                "observable_dofs": list(obs["observable_dofs"]),
                "per_target": obs["per_target"],
                "per_dof": obs["per_dof"],
                "rationale": obs["rationale"],
            },
            "fea_terms": fea_terms,
            "registration": {
                "tolerance_px": float(config.register_tolerance_px),
                "max_shift_px": float(max_shift),
                "poses": registrations,
            },
            "stages": {k: float(v) for k, v in stages.items()},
            "warnings": elevation_warnings,
        }
        if config.keep_frames:
            rec["frames"] = kept_frames
        # 绑定内存：单次俯仰角处理完即丢弃帧（keep_frames=False 时无引用）
        del poses, kept_frames
        return rec, rows, fea_terms, stages

    # ------------------------------------------------------- per-capture pre
    def _preprocess_capture(self, raw: np.ndarray, master: MasterFrames,
                            calib_meta: Dict[str, Any],
                            warnings: List[str]
                            ) -> Tuple[np.ndarray, Dict[str, Any]]:
        config = self.config
        raw = np.asarray(raw, dtype=float)
        if raw.ndim != 3:
            raise ValueError("camera.capture must return [K,N,N]")
        window = int(self._psf_window)
        if raw.shape[1] < window or raw.shape[2] < window:
            raise ValueError(
                f"camera frame {raw.shape[1:]} smaller than PSF window "
                f"{window}")

        adu_max = float(self._camera_meta.get("adu_max", 65535))
        # 饱和/宇宙线必须在 raw ADU 上判定（bias 扣除后尖峰可能低于 adu_max，
        # 只看定标后帧会漏掉）。这里对 raw 判饱和，再用定标后的帧做修复。
        raw_saturated = np.stack([
            saturation_mask(raw[k], adu_max) for k in range(raw.shape[0])])
        calibrated = calibrate(raw, master, config.exposure_s)
        base_bad = np.zeros(calibrated.shape[1:], dtype=bool)
        if calib_meta.get("bad_pixel_mask") is not None:
            base_bad |= np.asarray(calib_meta["bad_pixel_mask"], dtype=bool)
        if calib_meta.get("defect_mask") is not None:
            base_bad |= np.asarray(calib_meta["defect_mask"], dtype=bool)

        n_sat = 0
        repaired: List[np.ndarray] = []
        for k in range(calibrated.shape[0]):
            frame = calibrated[k]
            sat = raw_saturated[k]
            n_sat += int(np.count_nonzero(sat))
            mask = base_bad | sat
            repaired.append(repair_bad_pixels(frame, mask) if mask.any()
                            else np.array(frame, copy=True))

        # 星点检测：WFS 相机是"以恒星为中心的 star-window ROI"，强离焦帧的
        # 环状结构会被阈值分割成弧段，因此只接受窗口中心附近的候选，否则
        # 回退到几何中心（真实整幅相机应由 adapter 提供 ROI/目标星约定）。
        frames = np.asarray(repaired)
        qc_frame = int(np.argmin(np.abs(np.asarray(config.defocus, dtype=float))))
        bg0, _ = estimate_background(frames[0])
        stars = detect_stars(frames[qc_frame], threshold_sigma=5.0,
                             min_pixels=3)
        n = frames.shape[1]
        centre = ((n - 1) / 2.0, (n - 1) / 2.0)
        near = [s for s in stars
                if float(np.hypot(s["x"] - centre[0], s["y"] - centre[1]))
                <= 0.3 * float(n)]
        if near:
            target = max(near, key=lambda s: float(s["flux"]))
        elif stars:
            target = max(stars, key=lambda s: float(s["flux"]))
        else:
            target = {"x": centre[0], "y": centre[1], "flux": 0.0,
                      "peak": 0.0, "npix": 0}
            warnings.append(
                "no star detected in a captured frame; falling back to the "
                "window centre for the PSF cutout")
        yy, xx = np.indices(frames.shape[1:], dtype=float)
        protect = ((xx - target["x"]) ** 2 + (yy - target["y"]) ** 2
                   <= max(6.0, window * 0.15) ** 2)

        cleaned = np.empty_like(frames)
        for k in range(frames.shape[0]):
            cleaned[k] = remove_cosmic_rays(frames[k], sigma=config.cr_sigma,
                                            protect=protect)
        backgrounds = np.empty(frames.shape[0], dtype=float)
        noises = np.empty(frames.shape[0], dtype=float)
        for k in range(frames.shape[0]):
            backgrounds[k], noises[k] = estimate_background(cleaned[k])
        cleaned = cleaned - backgrounds[:, None, None]

        psf_frames = np.empty((frames.shape[0], window, window), dtype=float)
        for k in range(frames.shape[0]):
            cut = cutout(cleaned[k], target["x"], target["y"], window,
                         clamp=True)
            psf_frames[k] = clean_psf(cut)
        totals = psf_frames.sum(axis=(1, 2), keepdims=True)
        psf_frames /= np.maximum(totals, 1e-12)
        info = {
            "star": {k: float(target[k]) for k in ("x", "y", "flux", "peak")},
            "background_adu": backgrounds.tolist(),
            "noise_adu": noises.tolist(),
            "n_saturated_px": int(n_sat),
            "psf_window": int(window),
            "n_stars_detected": int(len(stars)),
            "initial_background_adu": float(bg0),
        }
        return psf_frames, info

    # -------------------------------------------------------- table/fea build
    def _compensation_rows(self, elevation: float,
                           dofs_used: Sequence[str],
                           units: Dict[str, str], u: np.ndarray,
                           lo: np.ndarray, hi: np.ndarray,
                           target: np.ndarray, predicted: np.ndarray,
                           residual: np.ndarray, terms: Sequence[str],
                           active_bounds: Sequence[str]
                           ) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        target_rms = _rms(target)
        resid_rms = _rms(residual)
        max_resid = float(np.max(np.abs(residual)))
        for j, dof in enumerate(dofs_used):
            row: Dict[str, Any] = {
                "elevation_deg": float(elevation),
                "dof": str(dof),
                "unit": units[dof],
                "command": float(u[j]),
                "lower": float(lo[j]),
                "upper": float(hi[j]),
                "at_bound": bool(dof in active_bounds),
                "target_rms_waves": target_rms,
                "residual_rms_waves": resid_rms,
                "max_abs_residual_waves": max_resid,
            }
            for i, term in enumerate(terms):
                row[f"target_{term}_waves"] = float(target[i])
                row[f"predicted_{term}_waves"] = float(predicted[i])
                row[f"residual_{term}_waves"] = float(residual[i])
            rows.append(row)
        return rows

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
