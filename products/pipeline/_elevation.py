"""单俯仰角处理流程（自 orchestrator.WavefrontPipeline._run_elevation 拆出）。

本模块只做一件事：处理**一个**俯仰角——机构 DOF 声明快照 -> 逐 DOF ±激励
采集（含逐帧预处理）-> PD 反演/实测灵敏度 -> 有界最小二乘补偿 ->
可观测性/不可补偿项 -> 帧间配准 QC -> 单俯仰角结果 dict 装配。

拆分约定：

* 步骤之间共享的中间数据用 :class:`_ElevationRun` 按数据流分组传递，
  避免长参数表；各步骤是本模块的私有函数（采集/预处理/反演/补偿/
  可观测性/配准 QC/报告装配）。
* ``WavefrontPipeline._run_elevation`` 是类上的稳定入口（薄委托），其
  签名与返回值 ``(rec, rows, fea_terms, stages)`` 与拆分前完全一致。
* 这是**纯行为保持**重构：``run_pipeline()`` 对外契约（返回 dict 的每
  一个键、每个键的值语义与顺序）不变；``products.lsq.bounded_least_squares``
  调用点语义不变（补偿约定 ``S u ≈ -target``，见 :mod:`products.lsq`）。
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from products.lsq import bounded_least_squares
from products.preprocess import (MasterFrames, calibrate, centroid_subpixel,
                                 clean_psf, cutout, detect_stars,
                                 estimate_background, register_shift,
                                 remove_cosmic_rays, repair_bad_pixels,
                                 saturation_mask)
from products.sensitivity.measured import Pose, measure_sensitivity
from products.sensitivity.observability import (WAVELENGTH_NM,
                                                analyze_control_space)

if TYPE_CHECKING:                                    # pragma: no cover
    from products.pipeline.orchestrator import WavefrontPipeline

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


# ===========================================================================
# 步骤间共享状态（私有）
# ===========================================================================

@dataclass
class _DofLayout:
    """机构 DOF 声明快照（``mechanism.dofs()`` 的逐项浅拷贝与派生索引）。"""

    specs: List[Dict[str, Any]]        # 原始 DOF 声明（浅拷贝）
    names: List[str]                   # DOF 名，采集循环的激励顺序
    units: Dict[str, str]              # name -> unit
    lows: Dict[str, float]             # name -> 下界
    highs: Dict[str, float]            # name -> 上界


@dataclass
class _Compensation:
    """有界最小二乘补偿解与派生量（补偿约定：``residual = target + S u``）。"""

    S: np.ndarray
    dofs_used: List[str]
    lo: np.ndarray
    hi: np.ndarray
    u: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    active_bounds: List[str]


@dataclass
class _ElevationRun:
    """单个俯仰角的中间状态，按数据流在各步骤函数间传递（本模块私有）。

    字段按处理阶段分组填充：telescope 状态 -> DOF 布局 -> 采集 ->
    灵敏度/反演 -> 补偿 -> 可观测性 -> 配准 QC -> 报告装配。
    """

    elevation: float
    label: str
    master: MasterFrames
    calib_meta: Dict[str, Any]
    terms: List[str]                   # config.terms 的本俯仰角副本
    defocus: List[float]               # config.defocus 的本俯仰角副本
    stages: Dict[str, float]           # 阶段计时（键序即报告顺序）
    elevation_warnings: List[str] = field(default_factory=list)
    # --- telescope 状态 ---
    tel_status: Any = None
    pointing: Any = None
    condition: Any = None
    # --- 机构 DOF ---
    layout: Optional[_DofLayout] = None
    # --- 采集 ---
    poses: List[Pose] = field(default_factory=list)
    ref_indices: List[int] = field(default_factory=list)
    kept_frames: List[np.ndarray] = field(default_factory=list)
    n_captures: int = 0
    readback: Any = None
    # --- 灵敏度/反演 ---
    sens: Dict[str, Any] = field(default_factory=dict)
    target: Optional[np.ndarray] = None
    n_inversions: int = 0
    # --- 补偿 ---
    comp: Optional[_Compensation] = None
    # --- 可观测性 ---
    obs: Dict[str, Any] = field(default_factory=dict)
    fea_terms: List[Dict[str, Any]] = field(default_factory=list)
    # --- 配准 QC ---
    registrations: List[Dict[str, Any]] = field(default_factory=list)
    max_shift: float = 0.0


# ===========================================================================
# 入口：单俯仰角处理（WavefrontPipeline._run_elevation 的实现）
# ===========================================================================

def run_elevation(pipeline: "WavefrontPipeline", elevation: float,
                  master: MasterFrames, calib_meta: Dict[str, Any],
                  label: str, warnings: List[str]
                  ) -> Tuple[Dict[str, Any], List[Dict[str, Any]],
                             List[Dict[str, Any]], Dict[str, float]]:
    """处理单个俯仰角，返回 ``(rec, rows, fea_terms, stages)``。

    按数据流依次执行各步骤；步骤顺序、告警汇入 ``warnings`` 的时机与
    返回 dict 的键序均与拆分前保持一致。
    """
    config = pipeline.config
    run = _ElevationRun(
        elevation=float(elevation),
        label=label,
        master=master,
        calib_meta=calib_meta,
        terms=list(config.terms),
        defocus=list(config.defocus),
        stages={"capture": 0.0, "preprocess": 0.0, "invert": 0.0,
                "sensitivity": 0.0, "compensation": 0.0,
                "observability": 0.0},
    )

    run.tel_status = pipeline.telescope.goto_elevation(
        run.elevation, timeout_s=config.slew_timeout_s)
    pipeline.mechanism.home()
    run.pointing = pipeline.telescope.pointing()
    run.condition = None
    if hasattr(pipeline.telescope, "operating_condition"):
        run.condition = pipeline.telescope.operating_condition()

    run.layout = _dof_layout(pipeline)
    _capture_poses(pipeline, run)
    _solve_sensitivity(pipeline, run)
    run.comp = _solve_compensation(run)
    _assess_observability(pipeline, run)
    _registration_qc(pipeline, run)
    _emit_elevation_warnings(run, warnings)

    rows = _compensation_rows(run)
    rec = _build_elevation_report(pipeline, run)
    if config.keep_frames:
        rec["frames"] = run.kept_frames
    # 绑定内存：单次俯仰角处理完即丢弃帧（keep_frames=False 时无引用）
    del run.poses, run.kept_frames
    return rec, rows, run.fea_terms, run.stages


# ===========================================================================
# 步骤：机构 DOF 快照
# ===========================================================================

def _dof_layout(pipeline: "WavefrontPipeline") -> _DofLayout:
    """快照机构 DOF 声明（逐项浅拷贝）并派生名字/单位/界索引。"""
    dof_specs = [dict(spec) for spec in pipeline.mechanism.dofs()]
    return _DofLayout(
        specs=dof_specs,
        names=[spec["name"] for spec in dof_specs],
        units={spec["name"]: spec["unit"] for spec in dof_specs},
        lows={spec["name"]: float(spec["low"]) for spec in dof_specs},
        highs={spec["name"]: float(spec["high"]) for spec in dof_specs},
    )


# ===========================================================================
# 步骤：采集（逐 DOF ±激励）与逐帧预处理
# ===========================================================================

def _capture_poses(pipeline: "WavefrontPipeline", run: _ElevationRun) -> None:
    """逐 DOF 施加 +A/-A 激励并采集位姿，填充 run 的采集字段与计时。

    每个 repeat：参考位姿 -> 对每个 DOF 依次 +A、回零、-A、回零（每次
    机构移动后各采一帧）。``ref_indices`` 记录全部参考位姿（u=0）的下标，
    供 target 平均与配准 QC 使用。
    """
    config = pipeline.config
    stages = run.stages
    poses: List[Pose] = []
    ref_indices: List[int] = []
    captures: List[Dict[str, Any]] = []   # 采集记录（与拆分前实现保持一致）
    kept_frames: List[np.ndarray] = []
    timestamp = 0.0
    n_captures = 0
    amplitude = float(config.pose_amplitude)

    def capture_pose(dof: Optional[str], level: float, repeat: int) -> None:
        nonlocal timestamp, n_captures
        t0 = time.perf_counter()
        raw = pipeline.camera.capture(run.defocus, config.exposure_s)
        stages["capture"] += time.perf_counter() - t0
        n_captures += 1
        t1 = time.perf_counter()
        psf, info = _preprocess_capture(pipeline, raw, run.master,
                                        run.calib_meta,
                                        run.elevation_warnings)
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
        for dof in run.layout.names:
            for sign in (1.0, -1.0):
                pipeline.mechanism.move(dof, sign * amplitude)
                capture_pose(dof, sign * amplitude, repeat)
                pipeline.mechanism.move(dof, 0.0)
                capture_pose(None, 0.0, repeat)
                ref_indices.append(len(poses) - 1)
    run.poses = poses
    run.ref_indices = ref_indices
    run.kept_frames = kept_frames
    run.n_captures = n_captures
    run.readback = pipeline.mechanism.readback()


def _preprocess_capture(pipeline: "WavefrontPipeline", raw: np.ndarray,
                        master: MasterFrames, calib_meta: Dict[str, Any],
                        warnings: List[str]
                        ) -> Tuple[np.ndarray, Dict[str, Any]]:
    config = pipeline.config
    raw = np.asarray(raw, dtype=float)
    if raw.ndim != 3:
        raise ValueError("camera.capture must return [K,N,N]")
    window = int(pipeline._psf_window)
    if raw.shape[1] < window or raw.shape[2] < window:
        raise ValueError(
            f"camera frame {raw.shape[1:]} smaller than PSF window "
            f"{window}")

    adu_max = float(pipeline._camera_meta.get("adu_max", 65535))
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


# ===========================================================================
# 步骤：PD 反演 + 实测灵敏度
# ===========================================================================

def _solve_sensitivity(pipeline: "WavefrontPipeline",
                       run: _ElevationRun) -> None:
    """实测灵敏度（复用 products.sensitivity.measured，不重写估计器）。

    ``invert_hook`` 逐位姿调用 PD 反演并记录 Zernike 对数；``run.target``
    是各参考位姿反演结果的按项平均（参考位姿应接近零）。
    """
    config = pipeline.config
    z_log: List[Dict[str, float]] = []
    invert_time = [0.0]

    def invert_hook(frames: Any) -> Dict[str, Any]:
        # 延迟 import：没有 nlopt 时管线模块仍可 import（测试可跳过）。
        from products.pd.service import invert_psf
        t0 = time.perf_counter()
        res = invert_psf(np.asarray(frames, dtype=float), run.defocus,
                         terms=run.terms, maxeval=config.pd_maxeval,
                         n_starts=config.pd_n_starts, seed=config.seed)
        invert_time[0] += time.perf_counter() - t0
        z_log.append({t: float(res["zernike"][t]) for t in run.terms})
        return res

    t0 = time.perf_counter()
    run.sens = measure_sensitivity(run.poses, invert=invert_hook,
                                   terms=run.terms, defocus=run.defocus,
                                   label=run.label)
    run.stages["invert"] = invert_time[0]
    run.stages["sensitivity"] = max(
        0.0, time.perf_counter() - t0 - invert_time[0])

    run.target = np.mean(
        np.asarray([[z_log[i][t] for t in run.terms]
                    for i in run.ref_indices], dtype=float), axis=0)
    run.n_inversions = len(z_log)


# ===========================================================================
# 步骤：有界最小二乘补偿
# ===========================================================================

def _solve_compensation(run: _ElevationRun) -> _Compensation:
    """有界最小二乘求补偿量（scipy lsq_linear，不 import GUI）。"""
    t0 = time.perf_counter()
    layout = run.layout
    S = np.asarray(run.sens["S"], dtype=float)
    dofs_used = list(run.sens["dofs"])
    lo = np.asarray([layout.lows[d] for d in dofs_used], dtype=float)
    hi = np.asarray([layout.highs[d] for d in dofs_used], dtype=float)
    # 机构命令 u 产生的波前增量是 S u；目标是抵消 target => S u = -target
    # （bounded_least_squares 用拟合约定，这里传 -target；界约束在求解
    # 内部生效，不再是 lstsq 后 clip）
    u_raw, _bls_info = bounded_least_squares(S, -run.target, lo, hi)
    u = np.nan_to_num(np.asarray(u_raw, dtype=float), nan=0.0,
                      posinf=0.0, neginf=0.0)
    predicted = S @ u
    residual = run.target + predicted
    active_bounds = [
        dofs_used[j] for j in range(len(dofs_used))
        if bool(np.isclose(u[j], lo[j]) or np.isclose(u[j], hi[j]))]
    run.stages["compensation"] = time.perf_counter() - t0
    return _Compensation(S=S, dofs_used=dofs_used, lo=lo, hi=hi, u=u,
                         predicted=predicted, residual=residual,
                         active_bounds=active_bounds)


# ===========================================================================
# 步骤：可观测性 / 不可补偿项
# ===========================================================================

def _assess_observability(pipeline: "WavefrontPipeline",
                          run: _ElevationRun) -> None:
    """可观测性分析 + 不可补偿项 -> FEA/装调前处理请求记录。"""
    config = pipeline.config
    comp = run.comp
    t0 = time.perf_counter()
    run.obs = analyze_control_space(comp.S, run.terms, comp.dofs_used,
                                    rel_threshold=config.rel_threshold)
    run.stages["observability"] = time.perf_counter() - t0

    fea_terms: List[Dict[str, Any]] = []
    for term in run.obs["dropped_rows"]:
        i = run.terms.index(term)
        amp = float(abs(run.target[i]))
        gain = float(run.obs["per_target"][term]["max_abs"])
        fea_terms.append({
            "term": term,
            "elevation_deg": run.elevation,
            "target_waves": float(run.target[i]),
            "magnitude_waves": amp,
            "magnitude_nm_wavefront": amp * WAVELENGTH_NM,
            "magnitude_nm_surface": amp * WAVELENGTH_NM / 2.0,
            "max_dof_gain_waves_per_unit": gain,
            "relative_to_global": (
                float(gain / run.obs["global_max"])
                if run.obs["global_max"] > 0 else 0.0),
            "threshold_waves": float(run.obs["threshold_abs"]),
            "rel_threshold": float(config.rel_threshold),
            "reason": (
                "所有机构 DOF 在该项上的最大响应低于 "
                f"{config.rel_threshold:g} x 全局最大 |S| "
                f"({run.obs['threshold_abs']:.6g} waves/unit)"),
            "suggested_action": _suggested_action(term),
        })
    run.fea_terms = fea_terms


# ===========================================================================
# 步骤：帧间配准 QC（参考位姿漂移）
# ===========================================================================

def _registration_qc(pipeline: "WavefrontPipeline",
                     run: _ElevationRun) -> None:
    """比较各次参考位姿（u=0）的漂移，超差记入 elevation_warnings。"""
    # 不同离焦帧的 PSF 形状本来就不同，不能互相配准；这里固定用 |defocus|
    # 最小的一帧，比较各次 reference pose（u=0）的漂移。离焦环状 PSF 的
    # 相位相关可能出现对称歧义，所以用质心漂移做判据，同时记录相位相关
    # 结果作为诊断信息。
    config = pipeline.config
    poses = run.poses
    qc_frame = int(np.argmin(np.abs(np.asarray(run.defocus, dtype=float))))
    ref_frames = poses[run.ref_indices[0]].frames
    ref_cx, ref_cy = centroid_subpixel(ref_frames[qc_frame])
    registrations: List[Dict[str, Any]] = []
    for i in run.ref_indices:
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
        run.elevation_warnings.append(
            f"registration: max reference-pose centroid drift "
            f"{max_shift:.3f} px > {config.register_tolerance_px} px "
            "tolerance")
    run.registrations = registrations
    run.max_shift = max_shift


# ===========================================================================
# 步骤：告警汇总 / 报告装配
# ===========================================================================

def _emit_elevation_warnings(run: _ElevationRun,
                             warnings: List[str]) -> None:
    """补偿触界 / S 秩亏 / 无需 FEA 告警，并以俯仰角前缀汇入总 warnings。"""
    comp = run.comp
    obs = run.obs
    elevation_warnings = run.elevation_warnings
    if comp.active_bounds:
        elevation_warnings.append(
            f"compensation hit mechanism bounds for {comp.active_bounds}; "
            "residual may be limited by travel, not by the solve")
    if int(obs["rank"]) < len(comp.dofs_used):
        elevation_warnings.append(
            f"measured S rank {obs['rank']} < n_dofs {len(comp.dofs_used)}; "
            "some DOF directions are degenerate")
    if not run.fea_terms:
        elevation_warnings.append(
            "every target term is observable above rel_threshold; no "
            "FEA/alignment request needed at this elevation")
    warnings.extend(f"elevation {run.elevation:g}: {w}" for w in
                    elevation_warnings)


def _compensation_rows(run: _ElevationRun) -> List[Dict[str, Any]]:
    """补偿表行：每 DOF 一行 + 逐项 target/predicted/residual 列。"""
    comp = run.comp
    layout = run.layout
    target = run.target
    predicted = comp.predicted
    residual = comp.residual
    terms = run.terms
    rows: List[Dict[str, Any]] = []
    target_rms = _rms(target)
    resid_rms = _rms(residual)
    max_resid = float(np.max(np.abs(residual)))
    for j, dof in enumerate(comp.dofs_used):
        row: Dict[str, Any] = {
            "elevation_deg": run.elevation,
            "dof": str(dof),
            "unit": layout.units[dof],
            "command": float(comp.u[j]),
            "lower": float(comp.lo[j]),
            "upper": float(comp.hi[j]),
            "at_bound": bool(dof in comp.active_bounds),
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


def _build_elevation_report(pipeline: "WavefrontPipeline",
                            run: _ElevationRun) -> Dict[str, Any]:
    """装配单俯仰角结果 dict（键集合/顺序/值语义即对外契约，保持不变）。"""
    config = pipeline.config
    comp = run.comp
    layout = run.layout
    sens = run.sens
    obs = run.obs
    rec: Dict[str, Any] = {
        "elevation_deg": run.elevation,
        "telescope": {"status": run.tel_status, "pointing": run.pointing,
                      "operating_condition": run.condition},
        "dofs": layout.specs,
        "terms": run.terms,
        "defocus": run.defocus,
        "n_poses": len(run.poses),
        "n_captures": int(run.n_captures),
        "n_inversions": run.n_inversions,
        "reference_poses": len(run.ref_indices),
        "mechanism_readback": run.readback,
        "target_aberration_waves": {t: float(run.target[i])
                                    for i, t in enumerate(run.terms)},
        "target_aberration_nm_wavefront": {
            t: float(run.target[i]) * WAVELENGTH_NM
            for i, t in enumerate(run.terms)},
        "measured_sensitivity": {
            "terms": list(sens["terms"]),
            "dofs": comp.dofs_used,
            "S": sens["S"],
            "S_std": sens.get("S_std"),
            "rank": int(sens["rank"]),
            "condition": float(sens["condition"]),
            "singular_values": sens.get("singular_values"),
            "hysteresis": sens.get("hysteresis"),
            "nonlinearity": sens.get("nonlinearity"),
            "drift_rms": float(sens.get("drift_rms", 0.0)),
            "unbracketed_poses": int(sens.get("unbracketed_poses", 0)),
            "label": sens.get("label", run.label),
            "method": sens.get("method"),
        },
        "compensation": {
            "commands": {d: float(comp.u[j])
                         for j, d in enumerate(comp.dofs_used)},
            "units": {d: layout.units[d] for d in comp.dofs_used},
            "bounds": {d: [layout.lows[d], layout.highs[d]]
                       for d in comp.dofs_used},
            "predicted_correction_waves": {
                t: float(comp.predicted[i])
                for i, t in enumerate(run.terms)},
            "residual_waves": {t: float(comp.residual[i])
                               for i, t in enumerate(run.terms)},
            "residual_rms_waves": _rms(comp.residual),
            "max_abs_residual_waves": float(np.max(np.abs(comp.residual))),
            "active_bounds": comp.active_bounds,
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
        "fea_terms": run.fea_terms,
        "registration": {
            "tolerance_px": float(config.register_tolerance_px),
            "max_shift_px": float(run.max_shift),
            "poses": run.registrations,
        },
        "stages": {k: float(v) for k, v in run.stages.items()},
        "warnings": run.elevation_warnings,
    }
    return rec
