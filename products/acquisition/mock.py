"""显式 mock 硬件：望远镜终端 / 天文相机 / 补偿机构。

**今天没有真实望远镜控制 API，也没有工况（operating condition）数据**。
本模块提供唯一可运行的实现，用于把端到端管线（采集 -> 预处理 -> PD 反演
-> 实测灵敏度 -> 俯仰角扫描补偿 -> FEA/装调请求）跑通、做回归测试和
GUI 演示。所有对象的 provenance 字段固定为::

    mode = "mock"
    label = "SYNTHETIC-VALIDATED"
    simulated = True

任何消费者都必须能一眼看出这些数据不是实测。真实 adapter 到位后，
*不要* 改这些 mock 的语义去冒充硬件，而是实现
:mod:`products.acquisition.interfaces` 的对应类。

输入:
    - MockTelescopeTerminal: seed、 slew 速率、初始指向；
    - MockCamera: ``state_provider``（当前 3 项 Zernike 波前状态，waves）+
      探测器参数；内部调用 :func:`products.ccd.service.run_ccd_pipeline`
      的 PSF stack，并叠加自己的 mock 探测器链（bias/dark/增益/读出噪声/
      饱和/宇宙线/坏像元）；
    - MockMechanism: DOF 子集、真值灵敏度矩阵（默认取
      products.control.service 的 3x5 M2 代理矩阵）、步长/回差/到位延迟。

输出:
    - 相机: raw ADU ``[K, N, N]``（星点窗口 ROI，含 bias/dark，未定标）；
    - 望远镜: 指向/状态/工况遥测 dict（JSON 安全）；
    - 机构: ``dofs()``/``readback()`` 位置（mm, mrad）与 ``wavefront_state()``。

边界:
    - 无任何网络/串口/供应商 SDK 调用；全部 numpy/scipy 合成，固定 seed
      下确定性（相机噪声由 seed 决定，机构运动学完全确定）。
    - MockCamera 只支持已验证的 3 项 PD 状态 ``Z4/Z8/Z9``；扩展项需要真实
      forward/adapter，mock 会明确报错而不是静默忽略。
    - MockMechanism 的 ``truth_S`` 是合成真值（"台架真值"），只用于闭环
      联调；真实机构效应必须像产品流程那样通过相机实测标定。
    - 相机返回的是 **星点窗口 ROI**（mock 模拟 WFS 恒星的 ROI 读出），
      不是整幅 192x192 探测器；真实相机 adapter 若要返回整幅，需要在
      metadata 中声明 ROI/裁剪约定。
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from products.acquisition.interfaces import (CameraApi, MechanismApi,
                                             TelescopeTerminalApi, UNIT_SYSTEM)

#: 所有 mock 对象的固定 provenance。
MOCK_LABEL = "SYNTHETIC-VALIDATED"
MOCK_MODE = "mock"

#: 相机 mock 今天只支持的 PD 状态项（与 products.optics.forward 的默认态一致）。
MOCK_CAMERA_TERMS: Tuple[str, ...] = ("Z4", "Z8", "Z9")

#: 机构 DOF 默认最小步长（mm / mrad）。
_DEFAULT_STEPS = {
    "m2_focus": 0.005, "m2_decenter_x": 0.005, "m2_decenter_y": 0.005,
    "m2_tilt_x": 0.02, "m2_tilt_y": 0.02,
}
#: 机构默认回差（mm / mrad）：小于步长，模拟真实丝杠/导轨的背隙。
_DEFAULT_HYSTERESIS = {
    "m2_focus": 0.002, "m2_decenter_x": 0.002, "m2_decenter_y": 0.002,
    "m2_tilt_x": 0.008, "m2_tilt_y": 0.008,
}

_FALLBACK_CONTROL_NAMES = ["m2_focus", "m2_decenter_x", "m2_decenter_y",
                           "m2_tilt_x", "m2_tilt_y"]
_FALLBACK_CONTROL_UNITS = ["mm", "mm", "mm", "mrad", "mrad"]
_FALLBACK_S = np.array([
    [4.040, 0.083, -0.083, 0.976, -0.976],
    [0.000, -4.633, 0.000, 0.000, 0.000],
    [0.000, 0.000, -4.633, 0.000, 0.000],
], dtype=float)


def _control_defaults() -> Tuple[List[str], List[str], np.ndarray, str]:
    """M2 灵敏度真值（P1-1，口径治理）：

    优先加载真实 Zemax S_Z（data/zemax/sensitivity_zemax.json，ZOS-API 实测，
    标签 EXTERNAL-CHECK）；文件缺失时才回退到代理矩阵 A，并返回
    source="synthetic-proxy" 供调用方标注 UNVERIFIED——代理矩阵三层错误
    （focus 符号/量级、tilt 通道虚构），不得再作为 mock 闭环的"正确真值"。
    """
    try:                                        # pragma: no cover - trivial
        from products.control.service import (CONTROL_NAMES, CONTROL_UNITS,
                                              load_real_sensitivity)
        real = load_real_sensitivity()
        if real is not None:
            return (list(CONTROL_NAMES), list(CONTROL_UNITS),
                    real["S"].copy(), "zemax-zos-api")
    except Exception:                           # pragma: no cover - trivial
        pass
    try:                                        # pragma: no cover - trivial
        from products.control.service import A, CONTROL_NAMES, CONTROL_UNITS
        return (list(CONTROL_NAMES), list(CONTROL_UNITS), np.asarray(A, float),
                "synthetic-proxy")
    except Exception:
        return (list(_FALLBACK_CONTROL_NAMES), list(_FALLBACK_CONTROL_UNITS),
                _FALLBACK_S.copy(), "synthetic-proxy")


# ===========================================================================
# 望远镜控制终端
# ===========================================================================

class MockTelescopeTerminal(TelescopeTerminalApi):
    """mock 望远镜终端：模拟俯仰角/方位角 slew 时间并返回合理遥测。

    俯仰角扫描对补偿量的影响通过 :meth:`condition_aberration` 合成：*没有*
    真实工况数据，所以这里用明确标注的、关于仰角的确定性代理（重力/失焦
    趋势），只用于让管线在 15°/45°/75° 上产生不同目标像差。
    """

    mode = MOCK_MODE
    label = MOCK_LABEL
    simulated = True

    def __init__(self, seed: int = 7, elevation_deg: float = 45.0,
                 azimuth_deg: float = 0.0,
                 slew_rate_deg_per_s: float = 5.0,
                 azimuth_rate_deg_per_s: float = 10.0,
                 max_sleep_s: float = 0.02,
                 elevation_limits: Tuple[float, float] = (0.0, 90.0),
                 label: str = MOCK_LABEL) -> None:
        self.seed = int(seed)
        self.label = str(label)
        lo, hi = float(elevation_limits[0]), float(elevation_limits[1])
        if not (lo <= elevation_deg <= hi):
            raise ValueError("elevation_deg outside mock elevation limits")
        self._elev_limits = (lo, hi)
        self._elev = float(elevation_deg)
        self._az = float(azimuth_deg) % 360.0
        if slew_rate_deg_per_s <= 0 or azimuth_rate_deg_per_s <= 0:
            raise ValueError("slew rates must be > 0")
        self.slew_rate_deg_per_s = float(slew_rate_deg_per_s)
        self.azimuth_rate_deg_per_s = float(azimuth_rate_deg_per_s)
        # 真实望远镜 slew 需要数十秒；mock 只阻塞一个很小的比例，但把
        # 物理上合理的时长完整记录在遥测里（避免测试/演示不必要地慢）。
        self.max_sleep_s = max(0.0, float(max_sleep_s))
        self._slew_log: List[Dict[str, Any]] = []
        self._settled = True
        self._last_slew_s = 0.0
        self._last_axis = None

    # -- TelescopeTerminalApi ------------------------------------------------
    def status(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "label": self.label, "simulated": True,
            "connected": True,
            "state": "tracking" if self._settled else "slewing",
            "settled": bool(self._settled),
            "elevation_deg": float(self._elev),
            "azimuth_deg": float(self._az),
            "last_slew_s": float(self._last_slew_s),
            "last_axis": self._last_axis,
            "limits": {"elevation_deg": list(self._elev_limits),
                       "azimuth_deg": [0.0, 360.0]},
            "servo": {"tracking_error_arcsec": 0.0,
                      "slew_rate_deg_per_s": self.slew_rate_deg_per_s},
            "operating_condition": self.operating_condition(),
            "units": dict(UNIT_SYSTEM),
            "note": ("mock 终端：slew 时长是物理估算，实际只阻塞 "
                     "max_sleep_s；没有真实望远镜控制 API"),
        }

    def goto_elevation(self, deg: float, timeout_s: Optional[float] = None
                       ) -> Dict[str, Any]:
        deg = float(deg)
        lo, hi = self._elev_limits
        if not (lo <= deg <= hi):
            raise ValueError(f"elevation {deg} outside mock limits {lo}..{hi}")
        return self._slew("elevation", deg,
                          abs(deg - self._elev) / self.slew_rate_deg_per_s,
                          timeout_s)

    def goto_azimuth(self, deg: float, timeout_s: Optional[float] = None
                     ) -> Dict[str, Any]:
        deg = float(deg) % 360.0
        delta = abs(deg - self._az)
        delta = min(delta, 360.0 - delta)          # 最短路径
        return self._slew("azimuth", deg,
                          delta / self.azimuth_rate_deg_per_s, timeout_s)

    def pointing(self) -> Dict[str, Any]:
        # 合成指向：没有真实星表/编码器解算，只用确定性代理把
        # elevation/azimuth 映射到 J2000 RA/Dec（固定"本地恒星时"）。
        ra = (180.0 - self._az) % 360.0
        dec = float(np.clip(self._elev - 45.0, -90.0, 90.0))
        return {
            "mode": self.mode, "label": self.label, "simulated": True,
            "elevation_deg": float(self._elev),
            "azimuth_deg": float(self._az),
            "ra_deg": float(ra), "dec_deg": float(dec), "epoch": "J2000",
            "settled": bool(self._settled),
            "timestamp": "mock-T+0.000s",
            "units": dict(UNIT_SYSTEM),
            "note": "mock 指向：RA/Dec 是确定性占位，不是天测解",
        }

    # -- 工况（工况数据不存在，这里只是显式 mock） --------------------------
    def operating_condition(self) -> Dict[str, Any]:
        """返回合成工况遥测，并显式声明 "无真实工况数据"。

        ``aberration_waves`` 是关于仰角的确定性代理：仰角越低，重力/失焦
        项越大。真实系统必须由终端/环境监测提供实测工况，不能使用本值。
        """
        e = float(self._elev)
        d = float(np.clip((90.0 - e) / 75.0, 0.0, 1.5))
        aberration = self.condition_aberration()
        return {
            "mode": self.mode, "label": self.label, "simulated": True,
            "source": "NO_REAL_OPERATING_CONDITION_DATA",
            "reason": ("今天没有工况（operating condition）数据接口；"
                       "本记录是 mock 合成占位"),
            "elevation_deg": e,
            "azimuth_deg": float(self._az),
            "ambient_temperature_c": round(12.0 - 2.0 * d, 3),
            "detector_temperature_c": -35.0,
            "wind_speed_m_s": round(2.0 + 1.5 * d, 3),
            "seeing_arcsec": round(0.7 + 0.3 * d, 3),
            "gravity_body_g": [0.0, 0.0, -1.0],
            "aberration_waves": aberration,
            "units": {"aberration_waves": "waves @632.8 nm wavefront",
                      "temperature_c": "degC", "wind_speed_m_s": "m/s"},
            "note": ("SYNTHETIC-VALIDATED 工况代理；仅用于让俯仰角扫描产生"
                     "不同的静态像差目标，不是实测/热模型/FEA 结果"),
        }

    def condition_aberration(self) -> Dict[str, float]:
        """仰角 -> 静态像差（waves，Z4/Z8/Z9）的确定性 mock 代理。"""
        e = float(self._elev)
        d = float(np.clip((90.0 - e) / 75.0, 0.0, 1.5))
        return {
            "Z4": 0.08 + 0.14 * d,
            "Z8": 0.04 + 0.05 * d,
            "Z9": 0.03 + 0.04 * d,
        }

    # -- internal ------------------------------------------------------------
    def _slew(self, axis: str, target: float, duration_s: float,
              timeout_s: Optional[float]) -> Dict[str, Any]:
        if timeout_s is not None and duration_s > float(timeout_s):
            raise TimeoutError(
                f"mock slew {axis}->{target:.3f} would take {duration_s:.3f}s "
                f"> timeout {float(timeout_s):.3f}s")
        sleep_s = min(max(0.0, float(duration_s)), self.max_sleep_s)
        self._settled = False
        if sleep_s > 0:
            time.sleep(sleep_s)
        if axis == "elevation":
            self._elev = float(target)
        else:
            self._az = float(target)
        self._settled = True
        self._last_slew_s = float(duration_s)
        self._last_axis = axis
        self._slew_log.append({"axis": axis, "target": float(target),
                               "simulated_duration_s": float(duration_s),
                               "blocked_s": float(sleep_s)})
        return self.status()


# ===========================================================================
# 天文相机
# ===========================================================================

class MockCamera(CameraApi):
    """mock 天文相机：PSF stack 来自 products.ccd，再走 mock 探测器链。

    数据来源与物理链条
    ------------------
    1. 当前波前状态 ``state_provider() -> {"Z4":..., "Z8":..., "Z9":...}``
       （waves）传给 :func:`products.ccd.service.run_ccd_pipeline`，得到
       **PSF stack** ``[K, n, n]``（已包含离焦、Fourier 光学与 seeing）；
    2. mock 探测器链把 PSF stack 变成 raw ADU：光子泊松噪声 + 读出噪声 +
       pixel response（flat）+ 增益 + 读出 bias + 位深量化 + 宇宙线/坏像元；
    3. 返回 raw ADU ``[K, n, n]``（bias/dark 已包含，未定标）。

    这里刻意模拟 WFS 恒星的 **ROI 读出**（star window），所以帧尺寸就是
    PSF 窗口（默认 64x64），而不是整幅探测器；真实 adapter 若返回整幅，
    必须在 ``metadata()["roi"]`` 中声明裁剪约定。

    限制：只支持 3 项已验证状态 Z4/Z8/Z9；其它 term 需要真实相机/前向模型。
    """

    mode = MOCK_MODE
    label = MOCK_LABEL
    simulated = True

    def __init__(self, state_provider: Optional[Callable[[], Any]] = None,
                 seed: int = 7,
                 terms: Sequence[str] = MOCK_CAMERA_TERMS,
                 star_flux_e_per_s: float = 3.0e5,
                 sky_background_e_per_pix_s: float = 50.0,
                 dark_current_e_per_s: float = 0.01,
                 read_noise_e: float = 4.0,
                 adu_gain_e_per_adu: float = 1.0,
                 readout_bias_adu: float = 500.0,
                 bit_depth: int = 16,
                 saturation_adu: int = 65535,
                 cosmic_ray_rate: float = 1e-4,
                 bad_pixel_fraction: float = 0.0,
                 flat_field_variation: float = 0.05,
                 psf_window: int = 64,
                 seeing_arcsec: float = 0.05,
                 obscuration: float = 0.0,
                 cr_exclusion_radius: float = 12.0,
                 label: str = MOCK_LABEL) -> None:
        self.terms = tuple(str(t) for t in terms)
        if not set(self.terms) <= set(MOCK_CAMERA_TERMS):
            raise ValueError(
                "MockCamera only supports MOCK_CAMERA_TERMS="
                f"{MOCK_CAMERA_TERMS}, got {self.terms}; an extended state "
                "needs a real camera/forward-model adapter")
        self.state_provider = state_provider
        self.seed = int(seed)
        self.label = str(label)
        self.star_flux_e_per_s = float(star_flux_e_per_s)
        self.sky_background_e_per_pix_s = float(sky_background_e_per_pix_s)
        self.dark_current_e_per_s = float(dark_current_e_per_s)
        self.read_noise_e = float(read_noise_e)
        self.adu_gain_e_per_adu = float(adu_gain_e_per_adu)
        self.readout_bias_adu = float(readout_bias_adu)
        self.bit_depth = int(bit_depth)
        self.saturation_adu = int(saturation_adu)
        self.adu_max = int(min(int(saturation_adu), 2 ** int(bit_depth) - 1))
        self.cosmic_ray_rate = max(0.0, float(cosmic_ray_rate))
        self.bad_pixel_fraction = float(bad_pixel_fraction)
        self.psf_window = int(psf_window)
        self.seeing_arcsec = float(seeing_arcsec)
        self.obscuration = float(obscuration)
        self.cr_exclusion_radius = float(cr_exclusion_radius)
        if self.adu_gain_e_per_adu <= 0:
            raise ValueError("adu_gain_e_per_adu must be > 0")
        if self.psf_window < 16:
            raise ValueError("psf_window must be >= 16")

        self._rng = np.random.default_rng(self.seed)
        self._capture_index = 0
        self._last_capture: Dict[str, Any] = {}
        self._flat_field = self._make_flat_field(float(flat_field_variation))
        self._defect_mask = self._make_defect_mask(self.bad_pixel_fraction)
        self._dead_mask = self._defect_mask & (self._rng.random(
            self._defect_mask.shape) < 0.5) if self._defect_mask.any() \
            else np.zeros_like(self._defect_mask)
        self._hot_mask = self._defect_mask & ~self._dead_mask
        self._meta_cache: Optional[Dict[str, Any]] = None

    # -- CameraApi -----------------------------------------------------------
    def capture(self, defocus_list: Sequence[float], exposure_s: float
                ) -> np.ndarray:
        defocus = [float(d) for d in defocus_list]
        if not defocus:
            raise ValueError("defocus_list must contain at least one value")
        exposure_s = float(exposure_s)
        if not math.isfinite(exposure_s) or exposure_s < 0:
            raise ValueError("exposure_s must be finite and >= 0")
        state = self._resolve_state()
        c4 = float(state["Z4"])
        c8 = float(state["Z8"])
        c9 = float(state["Z9"])

        # PSF stack 来自 products.ccd 的已验证前向模型（不使用其 raw 帧）。
        # 注意：不能强制 fast=True —— 高斯近似会丢掉 Zernike 信息。
        from products.ccd.service import run_ccd_pipeline
        res = run_ccd_pipeline(
            c4, c8, c9,
            defocus=defocus,
            seed=self.seed,
            exposure_s=exposure_s,
            return_frames=False,
            seeing_arcsec=self.seeing_arcsec,
            obscuration=self.obscuration,
            jitter_arcsec=0.0,
            read_noise_e=0.0,
            cosmic_ray_rate=0.0,
            bad_pixel_fraction=0.0,
        )
        psf = np.asarray(res["psf"], dtype=float)
        if psf.ndim != 3 or psf.shape[0] != len(defocus):
            raise RuntimeError("ccd PSF stack shape mismatch")
        n = int(psf.shape[1])

        # --- mock 探测器链 ---
        star_e = psf * (self.star_flux_e_per_s * exposure_s)
        bg_e = (self.sky_background_e_per_pix_s
                + self.dark_current_e_per_s) * exposure_s
        lam = np.clip(star_e + bg_e, 0.0, None)
        electrons = self._rng.poisson(lam).astype(float)
        if self.read_noise_e > 0:
            electrons += self._rng.normal(0.0, self.read_noise_e,
                                          electrons.shape)
        adu = (electrons * self._flat_field / self.adu_gain_e_per_adu
               + self.readout_bias_adu)
        adu = np.rint(np.clip(adu, 0.0, float(self.adu_max)))

        # 永久坏像元（dead/hot）；默认 0，开启时通过 metadata 的 defect_map
        # 交给预处理修复。
        if self._defect_mask.any():
            adu[:, self._dead_mask] = 0.0
            adu[:, self._hot_mask] = float(self.adu_max)
        # 单像素宇宙线：注入在 WFS 星点 ROI 之外，避免合成 PSF 被破坏。
        cr_count = self._inject_cosmic_rays(adu)

        self._capture_index += 1
        self._last_capture = {
            "defocus": defocus,
            "exposure_s": exposure_s,
            "shape": [int(n), int(n)],
            "n_frames": len(defocus),
            "state_waves": {k: float(v) for k, v in state.items()},
            "cosmic_ray_count": int(cr_count),
            "capture_index": int(self._capture_index),
            "units": dict(UNIT_SYSTEM),
        }
        return np.asarray(adu, dtype=float)

    def metadata(self) -> Dict[str, Any]:
        if self._meta_cache is None:
            self._meta_cache = {
                "mode": self.mode, "label": self.label, "simulated": True,
                "source": ("products.ccd.service.run_ccd_pipeline PSF stack + "
                           "mock detector chain"),
                "raw_units": "ADU (integer-valued float; bias/dark included, "
                             "no flat/bad-pixel/CR calibration applied)",
                "shape": [self.psf_window, self.psf_window],
                "terms": list(self.terms),
                "supported_terms": list(MOCK_CAMERA_TERMS),
                "bit_depth": int(self.bit_depth),
                "adu_max": int(self.adu_max),
                "saturation_adu": int(self.saturation_adu),
                "adu_gain_e_per_adu": float(self.adu_gain_e_per_adu),
                "read_noise_e": float(self.read_noise_e),
                "readout_bias_adu": float(self.readout_bias_adu),
                "dark_current_e_per_s": float(self.dark_current_e_per_s),
                "sky_background_e_per_pix_s": float(
                    self.sky_background_e_per_pix_s),
                "star_flux_e_per_s": float(self.star_flux_e_per_s),
                "cosmic_ray_rate": float(self.cosmic_ray_rate),
                "seeing_arcsec": float(self.seeing_arcsec),
                "obscuration": float(self.obscuration),
                "roi": ("star-window ROI centred on the WFS star "
                        "(mock emulates ROI readout)"),
                "flat_field": self._flat_field.tolist(),
                "flat_field_units": "relative pixel response, median = 1.0",
                "defect_map": self._defect_mask.astype(int).tolist(),
                "defect_count": int(self._defect_mask.sum()),
                "last_capture": dict(self._last_capture),
                "units": dict(UNIT_SYSTEM),
                "note": ("SYNTHETIC-VALIDATED mock: no vendor camera SDK; "
                         "frames are on-sky-like raw ADU but not real data"),
            }
        return self._meta_cache

    # -- helpers -------------------------------------------------------------
    def _resolve_state(self) -> Dict[str, float]:
        raw: Any = None
        if self.state_provider is not None:
            raw = self.state_provider()
        if raw is None:
            state = {t: 0.0 for t in MOCK_CAMERA_TERMS}
        elif isinstance(raw, Mapping):
            state = {t: float(raw.get(t, 0.0)) for t in MOCK_CAMERA_TERMS}
        else:
            vals = list(raw)
            if len(vals) != len(MOCK_CAMERA_TERMS):
                raise ValueError(
                    "state_provider sequence must be ordered as "
                    f"{MOCK_CAMERA_TERMS}")
            state = {t: float(v) for t, v in zip(MOCK_CAMERA_TERMS, vals)}
        if not all(math.isfinite(v) for v in state.values()):
            raise ValueError("state_provider returned non-finite Zernike state")
        return state

    def _make_flat_field(self, variation: float) -> np.ndarray:
        n = int(self.psf_window)
        yy, xx = np.mgrid[0:n, 0:n].astype(float)
        r2 = ((xx - (n - 1) / 2.0) / ((n - 1) / 2.0)) ** 2 + \
             ((yy - (n - 1) / 2.0) / ((n - 1) / 2.0)) ** 2
        # 平滑的 pixel-response 模式（真实 flat 的简化代理），中位数归一。
        flat = 1.0 - max(0.0, variation) * 0.5 * np.clip(r2, 0.0, 4.0)
        flat = np.clip(flat, 0.5, 1.5)
        med = float(np.median(flat))
        return flat / med if med > 0 else np.ones_like(flat)

    def _make_defect_mask(self, fraction: float) -> np.ndarray:
        n_pix = int(self.psf_window) ** 2
        n_bad = int(round(max(0.0, min(1.0, fraction)) * n_pix))
        mask = np.zeros(n_pix, dtype=bool)
        if n_bad > 0:
            idx = self._rng.choice(n_pix, size=min(n_bad, n_pix), replace=False)
            mask[idx] = True
        return mask.reshape(self.psf_window, self.psf_window)

    def _inject_cosmic_rays(self, adu: np.ndarray) -> int:
        """在 ROI 边缘区域注入单像素宇宙线（饱和 ADU 上限），返回命中数。"""
        if self.cosmic_ray_rate <= 0 or adu.size == 0:
            return 0
        n = int(adu.shape[1])
        n_hits = int(self._rng.poisson(self.cosmic_ray_rate * (n * n)
                                       * adu.shape[0]))
        if n_hits <= 0:
            return 0
        yy, xx = np.mgrid[0:n, 0:n]
        cy = cx = (n - 1) / 2.0
        allowed = np.flatnonzero(
            (yy - cy) ** 2 + (xx - cx) ** 2
            > self.cr_exclusion_radius ** 2)
        if allowed.size == 0:
            return 0
        for k in range(adu.shape[0]):
            picks = self._rng.choice(allowed, size=n_hits, replace=True)
            adu[k].ravel()[picks] = float(self.adu_max)
        return int(n_hits * adu.shape[0])


# ===========================================================================
# 补偿机构
# ===========================================================================

class MockMechanism(MechanismApi):
    """mock 补偿机构：量化步长 + 背隙/回差 + 到位延迟。

    ``truth_S`` 是该 mock 台架的合成真值灵敏度矩阵（waves / DOF 单位），
    只用于闭环联调时把机构位置映射为相机看到的波前状态；真实装置的效应
    必须通过相机实测标定，不能假设该矩阵。
    """

    mode = MOCK_MODE
    label = MOCK_LABEL
    simulated = True

    def __init__(self, n_dofs: int = 2,
                 names: Optional[Sequence[str]] = None,
                 units: Optional[Sequence[str]] = None,
                 truth_S: Optional[Sequence[Sequence[float]]] = None,
                 base_state: Optional[Mapping[str, float]] = None,
                 bounds: Tuple[float, float] = (-0.5, 0.5),
                 steps: Optional[Mapping[str, float]] = None,
                 hysteresis: Optional[Mapping[str, float]] = None,
                 slew_latency_s: float = 0.001,
                 label: str = MOCK_LABEL) -> None:
        ctrl_names, ctrl_units, default_S, s_source = _control_defaults()
        # P1-1 provenance: the truth matrix is the real Zemax S_Z when the
        # JSON is present; a caller-supplied truth_S or the proxy fallback is
        # recorded in `truth_source` so mock results are never mistaken for
        # real-S evidence.
        self.truth_source = ("caller-supplied" if truth_S is not None
                             else s_source)
        n = len(names) if names is not None else int(n_dofs)
        if n < 1:
            raise ValueError("MockMechanism needs at least one DOF")
        if n > len(ctrl_names):
            raise ValueError(f"at most {len(ctrl_names)} M2 DOFs are modelled")
        self.names = list(names) if names is not None else ctrl_names[:n]
        self.units = (list(units) if units is not None else ctrl_units[:n])
        if len(self.units) != len(self.names):
            raise ValueError("units length must match names")
        flags = np.zeros((3, len(self.names)), dtype=bool)
        for j, name in enumerate(self.names):
            if name in ctrl_names:
                flags[:, j] = True
        dense = default_S if truth_S is None else np.asarray(truth_S, float)
        dense = np.asarray(dense, dtype=float)
        if dense.ndim != 2 or dense.shape[0] != 3:
            raise ValueError("truth_S must be a 3 x n_dof matrix (Z4/Z8/Z9)")
        if dense.shape[1] == len(self.names):
            self._truth_S = dense.copy()
        elif dense.shape[1] == len(ctrl_names):
            keep = [ctrl_names.index(nm) for nm in self.names
                    if nm in ctrl_names]
            if len(keep) != len(self.names):
                raise ValueError("cannot slice truth_S to requested DOFs")
            self._truth_S = dense[:, keep].copy()
        else:
            raise ValueError("truth_S column count does not match DOFs")
        self.base_state = {t: float((base_state or {}).get(t, 0.0))
                           for t in MOCK_CAMERA_TERMS}
        self.low, self.high = float(bounds[0]), float(bounds[1])
        if not self.low < self.high:
            raise ValueError("bounds must satisfy low < high")
        default_steps = _DEFAULT_STEPS
        default_hyst = _DEFAULT_HYSTERESIS
        steps = dict(steps or {})
        hysteresis = dict(hysteresis or {})
        self._spec: List[Dict[str, Any]] = []
        for name, unit in zip(self.names, self.units):
            step = float(steps.get(name, default_steps.get(name, 0.005)))
            hyst = float(hysteresis.get(name, default_hyst.get(name, 0.0)))
            if step <= 0:
                raise ValueError(f"{name}: step must be > 0")
            self._spec.append({
                "name": name, "unit": unit,
                "low": self.low, "high": self.high, "step": step,
                "home": 0.0, "hysteresis": hyst,
            })
        self.slew_latency_s = max(0.0, float(slew_latency_s))
        self._pos = {name: 0.0 for name in self.names}
        self._last_q = {name: 0.0 for name in self.names}
        self._last_dir = {name: 0 for name in self.names}
        self._n_moves = 0
        self._total_slew_s = 0.0

    # -- MechanismApi --------------------------------------------------------
    def dofs(self) -> List[Dict[str, Any]]:
        return [dict(spec) for spec in self._spec]

    def move(self, name: str, value: float) -> None:
        if name not in self._pos:
            raise KeyError(f"unknown DOF {name!r}; known: {self.names}")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError("move value must be finite")
        if value < self.low - 1e-12 or value > self.high + 1e-12:
            unit = self._spec[self.names.index(name)]["unit"]
            raise ValueError(f"{name}={value} outside [{self.low}, "
                             f"{self.high}] {unit}")
        target = min(max(value, self.low), self.high)
        spec = self._spec[self.names.index(name)]
        step = float(spec["step"])
        # 1) 指令量化到机构最小步长
        q = round(target / step) * step
        q = min(max(q, self.low), self.high)
        prev_q = self._last_q[name]
        prev_dir = self._last_dir[name]
        if q > prev_q + 1e-15:
            direction = 1
        elif q < prev_q - 1e-15:
            direction = -1
        else:
            direction = 0
        # 2) 换向时先吃掉背隙：实测值滞后一个 hysteresis 在旧方向
        if direction == 0:
            actual = q
            used_dir = prev_dir
        else:
            lag = (prev_dir * float(spec["hysteresis"])
                   if prev_dir != 0 and direction != prev_dir else 0.0)
            actual = q + lag
            used_dir = direction
        actual = min(max(actual, self.low), self.high)
        # 3) 到位延迟（阻塞）
        if self.slew_latency_s > 0:
            time.sleep(self.slew_latency_s)
        self._pos[name] = float(actual)
        self._last_q[name] = float(q)
        self._last_dir[name] = int(used_dir)
        self._n_moves += 1
        self._total_slew_s += self.slew_latency_s

    def home(self) -> None:
        for name in self.names:
            self._pos[name] = 0.0
            self._last_q[name] = 0.0
            self._last_dir[name] = 0
        if self.slew_latency_s > 0:
            time.sleep(self.slew_latency_s)

    def readback(self) -> Dict[str, float]:
        return {name: float(self._pos[name]) for name in self.names}

    # -- 闭环联调辅助（真实机构没有这些方法） --------------------------------
    def positions_vector(self) -> np.ndarray:
        return np.asarray([self._pos[n] for n in self.names], dtype=float)

    def wavefront_state(self) -> Dict[str, float]:
        """合成真值状态：base + truth_S @ positions（waves, Z4/Z8/Z9）。"""
        z = self._truth_S @ self.positions_vector()
        return {t: float(self.base_state[t] + z[i])
                for i, t in enumerate(MOCK_CAMERA_TERMS)}

    def move_to_vector(self, values: Sequence[float]) -> None:
        vals = np.asarray(values, dtype=float).reshape(-1)
        if vals.size != len(self.names):
            raise ValueError("vector length must match DOF count")
        for name, value in zip(self.names, vals):
            self.move(name, float(value))

    def kinematics(self) -> Dict[str, Any]:
        """mock 运动学摘要（真实系统必须从编码器/标定得到）。"""
        return {
            "mode": self.mode, "label": self.label, "simulated": True,
            "dofs": self.dofs(),
            "bounds": [self.low, self.high],
            "slew_latency_s": self.slew_latency_s,
            "n_moves": int(self._n_moves),
            "truth_S": self._truth_S.tolist(),
            "truth_S_terms": list(MOCK_CAMERA_TERMS),
            "note": "truth_S 是 mock 台架真值，不是实测灵敏度矩阵",
        }


# ===========================================================================
# 一键组装
# ===========================================================================

@dataclass
class MockRig:
    """把三个 mock 连成一个可闭环的合成台架（真值状态 -> 相机）。"""

    telescope: MockTelescopeTerminal
    camera: MockCamera
    mechanism: MockMechanism
    terms: Tuple[str, ...] = MOCK_CAMERA_TERMS
    label: str = MOCK_LABEL
    mode: str = MOCK_MODE

    def state(self) -> Dict[str, float]:
        """当前合成波前状态 = 仰角工况代理 + 机构真值耦合（waves）。"""
        z = dict(self.telescope.condition_aberration())
        for term, value in self.mechanism.wavefront_state().items():
            z[term] = z.get(term, 0.0) + value
        return z

    def summary(self) -> Dict[str, Any]:
        return {
            "mode": self.mode, "label": self.label,
            "telescope": type(self.telescope).__name__,
            "camera": type(self.camera).__name__,
            "mechanism": type(self.mechanism).__name__,
            "terms": list(self.terms),
            "operating_condition": self.telescope.operating_condition(),
            "mechanism_kinematics": self.mechanism.kinematics(),
            "note": ("SYNTHETIC-VALIDATED 合成台架：无真实望远镜/相机/机构"
                     "SDK，也无真实工况数据"),
        }


def build_mock_rig(n_dofs: int = 2, seed: int = 7,
                   start_elevation_deg: float = 45.0,
                   camera_kwargs: Optional[Mapping[str, Any]] = None,
                   mechanism_kwargs: Optional[Mapping[str, Any]] = None,
                   telescope_kwargs: Optional[Mapping[str, Any]] = None
                   ) -> MockRig:
    """组装默认 mock 台架（相机状态 = 仰角工况 + 机构真值耦合）。"""
    mech_kw = dict(mechanism_kwargs or {})
    mech = MockMechanism(n_dofs=n_dofs, label=MOCK_LABEL, **mech_kw)
    tel_kw = dict(telescope_kwargs or {})
    tel_kw.setdefault("elevation_deg", start_elevation_deg)
    tel = MockTelescopeTerminal(seed=int(seed) + 1, label=MOCK_LABEL, **tel_kw)

    def _provider() -> Dict[str, float]:
        z = dict(tel.condition_aberration())
        for term, value in mech.wavefront_state().items():
            z[term] = z.get(term, 0.0) + value
        return z

    cam_kw = dict(camera_kwargs or {})
    cam = MockCamera(state_provider=_provider, seed=int(seed) + 2,
                     label=MOCK_LABEL, **cam_kw)
    return MockRig(telescope=tel, camera=cam, mechanism=mech)
