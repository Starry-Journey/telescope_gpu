"""CCD 数据管线预处理：定标帧、坏像元、背景、饱和、宇宙线、星点、配准。

本模块是"天文相机终端 -> 相位差反演"之间的全部图像预处理，全部用
**向量化 numpy/scipy** 实现（不对图像做逐像素 Python 循环；只有帧数 K
这一维允许小循环）。所有函数只依赖 numpy/scipy，不依赖 GUI/Qt/网络。

输入:
    - raw ADU 帧 ``[K, N, N]``（:class:`products.acquisition.interfaces.CameraApi`
      的约定：含 bias/dark，未定标）；
    - 定标帧栈（bias/dark/flat）与曝光时间；
    - 单帧图像（背景估计、宇宙线、饱和、星点检测、配准）。
输出:
    - MasterFrames（bias/dark/flat + 曝光元数据）；
    - 定标后的 ADU 帧；
    - 坏像元掩膜、背景值/噪声、星点列表、亚像素质心、帧间平移。

边界（boundaries）:
    - 没有真实相机 SDK / 实验室定标数据；``calib_dir`` 不存在时由
      :mod:`products.pipeline.orchestrator` 用前几帧+bias 曝光合成 master，
      并在结果里记录 ``calibration="synthetic"``。真实定标必须替换成实测
      master frames，不能把合成 flat 当作实测。
    - 本模块不判断科学可用性：它只做数学操作；数据 provenance 由调用方
      的 ``mode``/``label`` 负责。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import ndimage as ndi

#: 默认主帧文件（calib_dir 下）。真实定标目录也使用同一格式/键名。
MASTER_FILE = "master_frames.npz"

#: 所有输出 ADU 的统一单位说明。
ADU_UNIT = "ADU (bias/dark included before calibration; flat-normalised after)"


# ===========================================================================
# Master frames
# ===========================================================================

@dataclass
class MasterFrames:
    """bias/dark/flat 主帧 + 曝光元数据。

    Parameters
    ----------
    bias:
        主 bias（ADU，``bias_exposure_s`` 下的读出偏置；与曝光无关）。
    dark:
        **已扣 bias 的**暗电流信号（ADU，``dark_exposure_s`` 下）；校准时
        按 ``exposure_s / dark_exposure_s`` 线性缩放。
    flat:
        主平场（相对像素响应，中位数归一为 1；校准时直接相除）。
    bias_exposure_s / dark_exposure_s / flat_exposure_s:
        各主帧的采集曝光（秒），用于缩放与溯源。
    n_bias / n_dark / n_flat:
        每个主帧合成的原始帧数。
    label:
        provenance；今天默认 ``"SYNTHETIC-VALIDATED"``。
    """

    bias: Optional[np.ndarray] = None
    dark: Optional[np.ndarray] = None
    flat: Optional[np.ndarray] = None
    bias_exposure_s: float = 0.0
    dark_exposure_s: float = 1.0
    flat_exposure_s: float = 1.0
    n_bias: int = 0
    n_dark: int = 0
    n_flat: int = 0
    label: str = "SYNTHETIC-VALIDATED"
    units: Dict[str, str] = field(default_factory=lambda: {
        "bias": ADU_UNIT, "dark": "ADU (bias-subtracted dark current signal)",
        "flat": "relative pixel response (median normalised to 1)",
        "exposure": "s"})

    def __post_init__(self) -> None:
        for name in ("bias", "dark", "flat"):
            value = getattr(self, name)
            if value is not None:
                arr = np.asarray(value, dtype=float)
                if arr.ndim != 2:
                    raise ValueError(f"master {name} must be 2-D, got {arr.shape}")
                if not np.all(np.isfinite(arr)):
                    raise ValueError(f"master {name} contains NaN/Inf")
                setattr(self, name, arr)
        shapes = {arr.shape for arr in (self.bias, self.dark, self.flat)
                  if arr is not None}
        if len(shapes) > 1:
            raise ValueError(f"master frames must share one shape, got {shapes}")

    @property
    def shape(self) -> Optional[Tuple[int, int]]:
        for arr in (self.bias, self.dark, self.flat):
            if arr is not None:
                return (int(arr.shape[0]), int(arr.shape[1]))
        return None

    def to_dict(self, include_arrays: bool = False) -> Dict[str, Any]:
        """JSON 安全的元数据字典（默认不含图像数组）。"""
        out: Dict[str, Any] = {
            "shape": list(self.shape) if self.shape else None,
            "bias_exposure_s": float(self.bias_exposure_s),
            "dark_exposure_s": float(self.dark_exposure_s),
            "flat_exposure_s": float(self.flat_exposure_s),
            "n_bias": int(self.n_bias), "n_dark": int(self.n_dark),
            "n_flat": int(self.n_flat), "label": str(self.label),
            "units": dict(self.units),
        }
        if include_arrays:
            for name in ("bias", "dark", "flat"):
                arr = getattr(self, name)
                out[name] = None if arr is None else arr.tolist()
        return out

    def save(self, directory: Union[str, Path]) -> Path:
        """把主帧写成 ``<directory>/master_frames.npz`` 并返回路径。"""
        out = Path(directory)
        out.mkdir(parents=True, exist_ok=True)
        path = out / MASTER_FILE
        empty = np.zeros((0, 0), dtype=float)

        def _pack(arr: Optional[np.ndarray]) -> np.ndarray:
            return empty if arr is None else np.asarray(arr, dtype=float)

        np.savez(
            path,
            bias=_pack(self.bias), dark=_pack(self.dark),
            flat=_pack(self.flat),
            bias_exposure_s=float(self.bias_exposure_s),
            dark_exposure_s=float(self.dark_exposure_s),
            flat_exposure_s=float(self.flat_exposure_s),
            n_bias=int(self.n_bias), n_dark=int(self.n_dark),
            n_flat=int(self.n_flat), label=str(self.label),
            has_bias=int(self.bias is not None),
            has_dark=int(self.dark is not None),
            has_flat=int(self.flat is not None),
        )
        return path

    @classmethod
    def load(cls, directory_or_file: Union[str, Path]) -> "MasterFrames":
        """从 ``calib_dir``（或直接给 .npz）加载主帧。"""
        p = Path(directory_or_file)
        path = p / MASTER_FILE if p.is_dir() else p
        if not path.exists():
            raise FileNotFoundError(
                f"no master frames at {path}; expected {MASTER_FILE} in calib_dir")
        with np.load(path, allow_pickle=False) as doc:
            def _unpack(key: str, flag: str) -> Optional[np.ndarray]:
                if key not in doc or not int(doc[flag]):
                    return None
                arr = np.asarray(doc[key], dtype=float)
                return None if arr.size == 0 else arr

            return cls(
                bias=_unpack("bias", "has_bias"),
                dark=_unpack("dark", "has_dark"),
                flat=_unpack("flat", "has_flat"),
                bias_exposure_s=float(doc["bias_exposure_s"]),
                dark_exposure_s=float(doc["dark_exposure_s"]),
                flat_exposure_s=float(doc["flat_exposure_s"]),
                n_bias=int(doc["n_bias"]), n_dark=int(doc["n_dark"]),
                n_flat=int(doc["n_flat"]),
                label=str(doc["label"]) if "label" in doc else "SYNTHETIC-VALIDATED",
            )


# ===========================================================================
# 主帧构造
# ===========================================================================

def _stack(frames: Sequence[np.ndarray], name: str) -> np.ndarray:
    arr = np.asarray(frames, dtype=float)
    if arr.ndim != 3 or arr.shape[0] == 0:
        raise ValueError(f"{name} frames must be a non-empty [K, N, N] stack")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} frames contain NaN/Inf")
    return arr


def build_master_bias(frames: Sequence[np.ndarray]) -> np.ndarray:
    """由 bias 帧栈（``[K,N,N]``，曝光 0）逐像素中值得到主 bias。"""
    return np.median(_stack(frames, "bias"), axis=0)


def build_master_dark(frames: Sequence[np.ndarray], exposure_s: float,
                      bias: Optional[np.ndarray] = None) -> np.ndarray:
    """由 dark 帧栈得到主暗电流信号（ADU，已扣 bias 时则不含 bias）。

    Parameters
    ----------
    frames:
        dark 帧栈 ``[K,N,N]``（与 science 同曝光）。
    exposure_s:
        dark 帧的曝光时间（秒，``> 0``）。
    bias:
        可选主 bias；给定则从每帧扣除，返回纯暗电流信号。
    """
    exposure_s = float(exposure_s)
    if not np.isfinite(exposure_s) or exposure_s <= 0:
        raise ValueError("dark exposure_s must be finite and > 0")
    master = np.median(_stack(frames, "dark"), axis=0)
    if bias is not None:
        b = np.asarray(bias, dtype=float)
        if b.shape != master.shape:
            raise ValueError("bias shape must match dark frames")
        master = master - b
    return master


def build_master_flat(frames: Sequence[np.ndarray]) -> np.ndarray:
    """由平场帧栈得到主平场（相对像素响应，中位数归一为 1）。"""
    master = np.median(_stack(frames, "flat"), axis=0)
    if np.any(master <= 0):
        raise ValueError("flat frames must be positive everywhere")
    med = float(np.median(master))
    if med <= 0:
        raise ValueError("flat median must be > 0")
    return master / med


def bad_pixel_map(flat: np.ndarray, sigma: float = 5.0) -> np.ndarray:
    """由主平场标记坏像元（局部中值残差的鲁棒 sigma 阈值）。

    规则：像素响应非有限/非正，或与 5x5 局部中值的偏差超过
    ``sigma * 1.4826 * MAD``（MAD 为残差的中位绝对偏差；稀疏缺陷时用
    全局标准差兜底）。返回 ``[N,N]`` 布尔掩膜，``True = 坏像元``。
    """
    sig = float(sigma)
    if not np.isfinite(sig) or sig <= 0:
        raise ValueError("sigma must be finite and > 0")
    f = np.asarray(flat, dtype=float)
    if f.ndim != 2:
        raise ValueError("flat must be 2-D")
    finite = np.isfinite(f)
    resid = np.zeros_like(f)
    safe = np.where(finite, f, 1.0)
    resid = safe - ndi.median_filter(safe, size=5, mode="nearest")
    mask = ~finite | (f <= 0.0)
    vals = resid[finite & (f > 0.0)]
    if vals.size:
        med = float(np.median(vals))
        mad = 1.4826 * float(np.median(np.abs(vals - med)))
        scale = mad if mad > 1e-12 else float(np.std(vals))
        if scale > 1e-12:
            mask = mask | (np.abs(resid - med) > sig * scale)
    return mask


def repair_bad_pixels(frames: np.ndarray, bad_mask: np.ndarray) -> np.ndarray:
    """用 3x3 邻域中值替换坏像元（向量化，逐帧一次 ``median_filter``）。

    ``frames`` 可以是 ``[N,N]`` 或 ``[K,N,N]``；``bad_mask`` 是 ``[N,N]``。
    返回同形状的新数组（不修改输入）。
    """
    arr = np.asarray(frames, dtype=float)
    squeeze = arr.ndim == 2
    if squeeze:
        arr = arr[None, ...]
    if arr.ndim != 3:
        raise ValueError("frames must be [N,N] or [K,N,N]")
    mask = np.asarray(bad_mask, dtype=bool)
    if mask.shape != arr.shape[1:]:
        raise ValueError("bad_mask shape must match a frame")
    if not mask.any():
        return arr[0].copy() if squeeze else arr.copy()
    med = ndi.median_filter(arr, size=(1, 3, 3), mode="nearest")
    out = np.where(mask[None, :, :], med, arr)
    return out[0] if squeeze else out


# ===========================================================================
# 定标
# ===========================================================================

def calibrate(frames: np.ndarray, master: MasterFrames, exposure_s: float
              ) -> np.ndarray:
    """bias + dark（按曝光缩放）+ flat（归一）定标。

    ``calibrated = (raw - bias - dark * (exposure/master.dark_exposure)) / flat``

    Parameters
    ----------
    frames:
        raw ADU ``[K,N,N]`` 或 ``[N,N]``。
    master:
        :class:`MasterFrames`；bias/dark/flat 均可为 None（视为 0/0/1）。
    exposure_s:
        science 帧曝光（秒），用于暗电流缩放。

    Returns
    -------
    numpy.ndarray
        定标后 ADU（浮点，同输入形状；背景未扣除）。
    """
    raw = np.asarray(frames, dtype=float)
    squeeze = raw.ndim == 2
    if squeeze:
        raw = raw[None, ...]
    if raw.ndim != 3:
        raise ValueError("frames must be [N,N] or [K,N,N]")
    shape = master.shape
    if shape is not None and raw.shape[1:] != shape:
        raise ValueError(f"master shape {shape} does not match frames "
                         f"{raw.shape[1:]}")
    exp = float(exposure_s)
    if not np.isfinite(exp) or exp < 0:
        raise ValueError("exposure_s must be finite and >= 0")

    out = raw
    if master.bias is not None:
        out = out - master.bias[None, :, :]
    if master.dark is not None:
        dark_exp = float(master.dark_exposure_s)
        if dark_exp <= 0:
            raise ValueError("master.dark_exposure_s must be > 0")
        out = out - master.dark[None, :, :] * (exp / dark_exp)
    if master.flat is not None:
        flat = np.asarray(master.flat, dtype=float)
        med = float(np.median(flat))
        flat_norm = flat / med if med > 0 else flat
        flat_norm = np.where(np.abs(flat_norm) < 1e-6, 1.0, flat_norm)
        out = out / flat_norm[None, :, :]
    return out[0] if squeeze else out


# ===========================================================================
# 单帧处理
# ===========================================================================

def saturation_mask(frame: np.ndarray, adu_max: float) -> np.ndarray:
    """返回饱和/无效像元掩膜（``frame >= adu_max`` 或非有限）。"""
    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError("frame must be 2-D")
    lim = float(adu_max)
    if not np.isfinite(lim):
        raise ValueError("adu_max must be finite")
    return ~np.isfinite(arr) | (arr >= lim)


def estimate_background(frame: np.ndarray, sigma_clip: float = 3.0
                        ) -> Tuple[float, float]:
    """迭代 sigma-clip 估计背景值与噪声（星点/宇宙线会被裁掉）。

    Returns
    -------
    (value, noise):
        背景中值（ADU）与鲁棒噪声（``1.4826*MAD``，ADU）。
    """
    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError("frame must be 2-D")
    data = arr[np.isfinite(arr)]
    if data.size == 0:
        raise ValueError("frame has no finite pixels")
    sig = float(sigma_clip)
    if sig <= 0:
        raise ValueError("sigma_clip must be > 0")
    med = float(np.median(data))
    scale = 1.4826 * float(np.median(np.abs(data - med)))
    for _ in range(10):
        if scale <= 1e-12:
            break
        keep = np.abs(data - med) <= sig * scale
        if keep.sum() < max(10, int(0.1 * data.size)):
            break
        new_med = float(np.median(data[keep]))
        new_scale = 1.4826 * float(np.median(np.abs(data[keep] - new_med)))
        if (abs(new_med - med) <= 1e-12 * max(1.0, abs(med))
                and abs(new_scale - scale) <= 1e-12 * max(1.0, scale)):
            med, scale = new_med, new_scale
            break
        med, scale = new_med, new_scale
    return float(med), float(max(0.0, scale))


def remove_cosmic_rays(frame: np.ndarray, sigma: float = 5.0,
                       protect: Optional[np.ndarray] = None) -> np.ndarray:
    """单帧宇宙线剔除：局部中值 + 鲁棒 sigma 阈值，命中像素用中值替换。

    Parameters
    ----------
    frame:
        ``[N,N]`` 图像（ADU/已定标均可）。
    sigma:
        正向离群阈值（相对残差鲁棒 sigma）。
    protect:
        可选布尔掩膜：``True`` 的像素不参与剔除（例如 WFS 星点核心，
        避免把衍射极限 PSF 尖峰误判成宇宙线）。

    Returns
    -------
    numpy.ndarray
        新数组（不修改输入）。
    """
    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError("frame must be 2-D")
    sig = float(sigma)
    if not np.isfinite(sig) or sig <= 0:
        raise ValueError("sigma must be finite and > 0")
    med = ndi.median_filter(arr, size=3, mode="nearest")
    diff = arr - med
    finite = np.isfinite(diff)
    if not finite.any():
        return arr.copy()
    center = float(np.median(diff[finite]))
    scale = 1.4826 * float(np.median(np.abs(diff[finite] - center)))
    out = arr.copy()
    if scale <= 1e-12:
        return out
    hit = finite & (diff > center + sig * scale)
    if protect is not None:
        prot = np.asarray(protect, dtype=bool)
        if prot.shape != arr.shape:
            raise ValueError("protect shape must match frame")
        hit = hit & ~prot
    out[hit] = med[hit]
    return out


def cutout(frame: np.ndarray, x: float, y: float, size: int,
           clamp: bool = False) -> np.ndarray:
    """取以 ``(x, y)`` 为中心的 ``size x size`` 子图（浮点）。

    ``clamp=True`` 时把窗口平移到图像内部（返回完整 ``size x size``）；
    否则越界抛 ``ValueError``。``size`` 建议为偶数（PSF 窗口）。
    """
    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError("frame must be 2-D")
    size = int(size)
    if size < 2:
        raise ValueError("size must be >= 2")
    h, w = arr.shape
    if size > w or size > h:
        raise ValueError("cutout larger than frame")
    half = size // 2
    x0 = int(round(float(x))) - half
    y0 = int(round(float(y))) - half
    if clamp:
        x0 = int(np.clip(x0, 0, w - size))
        y0 = int(np.clip(y0, 0, h - size))
    elif x0 < 0 or y0 < 0 or x0 + size > w or y0 + size > h:
        raise ValueError("cutout extends outside the frame")
    return arr[y0:y0 + size, x0:x0 + size].copy()


def centroid_subpixel(cut: np.ndarray) -> Tuple[float, float]:
    """亚像素质心（背景用边框中值扣除，权重裁剪为非负）。

    Returns
    -------
    (x, y):
        以子图像素坐标表示的质心；全暗/无信号时返回几何中心。
    """
    arr = np.asarray(cut, dtype=float)
    if arr.ndim != 2 or arr.size == 0:
        raise ValueError("cutout must be a non-empty 2-D array")
    border = np.concatenate([arr[0, :], arr[-1, :], arr[:, 0], arr[:, -1]])
    finite = border[np.isfinite(border)]
    bg = float(np.median(finite)) if finite.size else 0.0
    weights = np.clip(np.where(np.isfinite(arr), arr, 0.0) - bg, 0.0, None)
    total = float(weights.sum())
    n0, n1 = arr.shape
    if total <= 1e-30:
        return (float((n1 - 1) / 2.0), float((n0 - 1) / 2.0))
    yy, xx = np.indices(arr.shape, dtype=float)
    x = float(np.sum(weights * xx) / total)
    y = float(np.sum(weights * yy) / total)
    return (x, y)


def detect_stars(frame: np.ndarray, threshold_sigma: float = 5.0,
                 min_pixels: int = 3) -> List[Dict[str, float]]:
    """星点检测（背景扣除 + 阈值分割 + 连通域），返回按流量排序的列表。

    Parameters
    ----------
    frame:
        ``[N,N]`` 图像（已扣/未扣背景均可，内部先估背景）。
    threshold_sigma:
        检测阈值（相对背景噪声的 sigma）。
    min_pixels:
        连通域最小像素数（过滤热像元/宇宙线残留）。

    Returns
    -------
    list of dict
        每项 ``{"x": float, "y": float, "flux": float, "peak": float,
        "npix": int}``；``flux`` 为连通域内背景扣除后的 ADU 总和，
        位置为流量加权亚像素质心。
    """
    arr = np.asarray(frame, dtype=float)
    if arr.ndim != 2:
        raise ValueError("frame must be 2-D")
    thr_sig = float(threshold_sigma)
    if not np.isfinite(thr_sig) or thr_sig <= 0:
        raise ValueError("threshold_sigma must be finite and > 0")
    min_pix = int(min_pixels)
    if min_pix < 1:
        raise ValueError("min_pixels must be >= 1")
    bg, noise = estimate_background(arr)
    resid = np.where(np.isfinite(arr), arr, 0.0) - bg
    if noise <= 1e-12:
        noise = float(np.std(resid[resid > 0])) if np.any(resid > 0) else 0.0
    mask = resid > thr_sig * max(noise, 1e-12)
    if not mask.any():
        return []
    labels, n_lab = ndi.label(mask)
    if n_lab == 0:
        return []
    yy, xx = np.indices(arr.shape, dtype=float)
    out: List[Dict[str, float]] = []
    # 逐个连通域是"星点数"级循环，不是逐像素循环；星点数很小。
    for lab, sl in enumerate(ndi.find_objects(labels), start=1):
        if sl is None:
            continue
        comp = (labels[sl] == lab)
        npix = int(comp.sum())
        if npix < min_pix:
            continue
        w = np.clip(np.where(comp, resid[sl], 0.0), 0.0, None)
        flux = float(w.sum())
        if flux <= 0:
            continue
        x = float(np.sum(w * xx[sl]) / flux)
        y = float(np.sum(w * yy[sl]) / flux)
        out.append({"x": x, "y": y, "flux": flux,
                    "peak": float(np.max(w)), "npix": npix})
    out.sort(key=lambda s: -s["flux"])
    return out


# ===========================================================================
# 帧间配准（相位相关）
# ===========================================================================

def _parabolic_offset(values_minus: float, values_0: float,
                      values_plus: float) -> float:
    # 抛物线顶点偏移：0.5*(f(-1)-f(+1)) / (f(-1)-2f(0)+f(+1))
    den = values_minus - 2.0 * values_0 + values_plus
    if abs(den) < 1e-15:
        return 0.0
    return float(np.clip(0.5 * (values_minus - values_plus) / den, -0.5, 0.5))


def register_shift(ref: np.ndarray, img: np.ndarray
                   ) -> Tuple[float, float, float]:
    """相位相关配准：返回 ``(dy, dx, error)``。

    约定：``img(y, x) ≈ ref(y - dy, x - dx)``，即 ``dy/dx`` 是 ``img``
    相对 ``ref`` 的整数+亚像元平移。亚像元用互相关峰的三点抛物线插值。

    Returns
    -------
    (dy, dx, error):
        ``dy``/``dx`` 单位像素；``error`` 是反平移后与 ``ref`` 的相对
        RMS 残差（0 表示完全一致，>0.5 通常表示配准失败）。
    """
    a = np.asarray(ref, dtype=float)
    b = np.asarray(img, dtype=float)
    if a.ndim != 2 or b.shape != a.shape:
        raise ValueError("ref/img must be same-shape 2-D arrays")
    if a.size == 0 or not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
        raise ValueError("ref/img must be finite and non-empty")
    a = a - float(a.mean())
    b = b - float(b.mean())
    fa = np.fft.fft2(a)
    fb = np.fft.fft2(b)
    # conj(F(ref)) * F(img) 的 ifft 峰位于 +s，其中 img(y,x)=ref(y-dy,x-dx)。
    cross = np.conj(fa) * fb
    cross /= np.maximum(np.abs(cross), 1e-12)
    cc = np.fft.ifft2(cross).real
    iy, ix = np.unravel_index(int(np.argmax(cc)), cc.shape)
    n0, n1 = cc.shape
    oy = _parabolic_offset(cc[(iy - 1) % n0, ix], cc[iy, ix],
                           cc[(iy + 1) % n0, ix])
    ox = _parabolic_offset(cc[iy, (ix - 1) % n1], cc[iy, ix],
                           cc[iy, (ix + 1) % n1])
    dy = float(iy) + oy
    dx = float(ix) + ox
    if dy > n0 / 2.0:
        dy -= n0
    if dx > n1 / 2.0:
        dx -= n1
    shifted = ndi.shift(b, (-dy, -dx), order=3, mode="wrap")
    denom = float(np.sqrt(np.mean(b * b)))
    err = float(np.sqrt(np.mean((shifted - a) ** 2)) / max(denom, 1e-12))
    return dy, dx, err


# ===========================================================================
# PSF 清理（预处理和 PD 反演之间的桥）
# ===========================================================================

def clean_psf(cut: np.ndarray, border: int = 6) -> np.ndarray:
    """PSF 子图清理：拟合/扣除边界平面、裁剪负值、归一为总流量 1。

    与 :func:`products.ccd.io.clean_cutout` 同一约定（PD 需要单位归一 PSF），
    但独立实现以保持 products/preprocess 不反向依赖 products/ccd。向量化。
    """
    arr = np.asarray(cut, dtype=float)
    if arr.ndim != 2 or min(arr.shape) < (2 * int(border) + 2):
        raise ValueError("cut must be 2-D and larger than 2*border")
    b = max(1, int(border))
    h, w = arr.shape
    yy, xx = np.indices(arr.shape, dtype=float)
    border_mask = np.zeros_like(arr, dtype=bool)
    border_mask[:b, :] = True
    border_mask[-b:, :] = True
    border_mask[:, :b] = True
    border_mask[:, -b:] = True
    design = np.column_stack([xx[border_mask], yy[border_mask],
                              np.ones(int(border_mask.sum()))])
    coeff, *_ = np.linalg.lstsq(design, arr[border_mask], rcond=None)
    plane = coeff[0] * xx + coeff[1] * yy + coeff[2]
    out = np.clip(arr - plane, 0.0, None)
    total = float(out.sum())
    if total <= 1e-30:
        return np.zeros_like(out)
    return out / total
