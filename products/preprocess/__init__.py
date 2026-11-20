"""数据管线预处理（preprocess）：定标、坏像元、背景、饱和、宇宙线、星点、配准。

本包位于 "天文相机 raw ADU" 和 "相位差反演（PD）" 之间，只做纯 numpy/
scipy 的确定性图像运算。

输入:
    - ``[K, N, N]`` raw ADU 帧栈（:mod:`products.acquisition.interfaces`）；
    - 主定标帧（bias/dark/flat）与曝光时间；
    - 单帧图像（背景、饱和、宇宙线、星点、配准）。
输出:
    - 定标后帧、坏像元掩膜、背景值/噪声、星点列表、亚像素质心、平移量、
      清理并归一的 PSF 子图。
边界:
    - 没有真实实验室定标数据；``calibration="synthetic"`` 的 master 由管线
      用 mock 相机的 bias 曝光/相机 metadata 合成，不能当作实测定标；
    - 不依赖 GUI/Qt/网络，不做逐像素 Python 循环。
"""
from __future__ import annotations

from .ccd_calib import (ADU_UNIT, MASTER_FILE, MasterFrames, bad_pixel_map,
                        build_master_bias, build_master_dark,
                        build_master_flat, calibrate, centroid_subpixel,
                        clean_psf, cutout, detect_stars, estimate_background,
                        register_shift, remove_cosmic_rays, repair_bad_pixels,
                        saturation_mask)

__all__ = [
    "ADU_UNIT",
    "MASTER_FILE",
    "MasterFrames",
    "bad_pixel_map",
    "build_master_bias",
    "build_master_dark",
    "build_master_flat",
    "calibrate",
    "centroid_subpixel",
    "clean_psf",
    "cutout",
    "detect_stars",
    "estimate_background",
    "register_shift",
    "remove_cosmic_rays",
    "repair_bad_pixels",
    "saturation_mask",
]
