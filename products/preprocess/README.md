# products/preprocess —— CCD 预处理

> raw ADU 帧 → 定标/坏像元/饱和/宇宙线/背景/星点/配准/归一化 PSF；
> 纯 numpy/scipy、确定性、无 GUI/网络，不逐像素 Python 循环。

## 职责

- master bias/dark/flat 合成与定标（按曝光缩放，flat 中位数归一）；
- 坏像元掩膜与修复、饱和掩膜、宇宙线剔除、背景估计与扣除；
- 星点检测、亚像素质心、帧间平移配准、PSF 子图清理与归一。

## 输入 / 输出

- 输入：`[K,N,N]` raw ADU 帧栈（`CameraApi` 约定）、定标帧栈与曝光时间、单帧图像；
- 输出：`MasterFrames`、定标后帧、坏像元掩膜、背景值/噪声、星点列表、
  亚像素质心、帧间平移量、清理并归一化的 PSF 子图；
- 单位：ADU（定标前含 bias/dark；flat 归一后为相对响应），曝光 s。

## 关键文件与函数

| 文件 | 关键接口 |
|---|---|
| `products/preprocess/ccd_calib.py` | `MasterFrames`、`build_master_bias`/`build_master_dark`/`build_master_flat`、`bad_pixel_map`/`repair_bad_pixels`、`calibrate`、`saturation_mask`、`estimate_background`、`remove_cosmic_rays`、`cutout`、`centroid_subpixel`、`detect_stars`、`register_shift`、`clean_psf` |
| `products/preprocess/__init__.py` | 导出上述公共 API、`MASTER_FILE`、`ADU_UNIT` |

## 依赖

- numpy、scipy（`scipy.ndimage`）；不依赖 Qt/astropy（FITS 由 `products.ccd` 负责）。

## 标签与验证状态

- 合成定标（`calib_dir` 缺失时用 mock 相机的 bias/dark/flat 合成）：
  `calibration="synthetic"`，不得当实测定标；
- 用真实台架 master frames 后仍需回归；当前整体 UNVERIFIED（待台架）。

## 常见坑

1. 合成 flat 不能当实测平场；报告里必须写 `calibration` 字段；
2. bias/dark 按曝光缩放（`dark` 是已扣 bias 的暗电流），别直接相减不缩放；
3. 输入可以是 2D 或 `[K,N,N]`，函数对维度敏感，调用前确认形状；
4. 饱和像素用掩膜排除，不要用 0 填充后当正常数据拟合；
5. 星点窗口/ROI 约定来自相机 metadata，换相机后先核对 ROI 与像素尺度。
