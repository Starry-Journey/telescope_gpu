# POPPY/HCIPy PSF 接入

> 脚本：`scripts/poppy_psf.py`
> 状态：Windows Python 3.13 上已安装并成功生成 5×256 PSF；
> **这是衍射计算，不是观测数据**。

## 1. 安装

```powershell
py -3.13 -m pip install --user poppy hcipy
```

已验证：

```text
poppy 1.1.2
hcipy 0.7.1
```

WSL Python 3.8 的 pip 因系统 OpenSSL 损坏暂不可用；当前在 Windows Python 运行。

## 2. 生成 PSF

```powershell
py -3.13 scripts\poppy_psf.py --n 256 --out data\psf\poppy_defocus_256.npy
```

模型：

- 2.5 m 口径、0.7 m 中心遮拦；
- 500 nm；
- 5 帧 Z4 离焦 `[-2,-1,0,1,2]` waves；
- 探测器 0.01 arcsec/pixel，2.56 arcsec → 256×256。

输出：

```text
poppy_defocus_256.npy    [5,256,256]
poppy_defocus_256.json   source/synthetic 元数据
```

## 3. 接到 PD batch

```bash
PYTHONPATH=. python3 scripts/pd_batch_256.py \
  --npy data/psf/poppy_defocus_256.npy \
  --n 256 --workers 4 --starts 3 --maxeval 20 --obscuration 0.28
```

`ml.traditional.forward_psf_numpy(..., obscuration=0.28)` 已支持中心遮拦；
不传时保持原行为。

## 4. 结果解释

用 POPPY 的 0.28 遮拦 PSF 跑 PD（`--starts 3 --maxeval 20 --obscuration 0.28`）：

```text
zernike = [-0.345, +0.0002, -0.0011] waves   cost = 2.1e-8   20 evals
```

（P2-11 之前的导出把 512² 探测器网格左上角裁成 256²，PSF 中心正好落在裁切角上，
只剩四分之一能量，旧值 Z4≈-0.22 是残缺帧的结果。现在中心裁切后 Z4≈-0.34，
Z8/Z9 已回到 0 附近，但 Z4 仍非零——**POPPY 的 pupil/采样与解析 forward 仍不完全
一致**（遮拦、像素尺度、Zernike 归一化）。

这只能说明：

- POPPY→npy→PD batch 的数据通路已打通；
- 不能把该结果当真实 Zemax/观测精度；
- 下一步要统一 pupil 采样、遮拦、Zernike 归一化，再比较。

## 5. 边界

- POPPY 是物理衍射，不是真实离焦观测；
- HCIPy 已安装但尚未接入；
- 真实多离焦仍需要实验/望远镜数据；
- 生成文件 `data/psf/*.npy` 在 `.gitignore` 中，可随时重建。

## 6. 外部对比结果

`scripts/poppy_compare.py` 比较 POPPY 与自有 analytic forward（同 256²、0.28 遮拦，
两侧都经 `products.optics.forward.center_psf` 统一到 (n/2, n/2) 定心约定）：

```text
defocus  POPPY FWHM  ours FWHM  normalized SSE
    0.0      9.47 px    1.58 px     2.4e-6
   +1.0    157.09 px   21.51 px     2.1e-8
   +2.0    231.40 px   45.30 px     5.6e-9
```

两侧 FWHM 比值稳定在 5–7 倍，量级差来自**像素尺度不同**（POPPY 探测器在
0.01″/px 下按 2× 过采样输出 512²，而解析 forward 的 256² 网格对应更粗的采样），
不是定心问题：定心修好后，零离焦帧的峰值与质心都落在 (128,128) 附近。

FWHM 差 3.5 倍，说明两者**像素尺度/OTF 采样**不同；SSE 小只是因为能量被摊开，
不能据此说模型一致。POPPY 只用于外部检查，不接入求解器。

## 7. 采样对齐尝试

`scripts/poppy_align.py` 按 FWHM 比例重采样 POPPY（sampling-only）：

- 0/1/2 wave 的缩放约 0.106/0.282/0.303；
- 重采样后 PD 仍在 Z8/Z9 边界，cost 1.27e-6；
- 说明仅对齐像素尺度不够，pupil/OTF/遮拦归一化仍不同。

因此 POPPY 仍只做外部对照，不进入求解器。
