# CCD 星场与运动目标处理管线

本目录是 `telescope_gpu` 的 **CCD 数据前端/运动目标检测** 扩展，和原有的
相位差波前反演（`src/forward_model.cu`、`src/phase_diversity.cu`）并列：

```
仿真/采集 CCD 星场序列
        │
        ▼
暗场/平场/本底 + 背景估计 + 去噪        pipeline/preprocess.py
        │
        ▼
星点提取 + 质心（连通域/加权质心）      pipeline/detect.py
        │
        ├──────────────► 星表 catalog.csv
        ▼
帧差 + 形态学 + Hough + 轨迹关联         pipeline/detect.py
        │
        ▼
WCS + 畸变多项式 + 星表匹配              pipeline/astrometry.py
        │
        ▼
检测率/虚警率/质心与天体测量精度          pipeline/evaluate.py
        │
        ▼
诊断图 + metrics.json + Word 说明文档
```

> SEP / photutils / OpenCV 是工业界常用的星点提取库；当前 WSL 环境没有安装，
> 因此 `detect.py` 用 `numpy + scipy.ndimage` 实现了同等功能的阈值、连通域、
> 质心、形态学和 Hough，保证零额外依赖、每一步可检查。文档里会对比这些库的
> 适用场景。

## 目录

```
pipeline/
  config.py       配置 dataclass（仿真/预处理/检测/天体测量）
  simulate.py     CCD 星场 + 定标帧 + 星表 + 真值
  preprocess.py   bias/dark/flat、mesh 背景、Gaussian 去噪
  detect.py       星点提取/质心、相位/高通互相关配准、帧差、形态学、Hough、轨迹
  astrometry.py   TAN WCS、三次畸变多项式、球面最近邻匹配
  evaluate.py     检测率、虚警率、位置 RMSE、天体测量误差
  visualize.py    数据集/星点/运动目标/天体测量/指标图
  run.py          命令行入口
  tests/          unittest（几何、质心、Hough、端到端合成）
configs/pipeline.json
```

## 依赖

- Python ≥ 3.8
- numpy、scipy、astropy、matplotlib

不需要 CUDA；CUDA 只用于原项目的正向模型。若要把本管线 GPU 化，见 Word
文档第 8 章“并行程序设计”。

## 快速开始

```bash
cd telescope_gpu
python3 -m pipeline.run --config configs/pipeline.json --force-sim
```

### 真实感 CCD 仿真

默认 `simulate.py` 用高斯 PSF 保证指标可复现。要接近 phosim 的视觉效果（Moffat seeing、衍射星芒、饱和/ blooming、宇宙线轨迹、热像素、背景梯度、平场灰尘）：

```bash
python3 -m pipeline.realistic --out data/realistic --frames 5 --stars 40
```

输出 `raw/`、`calib/`、`catalog.csv`、`truth.csv`、`preview.png`；说明见
`docs/tutorials/10_CCD图像真实感.md`。它不是 phosim 的完整光学模型，但比高斯 PSF 更接近真实 CCD。

输出在 `results/ccd_pipeline/`：

```
preprocessed/            定标后的 FITS + calibrated_stack.npz
stars.csv                星点表（x/y/flux/snr/area）
detections.csv           轨迹确认后的运动目标检测
astrometry_errors.csv    天体测量残差
metrics.json             检测率/虚警率/RMSE
fig_*.png                诊断图
run_summary.json         本次运行汇总
```

只跑已有数据：

```bash
python3 -m pipeline.run --config configs/pipeline.json --skip-sim
```

单元测试：

```bash
python3 -m unittest discover -s pipeline/tests -v
```

## 当前默认结果

默认配置：512×512、20 帧、20 颗星、目标流量 1200 e⁻、读出噪声 5 e⁻、
1″/pixel、三次光学畸变、0.6 pixel 指向抖动。

| 指标 | 结果 |
|---|---|
| 星点完备率 | 1.000（20/20） |
| 星点质心 RMSE | 0.67 pixel（含 0.6 pixel 指向抖动） |
| 运动目标检测率 | 0.85（17/20 帧） |
| 虚警率 | 0.000（轨迹确认后） |
| 目标位置 RMSE | 1.09 pixel |
| 星表匹配数 / 残差 | 20 / 0.080 arcsec |
| 目标天体测量 RMSE | 1.39 arcsec |

注意：这些是 **已知真值的仿真结果**，不是实测精度；目标在个别帧靠近亮星或
被星掩膜覆盖时会漏检，这是当前阈值+轨迹确认策略的边界。

## 模块职责

| 模块 | 输入 | 输出 | 关键点 |
|---|---|---|---|
| `simulate.py` | `SimConfig` | raw FITS、bias/dark/flat、catalog.csv、truth.csv | 天空→理想像素→畸变→PSF→flat/dark/bias；泊松+读出噪声 |
| `preprocess.py` | raw FITS + 定标帧 | 定标帧、背景图、`calibrated_stack.npz` | `(raw-bias-dark*t)/flat`；mesh sigma-clip 背景；Gaussian 去噪 |
| `detect.py` | 定标帧栈 | 星表、候选、轨迹 | 阈值/连通域/加权质心；高通互相关配准；中值栈帧差；形态学；Hough；最近邻轨迹 |
| `astrometry.py` | 星表 + FITS WCS | 畸变系数、匹配、RA/Dec | TAN WCS；三次多项式；球面单位向量最近邻；迭代最小二乘 |
| `evaluate.py` | 真值 + 检测 | metrics.json/CSV | 检测率、虚警率、位置 RMSE、天体测量 RMS |
| `visualize.py` | 上述结果 | PNG | 数据/星点/轨迹/残差/指标 |

## 和原相位差项目的关系

- 原项目解决“**已知离焦星场 → 反演低阶 Zernike**”，是本管线的下游；
- 本管线解决“**CCD 原始帧 → 星表/运动目标/天球坐标**”，是上游数据准备；
- 两者共用 FITS 约定和 2.5 m 望远镜参数，但算法目标不同，代码解耦，互不依赖。

## 已知边界

1. 星点提取是阈值+连通域，不是 PSF 拟合；密集星场/严重拖尾会混叠，生产环境应换 SEP/photutils。
2. 运动目标检测假设目标在帧间有明显位移、静止星场占主导；对地静止目标会被中值栈吸收。
3. Hough 只做直线 trail 验证，不支持曲线轨迹；曲线轨迹应改用卡尔曼/多假设跟踪。
4. 畸变模型是三次多项式，未包含大气折射/颜色项；高精度天体测量需要 Gaia 星表 + SCAMP/Astrometry.net。
5. 当前是 CPU 实现；GPU 化策略见 Word 文档第 8 章。
