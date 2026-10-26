# telescope_gpu

大口径望远镜重力变形的**相位差法波前反演**仿真平台（v2）。

正向生成离焦星点图像 → 用相位差法反演出 Zernike 系数 → 与真值比对，
形成闭环自检。对应研究报告《基于相位差法与最小二乘影响矩阵的
大口径望远镜重力变形标定技术研究》技术路线的第 ① 步。

> **v2 状态（2026-10）**：CPU/GPU 分层、配置与 CLI、FITS 星表自描述、
> 依赖无关单元测试、批量 cuFFT、GPU 归约。默认配置下
> θ=45°、5 帧、15 颗星，反演波前 RMS = **0.0017λ**（v1 为 0.0155λ）。
> 这是**已知真值的仿真数据**，不是实测精度。

## 架构

```
include/telescope/
  types.hpp            光学参数、Zernike 系数、星场、搜索边界
  zernike.hpp          Zernike 基函数、光瞳网格、重力变形模型
  fft_cpu.hpp          radix-2 CPU FFT（参考实现）
  cpu_reference.hpp    CPU 正向模型（逐像素对照）
  fits_io.hpp          最小 FITS 读写（float32 + 元数据/星表卡片）
  config.hpp           key=value 配置 + 命令行覆盖
  forward_model.cuh    GPU 正向模型（PSF / 星点图 / 多帧）
  phase_diversity.hpp  预处理、代价函数、多起点 + BOBYQA 反演

src/
  zernike.cpp          8 项 Noll Zernike（实际拟合 Z4/Z8/Z9）
  fft_cpu.cpp          CPU FFT
  cpu_reference.cpp    CPU 正向模型
  fits_io.cpp          FITS I/O
  config.cpp           配置/CLI
  forward_model.cu     光瞳 → cuFFT → PSF → 降采样 → 星点叠加
  phase_diversity.cu   代价函数与优化器

apps/
  forward_generate.cu  生成 5 帧带噪声 FITS
  inverse_solve.cu     读 FITS、反演、写结果/报告

tests/                 自建轻量测试框架 + CTest
configs/default.conf   默认参数
scripts/               可视化（沿用 v1 脚本，读取 6 数结果文件）
baseline/              v1 结果与图片，仅作对照
```

构建目标分两层：

- `telescope_cpu`：不依赖 CUDA，可在无 GPU 环境编译和测试。
- `telescope_cuda`：链接 `telescope_cpu` + CUDA/cuFFT + NLopt。

## 构建

依赖：CMake ≥ 3.18、C++17、CUDA Toolkit、cuFFT、NLopt。

```bash
cd telescope_gpu
cmake -S . -B build
cmake --build build -j4
```

CUDA 架构默认 `86`（RTX 3050）。换机器时显式指定：

```bash
cmake -S . -B build -DCMAKE_CUDA_ARCHITECTURES=89
```

## 运行

两个程序按 `theta_XXXdeg/` 相对路径读写数据，建议在 `build/` 下运行：

```bash
cd build

# 1. 生成仿真数据：theta_045deg/ 下 5 帧 FITS
./forward_generate --theta 45

# 2. 反演
./inverse_solve --theta 45

# 3. 可视化（可选；需要 matplotlib/astropy，输出写在当前目录）
python3 ../scripts/visualize.py
```

常用覆盖参数：

```bash
./forward_generate --theta 45 --star-count 15 --seed 42
./inverse_solve   --theta 45 --solve-starts 40 --solve-refine 3
./inverse_solve   --config ../configs/default.conf --theta 45
```

`--help` 查看完整说明。所有配置项都在 `configs/default.conf` 中，
命令行 `--key value` 或 `--key=value` 可覆盖任意一项。

## 输出

- `inverse_result.txt`：6 个数（真值 Z4/Z8/Z9 + 反演值），兼容 v1 可视化脚本。
- `inverse_report.txt`：帧数、`cost(true)`、`cost(zero)`、最终代价、RMS、
  伪解标记、逐项误差。

`inverse_solve` 会先计算 `cost(true)`（用 FITS 头部记录的真值）和
`cost(zero)`。若最终代价**比 `cost(true)` 低超过 1%**，程序会标记
`pseudo_solution`：这通常意味着正向模型与反演变量集合不一致，
结果是伪解。

## 测试

```bash
cd build
ctest --output-on-failure
```

- `cpu_unit`：Zernike、FFT、FITS、配置。
- `gpu_unit`：GPU vs CPU 正向模型、归一化、真值处代价最小、
  端到端反演恢复（含噪声/无噪声两种）。

测试不依赖 GTest（环境未安装），使用 `tests/test_framework.hpp` 中的
轻量注册/断言框架。

## 数值自检与已知边界

1. **背景扣除后不要 clamp 负值。** 这是 v2 调试中定位到的关键 bug：
   背景 200 count、泊松噪声 σ≈14 count，若把 `data - background`
   截断到 0，会在每像素产生约 5.6 count 的正偏置；总偏置（约 37 万
   count）远大于真实信号（约 12 万 count），归一化后科学信号被稀释
   约 4 倍，优化器会把解推向“最锐利”的伪解。v1 的 `cost(true)` 偏大、
   RMS 0.0155λ 主要来自这里。
2. **只拟合 Z4/Z8/Z9。** 其余 Zernike 项在正向模型中为 0；如果扩展
   变量集合，必须同时改 `active_indices()`、`kNumZernike` 和边界。
3. **搜索边界必须允许负值。** 重力变形模型在 θ>225° 时 Z9 为负；
   `ZernikeBounds::default_bounds()` 已改为对称区间 `[-3,3]`、
   `[-1.5,1.5]`。20 个起点在对称区间下可能漏掉真解盆地，默认改为 40。
4. **噪声模型**：泊松(信号+背景) + 高斯读出噪声；背景用 FITS 头部 `BKG`
   （仿真真值）。中位数背景是后备方案，会被星点翼抬高，偏差更大。
5. **FFT 归一化**：PSF 在 GPU 上归一化为 sum=1，再叠加星点通量；
   `generate_frames` 输出未归一化图像，反演前逐帧去背景并归一化。
6. **星点坐标**：生成时取整，FITS 头部记录同一组整数坐标，避免
   “头部坐标 vs 实际叠加坐标”不一致。

## 与 v1 的差异

- 目录拆分为 `include/telescope/`、`src/`、`apps/`、`tests/`、`configs/`。
- 新增 `telescope_cpu` 层和 CPU reference，可在无 GPU 环境跑测试。
- 新增配置/CLI、FITS 星表卡片、结果报告。
- 正向模型改为预分配 workspace + 批量 `cufftPlanMany` +
  块内归约 + GPU 归一化。
- 反演代价函数改为多帧批量生成，去除逐帧同步。
- 修复背景扣除 clamp 负值导致的伪解问题。
- v1 的 6 数结果文件格式保留，`scripts/visualize.py` 可直接使用。

## 文档

- `docs/代码逐行数学对照.md`、`docs/代码详解.md`、
  `docs/研究报告对照说明.md`：v1 时期的数学/代码对照，
  **文件路径与函数名以 v1 为准**；v2 的结构变化见本 README 和
  `projectcpp_learning/telescope_gpu工程化说明.md`。
- 求职规划与工程化叙事：`projectcpp_learning/`。
