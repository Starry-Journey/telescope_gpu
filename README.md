# telescope_gpu

大口径望远镜重力变形的**相位差法波前反演**仿真平台（v2.1）。

正向生成离焦星点图像 → 用相位差法反演出 Zernike 系数 → 与真值比对，
形成闭环自检。对应研究报告《基于相位差法与最小二乘影响矩阵的
大口径望远镜重力变形标定技术研究》技术路线的第 ① 步。

> **v2.1 状态（2026-10）**：CPU/GPU 分层、配置与 CLI、FITS 星表自描述、
> 依赖无关单元测试、批量 cuFFT、GPU 归约。默认配置下
> θ=45°、5 帧、15 颗星，反演波前 RMS = **0.0017λ**。
> 按“三个系数误差的算术 RMS”这一同一口径：v1 为 0.0155λ，v2 为 0.00096λ
> （波前 RMS 与系数 RMS 相差 √3，不能直接混比）。
> 这是**已知真值的仿真数据**，不是实测精度。
>
> 详细架构、逐模块讲解与面试讲法见 `docs/代码说明.md`。

## 架构

```
include/telescope/
  types.hpp            光学参数、Zernike 槽位/边界、帧数上限
  zernike.hpp          Zernike 基函数、光瞳网格、重力变形模型
  fft_cpu.hpp          radix-2 CPU FFT（参考实现）
  cpu_reference.hpp    CPU 正向模型（逐像素对照）
  fits_io.hpp          最小 FITS 读写（float32 + 元数据/星表卡片）
  config.hpp           key=value 配置 + 命令行覆盖
  forward_model.cuh    GPU 正向模型（PSF / 星点图 / 多帧）
  phase_diversity.hpp  预处理、代价函数、多起点 + BOBYQA 反演

src/
  zernike.cpp          8 个基函数槽位（实际拟合 Z4/Z8/Z9）
  fft_cpu.cpp          CPU FFT
  cpu_reference.cpp    CPU 正向模型
  fits_io.cpp          FITS I/O
  config.cpp           配置/CLI + 参数校验
  forward_model.cu     光瞳 → cuFFT → PSF → 降采样 → 星点叠加
  phase_diversity.cu   代价函数与优化器

apps/
  forward_generate.cu  生成 5 帧带噪声 FITS
  inverse_solve.cu     读 FITS、反演、写结果/报告
tests/                 自建轻量测试框架 + CTest
configs/default.conf   默认参数
scripts/               可视化与灵敏度矩阵示意图
baseline/              v1 结果与图片，仅作对照
docs/                  代码说明（本版新增）
```

构建目标分两层，依赖方向单一（CUDA → CPU）：

- `telescope_cpu`：不依赖 CUDA，可在无 GPU 环境编译和测试。
- `telescope_cuda`：链接 `telescope_cpu` + CUDA/cuFFT + NLopt。
  CUDA 或 NLopt 缺失时自动降级为 CPU-only，不会让整个构建失败。

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

只编译/测试 CPU 层（无 GPU 的 CI、笔记本）：

```bash
cmake -S . -B build-cpu -DTEL_BUILD_CUDA=OFF
cmake --build build-cpu -j4
ctest --test-dir build-cpu --output-on-failure
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
python3 ../scripts/visualize.py --theta 45
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
- `inverse_report.txt`：帧数、背景、`cost(true)`、`cost(zero)`、最终代价、
  RMS、伪解标记、逐项误差、正向模型调用次数。

`inverse_solve` 会先计算 `cost(true)`（用 FITS 头部记录的真值）和
`cost(zero)`。若最终代价**比 `cost(true)` 低超过 1%**，程序会标记
`pseudo_solution`：这通常意味着正向模型与反演变量集合不一致，
结果是伪解。

## 测试

```bash
cd build
ctest --output-on-failure
```

- `cpu_unit`（28 个用例）：Zernike、FFT、FITS、配置与参数校验。
- `gpu_unit`（11 个用例）：GPU vs CPU 正向模型、归一化、真值处代价最小、
  端到端反演恢复、预处理、FITS 帧一致性校验、错误路径。

测试不依赖 GTest（环境未安装），使用 `tests/test_framework.hpp` 中的
轻量注册/断言框架。

## 编号约定（重要）

代码/报告里的 `Z8`、`Z9` 是**工程槽位号**，不是严格 Noll 编号：

| 槽位 | 标签 | 真实 Noll | 是否拟合 |
|---|---|---|---|
| 0 | Z2 tilt-x | 2 | 否 |
| 1 | Z3 tilt-y | 3 | 否 |
| 2 | Z4 defocus | 4 | **是** |
| 3 | Z5 astig-45 | 5 | 否 |
| 4 | Z6 astig-0 | 6 | 否 |
| 5 | Z7 trefoil | 9 | 否 |
| 6 | Z8 coma-x | 8 | **是** |
| 7 | Z9 coma-y | 7 | **是** |

也就是说，本项目拟合的是“离焦 + x 彗差 + y 彗差”，物理含义没问题，
但 `Z9` 对应文献里的 Noll Z7。与 Zemax 对照时：
Zemax Z7（垂直彗差）→ 代码 Z9，Zemax Z8（水平彗差）→ 代码 Z8。
详见 `docs/代码说明.md`。

## 数值自检与已知边界

1. **背景扣除后不要 clamp 负值。** 这是 v2 调试中定位到的关键 bug：
   背景 200 count、泊松噪声 σ≈14 count，若把 `data - background`
   截断到 0，会在每像素产生约 5.6 count 的正偏置；总偏置（约 37 万
   count）远大于真实信号（约 12 万 count），归一化后科学信号被稀释
   约 4 倍，优化器会把解推向“最锐利”的伪解。v1 的 `cost(true)` 偏大、
   系数 RMS 0.0155λ 主要来自这里。
2. **只拟合 Z4/Z8/Z9。** 其余槽位在正向模型中为 0；如果扩展变量集合，
   必须同时改 `active_zernike_indices()`、`kNumZernike` 和边界。
3. **搜索边界必须允许负值。** 重力变形模型在 θ>225° 时 Z9 为负；
   `ZernikeBounds::default_bounds()` 使用对称区间 `[-3,3]`、
   `[-1.5,1.5]`。20 个起点在对称区间下可能漏掉真解盆地，默认 40。
4. **噪声模型**：泊松(信号+背景) + 高斯读出噪声；背景用 FITS 头部 `BKG`
   （仿真真值）。中位数背景是后备方案，会被星点翼抬高，偏差更大。
5. **FFT 归一化**：PSF 在 GPU 上归一化为 sum=1，再叠加星点通量；
   `generate_frames` 输出未归一化图像，反演前逐帧去背景并归一化。
6. **星点坐标**：生成时取整，FITS 头部记录同一组整数坐标；反演端
   `set_stars` 再次取整，避免“头部坐标 vs 实际叠加坐标”不一致。
7. **FITS 自描述**：反演端会校验每帧的 `THETA`、`DEFOCUS`、图像尺寸、
   星表和真值是否一致；不一致直接失败，避免混帧后“看起来收敛”。

## 报告与代码数值的差异

当前仓库 v2.1 在 θ=45° 的复现结果是：

```
Z4  +1.4728 -> +1.4729   (误差 0.0001)
Z8  +0.1714 -> +0.1723   (误差 0.0009)
Z9  +0.1800 -> +0.1786   (误差 0.0014)
wavefront RMS = 0.0017 λ
final cost = 5.3124e-03 (cost(true) = 5.3125e-03)
```

研究报告 PPT 中 `RMS=0.0631λ`、`Z4 误差 0.0960λ` 等数字来自早期版本，
与当前代码不一致。答辩前应二选一：用当前代码重跑并更新 PPT，或在 PPT
中明确标注“早期版本/不同参数”。`docs/代码说明.md` 第 11 章给出了对照表
和更新建议。

## 文档

- `docs/代码说明.md`：架构、模块职责、逐函数讲解、数据流、代码审查记录、
  性能分析、面试讲法与追问。
- `README.md`：构建/运行/边界。
