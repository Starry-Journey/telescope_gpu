# telescope-gpu

大口径望远镜重力变形的**相位差法波前反演**仿真平台。

正向生成离焦星点图像 → 用相位差法反演出 Zernike 系数 → 与真值比对，
形成闭环自检。对应研究报告《基于相位差法与最小二乘影响矩阵的
大口径望远镜重力变形标定技术研究》技术路线的 **第 ① 步**。

## 文档

| 文档 | 内容 |
|---|---|
| [docs/代码逐行数学对照.md](docs/代码逐行数学对照.md) | **逐行**：代码行 → 数学语言的对照表，每个 kernel 一段 |
| [docs/代码详解.md](docs/代码详解.md) | 逐模块的数学物理原理，说明代码**为什么这样写** |
| [docs/研究报告对照说明.md](docs/研究报告对照说明.md) | 报告 ↔ 代码的对应关系、参数对照、**差异清单** |

> ⚠ **先读对照说明的 §4 和 §5。** 报告的 `Z7/Z8/Z9` 编号与代码不一致；
> 报告 §3.5 的结果数字（RMS 0.0631λ）来自修改前的代码，当前为 0.0155λ。

## 结构

```
include/forward_model.cuh   公共声明：OpticalParams / ZernikeTable / ForwardModelGPU
src/forward_model.cu        正向模型：7 个 CUDA kernel（Zernike → 光瞳 → FFT → PSF → 星点叠加）
src/forward_generate.cu     生成 5 帧仿真数据，写 FITS
src/inverse_solve.cu        读 FITS + 三阶段优化反演（20 起点 → 前 3 → BOBYQA）
build/visualize.py          可视化（fig1~fig4）
build/plot_sensitivity.py   灵敏度矩阵示意图（纯绘图，无计算）
docs/                       文档
```

## 构建与运行

依赖：CUDA Toolkit（`CUDA_ARCHITECTURES` 硬编码为 `86`）、cuFFT、NLopt、CMake ≥ 3.18。

```bash
mkdir -p build && cd build
cmake .. && make -j

./forward_generate 45    # 生成 build/theta_045deg/ 下 5 帧 FITS
./inverse_solve 45       # 反演，输出 inverse_result.txt
python3 visualize.py     # 生成 fig1~fig4
```

两个可执行文件按 `theta_XXXdeg/` 相对路径读写数据，**必须在 `build/` 下运行**。

## 参数

图像 256×256、FFT 512²、光瞳半径 0.45、过采样 2、15 颗星（通量 3000~15000 e⁻）、
背景 200 e⁻、读出噪声 5 e⁻、离焦帧 0/±1/±2 λ、优化变量 Z4/Z8/Z9。

## 自洽性自检

`inverse_solve` 开头会打印 `cost(true Z4,Z8,Z9)`。**若最终 `final cost` 低于它，
说明正向生成端与反演端的 Zernike 项集合不一致，反演结果是伪解。**
详见 [docs/研究报告对照说明.md §5.3](docs/研究报告对照说明.md)。
