# 4–6 周工程化 MVP 路线

> 目标：把 `telescope_gpu` 从“能跑的课题代码”推进到 **可构建、可测试、可测速、可复现、可展示** 的工程 MVP；不是把 L1–L5 五条线同时做完。
>
> 原则：先一条主线、每阶段可验收、不要并行堆技术栈；本文给出 6 周表和硬闸门。

## 0. 一句话

**4 周做出“C++ 数值库 + CCD 检测管线”的闭环，第 5–6 周加“PyTorch 可微前向 + CNN 对比 + 可视化 demo”；每周只交付一个可验收的 vertical slice。**

## 1. 产出物

| 类别 | 具体产出 | 验收 |
|---|---|---|
| 代码仓库 | `telescope_gpu`：C++/CUDA + `pipeline/` + `numerics/` + `ml/` + `web/` | `git clone` 后按 README 能构建 |
| 数据集 | `data/ccd_pipeline/` 仿真 FITS（raw/bias/dark/flat/catalog/truth） | `--force-sim` 可重建；`dataset.json` 记录参数 |
| 指标 | `results/ccd_pipeline/metrics.json`、`ml/results/comparison.csv`、benchmark CSV | 数字能由命令复现 |
| 可视化 | `fig_*.png`、FastAPI/React 小程序、Word/PDF 技术说明 | 一页能讲清数据流和结果 |
| 文档 | 每个模块一篇教程（科学原理 + 代码设计 + 技术栈 + 测试） | `docs/tutorials/` 索引完整 |

## 2. 分层与优先级

| 层 | 本周做什么 | 不做什么 |
|---|---|---|
| L1 工程化 | CMake 子目录、CLI/config、日志、Git 忽略、README | 不重写全部旧代码 |
| L2 正确性 | GTest 10 项、CPU reference、ASan 选项、回归指标 | 不追求 100% 覆盖率 |
| L3 性能 | Google Benchmark、批量 FFT/预分配说明、Nsight 记录模板 | 不做大规模 kernel 重写 |
| L4 数据/算法 | CCD 预处理、星点、帧差/Hough、WCS/评估 | 不接真实望远镜控制 |
| L5 部署/AI | pybind11、Docker、CI、PyTorch 可微前向 + CNN | 不做在线训练/大模型 |

## 3. 六周表

| 周 | 主线 | 交付 | 闸门 |
|---:|---|---|---|
| W1 | L1 + C++ 数值库骨架 | `numerics/`：LU/QR/SVD/FFT/RK4 + CMake | `numerics_tests` 10/10 通过 |
| W2 | L2 正确性 | GTest 边界/异常、ASan 构建、与 NumPy/Eigen 对比 | ASan 无报错；误差 <1e-10 |
| W3 | L3 性能 | Google Benchmark：solve/lstsq/FFT/RK4；批量 FFT/预分配说明 | 有 CSV/表格；能解释复杂度 |
| W4 | L4 数据/算法 | `pipeline/` 跑通：检测率/虚警率/位置/天体测量 | `metrics.json` 稳定；图可复现 |
| W5 | L5 部署 | pybind11 模块、Dockerfile、GitHub Actions | `import telescope_numerics_py` 成功；CI 绿 |
| W6 | L5 AI | PyTorch 可微前向、CNN 训练、与传统对比 | 有 comparison.csv/图/误差表 |

> 如果只有 4 周：砍掉 W6 的 CNN 训练，保留“可微前向 + 一个小样本 sanity check”；不要砍 L2 测试和 L4 指标，它们是可信度的基础。

## 4. 验收闸门（硬标准）

1. **构建闸门**：CPU-only 与 CUDA 构建都能过；无数值库依赖时主项目仍可构建（`TEL_BUILD_NUMERICS=OFF`）。
2. **测试闸门**：`ctest` 全绿；数值库 10 项；pipeline 8 项；ML 4 项；ASan 构建无 heap/stack 报错。
3. **数值闸门**：线性解残差 <1e-10；最小二乘与 NumPy `lstsq` 误差 <1e-8；FFT 与 `numpy.fft` 误差 <1e-10；RK4 与解析解误差 <1e-6。
4. **性能闸门**：benchmark 能重复；至少能解释 `solve` 的 O(N³)、FFT 的 O(N log N)、RK4 的 O(steps)。
5. **数据闸门**：`--force-sim` 重建；指标不漂移；图与 metrics 一致。
6. **AI 闸门**：CNN 在固定测试集上优于“预测均值”基线；训练曲线/对比表可复现；明确“仿真、不是实测”。
7. **文档闸门**：每个模块含科学原理、代码结构、技术栈、运行命令、结果、边界。

## 5. 数据集策略

- **现在**：全仿真（已知真值），用于验证算法链路和指标口径。
- **半真实**：用公开 FITS/CCD 星场替换 `data/ccd_pipeline/raw/`，保留同一套 `catalog.csv/truth.csv` 接口；没有真值时只报检测数/重复性，不报“精度”。
- **真实**：需要望远镜观测日志、WCS 定标、星表（Gaia）；在文档中单独标注，不能和仿真数字混表。

## 6. 可视化/前端

```
metrics.json / figures
        │
        ▼
FastAPI (Python)  ── /api/metrics, /api/simulate, /api/invert
        │
        ▼
前端（React + Plotly）
  ├─ 数据流/模块图
  ├─ 原始帧/定标帧/残差/轨迹叠加
  ├─ 检测率/虚警率/误差曲线
  └─ CNN vs 传统 的精度/耗时对比
```

MVP 先做**单页 + 只读/仿真 API**；不做用户系统/数据库。

## 7. 明确不做

- 不做完整 EDA/CAE/遥感平台；
- 不做在线训练、分布式训练、模型压缩；
- 不做真实望远镜闭环控制；
- 不做 5 条技术栈并行学习；
- 不把仿真精度写成实测精度。

## 8. 与主技术报告的关系

- 主报告解决“5 帧相位差 → Z4/Z8/Z9 → 波前 RMS”的科学问题；
- 本 MVP 的 `numerics/` 是可复用的数值核心，`ml/` 是学习型反演的**方法验证**（64×64 PSF 代理模型），`pipeline/` 是 CCD 前端；
- 三者不替代主求解器，数字不能混表；对齐关系见 `docs/工程化技术说明_代码对齐版.docx` 第 9 章。

