# 多终端通用望远镜 · 波前反演与补偿工程（TelescopeEngineering）

> 课题：**基于相位差法与高性能计算的大口径望远镜重力变形标定研究**
> （数据链路工程版：GPU/CUDA 反演引擎 + CNN 加速，HPC · AI4S 方向）

面向 2.5 m R-C 望远镜的**离焦 CCD → 相位差反演 → 灵敏度矩阵 → 补偿控制**工程实现，
含 FEA/Zemax 数据链、可观测性分析、装调前处理与原生 Qt 桌面前端。

> 当前主路径：**PySide6 原生桌面**（`desktop/`）+ `products/` 服务层。
> 旧 Web/React/FastAPI 架构已归档到 `archive/legacy-web/`，不再维护。

## 1. 能力一览

| 能力 | 入口 | 状态标签 |
|---|---|---|
| CCD 合成（物理探测器链）与 FITS 接入 | `products/ccd/` | SYNTHETIC-VALIDATED |
| CCD 预处理（bias/dark/flat、坏像元、宇宙线、背景、星点、配准） | `products/preprocess/` | SYNTHETIC-VALIDATED |
| 相位差反演（3 项 / 6 项 Zernike，BOBYQA 多起点） | `products/pd/` | CORE-VERIFIED |
| 实测灵敏度标定（括阻位姿 + 漂移消除 + 交叉验证） | `products/sensitivity/measured.py` | UNVERIFIED（待台架） |
| 真实 Zemax 3×5 灵敏度 + 适用范围 | `data/zemax/` | EXTERNAL-CHECK |
| Abaqus M1 板级/文档版 FEA + S 区间 | `data/fea_cases/` | FEA-PROVISIONAL |
| 机构可观性 + 像散装调方案 | `products/sensitivity/observability.py` | FEA-PROVISIONAL |
| 端到端管线（俯仰扫描 → 补偿表 → FEA 请求） | `products/pipeline/` | mock / 待硬件 API |
| 硬件 API 接口层（相机/望远镜终端/补偿机构） | `products/acquisition/` | 接口已定义，mock 可用 |
| 桌面 GUI（6 页签 + 3D 可视化 + 导出） | `desktop/` | GUI-BETA |

## 2. 架构

```text
desktop/gui.py (PySide6, 6 tabs)
   │  QProcess job.json/result.json（可取消）
desktop/worker.py  无 Qt 依赖的任务分发
   │
products/  ccd · preprocess · optics · pd · sensitivity · zemax · control ·
           acquisition(接口+mock) · pipeline(编排) · jobs · fea
   │
data/     zemax/（真实 ZOS-API 结果）· fea_cases/（Abaqus 工况 + manifest）
framework/ numerics/   C++ 通用框架与数值库（GTest/Benchmark/pybind11）
```

详细分层与契约见 [`docs/架构与模块说明.md`](docs/架构与模块说明.md)，
逐文件注释见 [`docs/代码模块注释索引.md`](docs/代码模块注释索引.md)。

## 3. 快速开始

```powershell
# 打包版（推荐内部演示）
dist\TelescopeEngineeringQtNative\TelescopeEngineeringQtNative.exe
# 或源码
py -3.13 desktop\qt_app.py
```

端到端管线（当前用 mock 硬件，输出补偿表与 FEA 请求）：

```powershell
py -3.13 scripts\run_pipeline.py --elevations 15,45,75 --out results\pipeline\demo
```

WSL/Linux（C++/CUDA/HPC 开发）：

```bash
bash scripts/setup_wsl_dev.sh --verify   # 幂等：apt + .venv + CUDA torch + 自检
cmake --preset linux-cuda
cmake --build --preset linux-cuda
ctest --preset linux-cuda --output-on-failure
```

环境版本、刻意不装的重复工具与排错见
[`docs/WSL开发环境.md`](docs/WSL开发环境.md)；HPC/AI4S 差距与路线见
[`docs/代码审阅-2026-11-22.md`](docs/代码审阅-2026-11-22.md)。

重新生成数据（需要许可）：

```powershell
# Zemax 3×5 S
py -3.13 scripts\zos_sensitivity.py --out data\zemax --surface 4 --delta 0.01
# Abaqus M1 板级 5 工况（正式数据用 seed=30 —— 缺省会得到 seed=45 的不同网格）
$env:TEL_ABAQUS_OUT="$PWD\data\fea_cases\abaqus_m1"
$env:TEL_ABAQUS_SEED='30'
D:\SIMULIA\Commands\abq2025.bat cae noGUI=scripts\abaqus_m1_whiffletree.py
```

## 4. 测试

```powershell
py -3.13 -m unittest discover -s products/tests
$env:QT_QPA_PLATFORM='offscreen'; py -3.13 -m unittest discover -s desktop/tests
py -3.13 scripts\check_doc_links.py
$env:QT_QPA_PLATFORM='offscreen'; py -3.13 desktop\e2e_smoke.py
```

基线见 [`docs/测试矩阵与回归.md`](docs/测试矩阵与回归.md)。

## 5. 数据与标签

| 数据 | 来源 | 标签 |
|---|---|---|
| `data/zemax/sensitivity_zemax.json` | OpticStudio 2023 R1 ZOS-API（真实处方） | EXTERNAL-CHECK |
| `data/fea_cases/abaqus_m1/` | Abaqus 2025 板级弹簧垫模型 | FEA-PROVISIONAL |
| `data/fea_cases/abaqus_design/` | 紫台文档参数（OD2560/±5 K） | FEA-PROVISIONAL |
| `release/telescope_sim_inputs_*/` | 仿真前置数据包（ZMX+FEA+Zemax S） | - |
| 实测 S | 台架 CCD+PD | UNVERIFIED |

不要用合成/代理数据替代实测结论；标签随数据一起引用。

## 6. 文档

- [`docs/README.md`](docs/README.md) 文档总索引（CURRENT / REFERENCE / ARCHIVED）
- [`docs/架构与模块说明.md`](docs/架构与模块说明.md)
- [`docs/测试矩阵与回归.md`](docs/测试矩阵与回归.md)
- [`docs/实测Zemax与Abaqus建模报告.md`](docs/实测Zemax与Abaqus建模报告.md)
- [`docs/实测灵敏度标定设计.md`](docs/实测灵敏度标定设计.md)
- [`docs/紫台设计参数提取.md`](docs/紫台设计参数提取.md)
- 历史材料：[`docs/archive/`](docs/archive/)、`archive/legacy-web/`

## 7. 已知限制

1. 无望远镜控制终端/相机厂商 API：`products/acquisition/` 定义接口并提供 mock，
   接入时只需实现三个适配器；
2. 无工况图与俯仰实测：管线接受 elevation 列表，补偿表按角度输出，待实测填充；
3. Abaqus 文档版为平面环盘（本机 CAE 旋转特征不可用），锥面矢高未建模；
4. Zemax S 覆盖 Z4/Z8/Z9、单视场 0.01°、0.7 µm；
5. 平移机构对像散/彗差盲区已由可观性分析删除，需 FEA 装调方案补偿。
