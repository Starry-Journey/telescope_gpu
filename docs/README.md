# 文档索引（当前架构的单一事实源）

> **当前架构**：PySide6 原生桌面（`desktop/`）+ worker 子进程 + `products/` 服务层。
> 采集侧新增 `products/acquisition`、preprocess、pipeline 三段（厂商接口 + 显式 mock
> → CCD 预处理 → 端到端编排）。
> **旧 Web/React + FastAPI 已删除**（代码在 `archive/legacy-web/`，文档在
> `docs/archive/legacy-web/`）；旧 C++/ML 报告只在 `docs/archive/` 保留追溯。
> 本索引中的每个条目都是可点开的仓库内链接；状态标签：`CURRENT`（描述当前实现）/
> `REFERENCE`（仍可参考的背景/边界）/ `ARCHIVED`（仅追溯）。

## 1. 架构与运行

| 文档 | 一句话用途 | 状态 |
|---|---|---|
| [架构与模块说明.md](架构与模块说明.md) | 分层、端到端数据流、厂商接口契约、数据标签（权威架构文档） | CURRENT |
| [Qt桌面前端.md](Qt桌面前端.md) | PySide6 页签、QProcess/job 协议、取消语义、onedir 打包 | CURRENT |
| [代码模块注释索引.md](代码模块注释索引.md) | 逐文件职责 / 关键函数 / 输入输出 / 注释入口 | CURRENT |
| [后续优化方向.md](后续优化方向.md) | P0–P5 方向与验收记录；Web/ML 清理项已标注“已归档/已删除” | REFERENCE |

## 2. 模块说明（每个代码模块一篇 README）

| 文档 | 一句话用途 | 状态 |
|---|---|---|
| [products/README.md](../products/README.md) | 产品服务层总览：子包边界、调用链、公共常量、标签 | CURRENT |
| [products/acquisition/README.md](../products/acquisition/README.md) | 厂商接口（望远镜/相机/机构）+ 显式 mock + 单位/时序/错误码 | CURRENT |
| [products/preprocess/README.md](../products/preprocess/README.md) | CCD 预处理：定标/坏像元/饱和/宇宙线/背景/星点/配准 | CURRENT |
| [products/pipeline/README.md](../products/pipeline/README.md) | 端到端编排：俯仰扫描 → 采集 → PD → 实测 S → 补偿表 → FEA 请求 | CURRENT |
| [products/ccd/README.md](../products/ccd/README.md) | CCD 合成仿真 + 真实 FITS → PSF 提取 | CURRENT |
| [products/optics/README.md](../products/optics/README.md) | 唯一 Zernike 项表 + Fourier 光学前向 PSF | CURRENT |
| [products/pd/README.md](../products/pd/README.md) | 相位差反演（BOBYQA 多起点，3/6 项）与 256² 批处理 | CURRENT |
| [products/sensitivity/README.md](../products/sensitivity/README.md) | FEA 面形 → Zernike/环境 S、实测 S、可观测性 | CURRENT |
| [products/zemax/README.md](../products/zemax/README.md) | ZMX 解析、光路出图、ZOS-API/rayoptics/自研 tracer 边界 | CURRENT |
| [products/control/README.md](../products/control/README.md) | S → u：加权最小二乘 + Tikhonov + 边界投影 | CURRENT |
| [products/jobs/README.md](../products/jobs/README.md) | SQLite 作业队列（提交/claim/完成/失败） | CURRENT |
| [products/fea/README.md](../products/fea/README.md) | CalculiX 简化主镜板级模型（无商业许可时的 FEA） | CURRENT |
| [desktop/README.md](../desktop/README.md) | Qt 入口/GUI/worker/TaskRunner/协议/自检 | CURRENT |
| [scripts/README.md](../scripts/README.md) | 外部计算、打包、性能、文档检查类脚本清单 | CURRENT |
| [data/README.md](../data/README.md) | FEA 工况/Zemax S/PSF 数据的目录、单位与标签 | CURRENT |

## 3. 设计与建模

| 文档 | 一句话用途 | 状态 |
|---|---|---|
| [紫台设计参数提取.md](紫台设计参数提取.md) | 从设计文档提取的 M1 参数、缺失项假设、Abaqus 设计工况复现 | CURRENT |
| [实测Zemax与Abaqus建模报告.md](实测Zemax与Abaqus建模报告.md) | 真实 ZOS-API 3×5 S 与 Abaqus M1 面形（EXTERNAL-CHECK / FEA-PROVISIONAL） | CURRENT |
| [实测灵敏度标定设计.md](实测灵敏度标定设计.md) | 用 CCD + PD 在仪器上测 S 的测量协议与精度预算 | CURRENT |
| [Zemax3A.md](Zemax3A.md) | ZMX 解析能力、ZOS-API 控制量/单位与“不伪造数值”边界 | REFERENCE |
| [自研光线追迹.md](自研光线追迹.md) | conic tracer 实现与 rayoptics 差异（TRACER-EXPERIMENTAL） | REFERENCE |
| [数据来源与真实FITS.md](数据来源与真实FITS.md) | 真实 FITS/Zemax/FEA 数据与合成数据的口径、可做/不可做 | REFERENCE |

## 4. 验证与测试

| 文档 | 一句话用途 | 状态 |
|---|---|---|
| [测试矩阵与回归.md](测试矩阵与回归.md) | 本地测试命令、按功能覆盖矩阵、打包验证清单 | CURRENT |
| [perf_notes.md](perf_notes.md) | 性能基准与优化前后对比（Zernike/前向/CCD/拟合、内存） | CURRENT |
| [实测灵敏度标定-实施报告.md](实测灵敏度标定-实施报告.md) | 实测 S 估计器的实现、闭环证据与台架待办 | CURRENT |
| [最终验收标准.md](最终验收标准.md) | Gate 定义、状态标签语义、验收矩阵（冻结版） | REFERENCE |

## 5. 历史归档

| 文档 | 一句话用途 | 状态 |
|---|---|---|
| [archive/README.md](archive/README.md) | 归档总表：每个归档文件的原因与取代者 | ARCHIVED |
| [archive/legacy-web/Web小程序技术说明.md](archive/legacy-web/Web小程序技术说明.md) | 旧 FastAPI + React 小程序说明（含 docx/pdf） | ARCHIVED |
| [archive/legacy-web/框架与Web集成.md](archive/legacy-web/框架与Web集成.md) | 旧 Web API/前端集成契约 | ARCHIVED |
| [archive/reports/工程化技术说明_代码对齐版.md](archive/reports/工程化技术说明_代码对齐版.md) | 旧工程化总说明（C++ 核心 + Web 时代） | ARCHIVED |
| [archive/reports/模块1_CCD数据管线.md](archive/reports/模块1_CCD数据管线.md) | 旧模块1技术说明（docx/pdf 同目录） | ARCHIVED |
| [archive/legacy-architecture/目录与模块架构.md](archive/legacy-architecture/目录与模块架构.md) | 旧目录树（`include/ src/ apps/ framework/` 已不存在） | ARCHIVED |
| [archive/legacy-architecture/参数与控制量对照.md](archive/legacy-architecture/参数与控制量对照.md) | 旧 Web 页面参数/单位对照 | ARCHIVED |

## 6. 维护规则

1. **唯一事实源**：架构变化先改 `docs/架构与模块说明.md` 与对应模块 `README.md`，
   再改报告/索引；旧版本移入 `docs/archive/` 并在 `docs/archive/README.md` 登记；
2. **不伪造数据口径**：每张图/表/结论带标签（`CORE-VERIFIED` / `SYNTHETIC-VALIDATED` /
   `EXTERNAL-CHECK` / `FEA-PROVISIONAL` / `TRACER-EXPERIMENTAL` / `GUI-BETA` / `UNVERIFIED`），
   采集侧 mock 一律带 `mode=mock`，不得当实测证据；
3. **链接必须可点**：提交前运行 `py -3.13 scripts/check_doc_links.py`（归档目录自动跳过）；
4. **文档不再描述已删除的 Web 路径**：需要追溯时引用 `docs/archive/legacy-web/`。
