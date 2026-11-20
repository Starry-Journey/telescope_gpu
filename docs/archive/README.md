# 历史归档（ARCHIVED）

> 本目录只做追溯，**不作为当前结论**。当前架构与文档入口见 `docs/README.md`。
> 2026-11-20 文档重构：项目已从“Web/React + FastAPI”迁到“PySide6 桌面 + worker
> 子进程 + products/ 服务层”。旧 Web 包不再提供启动入口，旧 C++/CUDA 主求解器与
> `framework/` 源码已不在仓库中，因此相关文档全部移入本目录。归档采用“移动而非删除”。

归档分组：

- `legacy-web/`：Web/桌面壳（FastAPI + React、PyWebView、Docker 部署）时代的文档；
- `reports/`：旧版技术报告、旧模块说明、C++/ML 专题说明（多为 .docx/.pdf 生成物）；
- `legacy-architecture/`：旧目录/架构/路线/性能记录。

说明：带 `-根目录副本` 后缀的文件与原文件内容逐字节相同，是重构前 `docs/`
根目录与 `docs/archive/` 各存一份造成的重复副本，保留副本以遵守“只归档不删除”。

## 1. legacy-web/（旧 Web / 桌面壳路径）

| 文件 | 归档原因 | 被什么取代 |
|---|---|---|
| `Web小程序技术说明.md` / `.docx` / `.pdf` | FastAPI + React + Plotly 小程序的技术说明；`web/` 包已删除，不再有 Web 启动入口 | `docs/Qt桌面前端.md`、`desktop/README.md` |
| `框架与Web集成.md`（含 `-根目录副本`） | 旧 Web API / 前端集成契约（`/api/*`、页面状态） | `docs/架构与模块说明.md` |
| `桌面GUI.md`（含 `-根目录重复副本`） | PyWebView 包壳方案（内嵌 FastAPI + React），已废弃 | `docs/Qt桌面前端.md` |
| `Docker构建记录.md` | Web/CI 时代的容器交付尝试（本机未构建成功）；当前交付形态为本地 Qt onedir EXE | `docs/Qt桌面前端.md`（打包章节） |

## 2. reports/（旧技术报告与旧模块说明）

| 文件 | 归档原因 | 被什么取代 |
|---|---|---|
| `大口径望远镜重力变形标定技术研究_技术报告_代码对齐版.docx`、`_v3.docx`、`_old.docx` | 旧版主报告，口径为“C++/CUDA 主求解器 + Web 前端”，与当前桌面/产品层不一致 | `docs/实测Zemax与Abaqus建模报告.md`、`docs/实测灵敏度标定-实施报告.md` |
| `技术报告_代码对齐版_v2.docx` / `.pdf`（含副本） | 更早一版主报告，已被上面两份实测报告取代 | 同上 |
| `工程化技术说明_代码对齐版.md` / `.docx` / `.pdf`（含副本） | 旧工程化总说明：C++ 核心、FastAPI/React、旧 CI 目录均不再存在 | `docs/架构与模块说明.md`、`docs/测试矩阵与回归.md` |
| `模块1_CCD数据管线.md` / `.docx` / `.pdf` | 旧模块1说明，按旧 `pipeline/realistic.py` + Web API 口径写 | `products/ccd/README.md` |
| `模块2_相位差反演.md` / `.docx` / `.pdf` | 旧模块2说明，混用 C++ 主求解器与 Web 请求口径 | `products/pd/README.md` |
| `模块3_灵敏度与控制.md` / `.docx` / `.pdf` | 旧模块3说明，三套 S 的 Web 页面/API 描述已失效 | `products/sensitivity/README.md`、`products/control/README.md` |
| `学习型反演技术说明.md` / `.docx` / `.pdf`（含副本） | CNN/可微前向研究材料；`ml/` 研究层已移入 `archive/ml-research/`，桌面不依赖 | `ml/README.md`（研究层现状） |
| `数值库技术说明.md` / `.docx` / `.pdf` | C++ numerics 教学/参考库说明，未接入当前桌面主路径 | `numerics/README.md`（独立库仍保留） |
| `框架技术说明.md` / `.docx` / `.pdf`、`框架设计说明.md` | C++ `framework/` 契约与 demo，源码已不在仓库 | `docs/架构与模块说明.md` |
| `FEA案例与灵敏度.md` / `.docx` / `.pdf` | 合成 FEA 参考工况说明，已被真实 Abaqus 求解结果取代 | `docs/实测Zemax与Abaqus建模报告.md` |
| `代码说明.md` / `.docx` | 旧 C++ 相位差求解器逐函数说明（第 4/5 章调用链） | `docs/代码模块注释索引.md`（当前 Python 产品层） |
| `CCD星场处理管线说明.md` / `.docx` / `.pdf` | 旧离线 CCD 管线（`pipeline/`）报告 | `products/ccd/README.md`、`pipeline/README.md` |

## 3. legacy-architecture/（旧架构 / 路线 / 性能）

| 文件 | 归档原因 | 被什么取代 |
|---|---|---|
| `目录与模块架构.md` | 目录树仍写 `include/ src/ apps/ framework/ web/`，这些目录已删除或归档 | `docs/架构与模块说明.md` |
| `项目架构总览.md` | 旧总览仍把 `web/*_service.py` 当兼容层、把 `framework/` 当核心层 | `docs/README.md`、`docs/架构与模块说明.md` |
| `模块化文档索引.md` | 仅剩“重定向到 docs/README.md”的 stub | `docs/README.md` |
| `参数与控制量对照.md` | 按旧 Web 页面（③ 框架 / ④ FEA）写的参数表，且把 framework 当实现 | `products/control/README.md`、`docs/架构与模块说明.md` |
| `256²性能预算.md` | 面向 Web 同步请求的 PD 性能预算，环境为 WSL2 CPU；当前由桌面 worker/批处理承担 | `products/pd/README.md`（性能边界）；后续性能记录由 `docs/perf_notes.md` 维护 |
| `POPPY_PSF.md` | POPPY 外部对照历史（FWHM 76.4 vs 21.5 px，未对齐），仅记录失败尝试 | `docs/实测Zemax与Abaqus建模报告.md` 的外部对照结论 |
| `CalculiX简化主镜.md` | CalculiX 简化板/弹簧模型的多轮历史记录，已被真实 Abaqus 结果取代 | `docs/实测Zemax与Abaqus建模报告.md` |
| `ZOS与Abaqus实测.md` | 2026-11-16 的 license/API 探测失败记录；11-19 已用真实许可跑通 | `docs/实测Zemax与Abaqus建模报告.md` |
| `开源替代路线.md` | 早期“用 rayoptics/CalculiX/POPPY 替代商业软件”的路线评估，执行后已收敛 | `docs/实测Zemax与Abaqus建模报告.md`、`products/zemax/README.md` |

### 3.1 readme-snippets/（根目录 README 片段）

| 文件 | 归档原因 | 被什么取代 |
|---|---|---|
| `readme_3a_snippet.md`、`readme_docs_snippet.md`、`readme_engineering_section.md`、`readme_fea_snippet.md`、`readme_modules_snippet.md` | 旧 `update_readme_*.py` 一次性脚本生成/使用的 README 片段，内容按 Web API 与旧目录写，且指向已归档文档 | `docs/README.md`、各模块 `README.md` |

## 4. 归档规则

1. 旧版本不覆盖、不删除：移入 `docs/archive/` 对应分组，并在本表登记“原因 + 取代者”；
2. 归档文档里的链接与数字一律不作为当前结论；需要引用时先确认当前文档；
3. 代码归档在仓库根的 `archive/legacy-web/`、`archive/ml-research/`，与本目录分开；
4. `~$...docx` 是 Office 临时锁文件（非文档），重构时已清理。
