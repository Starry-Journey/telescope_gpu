# scripts/ — 环境、数据与验证脚本

这些脚本不是运行时依赖；每个脚本都能独立执行，输出写在 `results/` 或
`data/`（两者均被 `.gitignore` 忽略，需要时用 `scripts/package_sim_inputs.py`
打包交付）。

## 环境与质量

| 脚本 | 用途 |
|---|---|
| [setup_wsl_dev.sh](setup_wsl_dev.sh) | WSL/Linux 开发环境一键安装（apt + uv venv + CUDA torch），幂等 |
| [check_doc_links.py](check_doc_links.py) | 检查文档里反引号路径是否存在（CI 门禁） |
| [benchmark.py](benchmark.py) | Python 光学/CCD/灵敏度基准，输出 `results/bench/*.json` |
| [benchmark_pd.py](benchmark_pd.py) | 相位差反演的 CPU 预算（256² / 多起点） |

## 外部数据生成（需要商业软件许可）

| 脚本 | 需要 | 用途 |
|---|---|---|
| [zos_sensitivity.py](zos_sensitivity.py) | OpticStudio ZOS-API | 生成真实 3×5 灵敏度矩阵 S |
| [zos_s_scope.py](zos_s_scope.py) | OpticStudio ZOS-API | 检查 S 的线性适用范围（步长/视场/波长） |
| [zos_probe.ps1](zos_probe.ps1) | Windows PowerShell + ZOS-API | 探测 ZOS-API 环境 |
| [zemax_zos_worker.py](zemax_zos_worker.py) | OpticStudio ZOS-API | ZOS-API 子进程 worker |
| [abaqus_m1_whiffletree.py](abaqus_m1_whiffletree.py) | Abaqus CAE | 36 点 whiffletree M1 工况 |
| [abaqus_m1_design.py](abaqus_m1_design.py) | Abaqus CAE | 紫台文档参数（OD2560/±5 K）工况 |
| [abaqus_m1_simple.py](abaqus_m1_simple.py) | Abaqus CAE | 最小可复现板级模型 |
| [abaqus_m1_fit.py](abaqus_m1_fit.py) | — | 把 Abaqus 面形拟合为 Zernike |
| [abaqus_k_sweep_summary.py](abaqus_k_sweep_summary.py) | — | 刚度 k 扫描区间汇总 |
| [register_abaqus_cases.py](register_abaqus_cases.py) | — | 把新工况登记进 `data/fea_cases/manifest.json` |

## 可视化与外部对照

| 脚本 | 用途 |
|---|---|
| [zmx_layout.py](zmx_layout.py) | ZMX 2D 截面 / 3D 光路出图 |
| [plot_sensitivity.py](plot_sensitivity.py) | S 矩阵热力图 |
| [visualize.py](visualize.py) | FEA 面形/结构图 |
| [make_pipeline_doc_figs.py](make_pipeline_doc_figs.py) | 文档用管线示意图 |
| [poppy_psf.py](poppy_psf.py) / [poppy_compare.py](poppy_compare.py) / [poppy_otf_compare.py](poppy_otf_compare.py) / [poppy_align.py](poppy_align.py) | POPPY 外部 PSF/OTF 对照（EXTERNAL-CHECK，不作为主结论） |

## 端到端与交付

| 脚本 | 用途 |
|---|---|
| [run_pipeline.py](run_pipeline.py) | 俯仰扫描 → 采集(mock) → PD → 补偿表 → FEA 请求 |
| [pd_batch_256.py](pd_batch_256.py) | 256² PSF 的多起点批量反演 |
| [fits_defocus_test.py](fits_defocus_test.py) | 真实 FITS 离焦栈 → PD 回归测试 |
| [dump_engineering_terms.py](dump_engineering_terms.py) | 导出工程 Zernike 项表 |
| [package_sim_inputs.py](package_sim_inputs.py) | 把 ZMX/FEA/S 打包成仿真输入包 |
| [run_telescope.cmd](run_telescope.cmd) | Windows 演示启动器 |

> 环境准备请优先用 `docs/WSL开发环境.md` 和 `scripts/setup_wsl_dev.sh`；
> 不要为单个脚本再装一套 conda/virtualenv。
