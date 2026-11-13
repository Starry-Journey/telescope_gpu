# 教程索引

每个脚本/模块至少一篇 tutorial，包含：**科学原理 → 代码设计 → 技术栈 → 运行 → 结果 → 边界**。

| # | 教程 | 对应代码 | 你会学到 |
|---:|---|---|---|
| 00 | [工程化 MVP 路线](00_工程化MVP路线.md) | 全仓库 | 4–6 周计划、L1–L5、验收闸门 |
| 01 | [C++ 数值库](01_C++数值库.md) | `numerics/` | LU/QR/SVD、FFT、RK4、GTest、Benchmark、pybind11 |
| 02 | [可微相位差前向](02_可微相位差前向.md) | `ml/phase_torch.py` | 傅里叶光学、Zernike、torch.fft、autograd |
| 03 | [CNN 训练](03_CNN训练.md) | `ml/dataset.py`、`model.py`、`train_cnn.py` | 数据生成、归一化、CNN、训练曲线 |
| 04 | [传统对比](04_传统对比.md) | `ml/compare.py` | 多起点最小二乘、精度/速度/鲁棒性、two-stage |
| 05 | [工程化与部署](05_工程化与部署.md) | CMake、Docker、CI、web | L1–L5 落地、验收命令、坑 |
| 07 | [脚本 → 教程索引](07_脚本索引.md) | 全部脚本 | 每个脚本对应哪篇教程 |
| 08 | [CNN 初值 + BOBYQA](08_CNN初值BOBYQA.md) | `ml/two_stage.py` | 34 vs 1823 次评估的 two-stage 证明 |
| 09 | [FastAPI + React 小程序](09_FastAPI_React科学计算小程序.md) | `web/` | 仿真/反演 UI、API、部署 |

## 其它文档

- CCD 管线 32 页 Word/PDF：`docs/CCD星场处理管线说明.docx` / `.pdf`
- 模块注释索引：`docs/代码模块注释索引.md`
- 原相位差项目：`docs/代码说明.md`
- Docker 构建记录：`docs/Docker构建记录.md`
- 管线 README：`pipeline/README.md`
- 数值库 README：`numerics/README.md`
- 学习型反演 README：`ml/README.md`
- Web demo：`web/README.md`

## 读法

- **30 分钟**：00 + 01 的“构建/测试” + 04 的“结果表”；
- **2 小时**：01 → 02 → 03 → 04，跳过推导；
- **动手**：按 `00_工程化MVP路线.md` 的周表，每周改一个函数 + 加一个测试；
- **验证**：按 `08_CNN初值BOBYQA.md` 和 `05_工程化与部署.md` 的命令复现数字；
- **边界**：每篇末尾的“边界”一节引用时必须保留。
