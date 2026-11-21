## 工程化 MVP（L1–L5）

在原有相位差反演之外，仓库增加了可独立构建/测试的工程化层：

| 层 | 目录 | 内容 |
|---|---|---|
| L1–L3 | `numerics/` | Eigen LU/QR/SVD、radix-2 FFT、RK4；GTest 10 项、Google Benchmark、ASan、pybind11 |
| L4 | `pipeline/` | CCD 定标/星点/帧差/Hough/WCS/评估（numpy+scipy+astropy） |
| L5 | `ml/`、`web/` | PyTorch 可微相位差前向、CNN、传统对比；只读可视化 demo |
| 部署 | `Dockerfile`、`.github/workflows/ci.yml` | CPU MVP 镜像 + GitHub Actions |

快速验收：

```bash
# 数值库（Eigen/GTest/Benchmark/pybind11）
cmake -S numerics -B build-numerics -DCMAKE_BUILD_TYPE=Release
cmake --build build-numerics -j4
ctest --test-dir build-numerics --output-on-failure

# CCD 管线
python3 -m pipeline.run --config configs/pipeline.json --force-sim

# 学习型反演
python3 -m ml.train_cnn --epochs 20 --samples 2048 --batch 64 --n 64
python3 -m ml.compare --samples 4 --starts 16 --max-nfev 100

# 可视化
python3 web/serve.py --port 8000
```

文档索引：`docs/tutorials/README.md`；4–6 周路线：`docs/tutorials/00_工程化MVP路线.md`；
数值库教程：`docs/tutorials/01_C++数值库.md`；可微前向/CNN/对比：`02`–`04`。

> 所有 ML/管线数字都是**仿真**结果，不是 2.5 m 望远镜实测；边界写在每篇 tutorial 末尾。

