# ml：可微相位差 + CNN + 传统对比

依赖：`torch`（CPU 版足够）、numpy、scipy、matplotlib。

```bash
python3 -m ml.train_cnn --epochs 20 --samples 2048 --batch 64 --n 64
python3 -m ml.compare --samples 4 --starts 16 --max-nfev 100
python3 ml/plot_training.py
```

输出：

- `ml/runs/cnn.pt`：模型权重 + 归一化 scales + 数据配置；
- `ml/runs/train_log.csv`：逐 epoch 验证 RMSE；
- `ml/results/comparison.csv/json/png`：CNN vs 多起点最小二乘；
- `ml/results/two_stage.*`：two-stage 评估（与 `docs/data/two_stage.*` 是
  **同一次实验**，内容一致，仅换行符可能不同）；`docs/data/ml_comparison.csv`
  是 `ml/results/comparison.csv` 的文档快照。

默认结果（4 个验证集代表性样本——模型选择集，未做最终 held-out）：

| 方法 | c4 RMSE | c8 RMSE | c9 RMSE | 波前 RMS 误差 | 单样本耗时 |
|---|---:|---:|---:|---:|---:|
| CNN | 0.0281 | 0.0203 | 0.0160 | 0.0382 | 1.7 ms |
| 多起点 LSQ | 0.0217 | 0.0311 | 0.0081 | 0.0387 | 32.3 s |

复现：`python3 -m ml.compare --samples 4 --starts 16 --max-nfev 100`。
波前 RMS 误差按 `‖c_pred − c_true‖`（正交基上的波前 RMS）计算；旧版本用
`|‖c_pred‖ − ‖c_true‖|`，会低报约 45%，表中 0.038 才是可比的真值。

这是**仿真、同分布**结果，不是实测；传统法和 CNN 各有胜负，推荐
“CNN 初值 + 传统精修”的 two-stage。

详细教程：

- `docs/tutorials/02_可微相位差前向.md`
- `docs/tutorials/03_CNN训练.md`
- `docs/tutorials/04_传统对比.md`
