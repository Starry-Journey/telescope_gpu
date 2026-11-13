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
- `ml/results/comparison.csv/json/png`：CNN vs 多起点最小二乘。

默认结果（4 个独立测试样本）：

| 方法 | c4 RMSE | c8 RMSE | c9 RMSE | 波前 RMS 误差 | 单样本耗时 |
|---|---:|---:|---:|---:|---:|
| CNN | 0.0266 | 0.0205 | 0.0165 | 0.0290 | 1.1 ms |
| 多起点 LSQ | 0.0204 | 0.0307 | 0.0075 | 0.0098 | 94.8 s |

这是**仿真、同分布**结果，不是实测；传统法和 CNN 各有胜负，推荐
“CNN 初值 + 传统精修”的 two-stage。

详细教程：

- `docs/tutorials/02_可微相位差前向.md`
- `docs/tutorials/03_CNN训练.md`
- `docs/tutorials/04_传统对比.md`
