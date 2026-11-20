# CalculiX 简化主镜实测

> 2026-11-17，Windows conda-forge `calculix 2.23`

## 1. 安装

```bash
conda create -y -n ccx -c conda-forge calculix
conda run -n ccx ccx -v   # CalculiX 2.23
```

## 2. 模型

`products/fea/calculix_m1.py`：

- 环形 Zerodur 盘 OD 2500 / ID 700 / t 260 mm；
- 3600 节点、2736 个 C3D8 单元；
- 36 底支撑（6-12-12-6，固定）；
- 24 侧支撑（中厚，径向固定）；
- 重力 0°/90°。

## 3. 运行

```bash
python products/fea/calculix_m1.py --out data/fea_cases/calculix
conda run -n ccx ccx -i data/fea_cases/calculix/m1_0deg
conda run -n ccx ccx -i data/fea_cases/calculix/m1_90deg
```

输出：`*_top.csv`（720 个顶面节点，`x,y,z,u1,u2,dz_nm`）。

## 4. 结果

| case | dz mean | dz std | dz min | dz max |
|---|---:|---:|---:|---:|
| 0 deg | 0.0 nm | 30.7 nm | -99.9 nm | +99.9 nm |
| 90 deg | -170.4 nm | 79.4 nm | -405.0 nm | -80.2 nm |

66 项 Zernike 上传结果：

```text
0 deg:  Z8 ≈ -17.97 nm, Z4 ≈ 0, residual 2.15 nm
90 deg: Z4 ≈ -75.85 nm, Z8 ≈ 0, residual 4.05 nm
```

## 5. 解释与边界

- 这是**真实 CalculiX 解**（非合成 Zernike），但几何/支撑是简化板模型；
- 支撑固定节点会过约束，RMS 30/79 nm 不能直接与设计 6.5/23 nm 比较；
- 下一步：弹簧/接触支撑、网格收敛、热工况；
- 数据在 `data/fea_cases/calculix/`，可直接上传 `/api/fea/upload`。

## 6. 复现性

- `m1_0deg.inp` / `m1_90deg.inp` 已保存；
- `spooles.out`、`.dat` 等可重建；
- 记录 CalculiX 2.23、3600 节点、2736 单元。


## 6. 弹簧支撑与网格收敛（初步）

`products/fea/calculix_m1.py` 现在支持：

```bash
python products/fea/calculix_m1.py --support spring --k 1e7
```

- 36 个底支撑改为 `SPRING1`（z 向、k=1e7 N/mm）；
- 24 个侧支撑仍径向固定；
- 其他同上。

结果（0°/90°）：

| 工况 | 支撑 | 原始 dz std | 去平面后 std |
|---|---|---:|---:|
| 0° | 固定 | 30.7 nm | 22.2 nm |
| 90° | 固定 | 79.4 nm | 79.4 nm |
| 0° | z 弹簧 1e7 | 8.1 nm | 4.1 nm |
| 90° | z 弹簧 1e7 | 132.5 nm | 132.5 nm |
| 0° | z 弹簧 1e8 | 8.2 nm | ~4 nm |
| 90° | z 弹簧 1e8 | 97.3 nm | 97.3 nm |

初步判断：

- **0° 可以接近 6.5 nm**（去平面后约 4 nm）；
- **90° 仍远大于 23 nm**，单靠 z 向点弹簧不够；
- 需要 whiffletree/接触/侧向刚度，而不是继续调 k。

网格：

| NR/NT/NZ | 节点 | 0° 原始 std | 去平面 std |
|---|---:|---:|---:|
| 20/36/5 | 3600 | 30.7 nm | 22.2 nm |
| 28/48/7 | 9408 | 63.0 nm | 43.5 nm |

固定点支撑对网格敏感，**尚未收敛**；不能把 30.7 或 63 nm 当设计值。
下一阶段需要弹簧/接触支持下的网格收敛，而不是继续用固定节点。

## 7. 边界

- 真实 CalculiX 解，但模型是板级简化；
- 固定点/单点弹簧都会过约束或欠约束；
- 90° 未接近设计 23 nm；
- 不能覆盖原设计合成 M1 工况或宣称满足指标。

## 8. 侧向弹簧与网格敏感性（第二轮）

`--support spring --k 1e7 --k-side 1e6`：

- 36 个底支撑 z 向 SPRING1；
- 24 个侧支撑全局 x/y SPRING1（近似侧向 whiffletree）；
- 不再固定侧节点。

| 工况 | 原始 std | 去平面 std |
|---|---:|---:|
| 0° | 4.26 nm | 3.58 nm |
| 90° | 132.7 nm | 132.7 nm |

侧向弹簧让 0° 更接近 6.5 nm；90° 仍远大于 23 nm。

网格：

| 网格 | 节点 | 0° 原始/去平面 | 90° 原始/去平面 |
|---|---:|---:|---:|
| 20/36/5 | 3600 | 30.7 / 22.2 nm | 79.4 / 79.4 nm |
| 28/48/7 | 9408 | 53.5 / 40.2 nm | 140.1 / 140.1 nm |

**仍未收敛**：固定/点弹簧的支撑节点随网格变化，不能把某一网格值当设计结果。
接触/分布弹簧/whiffletree 是下一步，而不是继续调 k。

## 9. 结论

- 0° 的弹簧模型已到 4 nm 量级，可作为“接近 6.5 nm”的初步证据；
- 90° 仍失败，需要真实 whiffletree/接触；
- 网格敏感性说明当前支撑模型不足；
- 不覆盖原设计合成 M1 工况，不宣称满足指标。

## 10. Distributed spring（第三轮，关键结果）

`--support distributed`：

- 所有底面节点 z 向 SPRING1（弹性地基）；
- 所有外环节点 x/y SPRING1；
- 总刚度 `k=1e9`、`k_side=1e8` N/mm，按节点均分；
- 不是 36 点 whiffletree，也不是接触。

结果：

| 网格 | 0° raw / 去平面 | 90° raw / 去平面 |
|---|---:|---:|
| 20/36/5（3600） | 4.23 / **3.23 nm** | 8.90 / **8.90 nm** |
| 28/48/7（9408） | 4.08 / **2.82 nm** | 8.93 / **8.93 nm** |

两个网格差 <0.5 nm，明显好于点弹簧（30.7→53.5 nm）。

设计锚点：6.4873 nm（水平）/23.199 nm（垂直）。分布式弹簧模型低于目标，
但这是**弹性地基近似**，不是真实 whiffletree/接触，不能据此宣称设计已验证。

66 项 Zernike：

```text
0°: Z8 ≈ -0.74 nm, Z4 ≈ 0, 残差 0.37 nm
90°: Z4 ≈ -9.04 nm, Z8 ≈ 0, 残差 0.07 nm
```

## 11. 结论（更新）

- 分布式弹簧让 0°/90° 都进入设计锚点以下；
- 网格收敛性显著改善；
- 仍需真实 36 点 whiffletree/接触验证，不能把弹性地基当最终支撑；
- 数据在 `data/fea_cases/calculix_distributed/`。
