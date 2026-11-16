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
