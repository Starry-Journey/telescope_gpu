# Zemax 与 Abaqus 实测建模报告（真实许可，2026-11-19）

状态：**EXTERNAL-CHECK**（Zemax，真实处方求解）／**FEA-PROVISIONAL**（Abaqus，板级模型）
脚本：`scripts/zos_sensitivity.py`、`scripts/abaqus_m1_whiffletree.py`
数据：`data/zemax/sensitivity_zemax.{json,csv}`、`data/fea_cases/abaqus_m1/*_top.csv`

---

## 0. 结论摘要

| 项 | 之前 | 现在 |
|---|---|---|
| M2 5-DOF 灵敏度 S | 代码里硬编码的 synthetic 代理矩阵 | **真实 OpticStudio ZOS-API 求解**（3×5） |
| M1 重力/热 FEA | 合成 CSV + CalculiX 简化板 | **真实 Abaqus 2025 求解**（5 工况，6410 顶面节点） |
| 环境 S（3×2） | 仅合成 | `environment_sensitivity(source="abaqus")` 走真实 FEA |

两个重要物理发现（都不是拟合出来的，是求解结果）：

1. **代理矩阵结构错误**：代理里 `m2_tilt_x/y` 影响 Z4（离焦），真光学里倾斜
   产生的是**彗差**（Z8/Z9）；且 `m2_focus` 符号相反、量级差 5.8 倍。
2. **热方向在 Z4/Z8/Z9 子空间近乎不可观测**：Zerodur ±20 K 均匀浸泡后
   Z4/Z8/Z9 变化 ~1e-4 波，环境 S 的热列≈0 → 该子空间几乎只有重力方向，
   矩阵条件数 5.4e4。要控温影响必须扩展观测项（更高阶像差或直接测温）。

## 1. 环境确认

```text
OpticStudio   Ansys Zemax OpticStudio 2023 R1.00
              D:\Program Files\Ansys Zemax OpticStudio 2023 R1.00
              ZOS-API standalone 可用（CreateNewApplication + Wavefront/Zernike 分析）
Abaqus        SIMULIA Established Products 2025
              D:\SIMULIA\Commands\abq2025.bat
              license "cae" 从 Flexnet localhost 检出，9998/9999 可用
Python        pythonnet 3.1.0（ZOS-API 通道）
模型          C:\Users\17424\Documents\Zemax\Autosave\Recovered\2.5m主系统.zmx
              （不复制进仓库；仓库只存求解结果与脚本）
```

ZOS-API 关键调用（脚本内已封装）：

```python
app  = ZOSAPI.ZOSAPI_Connection().CreateNewApplication()
sys_ = app.PrimarySystem; sys_.LoadFile(zmx, False)
ana  = sys_.Analyses.New_Analysis(AnalysisIDM.ZernikeStandardCoefficients)
ana.ApplyAndWaitForCompletion(); ana.GetResults().GetTextFile(txt)   # 报告为 UTF-16
```

## 2. Zemax：真实 M2 5-DOF 灵敏度

**方法**：中心差分，每个探针重新 `LoadFile`（避免坐标断点状态残留）：

```text
z(-Δ) → z(0) → z(+Δ),   S[:,j] = (z(+Δ) − z(−Δ)) / (2Δ)
Δ = 0.01 mm（平移）/ 0.01 mrad（倾斜），波长 0.7 µm，LDE 第 4 面 = M2
```

**术语映射（本次新确认，重要）**：项目的 Z8/Z9 是 (cos, sin) 彗差对：

| 项目标签 | Zemax Noll 项 | 含义 |
|---|---|---|
| Z4 | 4 | 离焦 |
| Z8 | **8** `(3ρ³−2ρ)COS` | x 彗差 |
| Z9 | **7** `(3ρ³−2ρ)SIN` | y 彗差 |

Noll 9 是**三叶草**，不是 y 彗差。早期若按 Z9→Noll 9 取数会把 y 彗差测成 0
（本次排查中确实先踩到，已修正并在脚本注释里固化）。

**结果**（waves / (mm 或 mrad)，基线 Z4=−0.0874，Z8=Z9=0）：

| 控制 | Z4 | Z8 | Z9 |
|---|---|---|---|
| m2_focus (−23.2842) | **−23.2842** | 0 | 0 |
| m2_decenter_x | 0 | **−1.6564** | 0 |
| m2_decenter_y | 0 | 0 | **−1.6564** |
| m2_tilt_x | 0 | 0 | **−1.3868** |
| m2_tilt_y | 0 | **+1.3868** | 0 |

**与合成代理对比**：

| 控制 | 代理列 | 真实列 | 差异 |
|---|---|---|---|
| m2_focus→Z4 | +4.040 | −23.284 | 符号相反、5.8× |
| m2_decenter_x→Z8 | −4.633 | −1.656 | 2.8× |
| m2_decenter_y→Z9 | −4.633 | −1.656 | 2.8× |
| m2_tilt_x/y | 影响 Z4 ±0.976 | 影响彗差 ∓1.387 | **结构不同** |

倾斜在代理里被当成"交叉耦合到离焦"，真实光学的倾斜是彗差——这是代理矩阵
最严重的建模错误，现已由真实求解替代。

## 3. Abaqus：M1 主镜板级 FEA

**模型**（`scripts/abaqus_m1_whiffletree.py`）：

```text
几何   环形 Zerodur 盘 OD 2500 / ID 700 / 厚 260 mm，轴向 Z
支撑   36 个底支撑（6-12-12-6，r=375/625/875/1125）+ 24 个侧支撑（中厚面）
       每个垫把载荷分布在节点斑上（底 70 mm / 侧 45 mm 半径），弹簧接地
       k_bottom = 1e6 N/mm/垫，k_side = 5e5 N/mm/垫
材料   E=90 GPa, ν=0.24, ρ=2.53e-9 t/mm³, α=5e-8 /K
网格   C3D8R，seed 30 mm，6410 个顶面节点，约 5 万节点
工况   gravity_0deg（光轴水平 +X）、gravity_90deg（光轴竖直 −Z）
       thermal_plus20K / thermal_minus20K（均匀 ±20 K）
       thermal_grad20K（假设的径向浸泡梯度：内孔 0 K → 外缘 +20 K，10 环带）
```

**为什么用"垫面分布"而不是单点弹簧**：单点弹簧是点载荷奇异性，网格越细位移
越大（实测 seed45/30/20 → RMS 3006/3793/5000 nm，不收敛）。把每个垫的刚度
分摊到垫面节点后收敛：

| 网格 | 顶面节点 | gravity_90 RMS | Z4 |
|---|---|---|---|
| seed 45 | 2584 | 1171.4 nm | −391.2 nm |
| seed 30 | 6410 | 1122.0 nm | −380.5 nm |

seed45→30 差 4%，取 seed 30 作正式数据。

**结果**（seed 30，Zernike 由 66 项最小二乘拟合，rho 归一化 1250 mm）：

| 工况 | RMS | 拟合残差 | 峰值 |
|---|---|---|---|
| gravity_0deg | 3.15 nm | 1.41 nm | 14.4 nm |
| gravity_90deg | 1121.98 nm | 4.95 nm | 1675.0 nm |
| thermal_plus20K | 270.33 nm | 0.48 nm | 274.0 nm |
| thermal_minus20K | 270.33 nm | 0.48 nm | 274.0 nm |
| thermal_grad20K | 167.01 nm | 0.87 nm | 251.4 nm |

读数要点：

- gravity_90 的 ~1.1 µm RMS **主要是刚体下沉（活塞）+ 离焦**，来自
  36 个垫共 3.6e7 N/mm 的轴向柔度；真正不可补偿的面形误差是残差 4.95 nm 量级。
- gravity_0（光轴水平）只有 3.15 nm：侧支撑分布到垫面后局部夹持效应消失。
- 均匀 ±20 K 严格对称（±同值），符合 α 近零的线性响应。
- thermal_grad 的 Z4 = 0.129 波（@+20 K 梯度），是热梯度真正的低阶响应。

**环境灵敏度 S（3×2）**：`environment_sensitivity(source="abaqus")`

```text
S = [ (z(90°) − z(0°))/90 ,  (z(+20K) − z(−20K))/40 ]
= [[-0.006681,  0.000003],
   [ 0.000021, -0.000000],
   [ 0.000001,  0.000000]]   waves/(deg, K)
rank = 2, cond = 5.4e4
梯度工况单独报告：S_grad = [0.006447, -5e-6, -3e-6] waves/K
```

即：**在这 3 个低阶像差上，温度方向几乎不可观测**（Zerodur 近零膨胀的必然结果）。
这不是 bug，而是设计结论：热控要么靠更高阶像差/直接测温，要么靠机械热隔离。

## 4. 集成（TelescopeEngineering 侧）

| 位置 | 变化 |
|---|---|
| `products/control/service.py` | `load_real_sensitivity()` 读 `data/zemax/sensitivity_zemax.json`；`sensitivity(source="auto"/"proxy"/"zemax")`、`control(..., source=)`；响应带 `source`/`label`；缺文件自动回落代理并标注 |
| `products/sensitivity/service.py` | `environment_sensitivity(source="synthetic"/"abaqus")`；Abaqus 版输出 `label`/`model`/`nullspace_vectors`/`gradient` |
| `desktop/worker.py` | 任务 `s`/`u` 支持 `payload.source` 透传 |
| `desktop/gui.py` | ③C 按钮改为"计算 M2 灵敏度 S（Zemax 优先）"，日志显示 `source=`/`label=`，热图标题带来源 |
| `data/fea_cases/manifest.json` | 新增 5 个 `M1_abq_*` case（`synthetic: false`, `model`, `label`, `zernike_nm`）与 `abaqus_model` 元数据 |

标签规则：Zemax 结果 `EXTERNAL-CHECK`；Abaqus 结果 `FEA-PROVISIONAL`；
代理矩阵仍是 `UNVERIFIED`，三者不会互相冒充。

## 5. 验证

```text
products/tests  32/32 OK（新增 real-S 载入/控制/auto 优先/abaqus 环境 S 等 6 项）
desktop/tests   13 OK, 6 skipped
doc links       OK
```

关键断言：

- `test_real_sensitivity_loaded`：服务读出的矩阵与 JSON 逐元素一致（atol 1e-12）；
- `test_real_control_reaches_target`：用真实 S 反解目标，残差 < 1e-9；
- `test_auto_prefers_real`：文件存在时 `auto` 必须选真实源；
- `test_abaqus_environment_sensitivity`：3×2、`FEA-PROVISIONAL`、含梯度块。

## 6. 限制（必须随数据一起引用）

1. **Zemax**：真实求解，但只覆盖 Z4/Z8/Z9 三项、单视场（0°）、单波长（0.7 µm）、
   线性小量假设；未做公差/蒙特卡洛，也未验证 0.01 步长下的线性度（可用 0.005/0.02
   复算验证）。
2. **Abaqus**：板级模型，非整机；支撑是弹簧等效，不是真实柔性铰/接触；
   k 值与垫面半径改变了绝对量级（RMS 对 k 敏感），故只用于趋势/实验设计；
   热梯度剖面是假设值。
3. gravity_90 的 µm 级 RMS 与设计规范 6.3 nm 差两个数量级，原因是刚体柔度
   （活塞/离焦）计入 RMS 而规范指的是补偿后面形；引用时必须区分。
4. 与实测 S 的关系：FEA/Zemax 用来做**实验设计**（幅值选择）与
   **EXTERNAL-CHECK**（逐列交叉验证）；S 的真值仍以
   `products/sensitivity/measured.py` 的 CCD+PD 实测为准。

## 7. 复现

```powershell
# Zemax（约 2 s，需要许可）
py -3.13 scripts\zos_sensitivity.py --out data\zemax --surface 4 --delta 0.01

# Abaqus（每工况约 1.5 min，需要许可）
$env:TEL_ABAQUS_OUT="$PWD\data\fea_cases\abaqus_m1"; $env:TEL_ABAQUS_SEED="30"
D:\SIMULIA\Commands\abq2025.bat cae noGUI=scripts\abaqus_m1_whiffletree.py

# 环境 S（读真实 FEA）
py -3.13 -c "from products.sensitivity import service as s; print(s.environment_sensitivity(source='abaqus'))"

# 真实 Zemax S（读真实求解）
py -3.13 -c "from products.control import service as f; print(f.sensitivity()['source'])"
```

## 8. 下一步

1. Zemax 线性度检查（0.005/0.02 步长）+ 第二个视场/波长，确认 S 的适用范围；
2. Abaqus 支撑刚度敏感性扫描（k = 3e5/1e6/3e6），给出 S 的区间而非单值；
3. 把真实 S 接进 `measured.py` 的交叉验证列（实测 vs Zemax 逐列相对差）；
4. 若要把热方向变成可观测，扩展 PD 状态到球差/像散项或加温度传感器。
