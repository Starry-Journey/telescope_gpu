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

---

# 第二轮：适用范围、区间与可观测性（2026-11-19 晚）

## A. Zemax S 的适用范围（`scripts/zos_s_scope.py`）

同一套中心差分协议，分别改变步长、视场、波长：

| 变量 | 取值 | 结果 | 结论 |
|---|---|---|---|
| 步长 | 0.005 / 0.01 / 0.02 | S 最大相对散布 **6.0e-7** | 在 4 倍步长范围内 S 严格线性，0.01 是安全工作点 |
| 视场 | 轴上 0° vs 模型首个轴外场 0.01° | 最大相对差 **2.7%**（彗差列），离焦列不变 | 小视场下 S 近似不变；0.01° 太小，不能代表大视场，需更大离轴角再测 |
| 波长 | 0.70 µm（主） vs 0.40 µm | Z4 由 −23.284 → −40.747（比值 1.750 = 0.7/0.4） | **以"波"为单位的 S 按 1/λ 缩放**；换算成 nm 后与波长无关 |

工程含义：把 S 用于控制时，必须与 PD 反演使用同一波长；若 PD 换波段，
S 要按 λ_ref/λ 缩放，否则会引入 1.75× 的系统性增益误差。

## B. Abaqus 支撑刚度区间（`scripts/abaqus_k_sweep_summary.py`）

k_bottom = 3e5 / 1e6 / 3e6 N/mm（k_side = k/2），其余设置不变：

| k (N/mm) | S[Z4, 重力] waves/deg | S[Z4, 温度] waves/K | 条件数 |
|---|---|---|---|
| 3e5 | −0.011552 | −2.9e-6 | 3.4e5 |
| 1e6 | −0.006681 | +2.9e-6 | 5.4e4 |
| 3e6 | −0.003545 | +2.7e-5 | 8.0e3 |

**S 区间（3×2，应引用区间而非单值）**：

```text
S_gravity(Z4) ∈ [-0.01155, -0.00355] waves/deg   （≈ 1/k，跨 3.3 倍）
S_thermal      ≈ 0（3e-6 量级）                   （与 k 无关）
```

刚度只改变刚体柔度（Z4 的重力列），不改变热列——热列近零是材料属性，不是
支撑假设。

## C. 热方向可观测性：扩项 vs 温度传感器

用户设想"把 PD 状态扩到球差/像散，让热方向可观测"。用 6 项状态
（Z4, Z5, Z6, Z8, Z9, Z11；Z11 = (4,0) 球差，与 `PROJECT_TERM_MAP` 一致）
从真实 Abaqus 工况重新拟合：

```text
S (waves/deg, waves/K) =
[[-6.68133560e-03  2.90189217e-06]
 [-2.39468354e-05 -3.03121840e-07]
 [ 4.54478119e-05 -7.43087156e-07]
 [ 2.06815785e-05 -1.31592627e-07]
 [ 1.01475859e-06  1.28253832e-08]
 [ 8.89046738e-04 -5.04097546e-05]]
rank=2  cond=135.917  sv=[6.7405e-3, 4.9592e-5]
观测性阈值 5% 判定: dropped_rows=[Z5, Z6, Z8, Z9], dropped_cols=[thermal_K]
reduced: [Z4, Z11] x [gravity_deg], rank 1
```

**结论：扩项救不了均匀浸泡。** 热列范数只为重力列的 0.74%，按 5% 观测性阈值
被判为不可观测并从补偿矩阵中删除——即使把状态扩到 6 项、包含球差。
原因是 Zerodur α≈0：均匀 ΔT 在所有低阶项上响应都 <1e-4 波。

但**径向梯度是可观测的**：`thermal_grad20K` 的 Z4 = 0.1289 波 @20 K 梯度，
即 0.0064 波/K，与重力列同量级。因此热可观测性的正确做法是：

1. 以能被梯度激励的低阶项（Z4 主导）+ 至少两点温差信息建模，而不是单纯加项；
2. 或者直接加温度传感器（工程上最稳）；
3. 均匀浸泡工况不必进控制矩阵，作为偏置/标定项处理。

> 约定修正：本报告第一版 C 节曾按 `TERMS[11]=(4,-2)` 取 Z11，与
> `PROJECT_TERM_MAP["Z11"]=(4,0)` 不一致；上表为按 (4,0) 重算的结果，
> 差异已在此说明，代码为唯一来源。

## E. PD 状态扩展到 6 项（Z4/Z5/Z6/Z8/Z9/Z11）

`products/pd/service.py::invert_psf(..., terms=None)` 支持 3 或 6 项状态，
默认仍是 3 项且数值逐位不变（对快照做过 `array_equal` 校验）。

```python
from products.pd.service import invert_psf
r3 = invert_psf(stack, defocus)                                  # Z4/Z8/Z9
r6 = invert_psf(stack, defocus, terms=("Z4","Z5","Z6","Z8","Z9","Z11"))
r6["zernike"]   # {"Z4":..,"Z5":..,"Z6":..,"Z8":..,"Z9":..,"Z11":..}
```

- `products/optics/zernike.py` 新增 `PROJECT_TERM_MAP` / `resolve_terms` /
  `term_grids`，与项目标签约定一致（Z8=x 彗差 (3,1)，Z9=y 彗差 (3,-1)）；
- 6 项下的边界与容差：离焦 ±1.8、其余 ±0.35，`ftol=1e-8, xtol=1e-6`。
  理由：同一离焦栈上 6 个耦合系数使代价面更平，BOBYQA 的二次模型每维更噪；
  收紧非离焦边界避免串扰，实测无噪 6 项回环恢复误差 ≈1e-8 波；
- 新增 `products/tests/test_pd_terms.py` 10 项测试（3 项黄金像素、位兼容、
  6 项回环、未知标签报错、零额外系数等价性）。

## F. 机构自由度可观测性与像散装调方案

`products/sensitivity/observability.py` + `engineering_terms_sensitivity("abaqus")`：

- `analyze_control_space(S, targets, dofs, rel_threshold=0.05)`：给出每个
  自由度/像差项的最大灵敏度和可观性；低于阈值的方向从补偿矩阵删除，
  返回 reduced S 与原因；
- 实测结论（真实 Abaqus 数据）：6 项 × 2 控制中，**Z5/Z6（像散）、
  （口径 v2 撤回：当时把 S_F 当控制器用了——S_F 只说明 M1-FEA 能驱动
  Z8/Z9，M2 可控性由 S_Z 决定，Z8/Z9 **可控**；仅 Z5/Z6 全盲，见
  docs/控制口径_v2.md §5）
  `thermal_K` 列同时被判不可观测；
- `astigmatism_plan(...)`：从 FEA 最坏工况
  `M1_abq_gravity_90deg` 读出 Z5=−1.3643 nm、Z6=+2.5988 nm（面形），
  合成 |A| = 2.9352 nm（面形 0.00464 波 / 波前 0.00928 波 @632.8 nm），
  方位 `atan2(Z6,Z5)=117.70°` → 镜面像散轴 58.85°、校正轴 148.85°；
  给出 **4 个轴向调节点（148.85/238.85/328.85/58.85°，推拉交替）**、
  首步 1.4676 nm 面形、2 次迭代、目标残差 0.2935 nm，并明确
  **装在 M1 背部支撑环而非 M2 台**（M2 对 Z5/Z6 全盲——证据为实测 S_Z 像散行 S_astig≈0，
  见 docs/控制口径_v2.md；不是由 S_F 得出）。
  标签 `FEA-PROVISIONAL`，并注明 nm→垫片行程需台架影响系数标定。

## G. 模块可视化：上传有限元文件 → 3D 结构视图

- `desktop/worker.py` 新任务 `fea_viz`（无 Qt 依赖）：
  payload `{x,y,z,label,azim,elev}` → result `{png3d_b64, grid 48×48, stats{rms_nm,pv_nm,n}}`；
  matplotlib `plot_trisurf` + 半透明节点云 + 色标，输出 400×300 PNG（约 52 KB）。
- `desktop/gui.py` ③B 页签新增"3D 结构视图"与"导出 3D PNG"按钮，
  `_on_fea` 缓存节点数组，`export_all` 增加 `fea_3d.png`；
- `desktop/tests/test_worker.py` 新增 `test_fea_viz_task`；
- 限制：三角面覆盖节点凸包，中心遮拦在 3D 视图里会被桥接（2D 热图仍显示遮拦）。

## H. 仿真前置数据包（第 5 项交付）

```text
C:\Users\17424\Documents\deepseek-harness\default-workspace\telescope_engineering\release\telescope_sim_inputs_2026-11-19\
C:\Users\17424\Documents\deepseek-harness\default-workspace\telescope_engineering\release\telescope_sim_inputs_2026-11-19.zip
```

内容（21 个文件，6.14 MB；zip 2.2 MB）：

```text
model/2.5m主系统.zmx               真实光学处方
fea/*_top.csv                      13 个 Abaqus 顶面导出（基础 + k=3e5/3e6 扫描）
fea/manifest.json                  工况注册表（id/载荷/口径/RMS/Zernike/label）
fea/fit_summary.json               基础工况 66 项拟合摘要
fea/sensitivity_k_sweep.json       支撑刚度 S 区间
zemax/sensitivity_zemax.{json,csv} 真实 3×5 M2 灵敏度
zemax/sensitivity_zemax_scope.json 步长/视场/波长适用范围
README.md                          文件说明 + 放回 data/ 的方法 + 标签限定
```

复现：`py -3.13 scripts\package_sim_inputs.py`
