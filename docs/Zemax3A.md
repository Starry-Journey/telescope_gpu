# Zemax 3A：ZMX 解析与 ZOS-API 灵敏度

> 代码：`products/zemax/zmx_parser.py`
> 接口：`POST /api/zemax/inspect`
> 目标：ZMX 结构 → M2 5-DOF 扰动 → S（Z4/Z8/Z9）

## 1. 当前状态

已实现：

- 读取 **UTF-16LE/ASCII** ZMX；
- 解析 `ENPD`、`WAVM`、`SURF/TYPE/CURV/CONI/DISZ/GLAS/DIAM/STOP`；
- 返回面数、stop、口径、波长、面型；
- 前端 ③A 可上传 `.ZMX` 查看结构。

实测你给的 `2.5m主系统.ZMX`：

```text
ENPD = 2500 mm
surfaces = 8（0..7）
STOP = surface 2
wavelengths = 24
```

**未实现**：用 ZMX 直接算 Zernike 灵敏度。ZMX 只是处方，不是光线追迹器；
需要 OpticStudio ZOS-API 或自研光线追迹。当前不伪造 3A 数值。

## 2. 真正的 3A 流程

```text
ZMX
 -> OpticStudio ZOS-API（Windows）
 -> 设置 M2 surface 的 decenter_x/y, tilt_x/y, focus
 -> 每个控制量 ±δ
 -> Zernike/波前分析
 -> Z = (Zplus - Zminus)/(2δ)
 -> S (3x5)
 -> 上传到 Web ③C（或写入 data/zemax/sensitivity.csv）
```

### 2.1 控制量

| 控制量 | Zemax 操作 | 单位 |
|---|---|---|
| `m2_focus` | 改变 M2 厚度/像面 | mm |
| `m2_decenter_x/y` | M2 surface decenter | mm |
| `m2_tilt_x/y` | M2 surface tilt | mrad |

### 2.2 输出 CSV

```csv
target,m2_focus,m2_decenter_x,m2_decenter_y,m2_tilt_x,m2_tilt_y
Z4,...
Z8,...
Z9,...
```

这个 CSV 直接粘贴/上传到 ⑦“自定义 S”，即可求 rank/condition 和控制量。

## 3. ZOS-API 桥

ZOS-API 是 Windows/.NET + OpticStudio license；不能塞进 Linux Web 容器。

推荐部署：

```text
Linux Web/FastAPI
  -> job queue
  -> Windows worker（OpticStudio + ZOS-API）
  -> sensitivity.csv
  -> Web 读取
```

Windows worker 必须记录：

- OpticStudio 版本；
- ZMX 文件 sha256；
- 每个 control 的 δ、单位、surface 编号；
- Zernike 分析设置（波长/视场/归一化半径）；
- 输出 CSV 与 run_manifest。

## 4. 边界

- 解析 ≠ 追迹：当前只给结构摘要，不给 S；
- ZMX 坐标断点/非序列/自由曲面可能超出解析器；
- 真实 S 必须由 OpticStudio 或经过验证的光线追迹给出；
- 没有 ZOS-API 时，用 Zemax 导出的 CSV 走 ⑦，不伪造成“自动 3A”。
