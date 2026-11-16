# Zemax 3A：ZMX 解析与 ZOS-API 灵敏度

> 代码：`products/zemax/zmx_parser.py`、`web/job_queue.py`、`scripts/zemax_zos_worker.py`
> 接口：`POST /api/zemax/inspect`、`POST /api/zemax/jobs`、`GET /api/jobs/{id}`

## 1. 当前状态

已实现：

- 读取 **UTF-16LE/ASCII** ZMX（带 BOM）；
- 解析 `ENPD`、`WAVM`、`SURF/TYPE/CURV/CONI/DISZ/GLAS/DIAM/STOP`；
- 返回面数、stop、口径、波长、面型；
- 前端 ③A 可上传 `.ZMX` 查看结构；
- **SQLite job queue**：提交/claim/complete/fail；
- Windows worker 脚本 + `--mock` 队列测试。

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
 -> Web ③C / CSV
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

## 3. Job queue

```text
POST /api/zemax/jobs   {name,data_b64} -> {job_id,status:"queued"}
GET  /api/jobs/{id}                     -> 状态/结果（不返回 ZMX base64）
POST /api/jobs/claim   {kind:"zemax"}   -> worker 领取
POST /api/jobs/{id}/complete {result}   -> worker 完成
```

存储：`results/jobs.sqlite3`（可删，自动重建）。

### 3.1 队列测试（Linux，不需要 Zemax）

```bash
python3 scripts/zemax_zos_worker.py --mock
```

返回明确标 `mock: true` 的 3×5 占位 S，只用于验证 Web/队列。

### 3.2 真实 worker（Windows）

```bat
python scripts\zemax_zos_worker.py --api http://<linux-host>:8000
```

`products/zemax/zos_bridge.py` 在 Windows 上接 ZOS-API；未配置时
明确抛错，不返回假值。

## 4. 边界

- 解析 ≠ 追迹：当前只给结构摘要，不给 S；
- ZMX 坐标断点/非序列/自由曲面可能超出解析器；
- 真实 S 必须由 OpticStudio 或经过验证的光线追迹给出；
- `--mock` 只能做队列测试，不能进报告/论文。
