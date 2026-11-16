# 框架与 Web 集成（Framework × FastAPI × React）

> 代码：`web/framework_service.py`、`web/api/main.py`、`web/ui/index.html`
> 启动：`python3 -m uvicorn web.api.main:app --reload --port 8000`
> 页面：`http://127.0.0.1:8000/`

## 1. 为什么用 Python 适配层

C++ `framework/` 是参考实现；Web 后端是 Python。v1 在 `web/framework_service.py` 用**同一套 3×5 矩阵/单位/公式**做适配：

```text
ParameterSpace -> ForwardModel (3x5) -> finite-difference S
              -> weighted least squares control
```

这样前端可以先看到完整交互；后续可用 pybind11/C++ 框架替换 service，JSON 契约不变。

## 2. 接口

| 方法 | 路径 | 输入 | 输出 |
|---|---|---|---|
| GET | `/api/framework` | — | 参数/控制名、单位、默认 target、bounds |
| POST | `/api/framework/sensitivity` | `{step}` | 3×5 S、rank、condition、singular values、nullspace |
| POST | `/api/framework/control` | `{target,lam,use_bounds,lower?,upper?}` | u、S·u、residual、cost、rank、nullspace、active constraints |

示例：

```bash
curl -s http://127.0.0.1:8000/api/framework | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8000/api/framework/sensitivity \
  -H 'Content-Type: application/json' -d '{"step":0.001}' | python3 -m json.tool
curl -s -X POST http://127.0.0.1:8000/api/framework/control \
  -H 'Content-Type: application/json' \
  -d '{"target":[0.82483,0.4633,-0.23165],"lam":1e-8,"use_bounds":true}' | python3 -m json.tool
```

实测：

```text
S = [[4.04,0.083,-0.083,0.976,-0.976],
     [0,-4.633,0,0,0],
     [0,0,-4.633,0,0]]
rank=3, condition=1.0897, nullspace=2
u=[0.1856,-0.1,0.05,0.0448,-0.0448]
residual ~ -4.6e-10
```

## 3. 前端“③ 框架”区域

页面在原有 `① 参数 / 生成`、`② 仿真/反演` 下方新增：

- 目标 `Z4/Z8/Z9` 数字输入；
- 差分步长 `δ`、Tikhonov `λ` 滑杆；
- “重算灵敏度矩阵 S”“求解控制量 u”按钮；
- S 热力图（Plotly，带数值文本）；
- 控制量柱状图（uz/ux/uy/uα/uβ）；
- 指标卡：S 尺寸/rank、condition/nullspace、max|residual|、是否触及边界；
- target / S·u / residual 表；
- 页面明确提示：3×5 秩 3、零空间 2 维，目标可复现但 u 不唯一。

### 3.1 FEA 案例区域（④）

页面再往下是“④ FEA 案例”：

- 下拉选择 `M1/M2/M3` 重力/热工况；
- 镜面位移热力图（nm）；
- 15 项 Zernike 柱状图 + 拟合残差；
- Z4/Z8/Z9 卡片；
- 环境灵敏度 S（重力/温度）热力图；
- 合成参考黄条：**不是已求解的 2.5 m FEA 模型**。

接口：

```text
GET /api/fea/cases
GET /api/fea/cases/{id}
GET /api/fea/sensitivity
```

说明见 `docs/FEA案例与灵敏度.md`。

![framework sensitivity](../figs/fig_framework_sensitivity.png)

## 4. 运行

```bash
cd /home/legend-star/telescope_gpu
python3 -m uvicorn web.api.main:app --reload --port 8000
# 浏览器打开 http://127.0.0.1:8000/
```

自测：

```bash
python3 web/test_framework.py   # 默认 8000，可用 MINIAPP_URL 覆盖
```

## 5. 边界

- `framework_service.py` 是 Python 适配层，不是 C++ 框架本身；两者数值一致（同一 A/公式）；
- 3×5 是线性代理，不是 Zemax/实测 IM；
- `lam` 目前是最小二乘正则，不是硬约束 QP；
- 前端依赖 React/Babel/Plotly CDN；离线要 vendor；
- 真实数据接入时，把 `target` 换成 pipeline/phase 输出的 z 向量即可，接口不变。
