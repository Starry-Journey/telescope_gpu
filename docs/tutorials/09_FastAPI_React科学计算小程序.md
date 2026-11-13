# FastAPI + React + Plotly 科学计算小程序

> 代码：`web/api/main.py`、`web/ui/index.html`
> 功能：**CCD/PSF 仿真生成** + **CNN 初值 + BOBYQA 反演**
> 技术栈：FastAPI、PyTorch、NLopt、React 18、Plotly.js、Babel Standalone

## 1. 产品形态

一个单页小程序：

```
左栏：c4/c8/c9/noise 滑杆 + 生成/反演按钮
右栏：5 帧离焦 PSF 缩略图 + 指标卡 + Plotly 系数对比
```

两个核心动作：

1. **生成**：`POST /api/simulate` → 用 `ml/phase_torch.py` 的可微前向生成 5 帧 PSF，返回 PNG data URL；
2. **反演**：`POST /api/invert` → 生成带噪 PSF，CNN 预测初值，NLopt BOBYQA 精修，返回
   `truth / cnn / refined / nfev / cost`。

## 2. 后端设计

```python
@app.post("/api/simulate")
def simulate(req: SimRequest):
    coeff = torch.tensor([req.c4, req.c8, req.c9])
    stack = forward_stack(coeff, ForwardConfig(n=64, noise_sigma=req.noise),
                          add_noise=True).detach().numpy()
    return {"frames": [png_data_url(f) for f in stack], ...}

@app.post("/api/invert")
def invert(req: SimRequest):
    stack = forward_stack(torch.tensor(truth), cfg, add_noise=True).numpy()
    cnn = model(peak_normalise(stack)) * scales       # 初值
    x, nfev, cost = bobyqa(cnn, sum_normalise(stack)) # 精修
    return {"truth": ..., "cnn": ..., "refined": x.tolist(), "nfev": nfev}
```

关键点：

- **预处理一致性**：CNN 用峰值归一化，BOBYQA 用 sum 归一化；这是两个不同接口，不能混。
- **模型缺失兜底**：`ml/runs/cnn.pt` 不存在时，`init=0`，仍可跑 BOBYQA；返回 `model_loaded=false`。
- **返回图片**：matplotlib 生成小 PNG 再 base64，不把原始大数组传给前端。
- **同步阻塞**：BOBYQA 约 0.2 s，FastAPI 默认线程池可接受；更长任务应改任务队列。

## 3. 前端设计

- React 18 UMD + Babel Standalone（浏览器内 JSX），无 npm 构建；
- Plotly.js 画 `truth/CNN/BOBYQA` 三组柱状图；
- `useEffect` 首屏自动生成；滑杆 `input` 更新数值；
- 左栏按钮和右栏按钮都绑定同一组 `doSim/doInv`；
- 错误显示在 `#error`，不静默失败。

为什么不用 Vite/Next？MVP 目标是“一个文件能跑、无需 node_modules”；生产化再迁移到 Vite + TypeScript + 组件库。

## 4. 运行

```bash
cd telescope_gpu
# 先训练 CNN（只需一次）
python3 -m ml.train_cnn --epochs 20 --samples 2048 --batch 64 --n 64
python3 -m uvicorn web.api.main:app --reload --port 8000
# http://127.0.0.1:8000/
```

接口自测：

```bash
python3 web/test_api.py   # 需先把 BASE 改成实际端口
```

## 5. 实测

本机 API：

```text
GET  /api/health -> {"ok":true,"model":true}
POST /api/simulate -> 5 frames, [5,64,64]
POST /api/invert   -> truth [0.6,-0.2,0.1]
                      CNN  [0.589,-0.195,0.105]
                      BOBYQA [0.596,-0.221,0.110]
                      nfev 30, cost 1.2e-7
GET  / -> 200
```

## 6. 边界与下一步

- 后端只做 64×64 PSF、3 项 Zernike；接主项目要改成 256²/5 帧/15 星，并加任务队列；
- 前端 CDN 依赖 unpkg/plotly；离线部署需把 vendor 文件放本地；
- 单用户/无数据库；生产化需要 FastAPI + Celery/RQ + 对象存储；
- 下一步：把 `/api/metrics` 和 `results/` 图接进页面，做“完整 CCD 管线 dashboard”；
- 可加 WebSocket/SSE 显示 BOBYQA 迭代过程。
