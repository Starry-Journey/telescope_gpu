# web：FastAPI + React + Plotly 科学计算小程序

两个功能：

1. **CCD/PSF 仿真生成**：滑杆调 `c4/c8/c9/noise`，后端用 PyTorch 可微前向生成 5 帧离焦 PSF；
2. **反演**：CNN 给初值 + NLopt BOBYQA 精修，前端用 Plotly 对比 truth / CNN / refined，并显示评估次数与 cost。

## 启动

```bash
cd telescope_gpu
python3 -m uvicorn web.api.main:app --reload --port 8000
# open http://127.0.0.1:8000/
```

依赖：`fastapi`、`uvicorn`、`torch`、`nlopt`、`numpy`、`matplotlib`。
CNN checkpoint 默认读 `ml/runs/cnn.pt`；没有就先训练：

```bash
python3 -m ml.train_cnn --epochs 20 --samples 2048 --batch 64 --n 64
```

## API

| 方法/路径 | 内容 |
|---|---|
| `GET /api/health` | 服务/模型状态 |
| `POST /api/simulate` | `{c4,c8,c9,noise}` → 5 帧 PNG data URL |
| `POST /api/invert` | `{c4,c8,c9,noise}` → truth/CNN/BOBYQA/nfev/cost |
| `GET /api/metrics` | CCD 管线 `metrics.json` |
| `GET /api/figures` | 结果图列表 |
| `GET /` | React 单页 |

## 前端技术栈

- React 18（UMD CDN）+ Babel Standalone（浏览器内 JSX）；
- Plotly.js（CDN）画系数对比；
- 无 npm 构建，单文件 `web/ui/index.html`，适合 MVP/演示。
- 生产化建议：Vite + React 工程、把 Plotly 打包进 bundle、加 TypeScript/测试；后端保留同样 API。

## 旧版

`web/serve.py` 是标准库只读 dashboard，仍可用：

```bash
python3 web/serve.py --port 8000
```

## 边界

- 仿真 64×64 PSF、3 项 Zernike、无星点叠加；不是主项目 256²/5 帧/15 星的完整正向模型；
- 反演是教学 demo，不替代科学级相位差反演；
- 浏览器需要能访问 unpkg/plotly CDN；离线部署请把 React/Plotly 下载到 `web/ui/vendor/` 并改 `<script src>`。
