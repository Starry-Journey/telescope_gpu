## 三个独立功能模块（v1）

| 模块 | 入口 | 文档 |
|---|---|---|
| ① CCD 数据管线 + 真实感仿真 | `POST /api/ccd/pipeline`，页面“⑤” | `docs/模块1_CCD数据管线.docx` |
| ② 相位差反演 | `POST /api/pd/invert`，页面“⑤/⑥” | `docs/模块2_相位差反演.docx` |
| ③ 灵敏度矩阵 + 控制量 | `/api/fea/*`、`/api/sensitivity/custom`、`/api/control/custom`，页面“④/⑦” | `docs/模块3_灵敏度与控制.docx` |

运行：

```bash
python3 -m uvicorn web.api.main:app --reload --port 8000
# http://127.0.0.1:8000/
```

模块 1 生成 5 帧真实感 CCD 与 64×64 PSF；模块 2 用这批 PSF 做 PD/BOBYQA；
模块 3 支持 FEA 合成工况、3×5 代理 S、任意自定义 S 与控制量求解。
