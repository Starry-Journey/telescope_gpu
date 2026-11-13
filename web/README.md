# web：最小可视化 demo

只读后端，标准库 `http.server`，不需要 FastAPI/Flask。

```bash
# 先跑出 results/ccd_pipeline/
python3 -m pipeline.run --config configs/pipeline.json --force-sim
python3 web/serve.py --port 8000
# open http://127.0.0.1:8000/
```

接口：

| 路径 | 内容 |
|---|---|
| `/` | 单页 dashboard |
| `/api/metrics` | `results/ccd_pipeline/metrics.json` |
| `/api/figures` | 图片文件名列表 |
| `/figures/<name>` | 结果图 |

后续迁移到 FastAPI + React/Vue 时，保留 `/api/metrics`、`/api/figures` 两个接口即可；
前端只依赖 JSON 和 PNG，不耦合 Python 内部。

边界：当前是单机只读 demo，没有用户/数据库/任务队列；真实部署需要把
pipeline 放到 worker（Celery/RQ）并用对象存储保存 FITS/图。
