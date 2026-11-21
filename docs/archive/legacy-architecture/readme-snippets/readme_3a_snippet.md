## 3A 队列与 256² 预算

```text
POST /api/zemax/jobs          # 提交 ZMX 到 SQLite job queue
GET  /api/jobs/{id}           # 状态/结果（不返回 ZMX base64）
scripts/zemax_zos_worker.py   # Windows/OpticStudio worker; --mock 只测队列
PYTHONPATH=. python3 scripts/benchmark_pd.py
```

实测 5 帧前向：64² 0.014 s、128² 0.042 s、256² 0.180 s；
256² 100 次 BOBYQA 约 16.7 s/起点。预算与并行建议见
`docs/256²性能预算.md`，ZOS-API 流程见 `docs/Zemax3A.md`。
