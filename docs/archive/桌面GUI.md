# 桌面 GUI（PyWebView 壳）

## 1. 现状

当前前端已经是 **Web GUI**：

```text
React + Babel + Plotly
FastAPI /api/*
浏览器 http://127.0.0.1:8000/
```

它支持：

- ① CCD 合成/真实 FITS 上传；
- ② PD 反演；
- ③A ZMX/3A 队列/rayoptics S；
- ③B FEA 工况/CSV 上传；
- ③C 自定义 S → 控制量。

所以不是“没有 GUI”，而是“浏览器 GUI”。

## 2. 桌面壳

`web/desktop_app.py` 用 **PyWebView** 包一层：

```bash
py -3.13 -m pip install pywebview
py -3.13 web/desktop_app.py
```

它会：

1. 在 `127.0.0.1:8765` 启动现有 FastAPI；
2. 打开原生窗口；
3. 加载现有 React 页面。

不重写前端。`requirements-desktop.txt` 只加 `pywebview`。

## 3. 打包

```bash
pyinstaller --onefile --add-data "web/ui;web/ui" web/desktop_app.py
```

注意：

- FastAPI 静态路径、`data/`、`results/` 需要一起打包；
- CalculiX/rayoptics/POPPY 是外部依赖，不应塞进 exe；
- 打包后需保留 `data/fea_cases` 或让用户选择数据目录。

## 4. 桌面 GUI 建议

- 原生文件选择：FITS/ZMX/FEA CSV；
- job 队列状态/日志面板；
- “未安装 rayoptics/CalculiX”明确提示；
- 离线模式；
- 结果目录可配置。

## 5. 方案对比

| 方案 | 复用 React | 体积 | 适合 |
|---|---|---|---|
| PyWebView | 是 | 小 | 本地科研工具，推荐 |
| Tauri 2 | 是 | 最小 | 需要精致桌面包 |
| Electron | 是 | 大 | 生态熟 |
| PySide6 | 否 | 中 | 需要原生控件 |

当前先做 PyWebView 壳，不改变 Web 主路径。
