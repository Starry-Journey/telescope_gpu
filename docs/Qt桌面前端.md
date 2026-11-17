# Qt 桌面前端（主路径）

> 代码：`desktop/qt_app.py`
> EXE：`dist\TelescopeEngineeringQtNative\TelescopeEngineeringQtNative.exe`
> 桌面快捷方式：`C:\Users\17424\Desktop\Telescope Engineering.lnk`

## 1. 为什么改成 Qt

用户反馈 Web GUI 打不开；PyWebView 在 PyInstaller 后 `.NET/pythonnet` 初始化失败。
因此桌面主路径改为 **PySide6 原生控件**，不再嵌入 React，不再依赖浏览器/HTTP。

## 2. 结构

```text
PySide6 QMainWindow
  ├─ ① CCD / PSF    直接调用 products.ccd.service / fits_service
  ├─ ② 相位差反演   直接调用 products.pd.service
  ├─ ③B FEA / 66项 直接调用 products.sensitivity.service
  ├─ ③A ZMX        直接调用 products.zemax.zmx_parser
  └─ ③C S / 控制   直接调用 products.control.service
```

所有图像用 Qt QPixmap；FEA 图用 Matplotlib Agg 转 PNG，再交给 Qt。
原生文件对话框：FITS、ZMX、FEA CSV。

## 3. 运行

```powershell
py -3.13 desktop\qt_app.py
py -3.13 desktop\qt_app.py --self-test
```

EXE：

```powershell
dist\TelescopeEngineeringQtNative\TelescopeEngineeringQtNative.exe
dist\TelescopeEngineeringQtNative\TelescopeEngineeringQtNative.exe --self-test
```

## 4. 打包

```powershell
py -3.13 -m PyInstaller --noconfirm --clean --onedir ^
  --name TelescopeEngineeringQtNative ^
  --add-data "data/fea_cases;data/fea_cases" ^
  --add-data "data/zemax;data/zemax" ^
  --hidden-import nlopt ^
  --hidden-import pipeline.realistic ^
  --hidden-import web.job_queue ^
  --exclude-module torch --exclude-module rayoptics ^
  --exclude-module poppy --exclude-module hcipy ^
  --exclude-module PySide6.QtWebEngineWidgets ^
  --exclude-module PySide6.QtWebEngineCore ^
  desktop\qt_app.py
```

## 5. 当前状态

- `--self-test`：通过；
- 主窗口：可打开；
- 五个页签：CCD/PSF、PD、FEA、ZMX、S/控制；
- 无 React、无浏览器、无 FastAPI 依赖（桌面直连服务）；
- Web GUI 保留为旧路径，不再是桌面主路径。

## 6. 边界

- Qt 前端是本地单用户，不是多用户 Web；
- 3A 仍只有 ZMX 解析/rayoptics S 外部参考；
- 真实 FEA/离焦数据仍需外部输入；
- EXE 为 onedir，必须带 `_internal`。

## 11. 启动/柱状图/验证（2026-11-17）

- PD/FEA 新增 Zernike 柱状图；FEA 显示前 15 项。
- 启动时不再自动拉起 FEA worker；首屏只加载 Qt/numpy。
- QProcess 使用 job.json/result.json 文件协议，避免 MB 级 JSON 管道截断。
- 无头自检：

```powershell
py -3.13 desktop/qt_smoke.py
# QT_SMOKE_OK .../results/qt_smoke
```

会验证：星场帧、PSF 拼图、PD 柱状图、FEA 柱状图，并保存 PNG。

### SmartScreen

未签名 EXE 会被 SmartScreen 拦截：

```text
发布者：未知
```

这不是缺模块。处理：

```powershell
Unblock-File dist\TelescopeEngineeringQtNative\TelescopeEngineeringQtNative.exe
```

或弹窗点“运行”。彻底去掉需要代码签名证书；本地自签名不保证消除 SmartScreen。

## 12. Worker 拆分（P5-3）

```text
desktop/qt_app.py   薄入口：--worker 时只 import desktop.worker
desktop/gui.py      PySide6 QtApp/页签
desktop/worker.py   服务任务，不 import PySide6
```

验证：

```python
import desktop.worker
assert "PySide6" not in sys.modules
```

QProcess 仍启动同一 EXE：

```text
EXE --worker task --job ... --result ...
```

worker 分支不再加载 Qt，启动/内存更低。
