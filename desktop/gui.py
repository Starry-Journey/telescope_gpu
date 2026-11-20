"""Native Qt desktop frontend (PySide6).

Primary desktop path.  Does not embed React/WebView and does not need FastAPI.
Services are imported lazily so the window opens fast and PyInstaller can keep
the same service modules as the Web path.

Tabs: CCD/PSF, PD, FEA/66-term, ZMX, S/control, demo.
"""
from __future__ import annotations

import base64
import csv
import io
import json
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt, QThread, Signal, QSize, QSettings, QTimer
from PySide6.QtGui import QPixmap, QDesktopServices, QAction
from PySide6.QtCore import QUrl
from desktop.tasks import TaskRunner
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget, QDialog,
    QCheckBox, QToolBar,
)

DEFOCUS = "-2,-1,0,1,2"
QSS = """
QWidget { background:#0f141a; color:#dce6f2; font-family:'Segoe UI','Microsoft YaHei'; font-size:10pt; }
QMainWindow { background:#0f141a; }
QLabel#Title { font-size:20pt; font-weight:700; color:#eaf2ff; }
QLabel#Subtitle { color:#8aa3bc; font-size:9pt; }
QGroupBox { border:1px solid #263342; border-radius:9px; margin-top:14px; padding:12px 8px 8px 8px; }
QGroupBox::title { subcontrol-origin:margin; left:12px; padding:0 6px; color:#9dc2ff; font-weight:600; }
QPushButton { background:#1f6feb; color:white; border:0; border-radius:6px; padding:7px 12px; font-weight:600; }
QPushButton:hover { background:#2f81f7; }
QPushButton:disabled { background:#263342; color:#6b7f94; }
QPushButton#Ghost { background:#1a2530; color:#cfe0f2; border:1px solid #2d4054; }
QPushButton#Ghost:hover { background:#223344; }
QLineEdit,QComboBox,QDoubleSpinBox,QSpinBox {
  background:#0b1117; border:1px solid #2a3a4a; border-radius:5px; padding:4px 6px; selection-background-color:#1f6feb;
}
QComboBox QAbstractItemView { background:#111a22; selection-background-color:#1f6feb; }
QTextEdit { background:#0b1117; border:1px solid #263342; border-radius:6px; padding:6px; }
QTabWidget::pane { border:1px solid #263342; border-radius:8px; top:-1px; }
QTabBar::tab { background:#131c25; color:#9db2c8; padding:8px 14px; border:1px solid #263342; border-bottom:0; border-top-left-radius:6px; border-top-right-radius:6px; margin-right:2px; }
QTabBar::tab:selected { background:#1f6feb; color:white; }
QStatusBar { background:#0b1117; color:#8aa3bc; border-top:1px solid #263342; }
QToolBar { background:#0b1117; border:0; spacing:6px; }
"""
GOOD = "#3fb950"; BAD = "#f85149"; WARN = "#d29922"


class ScaledImageLabel(QLabel):
    """QLabel that keeps the original pixmap and rescales on resize.

    Fixes the "3D view shows nothing" class of bugs: a pixmap scaled to a
    label size captured while the tab was not laid out (or a 0x0 size) used to
    become invisible and never recovered.
    """

    def __init__(self, text="", min_size=(380, 300), parent=None):
        super().__init__(text, parent)
        self._source = None
        self.setAlignment(Qt.AlignCenter)
        self.setMinimumSize(*min_size)
        self.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;"
                           "color:#6b7f94;")

    def set_source_pixmap(self, pix: QPixmap):
        self._source = pix
        self._apply()

    def source_pixmap(self):
        return self._source

    def clear_source(self):
        self._source = None
        self.setText("")

    def _apply(self):
        if self._source is None or self._source.isNull():
            return
        size = self.size()
        target = QSize(max(size.width(), self.minimumWidth()),
                       max(size.height(), self.minimumHeight()))
        self.setPixmap(self._source.scaled(target, Qt.KeepAspectRatio,
                                           Qt.SmoothTransformation))

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._apply()


def _writable_dir(*candidates) -> Path:
    """First candidate directory that can actually be written to."""
    for cand in candidates:
        if not cand:
            continue
        p = Path(cand)
        try:
            p.mkdir(parents=True, exist_ok=True)
            probe = p / ".write_test"
            probe.write_text("ok", encoding="utf-8")
            probe.unlink()
            return p
        except Exception:
            continue
    return Path.cwd()


def _default_export_dir() -> Path:
    docs = Path.home() / "Documents"
    return _writable_dir(docs / "TelescopeEngineering" / "exports",
                         ROOT / "results" / "exports",
                         Path.cwd() / "telescope_exports")


class Worker(QThread):
    result = Signal(object)
    failed = Signal(str)

    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def run(self):
        try:
            self.result.emit(self.fn())
        except Exception as e:
            self.failed.emit(str(e) + "\n\n" + traceback.format_exc())


def _png(url):
    raw = base64.b64decode(url.split(",", 1)[1])
    pix = QPixmap(); pix.loadFromData(raw)
    return pix


def _fea_png(detail):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    z = np.asarray(detail["grid"]["z"], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(z, origin="lower", cmap="viridis")
    ax.set_title("FEA dz [nm]", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    pix = QPixmap(); pix.loadFromData(buf.getvalue())
    return pix


def _bar_png(labels, values, title="Zernike"):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(6, 3.2))
    vals = [float(v) for v in values]
    ax.bar(range(len(vals)), vals, color="#4da3ff")
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.axhline(0, color="#888", linewidth=0.8)
    ax.set_ylabel("nm")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    pix = QPixmap(); pix.loadFromData(buf.getvalue())
    return pix


def _frames_png(stack):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, len(stack), figsize=(10, 2.4))
    for ax, img, i in zip(np.ravel(axes), stack, range(len(stack))):
        ax.imshow(img, origin="lower", cmap="inferno")
        ax.set_title("frame %d" % i, fontsize=8); ax.set_xticks([]); ax.set_yticks([])
    fig.tight_layout()
    buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
    plt.close(fig)
    pix = QPixmap(); pix.loadFromData(buf.getvalue())
    return pix


class QtApp(QMainWindow):
    def __init__(self, startup_t0=None):
        super().__init__()
        self._startup_t0 = startup_t0
        self.setWindowTitle("Telescope Engineering · Qt")
        self.resize(1440, 940)
        self.psf = None
        self.defocus = None
        self.last_ccd = None
        self.last_pd = None
        self.last_fea = None
        self.last_fea3d = None
        self._fea_xyz = None  # node arrays (mm/mm/nm) backing the 3D view
        self.last_s = None
        self.frames = []
        self._workers = []
        self.active_worker = None
        self.runner = TaskRunner(self._task_root(), self)
        self._task_done = None
        self._task_label = ""
        self.runner.finished.connect(self._on_task_done)
        self.runner.failed.connect(self._on_task_failed)
        self.runner.cancelled.connect(self._on_task_cancelled)
        if getattr(sys, "frozen", False):
            self.log_path = Path(sys.executable).resolve().parent / "qt_gui.log"
        else:
            self.log_path = ROOT / "results" / "qt_gui.log"
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass

        central = QWidget(); root = QVBoxLayout(central)
        header = QHBoxLayout()
        left = QVBoxLayout()
        title = QLabel("Telescope Engineering"); title.setObjectName("Title")
        sub = QLabel("CCD / 相位差 / FEA / ZMX / 控制 · Qt native")
        sub.setObjectName("Subtitle")
        left.addWidget(title); left.addWidget(sub)
        header.addLayout(left, 1)
        self.badge = QLabel("READY"); self.badge.setStyleSheet(
            "background:#12351f;color:#3fb950;border:1px solid #1f6f3a;border-radius:11px;padding:3px 10px;font-weight:600;")
        header.addWidget(self.badge, 0, Qt.AlignTop)
        self.cancel_btn = QPushButton("取消任务")
        self.cancel_btn.setObjectName("Ghost")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_task)
        header.addWidget(self.cancel_btn, 0, Qt.AlignTop)
        self.log_btn = QPushButton("日志")
        self.log_btn.setObjectName("Ghost")
        self.log_btn.clicked.connect(self.show_log)
        header.addWidget(self.log_btn, 0, Qt.AlignTop)
        self.exp_btn = QPushButton("导出目录")
        self.exp_btn.setObjectName("Ghost")
        self.exp_btn.clicked.connect(self.export_all)
        header.addWidget(self.exp_btn, 0, Qt.AlignTop)
        root.addLayout(header)

        self.tabs = QTabWidget()
        self.tabs.addTab(self.tab_ccd(), "① CCD / PSF")
        self.tabs.addTab(self.tab_pd(), "② 相位差反演")
        self.tabs.addTab(self.tab_fea(), "③B FEA / 66 项")
        self.tabs.addTab(self.tab_zmx(), "③A ZMX")
        self.tabs.addTab(self.tab_ctrl(), "③C S / 控制")
        self.tabs.addTab(self.tab_demo(), "④ 演示")
        root.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.statusBar().showMessage("Ready")
        self._src_label = QLabel("-")
        self.statusBar().addPermanentWidget(self._src_label)
        self._build_menu()
        self._settings = QSettings("TelescopeEngineering", "QtNative")
        geo = self._settings.value("geometry")
        if geo is not None:
            try:
                self.restoreGeometry(geo)
            except Exception:
                pass
        self._log("Qt GUI started")

    # ---------- helpers ----------
    def _build_menu(self):
        bar = self.menuBar()
        m_file = bar.addMenu("文件")
        act_exp = QAction("导出全部...", self)
        act_exp.triggered.connect(lambda: self.export_all())
        m_file.addAction(act_exp)
        act_open = QAction("打开导出目录", self)
        act_open.triggered.connect(self.open_export_dir)
        m_file.addAction(act_open)
        m_file.addSeparator()
        act_quit = QAction("退出", self)
        act_quit.triggered.connect(self.close)
        m_file.addAction(act_quit)
        m_view = bar.addMenu("视图")
        for idx, name in enumerate(["CCD / PSF", "相位差反演", "FEA / 66 项",
                                    "ZMX", "S / 控制", "演示"]):
            act = QAction("%d. %s" % (idx + 1, name), self)
            act.triggered.connect(lambda _=False, i=idx: self.tabs.setCurrentIndex(i))
            m_view.addAction(act)
        m_help = bar.addMenu("帮助")
        act_about = QAction("关于 / 标签说明", self)
        act_about.triggered.connect(self.show_about)
        m_help.addAction(act_about)

    def open_export_dir(self):
        target = _default_export_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        self.status("打开导出目录 " + str(target), GOOD)

    def show_about(self):
        QMessageBox.information(
            self, "关于",
            "Telescope Engineering · Qt native\n\n"
            "标签约定：\n"
            "CORE-VERIFIED / SYNTHETIC-VALIDATED / EXTERNAL-CHECK /\n"
            "FEA-PROVISIONAL / GUI-BETA / UNVERIFIED\n\n"
            "数据来源: data/zemax（Zemax ZOS-API 真实求解）、\n"
            "data/fea_cases（Abaqus 求解）")

    def status(self, text, color=GOOD):
        self._log(text)
        self.badge.setText(text.upper() if len(text) < 18 else text)
        self.badge.setStyleSheet(
            "background:#0b1117;color:%s;border:1px solid #2a3a4a;border-radius:11px;padding:3px 10px;font-weight:600;" % color)
        self.statusBar().showMessage(text)

    def show_log(self):
        d = QDialog(self)
        d.setWindowTitle("Qt GUI log")
        d.resize(900, 520)
        lay = QVBoxLayout(d)
        te = QTextEdit(); te.setReadOnly(True)
        try:
            te.setPlainText(self.log_path.read_text(encoding="utf-8")[-200000:])
        except Exception as e:
            te.setPlainText(str(e))
        lay.addWidget(te)
        row = QHBoxLayout()
        copy = QPushButton("复制全部"); copy.setObjectName("Ghost")
        copy.clicked.connect(lambda: (QApplication.clipboard().setText(te.toPlainText()),
                                      self.status("日志已复制")))
        op = QPushButton("打开日志文件"); op.setObjectName("Ghost")
        op.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.log_path))))
        row.addWidget(copy); row.addWidget(op); row.addStretch(1)
        lay.addLayout(row)
        d.exec()

    def _log(self, text):
        try:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write("%s  %s\n" % (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), text))
        except Exception:
            pass

    def _show_stack(self, stack):
        stack = np.asarray(stack, dtype=float)
        if stack.ndim == 2:
            stack = stack[None]
        pix = _frames_png(stack)
        self.ccd_img.set_source_pixmap(pix)
        self.ccd_img.setToolTip("frames=%s" % (stack.shape,))

    def busy(self, text):
        self.status("运行中: " + text, WARN)
        self.tabs.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def ready(self):
        self.tabs.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.active_worker = None

    def _task_root(self):
        """Writable dir for job/result JSON (portable + sandbox friendly)."""
        candidates = [Path(sys.executable).resolve().parent / "qt_tasks",
                      ROOT / "results" / "qt_tasks"]
        for c in candidates:
            try:
                c.mkdir(parents=True, exist_ok=True)
                (c / ".write_test").write_text("ok", encoding="utf-8")
                (c / ".write_test").unlink()
                return c
            except Exception:
                continue
        raise RuntimeError("no writable qt_tasks directory")

    def _run_process(self, task, payload, done, label):
        if self.runner.busy():
            return
        self._task_done = done
        self._task_label = label
        self.busy(label)
        try:
            self.runner.start(task, payload)
        except Exception as e:
            self.ready()
            self.status("失败: " + label, BAD)
            QMessageBox.critical(self, label, str(e))

    def _on_task_done(self, data):
        done = self._task_done
        label = self._task_label
        self._task_done = None
        self.ready()
        self._log("TASK ok %s %.2fs" % (label, self.runner.last_elapsed))
        self.status("完成: " + label)
        if done:
            done(data or {})

    def _on_task_failed(self, msg):
        label = self._task_label or "任务"
        self._task_done = None
        self.ready()
        self.status("失败: " + label, BAD)
        self._log("TASK fail %s %.2fs" % (label, self.runner.last_elapsed))
        self._log("ERROR " + label + "\n" + msg)
        box = QMessageBox(QMessageBox.Critical, label, str(msg)[:500], QMessageBox.Ok, self)
        box.setDetailedText(msg)
        box.exec()

    def _on_task_cancelled(self):
        label = self._task_label or "任务"
        self._task_done = None
        self.ready()
        self._log("TASK cancel %s %.2fs" % (label, self.runner.last_elapsed))
        self.status("已取消: " + label, WARN)

    def closeEvent(self, event):
        try:
            self._settings.setValue("geometry", self.saveGeometry())
        except Exception:
            pass
        if self.runner.busy():
            self.runner.cancel()
        super().closeEvent(event)

    def cancel_task(self):
        if not self.runner.busy():
            if getattr(self, "active_worker", None) is not None:
                self.status("演示任务不可中断（短任务，请等待完成）", WARN)
            return
        if QMessageBox.question(self, "取消任务",
                                "终止当前子进程？服务计算会立即停止。") != QMessageBox.Yes:
            return
        self.runner.cancel()

    def _run(self, fn, done, label):
        self.busy(label)
        w = Worker(fn)
        self._workers.append(w)
        self.active_worker = w

        def finish(r):
            self.ready(); self.status("完成: " + label)
            done(r)

        def fail(e):
            self.ready(); self.status("失败: " + label, BAD)
            self._log("ERROR " + label + "\n" + e)
            box = QMessageBox(QMessageBox.Critical, label, e.split("\n")[0], QMessageBox.Ok, self)
            box.setDetailedText(e)
            box.exec()

        w.result.connect(finish); w.failed.connect(fail)
        w.finished.connect(lambda: self._workers.remove(w) if w in self._workers else None)
        w.start()

    def _export_png(self, pix, path, files, failures):
        """Save one PNG and record the outcome (never silent)."""
        if pix is None or pix.isNull():
            return False
        try:
            ok = pix.save(str(path), "PNG")
        except Exception as exc:            # pragma: no cover - Qt backend
            ok = False
            self._log("EXPORT error %s: %s" % (path.name, exc))
        if ok and path.exists() and path.stat().st_size > 0:
            files.append(path.name)
            return True
        failures.append(path.name)
        self._log("EXPORT failed " + str(path))
        return False

    def export_all(self, directory=None):
        if directory is None:
            suggested = _default_export_dir() / datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = QFileDialog.getExistingDirectory(
                self, "选择导出目录", str(suggested))
        if not directory:
            return None
        out = _writable_dir(directory)
        if out != Path(directory):
            self._log("EXPORT fallback dir " + str(out))
        files = []
        failures = []
        try:
            if self.last_ccd:
                # strip base64 payloads: images are exported as files, metrics stay small
                ccd = dict(self.last_ccd)
                ccd.pop("psf_png", None)
                ccd["frames"] = [{"defocus": f.get("defocus")} for f in ccd.get("frames", [])]
                (out / "ccd_metrics.json").write_text(
                    json.dumps(ccd, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                files.append("ccd_metrics.json")
            if self.psf is not None:
                np.save(out / "psf_stack.npy", np.asarray(self.psf))
                files.append("psf_stack.npy")
            if self.star_img.pixmap():
                self._export_png(self.star_img.pixmap(), out / "star_field.png", files, failures)
            if self.ccd_img.pixmap():
                self._export_png(self.ccd_img.pixmap(), out / "psf_montage.png", files, failures)
            if self.last_pd:
                (out / "pd_result.json").write_text(
                    json.dumps(self.last_pd, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                files.append("pd_result.json")
            if self.pd_img.pixmap():
                self._export_png(self.pd_img.pixmap(), out / "pd_bar.png", files, failures)
            if self.last_fea:
                (out / "fea_fit.json").write_text(
                    json.dumps(self.last_fea, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                files.append("fea_fit.json")
            if self.fea_img.pixmap():
                self._export_png(self.fea_img.pixmap(), out / "fea_surface.png", files, failures)
            if self.fea_bar.pixmap():
                self._export_png(self.fea_bar.pixmap(), out / "fea_bar.png", files, failures)
            if getattr(self, "fea3d_img", None) is not None and self.fea3d_img.pixmap():
                self._export_png(self.fea3d_img.pixmap(), out / "fea_3d.png", files, failures)
            if getattr(self, "last_s", None):
                (out / "sensitivity.json").write_text(
                    json.dumps(self.last_s, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                files.append("sensitivity.json")
            if hasattr(self, "ctrl_img") and self.ctrl_img.pixmap():
                self._export_png(self.ctrl_img.pixmap(), out / "sensitivity_heatmap.png", files, failures)
            manifest = {"schema_version": 1, "app": "TelescopeEngineeringQtNative",
                        "exported": datetime.now().isoformat(timespec="seconds"),
                        "inputs": {"ccd_star": (self.last_ccd or {}).get("star"),
                                   "pd_nfev": (self.last_pd or {}).get("nfev"),
                                   "fea_case": ((self.last_fea or {}).get("case") or {}).get("id")},
                        "labels": {"ccd": "SYNTHETIC-VALIDATED", "fea": "FEA-PROVISIONAL"},
                        "files": files, "failed_files": failures}
            (out / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if failures:
                manifest["failed_files"] = failures
                (out / "manifest.json").write_text(
                    json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                self._log("EXPORT partial: failed " + ", ".join(failures))
                self.status("部分导出失败: " + ",".join(failures), WARN)
            else:
                self.status("已导出到 " + str(out), GOOD)
                self._log("EXPORT ok " + str(out) + " files=" + str(len(files)))
            return out
        except Exception as e:
            box = QMessageBox(QMessageBox.Critical, "导出", str(e), QMessageBox.Ok, self)
            box.setDetailedText(traceback.format_exc()); box.exec()
            return None

    def _save_json(self, obj, default):
        start = str(_default_export_dir() / Path(default).with_suffix(".json").name)
        path, _ = QFileDialog.getSaveFileName(self, "导出 JSON", start,
                                              "JSON (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            self.status("已导出 " + path, GOOD)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败（无写权限）",
                                "%s\n\n请选择“文档”或“桌面”等可写目录。\n%s"
                                % (path, exc))

    def _save_pixmap(self, label, default):
        """Save a label image, verifying the write and falling back if needed.

        QPixmap.save() returns False (no exception) when the target is not
        writable, e.g. C:\\Windows or an install dir; the old code ignored that
        and the user saw nothing happen.
        """
        src = label.source_pixmap() if hasattr(label, "source_pixmap") else None
        pix = src if (src is not None and not src.isNull()) else label.pixmap()
        if pix is None or pix.isNull():
            self.status("没有可导出的图像", WARN)
            return False
        default_path = str(_default_export_dir() / default)
        path, _ = QFileDialog.getSaveFileName(self, "导出 PNG", default_path,
                                              "PNG (*.png)")
        if not path:
            return False
        candidates = [Path(path)]
        if Path(path).suffix.lower() != ".png":
            candidates = [Path(path).with_suffix(".png")]
        fallback = _default_export_dir() / Path(path).name
        if fallback.resolve() != candidates[0].resolve():
            candidates.append(fallback)
        last_err = ""
        for cand in candidates:
            try:
                ok = pix.save(str(cand), "PNG")
            except Exception as exc:          # pragma: no cover - Qt backend
                ok, last_err = False, str(exc)
            if ok and cand.exists() and cand.stat().st_size > 0:
                self.status("已导出 " + str(cand), GOOD)
                self._log("EXPORT png " + str(cand))
                return True
            last_err = last_err or "无法写入（目录只读或权限不足）"
        QMessageBox.warning(
            self, "导出失败",
            "PNG 写入失败：%s\n\n已尝试:\n%s\n\n请选择“文档”或“桌面”等可写目录。"
            % (last_err, "\n".join(str(c) for c in candidates)))
        self.status("导出失败（无写权限）", BAD)
        return False

    def _save_npy(self):
        if self.psf is None:
            return
        default = str(_default_export_dir() / "psf_stack.npy")
        path, _ = QFileDialog.getSaveFileName(self, "导出 NPY", default, "NumPy (*.npy)")
        if not path:
            return
        try:
            np.save(path, np.asarray(self.psf))
            self.status("已导出 " + path, GOOD)
        except Exception as exc:
            fallback = _default_export_dir() / Path(path).name
            try:
                np.save(fallback, np.asarray(self.psf))
                self.status("已导出到可写目录 " + str(fallback), WARN)
            except Exception:
                QMessageBox.warning(self, "导出失败", "%s\n%s" % (path, exc))

    # ---------- CCD ----------
    def tab_ccd(self):
        w = QWidget(); lay = QHBoxLayout(w)
        box = QGroupBox("输入"); form = QFormLayout(box)
        self.c4 = QDoubleSpinBox(); self.c4.setRange(-2, 2); self.c4.setSingleStep(0.05); self.c4.setValue(0.3)
        self.c8 = QDoubleSpinBox(); self.c8.setRange(-1, 1); self.c8.setSingleStep(0.02); self.c8.setValue(0.1)
        self.c9 = QDoubleSpinBox(); self.c9.setRange(-1, 1); self.c9.setSingleStep(0.02); self.c9.setValue(-0.05)
        self.noise = QDoubleSpinBox(); self.noise.setRange(0, 100); self.noise.setValue(5.0)
        self.stars = QSpinBox(); self.stars.setRange(3, 80); self.stars.setValue(25)
        self.defocus_edit = QLineEdit(DEFOCUS)
        form.addRow("Z4", self.c4); form.addRow("Z8", self.c8); form.addRow("Z9", self.c9)
        form.addRow("噪声 e-", self.noise); form.addRow("星数", self.stars)
        form.addRow("defocus waves", self.defocus_edit)

        phys = QGroupBox("探测器 / 星空物理参数")
        pf = QFormLayout(phys)
        self.exposure = QDoubleSpinBox(); self.exposure.setRange(0.01, 600); self.exposure.setValue(1.0)
        self.exposure.setSuffix(" s")
        self.read_noise = QDoubleSpinBox(); self.read_noise.setRange(0.1, 100); self.read_noise.setValue(5.0)
        self.read_noise.setSuffix(" e-")
        self.sky_bg = QDoubleSpinBox(); self.sky_bg.setRange(0, 1000); self.sky_bg.setValue(1.0)
        self.sky_bg.setSuffix(" e-/px/s")
        self.seeing = QDoubleSpinBox(); self.seeing.setRange(0.1, 5.0); self.seeing.setSingleStep(0.05)
        self.seeing.setValue(0.7); self.seeing.setSuffix(" arcsec")
        self.plate_scale = QDoubleSpinBox(); self.plate_scale.setDecimals(4)
        self.plate_scale.setRange(0.005, 2.0); self.plate_scale.setValue(0.05)
        self.plate_scale.setSuffix(" arcsec/px")
        self.density = QDoubleSpinBox(); self.density.setRange(0, 200); self.density.setValue(5.0)
        self.density.setSuffix(" /arcmin2")
        self.mag_limit = QDoubleSpinBox(); self.mag_limit.setRange(6, 24); self.mag_limit.setValue(16.0)
        self.fov = QDoubleSpinBox(); self.fov.setRange(0.5, 60); self.fov.setValue(2.56)
        self.fov.setSuffix(" arcmin")
        self.bad_px = QDoubleSpinBox(); self.bad_px.setDecimals(4)
        self.bad_px.setRange(0, 0.05); self.bad_px.setSingleStep(0.0005); self.bad_px.setValue(0.0005)
        self.cr_rate = QDoubleSpinBox(); self.cr_rate.setDecimals(5)
        self.cr_rate.setRange(0, 0.01); self.cr_rate.setSingleStep(0.0001); self.cr_rate.setValue(0.0001)
        self.seed_box = QSpinBox(); self.seed_box.setRange(0, 10 ** 6); self.seed_box.setValue(7)
        for label, widget in (("曝光", self.exposure),
                              ("读出噪声", self.read_noise),
                              ("天空背景", self.sky_bg),
                              ("视宁度", self.seeing),
                              ("像元比例", self.plate_scale),
                              ("星密度", self.density),
                              ("星等极限", self.mag_limit),
                              ("视场", self.fov),
                              ("坏像元比例", self.bad_px),
                              ("宇宙线率", self.cr_rate),
                              ("随机种子", self.seed_box)):
            pf.addRow(label, widget)
        form.addRow(phys)
        run = QPushButton("生成合成 CCD + 5 帧 PSF"); run.clicked.connect(self.run_ccd)
        fits = QPushButton("上传真实 FITS"); fits.setObjectName("Ghost"); fits.clicked.connect(self.run_fits)
        form.addRow(run); form.addRow(fits)
        lay.addWidget(box, 1)

        right = QVBoxLayout()
        card = QGroupBox("CCD / PSF preview"); rl = QVBoxLayout(card)
        self.preview_tabs = QTabWidget()
        # star-field tab
        star_w = QWidget(); sl = QVBoxLayout(star_w)
        srow = QHBoxLayout(); srow.addWidget(QLabel("帧"))
        self.star_combo = QComboBox(); self.star_combo.currentIndexChanged.connect(self.show_star)
        srow.addWidget(self.star_combo, 1); sl.addLayout(srow)
        self.star_img = ScaledImageLabel("星场帧", (420, 300))
        self.star_img.setMinimumSize(640, 360)
        self.star_img.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;color:#6b7f94;")
        sl.addWidget(self.star_img, 1)
        self.preview_tabs.addTab(star_w, "星空帧")
        # PSF tab
        psf_w = QWidget(); pl = QVBoxLayout(psf_w)
        self.ccd_img = ScaledImageLabel("PSF 5 帧", (420, 200))
        self.ccd_img.setMinimumSize(640, 360)
        self.ccd_img.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;color:#6b7f94;")
        pl.addWidget(self.ccd_img, 1)
        self.preview_tabs.addTab(psf_w, "PSF 5 帧")
        rl.addWidget(self.preview_tabs, 1)
        bar = QHBoxLayout()
        for t, fn in (("导出星场 PNG", self._save_star),
                      ("导出 PSF PNG", lambda: self._save_pixmap(self.ccd_img, "ccd_psf.png")),
                      ("导出 NPY", self._save_npy),
                      ("导出 metrics", lambda: self._save_json(self.last_ccd or {}, "ccd_metrics.json"))):
            b = QPushButton(t); b.setObjectName("Ghost"); b.clicked.connect(fn); bar.addWidget(b)
        rl.addLayout(bar)
        self.ccd_log = QTextEdit(); self.ccd_log.setReadOnly(True); self.ccd_log.setMaximumHeight(130)
        right.addWidget(card, 3); right.addWidget(self.ccd_log, 1)
        lay.addLayout(right, 3)
        return w

    def _defocus(self):
        return [float(x) for x in self.defocus_edit.text().split(",") if x.strip()]

    def _ccd_payload(self):
        return {
            "c4": self.c4.value(), "c8": self.c8.value(), "c9": self.c9.value(),
            "noise": self.noise.value(), "stars": self.stars.value(),
            "defocus": self._defocus(),
            "exposure_s": self.exposure.value(),
            "read_noise_e": self.read_noise.value(),
            "sky_background_e_per_pix_s": self.sky_bg.value(),
            "seeing_arcsec": self.seeing.value(),
            "plate_scale_arcsec_per_pix": self.plate_scale.value(),
            "star_density_per_arcmin2": self.density.value(),
            "mag_limit": self.mag_limit.value(),
            "fov_arcmin": self.fov.value(),
            "bad_pixel_fraction": self.bad_px.value(),
            "cosmic_ray_rate": self.cr_rate.value(),
            "seed": self.seed_box.value(),
        }

    def run_ccd(self):
        payload = self._ccd_payload()
        self._run_process("ccd", payload, self._on_ccd, "CCD 合成")

    def _populate_frames(self, frames):
        self.frames = frames or []
        self.star_combo.blockSignals(True); self.star_combo.clear()
        for i, f in enumerate(self.frames):
            d = f.get("defocus") if isinstance(f, dict) else None
            self.star_combo.addItem("frame %d%s" % (i, "" if d is None else "  (%+.1f waves)" % d))
        self.star_combo.blockSignals(False)
        self.show_star(0)

    def show_star(self, idx):
        if not self.frames or idx < 0 or idx >= len(self.frames):
            return
        f = self.frames[idx]
        url = f.get("raw") if isinstance(f, dict) else str(f)
        self.star_img.set_source_pixmap(_png(url))

    def _save_star(self):
        # reuse the checked exporter so a read-only target cannot fail silently
        self._save_pixmap(self.star_img, "ccd_star.png")

    def _on_ccd(self, r):
        self.last_ccd = r
        self.psf = np.asarray(r["psf"], dtype=float); self.defocus = r["defocus"]
        self._populate_frames(r.get("frames", []))
        self._show_stack(self.psf)
        meta = r.get("meta") or {}
        self.ccd_log.setPlainText(
            "frames=%s shape=%s label=%s\n"
            "stars=%s  seeing=%s\" 曝光=%s s  饱和=%s  坏像元=%s  宇宙线=%s\n"
            "background=%.1f e-/px  read_noise=%s e-\nmetrics=%s" % (
                r["metrics"]["n_frames"], self.psf.shape, r.get("label"),
                meta.get("n_stars_rendered"), meta.get("seeing_arcsec"),
                meta.get("exposure_s"), meta.get("saturation_count"),
                meta.get("bad_pixel_count"), meta.get("cosmic_ray_count"),
                float(meta.get("background_e_per_pix") or 0.0),
                meta.get("read_noise_e"), r["metrics"]))

    def run_fits(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select FITS", "", "FITS (*.fits *.fit)")
        if not files:
            return
        df = self._defocus()
        self._run_process("fits_paths", {"paths": files, "defocus": df},
                          self._on_fits, "真实 FITS")

    def _on_fits(self, r):
        self.last_ccd = r
        self.psf = np.asarray(r["psf"], dtype=float); self.defocus = r.get("defocus", self._defocus())
        self._populate_frames(r.get("frames", []))
        self._show_stack(self.psf)
        self.ccd_log.setPlainText("real FITS: %d frames, shape=%s, star=%s" % (len(self.psf), self.psf.shape, r.get("star")))

    # ---------- PD ----------
    def tab_pd(self):
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        btn = QPushButton("对当前 PSF 做 PD 反演"); btn.clicked.connect(self.run_pd)
        exp = QPushButton("导出结果 JSON"); exp.setObjectName("Ghost")
        exp.clicked.connect(lambda: self._save_json(self.last_pd or {}, "pd_result.json"))
        row.addWidget(btn); row.addWidget(exp); row.addStretch(1)
        lay.addLayout(row)
        self.pd_img = QLabel("Zernike 柱状图")
        self.pd_img.setAlignment(Qt.AlignCenter)
        self.pd_img.setMinimumSize(520, 280)
        self.pd_img.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;color:#6b7f94;")
        lay.addWidget(self.pd_img, 2)
        self.pd_log = QTextEdit(); self.pd_log.setReadOnly(True); self.pd_log.setMaximumHeight(150)
        lay.addWidget(self.pd_log, 1)
        return w

    def run_pd(self):
        if self.psf is None:
            QMessageBox.information(self, "PD", "先生成/上传 PSF")
            return
        psf, df = self.psf.tolist(), self.defocus

        self._run_process("pd", {"psf": psf, "defocus": df},
                          self._on_pd, "PD 反演")

    def _on_pd(self, r):
        self.last_pd = r
        self.pd_log.setPlainText("Z=%s\nnfev=%s cost=%s\nresidual_rms=%s" %
                                 (r["zernike"], r["nfev"], r["cost"], r.get("residual_rms")))
        try:
            z = r.get("zernike", {})
            if isinstance(z, dict):
                z = [z.get(k, 0.0) for k in ("Z4", "Z8", "Z9")]
            self.pd_img.setPixmap(_bar_png(["Z4", "Z8", "Z9"], z, "PD Zernike [waves]").scaled(
                self.pd_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.pd_log.append("bar error: " + str(e))

    # ---------- FEA ----------
    def tab_fea(self):
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.case_box = QComboBox(); self.case_box.setMinimumWidth(360)
        self.case_box.currentTextChanged.connect(self.load_case)
        refresh = QPushButton("刷新"); refresh.setObjectName("Ghost"); refresh.clicked.connect(self._refresh_cases)
        upload = QPushButton("上传 FEA CSV"); upload.clicked.connect(self.upload_fea)
        row.addWidget(QLabel("工况")); row.addWidget(self.case_box, 1)
        row.addWidget(refresh); row.addWidget(upload)
        lay.addLayout(row)
        self.fea_img = ScaledImageLabel("FEA 面形", (720, 420))
        self.fea3d_img = ScaledImageLabel("3D 结构视图（加载工况后自动生成）", (380, 300))
        img_row = QHBoxLayout()
        img_row.addWidget(self.fea_img, 3)
        img_row.addWidget(self.fea3d_img, 2)
        self.structure_img = ScaledImageLabel("FEA 模型结构（点击生成）", (380, 300))
        img_row.addWidget(self.structure_img, 2)
        lay.addLayout(img_row, 3)
        self.fea_bar = QLabel("Zernike 柱状图")
        self.fea_bar.setAlignment(Qt.AlignCenter)
        self.fea_bar.setMinimumSize(600, 220)
        self.fea_bar.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;color:#6b7f94;")
        lay.addWidget(self.fea_bar, 2)
        bar = QHBoxLayout()
        self.auto3d = QCheckBox("加载后自动 3D")
        self.auto3d.setChecked(True)
        row.addWidget(self.auto3d)
        struct = QPushButton("FEA 模型结构")
        struct.setObjectName("Ghost")
        struct.clicked.connect(self.run_fea_structure)
        row.addWidget(struct)
        viz = QPushButton("3D 结构视图"); viz.clicked.connect(self.run_fea3d)
        vpng = QPushButton("导出 3D PNG"); vpng.setObjectName("Ghost")
        vpng.clicked.connect(lambda: self._save_pixmap(self.fea3d_img, "fea_3d.png"))
        png = QPushButton("导出 PNG"); png.setObjectName("Ghost")
        png.clicked.connect(lambda: self._save_pixmap(self.fea_img, "fea_surface.png"))
        js = QPushButton("导出拟合 JSON"); js.setObjectName("Ghost")
        js.clicked.connect(lambda: self._save_json(self.last_fea or {}, "fea_fit.json"))
        bp = QPushButton("导出柱状图"); bp.setObjectName("Ghost")
        bp.clicked.connect(lambda: self._save_pixmap(self.fea_bar, "fea_zernike.png"))
        bar.addWidget(viz); bar.addWidget(vpng)
        bar.addWidget(png); bar.addWidget(js); bar.addWidget(bp); bar.addStretch(1)
        lay.addLayout(bar)
        self.fea_log = QTextEdit(); self.fea_log.setReadOnly(True); self.fea_log.setMaximumHeight(130)
        lay.addWidget(self.fea_log, 1)
        return w

    def _refresh_cases(self):
        try:
            from products.sensitivity import service as sens
            cases = sens.list_cases()["cases"]
            self.case_box.blockSignals(True); self.case_box.clear()
            for c in cases:
                self.case_box.addItem(c["id"])
            self.case_box.blockSignals(False)
        except Exception as e:
            self.fea_log.setPlainText("cases error: " + str(e))

    def load_case(self, cid):
        if not cid:
            return
        self._run_process("fea_case", {"cid": cid}, self._on_fea, "加载 FEA")

    def _remember_fea_xyz(self, d, raw=None):
        """Keep node arrays (mm/mm/nm) so the 3D button has data to render."""
        if raw and len(raw.get("x") or []) >= 4:
            case = (d or {}).get("case") or {}
            self._fea_xyz = {
                "x": raw["x"], "y": raw["y"], "z": raw["z"],
                "label": raw.get("label") or "uploaded",
                "obscuration": float(case.get("obscuration_ratio") or 0.0),
                "aperture_radius_mm": float(case.get("aperture_radius_mm") or 0.0),
            }
            return
        try:
            g = d.get("grid") or {}
            gx = np.asarray(g.get("x") or [], dtype=float)
            gy = np.asarray(g.get("y") or [], dtype=float)
            vals = np.asarray([np.nan if v is None else float(v)
                               for row in (g.get("z") or []) for v in row],
                              dtype=float)
            if gx.size < 2 or gy.size < 2 or vals.size != gx.size * gy.size:
                self._fea_xyz = None
                return
            xx, yy = np.meshgrid(gx, gy)
            xx, yy = xx.ravel(), yy.ravel()
            ok = np.isfinite(vals)
            case = d.get("case") or {}
            self._fea_xyz = {"x": xx[ok].tolist(), "y": yy[ok].tolist(),
                             "z": vals[ok].tolist(),
                             "label": str(case.get("id") or "FEA"),
                             "obscuration": float(case.get("obscuration_ratio") or 0.0),
                             "aperture_radius_mm": float(case.get("aperture_radius_mm") or 0.0)}
        except Exception:
            self._fea_xyz = None

    def _on_fea(self, d, raw=None):
        self.last_fea = d
        self._remember_fea_xyz(d, raw)
        self.fea_img.set_source_pixmap(_fea_png(d))
        self.fea_log.setPlainText("case=%s\nsynthetic=%s targets=%s\nresidual=%s" %
                                  (d["case"].get("id"), d["case"].get("synthetic"),
                                   d["fit"]["targets_nm"], d["fit"]["residual_rms_nm"]))
        try:
            terms = d["fit"]["terms"][:15]
            labels = [t["label"] for t in terms]
            vals = d["fit"]["coefficients_nm"][:15]
            self.fea_bar.setPixmap(_bar_png(labels, vals, "FEA Zernike [nm]").scaled(
                self.fea_bar.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.fea_log.append("bar error: " + str(e))
        # a new surface invalidates the previous 3D render; auto-render if asked
        self.fea3d_img.clear_source()
        self.fea3d_img.setText("3D 结构视图（正在生成…）")
        if getattr(self, "auto3d", None) is not None and self.auto3d.isChecked():
            self.run_fea3d()

    def run_fea_structure(self):
        """Render the FEA mirror model (documented geometry + support pads)."""
        case = (self.last_fea or {}).get("case") or {}
        r_outer = float(case.get("aperture_radius_mm") or 1280.0)
        obsc = float(case.get("obscuration_ratio") or (700.0 / 2560.0))
        payload = {
            "outer_radius_mm": r_outer,
            "inner_radius_mm": max(r_outer * obsc, 1.0),
            "thickness_mm": 260.0,
            "support_rings": [[375.0, 6], [625.0, 12], [875.0, 12], [1125.0, 6]],
            "side_supports": 24,
            "label": str(case.get("id") or "M1"),
        }
        self._run_process("fea_structure", payload, self._on_fea_structure,
                          "FEA 模型结构")

    def _on_fea_structure(self, r):
        import base64 as _b64
        try:
            pix = QPixmap(); pix.loadFromData(_b64.b64decode(r["png3d_b64"]))
            if pix.isNull():
                raise ValueError("PNG decode failed")
        except Exception as exc:
            self.fea_log.append("structure error: " + str(exc))
            return
        self.structure_img.set_source_pixmap(pix)
        geo = r.get("geometry") or {}
        self.fea_log.append(
            "structure: %s OD=%.0f ID=%.0f t=%.0f pads=%s/%s (%s)" % (
                r.get("label"), geo.get("outer_radius_mm", 0) * 2,
                geo.get("inner_radius_mm", 0) * 2, geo.get("thickness_mm", 0),
                geo.get("axial_pads"), geo.get("lateral_pads"),
                r.get("note", "")))

    def run_fea3d(self):
        data = getattr(self, "_fea_xyz", None) or {}
        if len(data.get("x") or []) < 4:
            self.fea_log.append("3D: 请先加载工况或上传 FEA CSV 再生成")
            self.status("3D 视图: 无面形数据", WARN)
            return
        payload = {"x": data["x"], "y": data["y"], "z": data["z"],
                   "label": data.get("label") or "FEA",
                   "azim": -60, "elev": 25,
                   "obscuration": float(data.get("obscuration") or 0.0),
                   "aperture_radius_mm": float(data.get("aperture_radius_mm") or 0.0)}
        self._run_process("fea_viz", payload, self._on_fea3d, "FEA 3D")

    def _on_fea3d(self, r):
        # The rendered PNG is already decoded into a QPixmap below; keep only the
        # small metadata/stats record here so the GUI does not retain a second
        # (base64) copy of the image for the lifetime of the window.
        self.last_fea3d = {k: v for k, v in r.items() if k != "png3d_b64"}
        try:
            raw = base64.b64decode(r["png3d_b64"])
            pix = QPixmap(); pix.loadFromData(raw)
            if pix.isNull():
                raise ValueError("PNG 解码失败")
        except Exception as e:
            self.fea_log.append("3D error: " + str(e))
            self.status("3D 视图失败", BAD)
            return
        self.fea3d_img.set_source_pixmap(pix)
        del raw
        st = r.get("stats") or {}
        try:
            rms = float(st.get("rms_nm", float("nan")))
            pv = float(st.get("pv_nm", float("nan")))
        except (TypeError, ValueError):
            rms = pv = float("nan")
        self.fea3d_img.setToolTip("label=%s  n=%s" % (r.get("label"), st.get("n")))
        self.fea_log.append("3D: label=%s rms=%.4f nm pv=%.4f nm n=%s" %
                            (r.get("label"), rms, pv, st.get("n")))

    def upload_fea(self):
        path, _ = QFileDialog.getOpenFileName(self, "FEA CSV", "", "CSV (*.csv *.txt)")
        if not path:
            return
        try:
            xs, ys, zs = [], [], []
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    xs.append(float(row["x_mm"])); ys.append(float(row["y_mm"])); zs.append(float(row["dz_nm"]))
        except Exception as e:
            QMessageBox.critical(self, "FEA", str(e)); return
        raw = {"x": xs, "y": ys, "z": zs, "label": Path(path).stem}
        self._run_process("fea_upload", dict(raw),
                          lambda d, raw=raw: self._on_fea(d, raw), "上传 FEA")

    # ---------- ZMX ----------
    def tab_zmx(self):
        w = QWidget(); lay = QVBoxLayout(w)
        btn = QPushButton("选择 ZMX 并解析"); btn.clicked.connect(self.load_zmx)
        row = QHBoxLayout(); row.addWidget(btn); row.addStretch(1)
        lay.addLayout(row)
        img_row = QHBoxLayout()
        self.zmx_img = ScaledImageLabel("ZMX 二维光路（选择 ZMX 后生成）", (420, 300))
        self.zmx3d_img = ScaledImageLabel("ZMX 三维光路（选择 ZMX 后生成）", (420, 300))
        img_row.addWidget(self.zmx_img, 1)
        img_row.addWidget(self.zmx3d_img, 1)
        lay.addLayout(img_row, 3)
        exp = QHBoxLayout()
        b2 = QPushButton("导出 2D 光路"); b2.setObjectName("Ghost")
        b2.clicked.connect(lambda: self._save_pixmap(self.zmx_img, "zmx_layout_2d.png"))
        b3 = QPushButton("导出 3D 光路"); b3.setObjectName("Ghost")
        b3.clicked.connect(lambda: self._save_pixmap(self.zmx3d_img, "zmx_layout_3d.png"))
        exp.addWidget(b2); exp.addWidget(b3); exp.addStretch(1)
        lay.addLayout(exp)
        self.zmx_log = QTextEdit(); self.zmx_log.setReadOnly(True)
        self.zmx_log.setMaximumHeight(160)
        lay.addWidget(self.zmx_log, 1)
        return w

    def load_zmx(self):
        path, _ = QFileDialog.getOpenFileName(self, "ZMX", "", "ZMX (*.zmx)")
        if not path:
            return
        # Parsing is fast (~ms); do it synchronously so a QThread/QFileDialog
        # interaction cannot leave the ZMX tab disabled/unresponsive.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            from products.zemax.zmx_parser import parse_zmx_bytes
            d = parse_zmx_bytes(Path(path).read_bytes())
            self._on_zmx((Path(path).name, d))
            self.status("完成: ZMX 解析")
        except Exception as e:
            self.status("ZMX 解析失败", BAD)
            box = QMessageBox(QMessageBox.Critical, "ZMX", str(e), QMessageBox.Ok, self)
            box.setDetailedText(traceback.format_exc())
            box.exec()
        finally:
            QApplication.restoreOverrideCursor()

    def _on_zmx(self, r):
        name, d = r
        self.last_zmx = {"name": name, "parsed": d}
        self.zmx_log.setPlainText(
            "name=%s\nEPD=%s stop=%s surfaces=%d\nwavelengths=%d" %
            (name, d.get("aperture_mm"), d.get("stop"),
             len(d.get("surfaces", [])), len(d.get("wavelengths_um", []))))
        self._run_process("zmx_layout", {"parsed": d}, self._on_zmx_layout, "ZMX 光路")

    def _on_zmx_layout(self, r):
        import base64 as _b64
        for key, attr, tag in (("png2d_b64", "zmx_img", "2D"),
                               ("png3d_b64", "zmx3d_img", "3D")):
            raw = r.get(key)
            if not raw:
                continue
            pix = QPixmap(); pix.loadFromData(_b64.b64decode(raw))
            label = getattr(self, attr, None)
            if label is not None and not pix.isNull():
                label.set_source_pixmap(pix)
        self.zmx_log.append("layout: track=%s mm stop=%s surfaces=%s" %
                            (r.get("track_mm"), r.get("stop"),
                             len(r.get("surfaces") or [])))
        for note in (r.get("notes") or [])[:5]:
            self.zmx_log.append("note: " + str(note))

    # ---------- S/control ----------
    def tab_ctrl(self):
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        s_btn = QPushButton("计算 M2 灵敏度 S（Zemax 优先）")
        s_btn.clicked.connect(self.run_s)
        u_btn = QPushButton("求解控制量 u"); u_btn.clicked.connect(self.run_u)
        cal_btn = QPushButton("实测标定演示（CCD+PD 合成闭环）")
        cal_btn.clicked.connect(self.run_sens_demo)
        row.addWidget(s_btn); row.addWidget(u_btn); row.addWidget(cal_btn)
        row.addStretch(1)
        lay.addLayout(row)
        self.ctrl_img = QLabel("S heatmap"); self.ctrl_img.setAlignment(Qt.AlignCenter)
        self.ctrl_img.setMinimumSize(560, 260)
        self.ctrl_img.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;color:#6b7f94;")
        lay.addWidget(self.ctrl_img, 2)
        self.ctrl_log = QTextEdit(); self.ctrl_log.setReadOnly(True); self.ctrl_log.setMaximumHeight(170)
        lay.addWidget(self.ctrl_log, 1)
        return w

    def _show_matrix_heatmap(self, S, row_labels, col_labels, title, target):
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            S = np.asarray(S, dtype=float)
            fig, ax = plt.subplots(figsize=(7, 3.2))
            im = ax.imshow(S, cmap="RdBu_r", aspect="auto")
            ax.set_xticks(range(S.shape[1]))
            ax.set_xticklabels(col_labels, rotation=25, ha="right")
            ax.set_yticks(range(S.shape[0])); ax.set_yticklabels(row_labels)
            for i in range(S.shape[0]):
                for j in range(S.shape[1]):
                    ax.text(j, i, "%.2f" % S[i, j], ha="center", va="center", fontsize=8)
            ax.set_title(title, fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
            buf = io.BytesIO(); fig.savefig(buf, format="png", dpi=130, bbox_inches="tight")
            plt.close(fig)
            pix = QPixmap(); pix.loadFromData(buf.getvalue())
            target.setPixmap(pix.scaled(target.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        except Exception as e:
            self.ctrl_log.append("heatmap error: " + str(e))

    def run_s(self):
        self._run_process("s", {}, self._on_s, "计算 S")

    def _on_s(self, d):
        self.last_s = d
        self.ctrl_log.setPlainText(
            "source=%s label=%s\nS=%s\nrank=%s condition=%s nullspace=%s\n%s" % (
                d.get("source"), d.get("label"), d.get("values"),
                d.get("rank"), d.get("condition"), d.get("nullspace"),
                d.get("note", "")))
        self._show_matrix_heatmap(d["values"], d.get("target_names", []),
                                  d.get("control_names", []),
                                  "Sensitivity S [waves/(mm or mrad)] — %s" %
                                  (d.get("source", "")), self.ctrl_img)

    def run_sens_demo(self):
        self._run_process("sens_demo", {}, self._on_sens_demo, "实测标定")

    def _on_sens_demo(self, d):
        S = d.get("S") or []
        self.ctrl_log.setPlainText(
            "实测标定（合成闭环） label=%s\n"
            "S_meas err_rms=%.4f corr=%.4f rank=%s cond=%.2f\n"
            "hysteresis=%s\nper-pose n=%s drift_rms=%.2e\n%s" % (
                d.get("label"), d.get("err_rms", float("nan")),
                d.get("corr", float("nan")), d.get("rank"),
                d.get("condition", float("nan")),
                d.get("hysteresis"), d.get("n_poses"),
                d.get("drift_rms", float("nan")), d.get("note", "")))
        self._show_matrix_heatmap(S, d.get("terms", []),
                                  d.get("dofs", d.get("dof_names", [])),
                                  "Measured S (synthetic closed loop) [waves/(mm or mrad)]",
                                  self.ctrl_img)

    def run_u(self):
        self._run_process("u", {"target": [0.23377, -0.09266, 0.04633]},
                          self._on_u, "求解 u")

    def _on_u(self, d):
        self.ctrl_log.setPlainText("u=%s\nresidual=%s" % (d.get("control"), d.get("residual")))

    # ---------- demo ----------
    def tab_demo(self):
        w = QWidget(); lay = QVBoxLayout(w)
        box = QGroupBox("PSF 形态演示")
        row = QHBoxLayout(box)
        self.d4 = QDoubleSpinBox(); self.d4.setRange(-1.5, 1.5); self.d4.setValue(0.3); self.d4.setSingleStep(0.05)
        self.d8 = QDoubleSpinBox(); self.d8.setRange(-0.5, 0.5); self.d8.setValue(0.1); self.d8.setSingleStep(0.02)
        self.d9 = QDoubleSpinBox(); self.d9.setRange(-0.5, 0.5); self.d9.setValue(-0.05); self.d9.setSingleStep(0.02)
        row.addWidget(QLabel("Z4")); row.addWidget(self.d4)
        row.addWidget(QLabel("Z8")); row.addWidget(self.d8)
        row.addWidget(QLabel("Z9")); row.addWidget(self.d9)
        btn = QPushButton("生成 5 帧"); btn.clicked.connect(self.run_demo)
        row.addWidget(btn); row.addStretch(1)
        lay.addWidget(box)
        self.demo_img = QLabel("5-frame PSF morphology"); self.demo_img.setAlignment(Qt.AlignCenter)
        self.demo_img.setMinimumSize(900, 250); self.demo_img.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;")
        lay.addWidget(self.demo_img, 1)
        self.demo_log = QTextEdit(); self.demo_log.setReadOnly(True); self.demo_log.setMaximumHeight(110)
        lay.addWidget(self.demo_log)
        return w

    def run_demo(self):
        c4, c8, c9 = self.d4.value(), self.d8.value(), self.d9.value()
        def fn():
            from products.optics.forward import forward_stack_numpy
            return forward_stack_numpy([c4, c8, c9], n=64)
        self._run(fn, self._on_demo, "PSF 演示")

    def _on_demo(self, stack):
        self.demo_img.setPixmap(_frames_png(stack).scaled(self.demo_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.demo_log.setPlainText("shape=%s, frames=5, defocus=-2..2 waves" % (stack.shape,))


def run_gui(t0=None):
    import time
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    try:
        w = QtApp(startup_t0=t0)
    except Exception as e:
        base = (Path(sys.executable).resolve().parent
                if getattr(sys, "frozen", False) else ROOT / "results")
        try:
            base.mkdir(parents=True, exist_ok=True)
            (base / "qt_startup_error.log").write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        box = QMessageBox(QMessageBox.Critical, "Qt startup", str(e), QMessageBox.Ok)
        box.setDetailedText(traceback.format_exc())
        box.exec()
        raise
    w.show()
    try:
        if w._startup_t0 is not None:
            w._log("startup %.2fs (entry->show)" % (time.perf_counter() - w._startup_t0))
    except Exception:
        pass
    app.exec()


