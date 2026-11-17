"""Native Qt desktop frontend (PySide6).

Primary desktop path.  Does not embed React/WebView and does not need FastAPI.
Services are imported lazily so the window opens fast and PyInstaller can keep
the same service modules as the Web path.

Tabs: CCD/PSF, PD, FEA/66-term, ZMX, S/control, demo.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QGroupBox, QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox,
    QPushButton, QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget,
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
            self.failed.emit(str(e))


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
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Telescope Engineering · Qt")
        self.resize(1440, 940)
        self.psf = None
        self.defocus = None
        self.last_ccd = None
        self.last_pd = None
        self.last_fea = None
        self._workers = []
        self.active_worker = None
        self.log_path = ROOT / "results" / "qt_gui.log"
        self.log_path.parent.mkdir(parents=True, exist_ok=True)

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
        self._log("Qt GUI started")
        self._refresh_cases()

    # ---------- helpers ----------
    def status(self, text, color=GOOD):
        self._log(text)
        self.badge.setText(text.upper() if len(text) < 18 else text)
        self.badge.setStyleSheet(
            "background:#0b1117;color:%s;border:1px solid #2a3a4a;border-radius:11px;padding:3px 10px;font-weight:600;" % color)
        self.statusBar().showMessage(text)

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
        self.ccd_img.setPixmap(pix.scaled(self.ccd_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.ccd_img.setToolTip("frames=%s" % (stack.shape,))

    def busy(self, text):
        self.status("运行中: " + text, WARN)
        self.tabs.setEnabled(False)
        self.cancel_btn.setEnabled(True)

    def ready(self):
        self.tabs.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.active_worker = None

    def cancel_task(self):
        w = self.active_worker
        if w is None or not w.isRunning():
            return
        if QMessageBox.question(self, "取消任务",
                                "强制终止当前任务？可能留下临时文件。") != QMessageBox.Yes:
            return
        w.requestInterruption()
        w.terminate()
        w.wait(1500)
        self.ready()
        self.status("已取消", WARN)

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
            QMessageBox.critical(self, label, e)

        w.result.connect(finish); w.failed.connect(fail)
        w.finished.connect(lambda: self._workers.remove(w) if w in self._workers else None)
        w.start()

    def _save_json(self, obj, default):
        path, _ = QFileDialog.getSaveFileName(self, "导出 JSON", str(Path(default).with_suffix(".json")),
                                              "JSON (*.json)")
        if path:
            Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
            self.status("已导出 " + path)

    def _save_pixmap(self, label, default):
        pix = label.pixmap()
        if pix is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 PNG", default, "PNG (*.png)")
        if path:
            pix.save(path, "PNG"); self.status("已导出 " + path)

    def _save_npy(self):
        if self.psf is None:
            return
        path, _ = QFileDialog.getSaveFileName(self, "导出 NPY", "psf_stack.npy", "NumPy (*.npy)")
        if path:
            np.save(path, np.asarray(self.psf))
            self.status("已导出 " + path)

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
        run = QPushButton("生成合成 CCD + 5 帧 PSF"); run.clicked.connect(self.run_ccd)
        fits = QPushButton("上传真实 FITS"); fits.setObjectName("Ghost"); fits.clicked.connect(self.run_fits)
        form.addRow(run); form.addRow(fits)
        lay.addWidget(box, 1)

        right = QVBoxLayout()
        card = QGroupBox("PSF preview"); rl = QVBoxLayout(card)
        self.ccd_img = QLabel("PSF preview"); self.ccd_img.setAlignment(Qt.AlignCenter)
        self.ccd_img.setMinimumSize(640, 420)
        self.ccd_img.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;color:#6b7f94;")
        rl.addWidget(self.ccd_img, 1)
        bar = QHBoxLayout()
        for t, fn in (("导出 PNG", lambda: self._save_pixmap(self.ccd_img, "ccd_psf.png")),
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

    def run_ccd(self):
        c4, c8, c9 = self.c4.value(), self.c8.value(), self.c9.value()
        noise, stars, df = self.noise.value(), self.stars.value(), self._defocus()

        def fn():
            from web.ccd_pipeline_service import run_ccd_pipeline
            return run_ccd_pipeline(c4, c8, c9, noise, stars, df)
        self._run(fn, self._on_ccd, "CCD 合成")

    def _on_ccd(self, r):
        self.last_ccd = r
        self.psf = np.asarray(r["psf"], dtype=float); self.defocus = r["defocus"]
        self._show_stack(self.psf)
        self.ccd_log.setPlainText("frames=%s shape=%s\nmetrics=%s" %
                                  (r["metrics"]["n_frames"], self.psf.shape, r["metrics"]))

    def run_fits(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select FITS", "", "FITS (*.fits *.fit)")
        if not files:
            return
        payload = [{"name": Path(p).name, "data_b64": base64.b64encode(Path(p).read_bytes()).decode()} for p in files]
        df = self._defocus()

        def fn():
            from web.fits_pipeline_service import run_fits_pipeline
            return run_fits_pipeline(payload, df)
        self._run(fn, self._on_fits, "真实 FITS")

    def _on_fits(self, r):
        self.last_ccd = r
        self.psf = np.asarray(r["psf"], dtype=float); self.defocus = r.get("defocus", self._defocus())
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
        self.pd_log = QTextEdit(); self.pd_log.setReadOnly(True)
        lay.addWidget(self.pd_log, 1)
        return w

    def run_pd(self):
        if self.psf is None:
            QMessageBox.information(self, "PD", "先生成/上传 PSF")
            return
        psf, df = self.psf.tolist(), self.defocus

        def fn():
            from web.pd_service import invert_psf
            return invert_psf(psf, df, None)
        self._run(fn, self._on_pd, "PD 反演")

    def _on_pd(self, r):
        self.last_pd = r
        self.pd_log.setPlainText("Z=%s\nnfev=%s cost=%s\nresidual_rms=%s" %
                                 (r["zernike"], r["nfev"], r["cost"], r.get("residual_rms")))

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
        self.fea_img = QLabel(); self.fea_img.setAlignment(Qt.AlignCenter)
        self.fea_img.setMinimumSize(720, 420); self.fea_img.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;")
        lay.addWidget(self.fea_img, 3)
        bar = QHBoxLayout()
        png = QPushButton("导出 PNG"); png.setObjectName("Ghost")
        png.clicked.connect(lambda: self._save_pixmap(self.fea_img, "fea_surface.png"))
        js = QPushButton("导出拟合 JSON"); js.setObjectName("Ghost")
        js.clicked.connect(lambda: self._save_json(self.last_fea or {}, "fea_fit.json"))
        bar.addWidget(png); bar.addWidget(js); bar.addStretch(1)
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
            if cases:
                self.load_case(cases[0]["id"])
        except Exception as e:
            self.fea_log.setPlainText("cases error: " + str(e))

    def load_case(self, cid):
        if not cid:
            return
        def fn():
            from products.sensitivity import service as sens
            return sens.case_detail(cid)
        self._run(fn, self._on_fea, "加载 FEA")

    def _on_fea(self, d):
        self.last_fea = d
        self.fea_img.setPixmap(_fea_png(d).scaled(self.fea_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.fea_log.setPlainText("case=%s\nsynthetic=%s targets=%s\nresidual=%s" %
                                  (d["case"].get("id"), d["case"].get("synthetic"),
                                   d["fit"]["targets_nm"], d["fit"]["residual_rms_nm"]))

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
        def fn():
            from products.sensitivity import service as sens
            return sens.custom_surface(xs, ys, zs, 1250.0, 0.28, Path(path).stem)
        self._run(fn, self._on_fea, "上传 FEA")

    # ---------- ZMX ----------
    def tab_zmx(self):
        w = QWidget(); lay = QVBoxLayout(w)
        btn = QPushButton("选择 ZMX 并解析"); btn.clicked.connect(self.load_zmx)
        row = QHBoxLayout(); row.addWidget(btn); row.addStretch(1)
        lay.addLayout(row)
        self.zmx_log = QTextEdit(); self.zmx_log.setReadOnly(True)
        lay.addWidget(self.zmx_log, 1)
        return w

    def load_zmx(self):
        path, _ = QFileDialog.getOpenFileName(self, "ZMX", "", "ZMX (*.zmx)")
        if not path:
            return
        def fn():
            from products.zemax.zmx_parser import parse_zmx_bytes
            return Path(path).name, parse_zmx_bytes(Path(path).read_bytes())
        self._run(fn, self._on_zmx, "解析 ZMX")

    def _on_zmx(self, r):
        name, d = r
        self.zmx_log.setPlainText("name=%s\nEPD=%s stop=%s surfaces=%d\nwavelengths=%d" %
                                  (name, d.get("aperture_mm"), d.get("stop"),
                                   len(d.get("surfaces", [])), len(d.get("wavelengths_um", []))))

    # ---------- S/control ----------
    def tab_ctrl(self):
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        s_btn = QPushButton("计算 M2 代理 S"); s_btn.clicked.connect(self.run_s)
        u_btn = QPushButton("求解控制量 u"); u_btn.clicked.connect(self.run_u)
        row.addWidget(s_btn); row.addWidget(u_btn); row.addStretch(1)
        lay.addLayout(row)
        self.ctrl_log = QTextEdit(); self.ctrl_log.setReadOnly(True)
        lay.addWidget(self.ctrl_log, 1)
        return w

    def run_s(self):
        def fn():
            from web import framework_service as fw
            return fw.sensitivity()
        self._run(fn, self._on_s, "计算 S")

    def _on_s(self, d):
        self.ctrl_log.setPlainText("S=%s\nanalysis=%s" % (d.get("matrix"), d.get("analysis", {})))

    def run_u(self):
        def fn():
            from web import framework_service as fw
            return fw.control([0.23377, -0.09266, 0.04633])
        self._run(fn, self._on_u, "求解 u")

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
            from ml.traditional import forward_stack_numpy
            return forward_stack_numpy([c4, c8, c9], n=64)
        self._run(fn, self._on_demo, "PSF 演示")

    def _on_demo(self, stack):
        self.demo_img.setPixmap(_frames_png(stack).scaled(self.demo_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        self.demo_log.setPlainText("shape=%s, frames=5, defocus=-2..2 waves" % (stack.shape,))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        print("Qt frontend imports OK")
        return
    if args.smoke_test:
        from web.ccd_pipeline_service import run_ccd_pipeline
        r = run_ccd_pipeline(0.3, 0.1, -0.05, 2.0, 8, [-1, 0, 1])
        print("smoke frames", len(r["psf"]), "shape", np.asarray(r["psf"]).shape)
        return
    app = QApplication(sys.argv)
    app.setStyleSheet(QSS)
    w = QtApp(); w.show(); app.exec()


if __name__ == "__main__":
    main()
