"""Native Qt desktop frontend (PySide6).

This is the primary desktop UI; it calls the Python services directly instead
of embedding React/HTTP.  Tabs:
  1) CCD/PSF: synthetic + real FITS
  2) PD: invert the current PSF stack
  3) FEA/3B: 66-term cases and CSV upload
  4) 3A: ZMX structure / job status

No browser, no PyWebView, no React required for the desktop path.
"""
from __future__ import annotations

import argparse
import base64
import csv
import io
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QMessageBox, QPushButton,
    QSpinBox, QTabWidget, QTextEdit, QVBoxLayout, QWidget, QGroupBox,
)

DEFOCUS = "-2,-1,0,1,2"


def _pixmap_from_data_url(url: str) -> QPixmap:
    raw = base64.b64decode(url.split(",", 1)[1])
    pix = QPixmap()
    pix.loadFromData(raw)
    return pix


def _fea_pixmap(detail: dict) -> QPixmap:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    g = detail["grid"]
    z = np.asarray(g["z"], dtype=float)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(z, origin="lower", cmap="viridis")
    ax.set_title("FEA surface dz [nm]", fontsize=9)
    fig.colorbar(im, ax=ax, shrink=0.8)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    pix = QPixmap(); pix.loadFromData(buf.getvalue())
    return pix


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Telescope Engineering · Qt")
        self.resize(1280, 860)
        self.psf = None
        self.defocus = None
        self.truth = None
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)
        self.tabs.addTab(self._ccd_tab(), "① CCD / PSF")
        self.tabs.addTab(self._pd_tab(), "② 相位差反演")
        self.tabs.addTab(self._fea_tab(), "③B FEA / 66 项")
        self.tabs.addTab(self._zmx_tab(), "③A ZMX")
        self.tabs.addTab(self._ctrl_tab(), "③C S / 控制")
        self._refresh_cases()

    # ---------------- CCD ----------------
    def _ccd_tab(self):
        w = QWidget(); lay = QHBoxLayout(w)
        form_box = QGroupBox("输入"); form = QFormLayout(form_box)
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
        fits = QPushButton("上传真实 FITS"); fits.clicked.connect(self.run_fits)
        form.addRow(run); form.addRow(fits)
        lay.addWidget(form_box, 1)
        right = QVBoxLayout()
        self.ccd_img = QLabel("PSF preview"); self.ccd_img.setAlignment(Qt.AlignCenter)
        self.ccd_img.setMinimumSize(420, 420); self.ccd_img.setStyleSheet("border:1px solid #aaa")
        self.ccd_log = QTextEdit(); self.ccd_log.setReadOnly(True)
        right.addWidget(self.ccd_img, 2); right.addWidget(self.ccd_log, 1)
        lay.addLayout(right, 2)
        return w

    def _defocus(self):
        return [float(x) for x in self.defocus_edit.text().split(",") if x.strip()]

    def run_ccd(self):
        try:
            from web.ccd_pipeline_service import run_ccd_pipeline
            r = run_ccd_pipeline(self.c4.value(), self.c8.value(), self.c9.value(),
                                 self.noise.value(), self.stars.value(), self._defocus())
            self.psf = np.asarray(r["psf"], dtype=float)
            self.defocus = r["defocus"]
            self.truth = np.array([self.c4.value(), self.c8.value(), self.c9.value()])
            self.ccd_img.setPixmap(_pixmap_from_data_url(r["psf_png"]).scaled(
                self.ccd_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.ccd_log.setPlainText("frames=%s shape=%s\nmetrics=%s" %
                                      (r["metrics"]["n_frames"], self.psf.shape, r["metrics"]))
        except Exception as e:
            QMessageBox.critical(self, "CCD", str(e))

    def run_fits(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select FITS", "", "FITS (*.fits *.fit)")
        if not files:
            return
        try:
            from web.fits_pipeline_service import run_fits_pipeline
            payload = [{"name": Path(p).name, "data_b64": base64.b64encode(Path(p).read_bytes()).decode()}
                       for p in files]
            r = run_fits_pipeline(payload, self._defocus())
            self.psf = np.asarray(r["psf"], dtype=float)
            self.defocus = r.get("defocus", self._defocus())
            self.truth = None
            self.ccd_img.setPixmap(_pixmap_from_data_url(r["psf_png"]).scaled(
                self.ccd_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.ccd_log.setPlainText("real FITS: %d frames, shape=%s, star=%s" %
                                      (len(self.psf), self.psf.shape, r.get("star")))
        except Exception as e:
            QMessageBox.critical(self, "FITS", str(e))

    # ---------------- PD ----------------
    def _pd_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        self.pd_btn = QPushButton("对当前 PSF 做 PD 反演")
        self.pd_btn.clicked.connect(self.run_pd)
        self.pd_log = QTextEdit(); self.pd_log.setReadOnly(True)
        lay.addWidget(self.pd_btn); lay.addWidget(self.pd_log, 1)
        return w

    def run_pd(self):
        if self.psf is None:
            QMessageBox.information(self, "PD", "先生成/上传 PSF")
            return
        try:
            from web.pd_service import invert_psf
            r = invert_psf(self.psf.tolist(), self.defocus, None)
            self.pd_log.setPlainText("Z=%s\nnfev=%s cost=%s" %
                                     (r["zernike"], r["nfev"], r["cost"]))
        except Exception as e:
            QMessageBox.critical(self, "PD", str(e))

    # ---------------- FEA ----------------
    def _fea_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        self.case_box = QComboBox(); self.case_box.setMinimumWidth(300)
        self.case_box.currentTextChanged.connect(self.load_case)
        refresh = QPushButton("刷新工况"); refresh.clicked.connect(self._refresh_cases)
        upload = QPushButton("上传 FEA CSV"); upload.clicked.connect(self.upload_fea)
        row.addWidget(QLabel("工况")); row.addWidget(self.case_box, 1)
        row.addWidget(refresh); row.addWidget(upload)
        lay.addLayout(row)
        self.fea_img = QLabel(); self.fea_img.setAlignment(Qt.AlignCenter)
        self.fea_img.setMinimumSize(480, 380)
        self.fea_log = QTextEdit(); self.fea_log.setReadOnly(True)
        lay.addWidget(self.fea_img, 2); lay.addWidget(self.fea_log, 1)
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
        try:
            from products.sensitivity import service as sens
            d = sens.case_detail(cid)
            self.fea_img.setPixmap(_fea_pixmap(d).scaled(
                self.fea_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.fea_log.setPlainText("case=%s\nsynthetic=%s targets=%s\nresidual=%s" %
                                      (cid, d["case"].get("synthetic"), d["fit"]["targets_nm"],
                                       d["fit"]["residual_rms_nm"]))
        except Exception as e:
            self.fea_log.setPlainText("load error: " + str(e))

    def upload_fea(self):
        path, _ = QFileDialog.getOpenFileName(self, "FEA CSV", "", "CSV (*.csv *.txt)")
        if not path:
            return
        try:
            xs, ys, zs = [], [], []
            with open(path, newline="", encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    xs.append(float(row["x_mm"])); ys.append(float(row["y_mm"])); zs.append(float(row["dz_nm"]))
            from products.sensitivity import service as sens
            d = sens.custom_surface(xs, ys, zs, 1250.0, 0.28, Path(path).stem)
            self.fea_img.setPixmap(_fea_pixmap(d).scaled(
                self.fea_img.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
            self.fea_log.setPlainText("uploaded %s: %d pts, coeff=%d" %
                                      (Path(path).name, len(xs), len(d["fit"]["coefficients_nm"])))
        except Exception as e:
            QMessageBox.critical(self, "FEA", str(e))


    # ---------------- S / control ----------------
    def _ctrl_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        row = QHBoxLayout()
        s_btn = QPushButton("计算 M2 代理 S")
        u_btn = QPushButton("求解控制量 u")
        s_btn.clicked.connect(self.run_s)
        u_btn.clicked.connect(self.run_u)
        row.addWidget(s_btn); row.addWidget(u_btn)
        lay.addLayout(row)
        self.ctrl_log = QTextEdit(); self.ctrl_log.setReadOnly(True)
        lay.addWidget(self.ctrl_log, 1)
        return w

    def run_s(self):
        try:
            from web import framework_service as fw
            d = fw.sensitivity()
            self.ctrl_log.setPlainText("S=%s\nrank/condition=%s" %
                                       (d.get("matrix"), d.get("analysis", {})))
        except Exception as e:
            QMessageBox.critical(self, "S", str(e))

    def run_u(self):
        try:
            from web import framework_service as fw
            d = fw.control([0.23377, -0.09266, 0.04633])
            self.ctrl_log.setPlainText("u=%s\nresidual=%s" %
                                       (d.get("control"), d.get("residual")))
        except Exception as e:
            QMessageBox.critical(self, "u", str(e))

    # ---------------- ZMX ----------------
    def _zmx_tab(self):
        w = QWidget(); lay = QVBoxLayout(w)
        btn = QPushButton("选择 ZMX 并解析"); btn.clicked.connect(self.load_zmx)
        self.zmx_log = QTextEdit(); self.zmx_log.setReadOnly(True)
        lay.addWidget(btn); lay.addWidget(self.zmx_log, 1)
        return w

    def load_zmx(self):
        path, _ = QFileDialog.getOpenFileName(self, "ZMX", "", "ZMX (*.zmx)")
        if not path:
            return
        try:
            from products.zemax.zmx_parser import parse_zmx_bytes
            d = parse_zmx_bytes(Path(path).read_bytes())
            self.zmx_log.setPlainText("name=%s\nEPD=%s stop=%s surfaces=%d\nwavelengths=%d" %
                                      (Path(path).name, d.get("aperture_mm"), d.get("stop"),
                                       len(d.get("surfaces", [])), len(d.get("wavelengths_um", []))))
        except Exception as e:
            QMessageBox.critical(self, "ZMX", str(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        print("Qt frontend imports OK")
        return
    app = QApplication(sys.argv)
    w = MainWindow(); w.show(); app.exec()


if __name__ == "__main__":
    main()
