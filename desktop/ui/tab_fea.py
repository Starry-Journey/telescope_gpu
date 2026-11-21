"""Tab ③B FEA / 66 项: case loading, surface + 3D + structure views, CSV upload.

Split from desktop/gui.py (old lines 837-1040) into the TabFEA mixin.
Shared ``self`` attributes: last_fea/last_fea3d/_fea_xyz come from
AppShell.__init__; case_box/fea_img/fea3d_img/structure_img/fea_bar/auto3d/
fea_log are created by this tab's ``tab_fea`` builder.  Cross-mixin helpers:
_run_process/status (AppShell), _save_pixmap/_save_json (ExportActions).
"""
from __future__ import annotations

import base64
import csv
from pathlib import Path

import numpy as np

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMessageBox,
    QPushButton, QTextEdit, QVBoxLayout, QWidget,
)

from desktop.ui.image_widgets import ScaledImageLabel
from desktop.ui.plot_helpers import _bar_png, _fea_png
from desktop.ui.theme import BAD, WARN


class TabFEA:
    """Methods for the ③B FEA / 66 项 tab (no own __init__)."""

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
