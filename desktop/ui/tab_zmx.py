"""Tab ③A ZMX: parse a Zemax file and render its 2D/3D optical layout.

Split from desktop/gui.py (old lines 1043-1110) into the TabZMX mixin.
Shared ``self`` attributes: zmx_img/zmx3d_img/zmx_log are created by this
tab's ``tab_zmx`` builder; ``last_zmx`` is first assigned in ``_on_zmx``
(unchanged from the original monolith, which also never pre-initialised it).
Cross-mixin helpers: _run_process/status (AppShell), _save_pixmap
(ExportActions).  The zmx_parser import stays lazy and synchronous (parsing
is ~ms; a QThread/QFileDialog interaction used to leave the tab disabled).
"""
from __future__ import annotations

import traceback
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QApplication, QFileDialog, QHBoxLayout, QMessageBox, QPushButton,
    QTextEdit, QVBoxLayout, QWidget,
)

from desktop.ui.image_widgets import ScaledImageLabel
from desktop.ui.theme import BAD


class TabZMX:
    """Methods for the ③A ZMX tab (no own __init__)."""

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
