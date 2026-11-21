"""Tab ② PD (phase diversity): invert the current PSF, show the Zernike bar.

Split from desktop/gui.py (old lines 797-834) into the TabPD mixin.
Shared ``self`` attributes: psf/defocus/last_pd come from AppShell.__init__,
pd_img/pd_log are created by this tab's ``tab_pd`` builder.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QLabel, QMessageBox, QPushButton, QTextEdit, QVBoxLayout, QWidget

from desktop.ui.plot_helpers import _bar_png


class TabPD:
    """Methods for the ② 相位差反演 tab (no own __init__)."""

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
