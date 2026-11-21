"""Tab ④ 演示: 5-frame PSF morphology rendered by the in-process Worker thread.

Split from desktop/gui.py (old lines 1196-1226) into the TabDemo mixin.
Shared ``self`` attributes: demo_img/demo_log are created by this tab's
``tab_demo`` builder; the short numpy render runs through AppShell._run
(Worker QThread), which is why closeEvent must shut workers down safely.
"""
from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QDoubleSpinBox, QGroupBox, QHBoxLayout, QLabel, QPushButton, QTextEdit,
    QVBoxLayout, QWidget,
)

from desktop.ui.plot_helpers import _frames_png


class TabDemo:
    """Methods for the ④ 演示 tab (no own __init__)."""

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
