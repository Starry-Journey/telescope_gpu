"""Image labels that keep the original pixmap and rescale on resize.

Split from desktop/gui.py (old lines 64-102).  ScaledImageLabel fixes the
"3D view shows nothing" class of bugs: a pixmap scaled to a label size captured
while the tab was not laid out (or a 0x0 size) used to become invisible and
never recovered.
"""
from __future__ import annotations

from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import QLabel


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
