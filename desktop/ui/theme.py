"""Shared GUI constants: default parameters, stylesheet and status colours.

Split from desktop/gui.py (old lines 37-61) so every ui module can import them
without depending on the desktop.gui facade (which imports this package).
"""
from __future__ import annotations

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
