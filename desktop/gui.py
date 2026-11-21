"""Native Qt desktop frontend (PySide6) — thin facade over desktop/ui/.

The implementation lives in the ``desktop.ui`` package; this module keeps the
composition and every historical import path working:

    from desktop.gui import QtApp, ScaledImageLabel, Worker, run_gui
    from desktop.gui import _writable_dir, _default_export_dir, _png, ...

Module map (former desktop/gui.py line ranges):
    desktop/ui/theme.py          DEFOCUS/QSS/GOOD/BAD/WARN        (37-61)
    desktop/ui/paths.py          ROOT/_writable_dir/_default_export_dir
                                                                  (22-24, 105-126)
    desktop/ui/image_widgets.py  ScaledImageLabel                 (64-102)
    desktop/ui/worker_thread.py  Worker                           (129-141)
    desktop/ui/plot_helpers.py   _png/_fea_png/_bar_png/_frames_png(144-196)
    desktop/ui/app_shell.py      AppShell: window shell, task glue,
                                 closeEvent worker shutdown       (199-465)
    desktop/ui/export_actions.py ExportActions: export/save      (304-307, 467-629)
    desktop/ui/tab_ccd.py        TabCCD   ① CCD/PSF               (354-360, 631-794)
    desktop/ui/tab_pd.py         TabPD    ② 相位差反演             (796-834)
    desktop/ui/tab_fea.py        TabFEA   ③B FEA/66 项             (836-1040)
    desktop/ui/tab_zmx.py        TabZMX   ③A ZMX                  (1042-1110)
    desktop/ui/tab_ctrl.py       TabCtrl  ③C S/控制                (1112-1194)
    desktop/ui/tab_demo.py       TabDemo  ④ 演示                   (1196-1226)

Services are imported lazily (inside the methods that use them) so the window
opens fast and PyInstaller can keep the same service modules as the Web path.
matplotlib likewise stays lazily imported with the Agg backend.

Tabs: CCD/PSF, PD, FEA/66-term, ZMX, S/control, demo.
"""
from __future__ import annotations

import sys
import traceback
from pathlib import Path

# repo root (this file sits at <root>/desktop/gui.py); needed before the
# desktop.* imports when gui.py is executed as a plain script
_HERE_ROOT = Path(__file__).resolve().parents[1]
if str(_HERE_ROOT) not in sys.path:
    sys.path.insert(0, str(_HERE_ROOT))

from PySide6.QtWidgets import QApplication, QMainWindow, QMessageBox  # noqa: E402

from desktop.ui.paths import ROOT  # noqa: E402  (== _HERE_ROOT; single definition)
from desktop.ui.theme import BAD, DEFOCUS, GOOD, QSS, WARN  # noqa: E402,F401
from desktop.ui.paths import _default_export_dir, _writable_dir  # noqa: E402,F401
from desktop.ui.image_widgets import ScaledImageLabel  # noqa: E402,F401
from desktop.ui.worker_thread import Worker  # noqa: E402,F401
from desktop.ui.plot_helpers import (  # noqa: E402,F401
    _bar_png, _fea_png, _frames_png, _png,
)
from desktop.ui.app_shell import AppShell  # noqa: E402,F401
from desktop.ui.export_actions import ExportActions  # noqa: E402,F401
from desktop.ui.tab_ccd import TabCCD  # noqa: E402,F401
from desktop.ui.tab_pd import TabPD  # noqa: E402,F401
from desktop.ui.tab_fea import TabFEA  # noqa: E402,F401
from desktop.ui.tab_zmx import TabZMX  # noqa: E402,F401
from desktop.ui.tab_ctrl import TabCtrl  # noqa: E402,F401
from desktop.ui.tab_demo import TabDemo  # noqa: E402,F401

__all__ = [
    # classes / entry point (imported by qt_app.py, qt_smoke.py,
    # e2e_smoke.py, desktop/tests/test_gui_helpers.py)
    "QtApp", "ScaledImageLabel", "Worker", "run_gui",
    # constants
    "DEFOCUS", "QSS", "GOOD", "BAD", "WARN", "ROOT",
    # plot helpers
    "_png", "_fea_png", "_bar_png", "_frames_png",
    # path helpers
    "_writable_dir", "_default_export_dir",
    # mixins (composition pieces of QtApp)
    "AppShell", "ExportActions",
    "TabCCD", "TabPD", "TabFEA", "TabZMX", "TabCtrl", "TabDemo",
]


class QtApp(TabCCD, TabPD, TabFEA, TabZMX, TabCtrl, TabDemo,
            ExportActions, AppShell, QMainWindow):
    """Composition root: window shell + per-tab mixins (see desktop/ui/).

    Only AppShell defines ``__init__`` (all cross-tab state lives there);
    every other mixin contributes methods, so the MRO has exactly one
    cooperative constructor chain ending at QMainWindow.
    """


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
