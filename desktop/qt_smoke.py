"""Headless Qt GUI smoke test (no window needed).

Run from repo:
    py -3.13 desktop/qt_smoke.py
It exercises CCD preview, star frame, PD bar, FEA bar.
"""
import os
import sys
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from desktop.gui import QtApp
from products.sensitivity import service as sens
from products.ccd.service import run_ccd_pipeline

app = QApplication([])
w = QtApp()
r = run_ccd_pipeline(0.3, 0.1, -0.05, 2.0, 8, [-1, 0, 1])
w._on_ccd(r)
assert w.star_img.pixmap() is not None, "star frame pixmap missing"
assert w.ccd_img.pixmap() is not None, "PSF montage missing"
assert w.star_combo.count() == 3, "star frame selector missing"
w._on_pd({"zernike": {"Z4": 0.306, "Z8": 0.091, "Z9": -0.045}, "nfev": 151, "cost": 1.2e-7,
          "residual_rms": 0.01})
assert w.pd_img.pixmap() is not None, "PD bar missing"
w._on_fea(sens.case_detail("M1_distributed_0deg"))
assert w.fea_bar.pixmap() is not None, "FEA bar missing"
out = ROOT / "results" / "qt_smoke"
out.mkdir(parents=True, exist_ok=True)
w.star_img.pixmap().save(str(out / "star.png"), "PNG")
w.ccd_img.pixmap().save(str(out / "psf_montage.png"), "PNG")
w.pd_img.pixmap().save(str(out / "pd_bar.png"), "PNG")
w.fea_bar.pixmap().save(str(out / "fea_bar.png"), "PNG")
print("QT_SMOKE_OK", out)
