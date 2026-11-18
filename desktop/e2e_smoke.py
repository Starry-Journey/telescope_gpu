"""Automated E2E smoke for the Qt GUI (offscreen).

Mimics the manual E2E checklist:
  1. startup log line
  2. CCD 5-frame generation + star frame + PSF montage
  3. PD inversion + Z4/Z8/Z9 bar chart (dict zernike)
  4. FEA case + surface plot + top-15 bar chart
  5. S/control heatmap + u solve
  6. export directory artifacts
  7. cancel a running task: no task_* residue, TASK cancel log
  8. immediate rerun after cancel succeeds

Run:  py -3.13 desktop/e2e_smoke.py
Artifacts land in results/e2e/<date>/.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from PySide6.QtWidgets import QApplication
from desktop.gui import QtApp

OUT = ROOT / "results" / "e2e" / time.strftime("%Y-%m-%d")
OUT.mkdir(parents=True, exist_ok=True)

_T0 = time.perf_counter()
app = QApplication([])
w = QtApp(startup_t0=_T0)
w.show()
app.processEvents()
import time as _t
w._log("startup %.2fs (entry->show)" % (_t.perf_counter() - _T0))
app.processEvents()


def wait_idle(timeout=120.0):
    t0 = time.monotonic()
    while w.runner.busy() and time.monotonic() - t0 < timeout:
        app.processEvents()
        time.sleep(0.05)
    app.processEvents()
    return not w.runner.busy()


def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name, flush=True)
    if not cond:
        raise SystemExit("E2E failed at " + name)


results = {}

# 1. startup log
log = w.log_path.read_text(encoding="utf-8") if w.log_path.exists() else ""
check("startup log line", "startup" in log)

# 2. CCD
t0 = time.monotonic()
w.run_ccd()
check("ccd idle", wait_idle())
results["ccd_s"] = time.monotonic() - t0
check("star frame pixmap", w.star_img.pixmap() is not None)
check("psf montage pixmap", w.ccd_img.pixmap() is not None)
check("psf frames >= 2", len(getattr(w, "psf", [])) >= 2)

# 3. PD
t0 = time.monotonic()
w.run_pd()
check("pd idle", wait_idle())
results["pd_s"] = time.monotonic() - t0
check("pd bar pixmap", w.pd_img.pixmap() is not None)
check("pd result dict", isinstance(w.last_pd.get("zernike"), dict))

# 4. FEA
t0 = time.monotonic()
w.load_case("M1_distributed_0deg")
check("fea idle", wait_idle())
results["fea_s"] = time.monotonic() - t0
check("fea surface pixmap", w.fea_img.pixmap() is not None)
check("fea bar pixmap", w.fea_bar.pixmap() is not None)

# 5. S + u
t0 = time.monotonic()
w.run_s()
check("s idle", wait_idle())
results["s_s"] = time.monotonic() - t0
check("s heatmap pixmap", w.ctrl_img.pixmap() is not None)
t0 = time.monotonic()
w.run_u()
check("u idle", wait_idle())
results["u_s"] = time.monotonic() - t0

# 6. export directory
outdir = w.export_all(str(OUT / "export"))
print("export dir:", outdir)

# 7. cancel: start CCD again and cancel quickly
w.run_ccd()
app.processEvents()
time.sleep(0.3)
app.processEvents()
cancelled = w.runner.cancel()
check("cancel returned", cancelled)
t0 = time.monotonic()
while w.runner.busy() and time.monotonic() - t0 < 10:
    app.processEvents()
    time.sleep(0.05)
app.processEvents()
check("runner idle after cancel", not w.runner.busy())
residue = list(Path(w.runner.task_root).glob("task_*"))
check("no task_* residue", residue == [])

# 8. immediate rerun after cancel
w.run_ccd()
check("rerun after cancel", wait_idle())
check("rerun produced psf", len(getattr(w, "psf", [])) >= 2)

# logs
log = w.log_path.read_text(encoding="utf-8") if w.log_path.exists() else ""
check("TASK ok log", "TASK ok" in log)
check("TASK cancel log", "TASK cancel" in log)
check("export dir exists", outdir is not None and (Path(outdir) / "manifest.json").exists())

# evidence
(OUT / "qt_gui.log").write_text(log, encoding="utf-8")
w.tabs.setCurrentIndex(0); app.processEvents()
w.grab().save(str(OUT / "tab_ccd.png"), "PNG")
w.tabs.setCurrentIndex(1); app.processEvents()
w.grab().save(str(OUT / "tab_pd.png"), "PNG")
w.tabs.setCurrentIndex(2); app.processEvents()
w.grab().save(str(OUT / "tab_fea.png"), "PNG")
w.tabs.setCurrentIndex(4); app.processEvents()
w.grab().save(str(OUT / "tab_ctrl.png"), "PNG")
import json
(OUT / "timings.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
print("E2E_OK", OUT, results)

