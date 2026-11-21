"""Tab ① CCD / PSF: input form, star-field + PSF montage previews, FITS upload.

Split from desktop/gui.py (old lines 354-360 and 632-794) into the TabCCD
mixin.  Mixins carry methods only; every shared ``self`` attribute is created
either in AppShell.__init__ (psf/defocus/last_ccd/frames/...) or by this
tab's own ``tab_ccd`` builder (c4..seed_box, star_img, ccd_img, ...).
Cross-mixin helpers used here: _run_process/_status live in AppShell,
_save_pixmap/_save_json/_save_npy live in ExportActions.
"""
from __future__ import annotations

import numpy as np

from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QFormLayout, QGroupBox,
    QHBoxLayout, QLabel, QLineEdit, QPushButton, QSpinBox, QTabWidget,
    QTextEdit, QVBoxLayout, QWidget,
)

from desktop.ui.image_widgets import ScaledImageLabel
from desktop.ui.plot_helpers import _frames_png, _png
from desktop.ui.theme import DEFOCUS


class TabCCD:
    """Methods for the ① CCD / PSF tab (no own __init__)."""

    def tab_ccd(self):
        w = QWidget(); lay = QHBoxLayout(w)
        box = QGroupBox("输入"); form = QFormLayout(box)
        self.c4 = QDoubleSpinBox(); self.c4.setRange(-2, 2); self.c4.setSingleStep(0.05); self.c4.setValue(0.3)
        self.c8 = QDoubleSpinBox(); self.c8.setRange(-1, 1); self.c8.setSingleStep(0.02); self.c8.setValue(0.1)
        self.c9 = QDoubleSpinBox(); self.c9.setRange(-1, 1); self.c9.setSingleStep(0.02); self.c9.setValue(-0.05)
        self.noise = QDoubleSpinBox(); self.noise.setRange(0, 100); self.noise.setValue(5.0)
        self.stars = QSpinBox(); self.stars.setRange(3, 80); self.stars.setValue(25)
        self.defocus_edit = QLineEdit(DEFOCUS)
        form.addRow("Z4", self.c4); form.addRow("Z8", self.c8); form.addRow("Z9", self.c9)
        form.addRow("噪声 e-", self.noise); form.addRow("星数", self.stars)
        form.addRow("defocus waves", self.defocus_edit)

        phys = QGroupBox("探测器 / 星空物理参数")
        pf = QFormLayout(phys)
        self.exposure = QDoubleSpinBox(); self.exposure.setRange(0.01, 600); self.exposure.setValue(1.0)
        self.exposure.setSuffix(" s")
        self.read_noise = QDoubleSpinBox(); self.read_noise.setRange(0.1, 100); self.read_noise.setValue(5.0)
        self.read_noise.setSuffix(" e-")
        self.sky_bg = QDoubleSpinBox(); self.sky_bg.setRange(0, 1000); self.sky_bg.setValue(1.0)
        self.sky_bg.setSuffix(" e-/px/s")
        self.seeing = QDoubleSpinBox(); self.seeing.setRange(0.1, 5.0); self.seeing.setSingleStep(0.05)
        self.seeing.setValue(0.7); self.seeing.setSuffix(" arcsec")
        self.plate_scale = QDoubleSpinBox(); self.plate_scale.setDecimals(4)
        self.plate_scale.setRange(0.005, 2.0); self.plate_scale.setValue(0.05)
        self.plate_scale.setSuffix(" arcsec/px")
        self.density = QDoubleSpinBox(); self.density.setRange(0, 200); self.density.setValue(5.0)
        self.density.setSuffix(" /arcmin2")
        self.mag_limit = QDoubleSpinBox(); self.mag_limit.setRange(6, 24); self.mag_limit.setValue(16.0)
        self.fov = QDoubleSpinBox(); self.fov.setRange(0.5, 60); self.fov.setValue(2.56)
        self.fov.setSuffix(" arcmin")
        self.bad_px = QDoubleSpinBox(); self.bad_px.setDecimals(4)
        self.bad_px.setRange(0, 0.05); self.bad_px.setSingleStep(0.0005); self.bad_px.setValue(0.0005)
        self.cr_rate = QDoubleSpinBox(); self.cr_rate.setDecimals(5)
        self.cr_rate.setRange(0, 0.01); self.cr_rate.setSingleStep(0.0001); self.cr_rate.setValue(0.0001)
        self.seed_box = QSpinBox(); self.seed_box.setRange(0, 10 ** 6); self.seed_box.setValue(7)
        for label, widget in (("曝光", self.exposure),
                              ("读出噪声", self.read_noise),
                              ("天空背景", self.sky_bg),
                              ("视宁度", self.seeing),
                              ("像元比例", self.plate_scale),
                              ("星密度", self.density),
                              ("星等极限", self.mag_limit),
                              ("视场", self.fov),
                              ("坏像元比例", self.bad_px),
                              ("宇宙线率", self.cr_rate),
                              ("随机种子", self.seed_box)):
            pf.addRow(label, widget)
        form.addRow(phys)
        run = QPushButton("生成合成 CCD + 5 帧 PSF"); run.clicked.connect(self.run_ccd)
        fits = QPushButton("上传真实 FITS"); fits.setObjectName("Ghost"); fits.clicked.connect(self.run_fits)
        form.addRow(run); form.addRow(fits)
        lay.addWidget(box, 1)

        right = QVBoxLayout()
        card = QGroupBox("CCD / PSF preview"); rl = QVBoxLayout(card)
        self.preview_tabs = QTabWidget()
        # star-field tab
        star_w = QWidget(); sl = QVBoxLayout(star_w)
        srow = QHBoxLayout(); srow.addWidget(QLabel("帧"))
        self.star_combo = QComboBox(); self.star_combo.currentIndexChanged.connect(self.show_star)
        srow.addWidget(self.star_combo, 1); sl.addLayout(srow)
        self.star_img = ScaledImageLabel("星场帧", (420, 300))
        self.star_img.setMinimumSize(640, 360)
        self.star_img.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;color:#6b7f94;")
        sl.addWidget(self.star_img, 1)
        self.preview_tabs.addTab(star_w, "星空帧")
        # PSF tab
        psf_w = QWidget(); pl = QVBoxLayout(psf_w)
        self.ccd_img = ScaledImageLabel("PSF 5 帧", (420, 200))
        self.ccd_img.setMinimumSize(640, 360)
        self.ccd_img.setStyleSheet("border:1px dashed #2a3a4a;border-radius:8px;color:#6b7f94;")
        pl.addWidget(self.ccd_img, 1)
        self.preview_tabs.addTab(psf_w, "PSF 5 帧")
        rl.addWidget(self.preview_tabs, 1)
        bar = QHBoxLayout()
        for t, fn in (("导出星场 PNG", self._save_star),
                      ("导出 PSF PNG", lambda: self._save_pixmap(self.ccd_img, "ccd_psf.png")),
                      ("导出 NPY", self._save_npy),
                      ("导出 metrics", lambda: self._save_json(self.last_ccd or {}, "ccd_metrics.json"))):
            b = QPushButton(t); b.setObjectName("Ghost"); b.clicked.connect(fn); bar.addWidget(b)
        rl.addLayout(bar)
        self.ccd_log = QTextEdit(); self.ccd_log.setReadOnly(True); self.ccd_log.setMaximumHeight(130)
        right.addWidget(card, 3); right.addWidget(self.ccd_log, 1)
        lay.addLayout(right, 3)
        return w

    def _show_stack(self, stack):
        stack = np.asarray(stack, dtype=float)
        if stack.ndim == 2:
            stack = stack[None]
        pix = _frames_png(stack)
        self.ccd_img.set_source_pixmap(pix)
        self.ccd_img.setToolTip("frames=%s" % (stack.shape,))

    def _defocus(self):
        return [float(x) for x in self.defocus_edit.text().split(",") if x.strip()]

    def _ccd_payload(self):
        return {
            "c4": self.c4.value(), "c8": self.c8.value(), "c9": self.c9.value(),
            "noise": self.noise.value(), "stars": self.stars.value(),
            "defocus": self._defocus(),
            "exposure_s": self.exposure.value(),
            "read_noise_e": self.read_noise.value(),
            "sky_background_e_per_pix_s": self.sky_bg.value(),
            "seeing_arcsec": self.seeing.value(),
            "plate_scale_arcsec_per_pix": self.plate_scale.value(),
            "star_density_per_arcmin2": self.density.value(),
            "mag_limit": self.mag_limit.value(),
            "fov_arcmin": self.fov.value(),
            "bad_pixel_fraction": self.bad_px.value(),
            "cosmic_ray_rate": self.cr_rate.value(),
            "seed": self.seed_box.value(),
        }

    def run_ccd(self):
        payload = self._ccd_payload()
        self._run_process("ccd", payload, self._on_ccd, "CCD 合成")

    def _populate_frames(self, frames):
        self.frames = frames or []
        self.star_combo.blockSignals(True); self.star_combo.clear()
        for i, f in enumerate(self.frames):
            d = f.get("defocus") if isinstance(f, dict) else None
            self.star_combo.addItem("frame %d%s" % (i, "" if d is None else "  (%+.1f waves)" % d))
        self.star_combo.blockSignals(False)
        self.show_star(0)

    def show_star(self, idx):
        if not self.frames or idx < 0 or idx >= len(self.frames):
            return
        f = self.frames[idx]
        url = f.get("raw") if isinstance(f, dict) else str(f)
        self.star_img.set_source_pixmap(_png(url))

    def _save_star(self):
        # reuse the checked exporter so a read-only target cannot fail silently
        self._save_pixmap(self.star_img, "ccd_star.png")

    def _on_ccd(self, r):
        self.last_ccd = r
        self.psf = np.asarray(r["psf"], dtype=float); self.defocus = r["defocus"]
        self._populate_frames(r.get("frames", []))
        self._show_stack(self.psf)
        meta = r.get("meta") or {}
        self.ccd_log.setPlainText(
            "frames=%s shape=%s label=%s\n"
            "stars=%s  seeing=%s\" 曝光=%s s  饱和=%s  坏像元=%s  宇宙线=%s\n"
            "background=%.1f e-/px  read_noise=%s e-\nmetrics=%s" % (
                r["metrics"]["n_frames"], self.psf.shape, r.get("label"),
                meta.get("n_stars_rendered"), meta.get("seeing_arcsec"),
                meta.get("exposure_s"), meta.get("saturation_count"),
                meta.get("bad_pixel_count"), meta.get("cosmic_ray_count"),
                float(meta.get("background_e_per_pix") or 0.0),
                meta.get("read_noise_e"), r["metrics"]))

    def run_fits(self):
        files, _ = QFileDialog.getOpenFileNames(self, "Select FITS", "", "FITS (*.fits *.fit)")
        if not files:
            return
        df = self._defocus()
        self._run_process("fits_paths", {"paths": files, "defocus": df},
                          self._on_fits, "真实 FITS")

    def _on_fits(self, r):
        self.last_ccd = r
        self.psf = np.asarray(r["psf"], dtype=float); self.defocus = r.get("defocus", self._defocus())
        self._populate_frames(r.get("frames", []))
        self._show_stack(self.psf)
        self.ccd_log.setPlainText("real FITS: %d frames, shape=%s, star=%s" % (len(self.psf), self.psf.shape, r.get("star")))
