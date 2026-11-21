"""Export/save actions of the Qt window (mixin).

Split from desktop/gui.py (old lines 304-307 and 467-629) into the
ExportActions mixin.  Mixin carries methods only; every shared ``self``
attribute it touches (last_ccd/psf/last_pd/last_fea/last_s, star_img,
ccd_img, pd_img, fea_img, fea_bar, ctrl_img) is created either in
AppShell.__init__ or by the tab builders.  Directory helpers come from
desktop.ui.paths; status colours from desktop.ui.theme.

Behaviour note: ``export_all`` writes manifest.json exactly once — the old
code wrote it a second time on partial failure with identical content
(failed_files was already part of the first manifest).
"""
from __future__ import annotations

import json
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

from PySide6.QtCore import QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import QFileDialog, QMessageBox

from desktop.ui.paths import _default_export_dir, _writable_dir
from desktop.ui.theme import BAD, GOOD, WARN


class ExportActions:
    """Methods for exporting images/metrics and opening the export dir."""

    def open_export_dir(self):
        target = _default_export_dir()
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(target)))
        self.status("打开导出目录 " + str(target), GOOD)

    def _export_png(self, pix, path, files, failures):
        """Save one PNG and record the outcome (never silent)."""
        if pix is None or pix.isNull():
            return False
        try:
            ok = pix.save(str(path), "PNG")
        except Exception as exc:            # pragma: no cover - Qt backend
            ok = False
            self._log("EXPORT error %s: %s" % (path.name, exc))
        if ok and path.exists() and path.stat().st_size > 0:
            files.append(path.name)
            return True
        failures.append(path.name)
        self._log("EXPORT failed " + str(path))
        return False

    def export_all(self, directory=None):
        if directory is None:
            suggested = _default_export_dir() / datetime.now().strftime("%Y%m%d_%H%M%S")
            directory = QFileDialog.getExistingDirectory(
                self, "选择导出目录", str(suggested))
        if not directory:
            return None
        out = _writable_dir(directory)
        if out != Path(directory):
            self._log("EXPORT fallback dir " + str(out))
        files = []
        failures = []
        try:
            if self.last_ccd:
                # strip base64 payloads: images are exported as files, metrics stay small
                ccd = dict(self.last_ccd)
                ccd.pop("psf_png", None)
                ccd["frames"] = [{"defocus": f.get("defocus")} for f in ccd.get("frames", [])]
                (out / "ccd_metrics.json").write_text(
                    json.dumps(ccd, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                files.append("ccd_metrics.json")
            if self.psf is not None:
                np.save(out / "psf_stack.npy", np.asarray(self.psf))
                files.append("psf_stack.npy")
            if self.star_img.pixmap():
                self._export_png(self.star_img.pixmap(), out / "star_field.png", files, failures)
            if self.ccd_img.pixmap():
                self._export_png(self.ccd_img.pixmap(), out / "psf_montage.png", files, failures)
            if self.last_pd:
                (out / "pd_result.json").write_text(
                    json.dumps(self.last_pd, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                files.append("pd_result.json")
            if self.pd_img.pixmap():
                self._export_png(self.pd_img.pixmap(), out / "pd_bar.png", files, failures)
            if self.last_fea:
                (out / "fea_fit.json").write_text(
                    json.dumps(self.last_fea, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                files.append("fea_fit.json")
            if self.fea_img.pixmap():
                self._export_png(self.fea_img.pixmap(), out / "fea_surface.png", files, failures)
            if self.fea_bar.pixmap():
                self._export_png(self.fea_bar.pixmap(), out / "fea_bar.png", files, failures)
            if getattr(self, "fea3d_img", None) is not None and self.fea3d_img.pixmap():
                self._export_png(self.fea3d_img.pixmap(), out / "fea_3d.png", files, failures)
            if getattr(self, "last_s", None):
                (out / "sensitivity.json").write_text(
                    json.dumps(self.last_s, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
                files.append("sensitivity.json")
            if hasattr(self, "ctrl_img") and self.ctrl_img.pixmap():
                self._export_png(self.ctrl_img.pixmap(), out / "sensitivity_heatmap.png", files, failures)
            manifest = {"schema_version": 1, "app": "TelescopeEngineeringQtNative",
                        "exported": datetime.now().isoformat(timespec="seconds"),
                        "inputs": {"ccd_star": (self.last_ccd or {}).get("star"),
                                   "pd_nfev": (self.last_pd or {}).get("nfev"),
                                   "fea_case": ((self.last_fea or {}).get("case") or {}).get("id")},
                        "labels": {"ccd": "SYNTHETIC-VALIDATED", "fea": "FEA-PROVISIONAL"},
                        "files": files, "failed_files": failures}
            # single write: failed_files is already part of the manifest above
            # (the old code wrote the byte-identical file a second time on
            # partial failure — plain duplicate work, no information change)
            (out / "manifest.json").write_text(
                json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
            if failures:
                self._log("EXPORT partial: failed " + ", ".join(failures))
                self.status("部分导出失败: " + ",".join(failures), WARN)
            else:
                self.status("已导出到 " + str(out), GOOD)
                self._log("EXPORT ok " + str(out) + " files=" + str(len(files)))
            return out
        except Exception as e:
            box = QMessageBox(QMessageBox.Critical, "导出", str(e), QMessageBox.Ok, self)
            box.setDetailedText(traceback.format_exc()); box.exec()
            return None

    def _save_json(self, obj, default):
        start = str(_default_export_dir() / Path(default).with_suffix(".json").name)
        path, _ = QFileDialog.getSaveFileName(self, "导出 JSON", start,
                                              "JSON (*.json)")
        if not path:
            return
        try:
            Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2),
                                  encoding="utf-8")
            self.status("已导出 " + path, GOOD)
        except Exception as exc:
            QMessageBox.warning(self, "导出失败（无写权限）",
                                "%s\n\n请选择“文档”或“桌面”等可写目录。\n%s"
                                % (path, exc))

    def _save_pixmap(self, label, default):
        """Save a label image, verifying the write and falling back if needed.

        QPixmap.save() returns False (no exception) when the target is not
        writable, e.g. C:\\Windows or an install dir; the old code ignored that
        and the user saw nothing happen.
        """
        src = label.source_pixmap() if hasattr(label, "source_pixmap") else None
        pix = src if (src is not None and not src.isNull()) else label.pixmap()
        if pix is None or pix.isNull():
            self.status("没有可导出的图像", WARN)
            return False
        default_path = str(_default_export_dir() / default)
        path, _ = QFileDialog.getSaveFileName(self, "导出 PNG", default_path,
                                              "PNG (*.png)")
        if not path:
            return False
        candidates = [Path(path)]
        if Path(path).suffix.lower() != ".png":
            candidates = [Path(path).with_suffix(".png")]
        fallback = _default_export_dir() / Path(path).name
        if fallback.resolve() != candidates[0].resolve():
            candidates.append(fallback)
        last_err = ""
        for cand in candidates:
            try:
                ok = pix.save(str(cand), "PNG")
            except Exception as exc:          # pragma: no cover - Qt backend
                ok, last_err = False, str(exc)
            if ok and cand.exists() and cand.stat().st_size > 0:
                self.status("已导出 " + str(cand), GOOD)
                self._log("EXPORT png " + str(cand))
                return True
            last_err = last_err or "无法写入（目录只读或权限不足）"
        QMessageBox.warning(
            self, "导出失败",
            "PNG 写入失败：%s\n\n已尝试:\n%s\n\n请选择“文档”或“桌面”等可写目录。"
            % (last_err, "\n".join(str(c) for c in candidates)))
        self.status("导出失败（无写权限）", BAD)
        return False

    def _save_npy(self):
        if self.psf is None:
            return
        default = str(_default_export_dir() / "psf_stack.npy")
        path, _ = QFileDialog.getSaveFileName(self, "导出 NPY", default, "NumPy (*.npy)")
        if not path:
            return
        try:
            np.save(path, np.asarray(self.psf))
            self.status("已导出 " + path, GOOD)
        except Exception as exc:
            fallback = _default_export_dir() / Path(path).name
            try:
                np.save(fallback, np.asarray(self.psf))
                self.status("已导出到可写目录 " + str(fallback), WARN)
            except Exception:
                QMessageBox.warning(self, "导出失败", "%s\n%s" % (path, exc))
