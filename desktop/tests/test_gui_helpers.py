"""Regression tests for the GUI hardening fixes (export + 3D label).

These cover the two reported field bugs:
  * PNG export silently failing on a read-only target (QPixmap.save -> False),
  * the 3D view staying blank because a pixmap was scaled to a stale label size.
"""
import json
import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from PySide6.QtGui import QPixmap, QColor
    from PySide6.QtWidgets import QApplication
    HAVE_QT = True
except Exception:          # pragma: no cover - headless machines without PySide6
    HAVE_QT = False


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class ExportPathTest(unittest.TestCase):
    def test_writable_dir_skips_invalid_candidate(self):
        from desktop.gui import _writable_dir
        with tempfile.TemporaryDirectory() as td:
            # a regular file cannot be a directory -> mkdir must fail and the
            # helper has to fall through to the next candidate (this exercises
            # the same code path as a read-only target on every OS)
            blocker = Path(td) / "blocker"
            blocker.write_text("x", encoding="utf-8")
            target = _writable_dir(blocker / "sub", Path(td) / "ok")
            self.assertEqual(target, Path(td) / "ok")
            self.assertFalse((target / ".write_test").exists())

    def test_default_export_dir_exists(self):
        from desktop.gui import _default_export_dir
        d = _default_export_dir()
        self.assertTrue(d.exists())
        probe = d / ".probe"
        probe.write_text("ok", encoding="utf-8")
        probe.unlink()

    def test_export_all_reports_failed_files_key(self):
        from desktop.gui import QtApp
        app = QApplication.instance() or QApplication([])
        w = QtApp(startup_t0=None)
        with tempfile.TemporaryDirectory() as td:
            out = w.export_all(td)
            self.assertIsNotNone(out)
            manifest = json.loads((Path(td) / "manifest.json").read_text(encoding="utf-8"))
            self.assertIn("files", manifest)
            self.assertIn("failed_files", manifest)
        w.close()


@unittest.skipUnless(HAVE_QT, "PySide6 not available")
class ScaledLabelTest(unittest.TestCase):
    def test_pixmap_survives_small_label_and_resizes(self):
        from desktop.gui import ScaledImageLabel
        app = QApplication.instance() or QApplication([])
        label = ScaledImageLabel("3D", (380, 300))
        label.show()
        app.processEvents()
        label.resize(10, 10)          # not laid out yet / collapsed
        pix = QPixmap(400, 300)
        pix.fill(QColor("#336699"))
        label.set_source_pixmap(pix)
        first = label.pixmap()
        self.assertIsNotNone(first)
        self.assertFalse(first.isNull())
        self.assertGreaterEqual(first.width(), 380)
        # aspect ratio preserved inside the (380, 300) box
        self.assertAlmostEqual(first.width() / first.height(),
                               pix.width() / pix.height(), places=1)
        label.resize(800, 600)
        app.processEvents()
        second = label.pixmap()
        self.assertGreater(second.width(), first.width())
        self.assertAlmostEqual(second.width() / second.height(),
                               pix.width() / pix.height(), places=1)

    def test_clear_source_removes_pixmap(self):
        from desktop.gui import ScaledImageLabel
        app = QApplication.instance() or QApplication([])
        label = ScaledImageLabel("x", (100, 80))
        pix = QPixmap(20, 20)
        pix.fill(QColor("#000000"))
        label.set_source_pixmap(pix)
        label.clear_source()
        self.assertIsNone(label.source_pixmap())


if __name__ == "__main__":
    unittest.main()
