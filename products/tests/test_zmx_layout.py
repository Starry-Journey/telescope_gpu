"""Tests for the ZMX 2-D/3-D optical-layout renderer."""
import base64
import tempfile
import unittest
from pathlib import Path

from products.zemax.layout import (
    model_geometry,
    render_layout_2d,
    render_layout_3d,
    sag_z,
)
from products.zemax.zmx_parser import parse_zmx_file

# The proprietary ZMX is not committed; the smoke test is skipped elsewhere.
REAL_ZMX = Path(r"C:\Users\17424\Documents\Zemax\Autosave\Recovered\2.5m主系统.zmx")


def synthetic_parsed():
    """plane -> concave mirror (R=-2000, k=-1, fold) -> image plane."""
    return {
        "aperture_mm": 200.0,
        "stop": 1,
        "wavelengths_um": [0.55],
        "surfaces": [
            {"index": 0, "type": "STANDARD", "curv": 0.0,
             "thickness": 100.0, "diameter": 0.0},
            {"index": 1, "type": "STANDARD", "curv": -1.0 / 2000.0,
             "thickness": -800.0, "glass": "MIRROR", "conic": -1.0,
             "diameter": 100.0},
            {"index": 2, "type": "STANDARD", "curv": 0.0,
             "thickness": 0.0, "diameter": 40.0},
        ],
    }


class ModelGeometryTest(unittest.TestCase):
    def test_vertex_track_and_mirror(self):
        geom = model_geometry(synthetic_parsed())
        surfaces = geom["surfaces"]
        self.assertEqual(geom["surface_count"], 3)
        # vertex z = running sum of the *previous* thicknesses
        self.assertEqual([s["vertex_z"] for s in surfaces], [0.0, 100.0, -700.0])
        # unfolded optical path length: |100| + |-800| + 0
        self.assertAlmostEqual(geom["track_mm"], 900.0, places=12)
        self.assertAlmostEqual(geom["folded_span_mm"], 800.0, places=12)
        self.assertEqual(geom["stop"], 1)
        self.assertAlmostEqual(geom["stop_z"], 100.0, places=12)

        mirror = surfaces[1]
        self.assertEqual(mirror["kind"], "mirror")
        self.assertTrue(mirror["is_mirror"])
        self.assertTrue(mirror["is_fold"])
        self.assertEqual(mirror["label"], "M1 (STOP)")
        self.assertAlmostEqual(mirror["radius"], -2000.0, places=9)
        self.assertAlmostEqual(mirror["conic"], -1.0, places=12)
        self.assertEqual(mirror["semi_diameter"], 100.0)
        self.assertEqual(mirror["clear_aperture"], 200.0)

        # plane: infinite radius -> radius None, sag identically zero
        first = surfaces[0]
        self.assertEqual(first["kind"], "dummy")
        self.assertIsNone(first["radius"])
        self.assertEqual(first["curv"], 0.0)
        self.assertEqual(first["sag"]["max_sag"], 0.0)
        self.assertEqual(float(sag_z(37.5, 0.0)), 0.0)
        # last surface is detected as the image plane
        self.assertEqual(surfaces[2]["label"], "IMAGE")

    def test_sag_sign_convention(self):
        # conic paraboloid k=-1: sag = c*r^2/2 = -r^2/(2R), negative for R<0
        r = 50.0
        self.assertAlmostEqual(sag_z(r, -1.0 / 2000.0, -1.0),
                               -(r * r) / (2.0 * 2000.0), places=12)
        self.assertLess(sag_z(r, -1.0 / 2000.0, -1.0), 0.0)
        self.assertGreater(sag_z(r, 1.0 / 2000.0, -1.0), 0.0)
        # array form keeps the same convention
        out = sag_z([10.0, 20.0], -1.0 / 2000.0, -1.0)
        self.assertAlmostEqual(float(out[0]), -(100.0) / 2000.0 / 2.0, places=12)

    def test_refractor_detection(self):
        parsed = {
            "stop": 0,
            "surfaces": [
                {"index": 0, "type": "STANDARD", "curv": 0.01,
                 "thickness": 5.0, "glass": "N-BK7", "diameter": 10.0},
                {"index": 1, "type": "STANDARD", "curv": -0.01,
                 "thickness": 0.0, "diameter": 10.0},
            ],
        }
        first = model_geometry(parsed)["surfaces"][0]
        self.assertEqual(first["kind"], "refractor")
        self.assertAlmostEqual(first["radius"], 100.0, places=9)
        self.assertEqual(first["label"], "S0 (STOP)")

    def test_zernike_phase_is_plane_with_note(self):
        parsed = {
            "surfaces": [
                {"index": 0, "type": "STANDARD", "curv": 0.0,
                 "thickness": 10.0, "diameter": 50.0},
                {"index": 1, "type": "SZERNPHA", "curv": 0.0,
                 "thickness": 0.0, "diameter": 20.0},
            ],
            "stop": None,
        }
        geom = model_geometry(parsed)
        phase = geom["surfaces"][1]
        self.assertTrue(phase["is_phase"])
        self.assertEqual(phase["kind"], "dummy")
        self.assertEqual(phase["label"], "PHASE")
        self.assertIsNone(phase["radius"])
        self.assertIn("Zernike", phase["note"])
        self.assertTrue(any("Zernike" in n for n in geom["notes"]))


class RenderLayoutTest(unittest.TestCase):
    def test_render_3d_synthetic(self):
        result = render_layout_3d(synthetic_parsed())
        self.assertEqual(len(result["surfaces"]), 3)
        self.assertEqual(result["stop"], 1)
        self.assertAlmostEqual(result["track_mm"], 900.0, places=12)
        self.assertGreater(len(result["png_b64"]), 5000)
        raw = base64.b64decode(result["png_b64"])
        self.assertTrue(raw.startswith(b"\x89PNG\r\n\x1a\n"))
        self.assertLessEqual(len(raw), 400 * 1024)
        self.assertEqual(result["png_size_bytes"], len(raw))

    def test_render_2d_synthetic(self):
        result = render_layout_2d(synthetic_parsed())
        self.assertEqual(len(result["surfaces"]), 3)
        self.assertGreater(len(result["png_b64"]), 5000)
        self.assertLessEqual(result["png_size_bytes"], 400 * 1024)
        raw = base64.b64decode(result["png_b64"])
        self.assertTrue(raw.startswith(b"\x89PNG"))

    def test_render_writes_png_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "nested" / "layout_3d.png"
            result = render_layout_3d(synthetic_parsed(), png_path=path)
            self.assertTrue(path.exists())
            self.assertEqual(path.read_bytes(), base64.b64decode(result["png_b64"]))
            self.assertLessEqual(result["png_size_bytes"], 400 * 1024)

    def test_deterministic_output(self):
        a = render_layout_3d(synthetic_parsed(), azim=-30, elev=25)
        b = render_layout_3d(synthetic_parsed(), azim=-30, elev=25)
        self.assertEqual(a["png_b64"], b["png_b64"])

    def test_degenerate_all_plane(self):
        parsed = {
            "aperture_mm": 100.0,
            "stop": None,
            "surfaces": [
                {"index": 0, "type": "STANDARD", "curv": 0.0,
                 "thickness": 50.0, "diameter": 50.0},
                {"index": 1, "type": "STANDARD", "curv": 0.0,
                 "thickness": 50.0, "diameter": 40.0},
                {"index": 2, "type": "STANDARD", "curv": 0.0,
                 "thickness": 0.0, "diameter": 30.0},
            ],
        }
        geom = model_geometry(parsed)
        self.assertEqual(geom["track_mm"], 100.0)
        self.assertEqual([s["vertex_z"] for s in geom["surfaces"]], [0.0, 50.0, 100.0])
        for renderer in (render_layout_3d, render_layout_2d):
            result = renderer(parsed)
            self.assertGreater(len(result["png_b64"]), 0)
            self.assertEqual(len(result["surfaces"]), 3)

    def test_degenerate_single_surface(self):
        parsed = {
            "aperture_mm": 25.0,
            "stop": 0,
            "surfaces": [
                {"index": 0, "type": "STANDARD", "curv": 0.0,
                 "thickness": 0.0, "diameter": 25.0},
            ],
        }
        geom = model_geometry(parsed)
        self.assertEqual(geom["surface_count"], 1)
        self.assertEqual(geom["track_mm"], 0.0)
        self.assertEqual(geom["stop"], 0)
        for renderer in (render_layout_3d, render_layout_2d):
            result = renderer(parsed)
            self.assertGreater(len(result["png_b64"]), 0)

    def test_empty_prescription(self):
        geom = model_geometry({"surfaces": []})
        self.assertEqual(geom["surface_count"], 0)
        self.assertIn("Empty surface table", " ".join(geom["notes"]))
        for renderer in (render_layout_3d, render_layout_2d):
            self.assertGreater(len(renderer({"surfaces": []})["png_b64"]), 0)


@unittest.skipUnless(REAL_ZMX.exists(), "real 2.5m ZMX not present on this machine")
class RealZmxLayoutTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.parsed = parse_zmx_file(str(REAL_ZMX))
        cls.geom = model_geometry(cls.parsed)

    def test_real_geometry(self):
        self.assertEqual(self.geom["surface_count"], 8)
        self.assertEqual(self.geom["stop"], 2)
        self.assertAlmostEqual(self.geom["stop_z"], 3100.0, places=6)
        # unfolded track = 3100 + 2783.7896 + 2083.7896 + 0 + 1250 + 1594.7047
        self.assertAlmostEqual(self.geom["track_mm"], 10812.28379849065, places=6)
        self.assertAlmostEqual(self.geom["folded_span_mm"], 5244.70466518518, places=6)
        self.assertAlmostEqual(self.geom["aperture_mm"], 2500.0, places=9)

        labels = [s["label"] for s in self.geom["surfaces"]]
        self.assertEqual(labels[2], "M1 (STOP)")
        self.assertEqual(labels[3], "M2")
        self.assertEqual(labels[4], "PHASE")
        self.assertEqual(labels[-1], "IMAGE")

        m1, m2 = self.geom["surfaces"][2], self.geom["surfaces"][3]
        self.assertTrue(any("Zernike" in n for n in self.geom["notes"]))
        self.assertTrue(m1["is_mirror"] and m2["is_mirror"])
        self.assertAlmostEqual(m1["radius"], 1.0 / -1.3640703860319192e-4, places=6)
        self.assertAlmostEqual(m1["conic"], -1.02347, places=12)
        self.assertAlmostEqual(m2["radius"], 1.0 / -4.6562955443907935e-4, places=6)
        self.assertAlmostEqual(m2["conic"], -2.23765, places=12)
        # negative thickness at M1 folds the vertex path
        self.assertTrue(m1["is_fold"])
        self.assertAlmostEqual(m1["vertex_z"], 3100.0, places=9)
        self.assertAlmostEqual(m2["vertex_z"], 3100.0 - 2783.7895666527347, places=9)

    def test_real_render_smoke(self):
        result = render_layout_3d(self.parsed, width=4, height=3)
        self.assertEqual(len(result["surfaces"]), 8)
        self.assertGreater(len(result["png_b64"]), 5000)
        self.assertLessEqual(result["png_size_bytes"], 400 * 1024)
        result2d = render_layout_2d(self.parsed)
        self.assertGreater(result2d["png_size_bytes"], 5000)
        self.assertLessEqual(result2d["png_size_bytes"], 400 * 1024)


if __name__ == "__main__":
    unittest.main()
