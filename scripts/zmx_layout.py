"""Render a ZMX prescription to 3-D/2-D optical-layout PNGs + geometry.json.

The ZMX is the proprietary OpticStudio model and is *not* committed; this
script only reads a caller-supplied path (default: the local recovered
2.5 m model) and writes derived artifacts under ``results/`` (git-ignored).

Usage
-----
    py -3.13 scripts/zmx_layout.py --zmx "<path>" --out results/zmx_layout

Outputs
-------
    <out>/layout_3d.png   rotationally-symmetric 3-D layout
    <out>/layout_2d.png   y-z cross-section (fallback / docs)
    <out>/geometry.json   display model + surface table (JSON-safe)
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from products.zemax.layout import model_geometry, render_layout_2d, render_layout_3d
from products.zemax.zmx_parser import parse_zmx_file

DEFAULT_ZMX = r"C:\Users\17424\Documents\Zemax\Autosave\Recovered\2.5m主系统.zmx"


def main(argv=None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        try:  # ZMX filenames are often Chinese; keep the console output UTF-8
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass
    ap = argparse.ArgumentParser(description="Render a ZMX optical layout.")
    ap.add_argument("--zmx", default=DEFAULT_ZMX, help="input .zmx file")
    ap.add_argument("--out", default="results/zmx_layout",
                    help="output directory (default: results/zmx_layout)")
    ap.add_argument("--azim", type=float, default=-60.0, help="3-D azimuth")
    ap.add_argument("--elev", type=float, default=18.0, help="3-D elevation")
    args = ap.parse_args(argv)

    zmx = Path(args.zmx)
    if not zmx.exists():
        print("ZMX not found:", zmx)
        return 1

    parsed = parse_zmx_file(str(zmx))
    geom = model_geometry(parsed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    r3 = render_layout_3d(parsed, out / "layout_3d.png",
                          azim=args.azim, elev=args.elev)
    r2 = render_layout_2d(parsed, out / "layout_2d.png")

    geometry = {
        "source": "products.zemax.zmx_parser",
        "zmx": zmx.name,  # basename only: the proprietary ZMX is never committed
        "units": geom["units"],
        "aperture_mm": geom["aperture_mm"],
        "wavelengths_um": geom["wavelengths_um"],
        "surface_count": geom["surface_count"],
        "track_mm": geom["track_mm"],
        "folded_span_mm": geom["folded_span_mm"],
        "stop": geom["stop"],
        "stop_z": geom["stop_z"],
        "extent": geom["extent"],
        "surfaces": geom["surfaces"],
        "notes": geom["notes"],
        "renders": {
            "layout_3d_png": "layout_3d.png",
            "layout_2d_png": "layout_2d.png",
            "layout_3d_bytes": r3["png_size_bytes"],
            "layout_2d_bytes": r2["png_size_bytes"],
            "azim": args.azim,
            "elev": args.elev,
        },
    }
    (out / "geometry.json").write_text(
        json.dumps(geometry, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print("ZMX:", zmx.name, "| surfaces:", geom["surface_count"])
    print("track: %.3f mm | folded span: %.3f mm | stop: %s @ z=%s" % (
        geom["track_mm"], geom["folded_span_mm"], geom["stop"], geom["stop_z"]))
    for surface in geom["surfaces"]:
        print("  SURF %-2s %-12s z=%10.3f mm  R=%s  k=%s  sd=%.3f" % (
            surface["index"], surface["label"], surface["vertex_z"],
            "None" if surface["radius"] is None else "%.3f" % surface["radius"],
            surface["conic"], surface["semi_diameter"]))
    print("wrote:", (out / "layout_3d.png").resolve(),
          "(%d bytes)" % r3["png_size_bytes"])
    print("wrote:", (out / "layout_2d.png").resolve(),
          "(%d bytes)" % r2["png_size_bytes"])
    print("wrote:", (out / "geometry.json").resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
