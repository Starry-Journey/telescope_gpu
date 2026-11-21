"""Thin entrypoint.

--worker dispatches to desktop.worker without importing PySide6.
Otherwise imports desktop.gui and starts the Qt window.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main():
    t0 = time.perf_counter()
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--smoke-test", action="store_true")
    ap.add_argument("--worker", default="")
    ap.add_argument("--job", default="")
    ap.add_argument("--result", default="")
    args = ap.parse_args()
    if args.worker:
        from desktop.worker import run_worker
        try:
            run_worker(args.worker, args.job or None, args.result or None)
        except Exception as e:
            if args.result:
                Path(args.result).write_text(
                    json.dumps({"error": str(e) + "\n" + traceback.format_exc()}),
                    encoding="utf-8")
            raise
        return
    if args.self_test:
        import desktop.gui  # import check only
        print("Qt frontend imports OK")
        return
    if args.smoke_test:
        import numpy as np
        from products.ccd.service import run_ccd_pipeline
        r = run_ccd_pipeline(0.3, 0.1, -0.05, 2.0, 8, [-1, 0, 1])
        print("smoke frames", len(r["psf"]), "shape", np.asarray(r["psf"]).shape)
        return
    from desktop.gui import run_gui
    run_gui(t0)


if __name__ == "__main__":
    main()
