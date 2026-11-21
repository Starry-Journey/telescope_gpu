"""Runtime + memory benchmark for the data-processing core (optics / CCD / sensitivity).

Everything here is pure NumPy/SciPy on the CPU and deterministic for a fixed
seed.  Every row is the median of ``--repeats`` warm samples (3 by default);
"cold" rows clear the module caches before each sample so they measure the
first-call path (CSV read + fit + interpolation), not a dictionary lookup.

Run from the repo root:

    py -3.13 scripts/benchmark.py                 # before/after comparison
    py -3.13 scripts/benchmark.py --label before  # tag the JSON result
    py -3.13 scripts/benchmark.py --repeats 5

The JSON lands in ``results/bench/benchmark.json`` (override with ``--out``).
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import statistics
import sys
import time
import tracemalloc
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np  # noqa: E402

from products.optics import forward as fwd  # noqa: E402
from products.optics import zernike as z  # noqa: E402
from products.ccd.service import run_ccd_pipeline  # noqa: E402
from products.sensitivity import service as sens  # noqa: E402
from products.sensitivity import zernike as sens_z  # noqa: E402

DEFAULT_OUT = ROOT / "results" / "bench" / "benchmark.json"

#: Five-frame default defocus state used by the CCD / PSF paths.
DEFOCUS = (-2.0, -1.0, 0.0, 1.0, 2.0)
SIX_TERMS = ("Z4", "Z5", "Z6", "Z8", "Z9", "Z11")


# ---------------------------------------------------------------------------
# cache control (works before *and* after the caching optimisation: old builds
# simply have no cache_clear attributes, which is a no-op here)
# ---------------------------------------------------------------------------

def _cacheables() -> list:
    funcs = [sens.case_detail]
    for name in ("_read_case", "_manifest"):
        funcs.append(getattr(sens, name, None))
    for name in ("pupil_geometry", "low_order_grid", "term_grids",
                 "_pupil_geometry_cached", "_term_grids_cached",
                 "_low_order_grid_cached"):
        funcs.append(getattr(z, name, None))
    return [f for f in funcs if f is not None and hasattr(f, "cache_clear")]


def clear_caches() -> None:
    """Drop every LRU cache that can affect a measured call."""
    for func in _cacheables():
        func.cache_clear()


# ---------------------------------------------------------------------------
# timing helpers
# ---------------------------------------------------------------------------

def _run(fn):
    t0 = time.perf_counter()
    fn()
    return time.perf_counter() - t0


def timed(fn, repeats: int = 3, cold: bool = False) -> dict:
    """Median/min of ``repeats`` samples; ``cold`` clears caches per sample."""
    if not cold:
        fn()  # warm-up (builds the caches / imports lazily imported modules)
    samples = []
    for _ in range(repeats):
        if cold:
            clear_caches()
        gc.collect()
        samples.append(_run(fn))
    return {
        "median_s": statistics.median(samples),
        "min_s": min(samples),
        "samples_s": samples,
    }


def peak_memory(fn) -> dict:
    """Peak Python allocation of one call (caches cleared first, cold path)."""
    clear_caches()
    gc.collect()
    tracemalloc.start()
    try:
        result = fn()
        _current, peak = tracemalloc.get_traced_memory()
    finally:
        del result
        tracemalloc.stop()
    return {"peak_bytes": int(peak)}


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------

def synthetic_surface(n_nodes: int = 6400, radius: float = 1280.0,
                      seed: int = 3) -> tuple:
    """Deterministic scattered mm/mm/nm surface for the 66-term fit."""
    rng = np.random.default_rng(seed)
    r = radius * np.sqrt(rng.uniform(0.0, 1.0, n_nodes))
    theta = rng.uniform(0.0, 2.0 * np.pi, n_nodes)
    x = r * np.cos(theta)
    y = r * np.sin(theta)
    z = 120.0 * (r / radius) ** 2 + 40.0 * np.cos(2.0 * theta) * (r / radius) + 7.0
    return x, y, z


def _cases() -> list:
    return [c["id"] for c in sens.list_cases()["cases"]]


# ---------------------------------------------------------------------------
# benchmark rows
# ---------------------------------------------------------------------------

def build_rows(repeats: int) -> list:
    rows: list = []

    def add(name: str, unit: str, fn, cold: bool = False) -> None:
        stats = timed(fn, repeats=repeats, cold=cold)
        row = {"name": name, "unit": unit, **stats}
        rows.append(row)
        print(f"  {name:<44} {row['median_s'] * 1e3:9.3f} ms"
              + (f"  (cold {row['min_s'] * 1e3:.3f} ms)" if cold else ""),
              flush=True)

    # ---- Zernike basis / grids -------------------------------------------
    for n in (64, 128):
        add(f"zernike.low_order_grid n={n}", "ms",
            lambda n=n: z.low_order_grid(n))
        add(f"zernike.low_order_grid n={n} (cold)", "ms",
            lambda n=n: z.low_order_grid(n), cold=True)
        add(f"zernike.term_grids n={n}", "ms",
            lambda n=n: z.term_grids(n))
        add(f"zernike.term_grids n={n} (cold)", "ms",
            lambda n=n: z.term_grids(n), cold=True)

    # ---- Fourier-optics forward model ------------------------------------
    add("forward.forward_psf_numpy 3-term n=64", "ms",
        lambda: fwd.forward_psf_numpy([0.3, 0.1, -0.05], 0.0, n=64))
    add("forward.forward_psf_numpy 6-term n=64", "ms",
        lambda: fwd.forward_psf_numpy([0.3, 0.0, 0.0, 0.1, -0.05, 0.0], 0.0,
                                      n=64, terms=SIX_TERMS))
    add("forward.forward_stack_numpy 3-term n=64", "ms",
        lambda: fwd.forward_stack_numpy([0.3, 0.1, -0.05], n=64, defocus=DEFOCUS))
    add("forward.forward_stack_numpy 6-term n=64", "ms",
        lambda: fwd.forward_stack_numpy([0.3, 0.0, 0.0, 0.1, -0.05, 0.0],
                                        n=64, defocus=DEFOCUS, terms=SIX_TERMS))

    # ---- CCD pipeline ----------------------------------------------------
    add("ccd.run_ccd_pipeline default (5 frames)", "ms", run_ccd_pipeline)
    add("ccd.run_ccd_pipeline 5 frames (no PNG)", "ms",
        lambda: run_ccd_pipeline(return_frames=False))
    add("ccd.run_ccd_pipeline 30 frames (no PNG)", "ms",
        lambda: run_ccd_pipeline(defocus=[float(d) for d in range(30)],
                                 return_frames=False))

    # ---- sensitivity -----------------------------------------------------
    case_id = "M1_design_gravity_0deg"  # 3761-node representative case
    add(f"sensitivity.case_detail {case_id} (cold)", "ms",
        lambda: sens.case_detail(case_id), cold=True)

    clear_caches()
    sens.case_detail(case_id)             # populate the lru_cache
    add(f"sensitivity.case_detail {case_id} (cached)", "ms",
        lambda: sens.case_detail(case_id))

    x, y, zz = synthetic_surface()
    add("sensitivity.fit_surface 6400 nodes / 66 terms", "ms",
        lambda: sens_z.fit_surface(x, y, zz, 1280.0, n_terms=66))

    return rows


def build_memory() -> list:
    out: list = []

    def add(name: str, fn) -> None:
        mem = peak_memory(fn)
        out.append({"name": name, **mem})
        print(f"  {name:<44} {mem['peak_bytes'] / 1e6:9.2f} MB peak",
              flush=True)

    add("memory.run_ccd_pipeline default (5 frames)",
        lambda: run_ccd_pipeline())
    x, y, zz = synthetic_surface()
    add("memory.fit_surface 6400 nodes / 66 terms",
        lambda: sens_z.fit_surface(x, y, zz, 1280.0, n_terms=66))
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repeats", type=int, default=3,
                        help="samples per row (default 3, median reported)")
    parser.add_argument("--label", default="", help="tag stored in the JSON")
    parser.add_argument("--out", default=str(DEFAULT_OUT),
                        help="JSON output path")
    args = parser.parse_args(argv)
    repeats = max(1, int(args.repeats))

    print("benchmark rows (median of %d warm samples)" % repeats, flush=True)
    started = time.perf_counter()
    rows = build_rows(repeats)
    print("memory rows (tracemalloc peak, cold caches)", flush=True)
    memory = build_memory()
    elapsed = time.perf_counter() - started

    payload = {
        "label": args.label,
        "created_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "python": platform.python_version(),
        "numpy": np.__version__,
        "repeats": repeats,
        "elapsed_s": elapsed,
        "rows": rows,
        "memory": memory,
    }
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    print("\n%s" % ("=" * 78))
    print("%-46s %12s %12s" % ("benchmark", "median ms", "min ms"))
    print("-" * 78)
    for row in rows:
        print("%-46s %12.3f %12.3f" % (row["name"], row["median_s"] * 1e3,
                                       row["min_s"] * 1e3))
    print("-" * 78)
    for mem in memory:
        print("%-46s %12.2f MB" % (mem["name"], mem["peak_bytes"] / 1e6))
    print("=" * 78)
    print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
