# Runtime / memory optimisation of the data-processing core

Optimisation pass over `products/optics`, `products/ccd`, `products/sensitivity`
and one Qt-side memory fix.  **No public API, returned key, or numeric contract
changed** — see "Behaviour-equivalence evidence" below.

Reference machine: Windows 11, 16 logical cores, Python 3.13.12,
NumPy 2.5.5, SciPy 1.18.1, PySide6 (offscreen for the GUI tests).
The box runs other agents/builds while measuring, so absolute milliseconds move
by 10–40 % between runs.  Every table below therefore reports the method used
and the observed range where relevant.

| Artefact | Path |
| --- | --- |
| Benchmark script | `scripts/benchmark.py` |
| Benchmark JSON (after) | `results/bench/benchmark.json` |
| Benchmark JSON (before) | `results/bench/benchmark_before.json` |
| Interleaved before/after A/B log | `results/bench/ab_paired.txt` |

```bash
py -3.13 scripts/benchmark.py                 # median of 3 warm samples + tracemalloc peaks
py -3.13 scripts/benchmark.py --repeats 5 --label after
```

## 1. Benchmark table (median of 3 warm samples)

`before` = the JSON captured on the unmodified tree; `after` = the
least-contended run after the changes.  "cold" rows clear every module cache
before each sample (they measure construction, not a cache hit).

| Benchmark | before (ms) | after (ms) | speed-up |
| --- | ---: | ---: | ---: |
| `zernike.low_order_grid` n=64 (warm) | 0.390 | 0.041 | **9.5x** |
| `zernike.low_order_grid` n=64 (cold) | 0.432 | 0.396 | 1.1x |
| `zernike.term_grids` n=64 (warm) | 0.550 | 0.060 | **9.2x** |
| `zernike.term_grids` n=64 (cold) | 0.495 | 0.523 | 0.95x |
| `zernike.low_order_grid` n=128 (warm) | 0.792 | 0.077 | **10.3x** |
| `zernike.low_order_grid` n=128 (cold) | 0.798 | 0.839 | 0.95x |
| `zernike.term_grids` n=128 (warm) | 1.130 | 0.098 | **11.5x** |
| `zernike.term_grids` n=128 (cold) | 1.040 | 1.087 | 0.96x |
| `forward.forward_psf_numpy` 3-term n=64 | 2.387 | 0.919 | **2.6x** |
| `forward.forward_psf_numpy` 6-term n=64 | 3.520 | 0.948 | **3.7x** |
| `forward.forward_stack_numpy` 3-term n=64 | 9.328 | 4.090 | **2.3x** |
| `forward.forward_stack_numpy` 6-term n=64 | 12.819 | 4.607 | **2.8x** |
| `ccd.run_ccd_pipeline` default (5 frames + PNG) | 56.648 | 44.651 | **1.27x** |
| `ccd.run_ccd_pipeline` 5 frames, no PNG | 40.276 | 30.062 | **1.34x** |
| `ccd.run_ccd_pipeline` 30 frames, no PNG | 225.207 | 173.536 | **1.30x** |
| `sensitivity.case_detail` M1_design (cold) | 107.012 | 83.741 | **1.28x** |
| `sensitivity.case_detail` M1_design (cached) | 0.009 | 0.008 | cache hit |
| `sensitivity.fit_surface` 6400 nodes / 66 terms | 28.478 | 10.055 | **2.8x** |

Notes on the most environmental-noise-sensitive rows:

* `case_detail` (cold): dominated by the SciPy scattered→grid interpolation
  (~80–120 ms of the cold call); only its fit/read parts got faster (interleaved
  A/B: `fit_surface` 5.7x, `_read_case` 2.0–2.3x, `_to_grid` unchanged).
* `fit_surface`: the benchmark row has a heap-state/contention bias (the CCD
  rows leave a fragmented heap, and the box is shared).  A fresh-process loop
  measures 6.0–6.5 ms and the interleaved old/new A/B gives 4.4–7.0x; treat
  2.8x as the conservative lower bound and ~5x as the typical value.

### Interleaved A/B (same process, alternating old/new, median of 3–5)

The pre-change implementations were reconstructed verbatim / loaded from the
preserved pre-change `.pyc`, then run alternately with the new ones in one
process so machine drift hits both sides equally (`results/bench/ab_paired.txt`).

| Operation | speed-up |
| --- | ---: |
| `low_order_grid` n=64 / n=128 (warm) | 26–30x / 26x |
| `term_grids` n=64 / n=128 (warm) | 36–42x / 38–45x |
| `forward_psf_numpy` 3-term / 6-term | 2.5–2.8x / 2.6–2.8x |
| `forward_stack_numpy` 3-term / 6-term | 2.2–2.6x / 3.2–3.4x |
| `ccd.model.simulate` default / 30 frames | 1.07–1.20x / 1.07–1.15x |
| `sensitivity.fit_surface` 6400 nodes / 66 terms | 4.4–7.0x |
| `sensitivity.fit_surface` M1_design nodes (3761) | 5.2–6.1x |
| `sensitivity._read_case` (CSV parse) | 1.7–2.3x |
| `_to_grid` (unchanged control) | 0.95–1.40x (noise) |

The A/B log predates the final 66x66 eigenvalue conditioning guard in
`_least_squares` (a ~0.1–0.5 ms addition to the new fit), so its `fit_surface`
ratios are up to ~10 % optimistic; the range quoted above is the observed spread
across three runs.

Note: the pre-change `.pyc` used for the `simulate` comparison still imports the
*new* `products.optics.forward`, so the measured 1.07–1.20x is the model-layer
win only.  The end-to-end CCD win (1.34x without PNG) also includes the cached
Zernike grids the forward model uses for every defocus frame.

### Memory (tracemalloc peak, cold caches)

| Measured call | before | after | change |
| --- | ---: | ---: | ---: |
| `run_ccd_pipeline` default (5 frames) | 7.91 MB | 5.95 MB | **−25 %** |
| `fit_surface` 6400 nodes / 66 terms | 6.93 MB | 5.34 MB | **−23 %** |

The CCD win is the removed unused per-frame `repaired_frames` list plus the
removed full-frame copies in the frame loop; the fit win is the design matrix
being built column-by-column instead of `list-of-66-arrays + np.stack` (same
3.4 MB result, half the transient peak).  The LRU grid caches add a bounded
~0.1–3 MB of read-only arrays per cached `(n, pupil_radius, obscuration, terms)`
key (max 8 entries).

## 2. What changed

### `products/optics/zernike.py`
* `pupil_geometry`, `low_order_grid`, `term_grids` are memoised through
  private `lru_cache` wrappers keyed by `(n, pupil_radius, obscuration[, terms])`
  (`maxsize=8`).  Cached arrays are `writeable=False`; the **public** functions
  return fresh writable copies, so the old "caller may mutate the result"
  semantics are preserved.
* `basis` / `fit_surface`: the 66-column design matrix is built from
  precomputed radial-polynomial coefficients and shared `rho**k` /
  `cos/sin(m*theta)` tables (`_design_matrix`) instead of 66 full polynomial
  evaluations.
* `fit_surface` solves normal equations through a Cholesky factorisation when
  the design matrix is well conditioned (`cond(A) ≈ 4–5` for every bundled FEA
  case and the benchmark surface) and falls back to the original SVD
  `numpy.linalg.lstsq` when Cholesky fails (e.g. a collinear uploaded surface)
  or when a cheap 66x66 eigenvalue check shows the squared condition number
  would exceed 1e12.

### `products/optics/forward.py`
* `forward_psf_numpy` uses the cached (read-only) basis grids and pupil mask
  directly, so a 5-frame defocus stack or a PD optimiser iteration evaluates
  the Zernike grids once per `(fft_n, terms, pupil)` instead of once per frame.
* Phase accumulation no longer starts from an explicit `zeros_like` (0 + x
  was bit-identical); the FFT/shift/bin sequence is unchanged.

### `products/ccd/model.py`
* `_repair_defects` gathers the 8 neighbours of each flagged pixel instead of
  shifting the full frame 8 times (bad-pixel masks are ~20 pixels, frames are
  192x192 → ~1 ms saved per frame; arithmetic and accumulation order unchanged).
* The seeing OTF is cached per `(canvas size, sigma)` (fixed for every frame of
  one simulation).
* The frame loop works in place (`scene += background`, `np.rint(..., out=)`,
  `np.clip(..., out=)`, no second `raw` copy) and the unused
  `repaired_frames` list is gone.

### `products/ccd/io.py`
* `clean_cutout` builds the border mask / coordinate grids / 3-column plane-fit
  design matrix once per cutout shape (LRU, read-only) instead of per call.
  (Matplotlib stays an eager module import here on purpose — see §4.)

### `products/sensitivity/service.py`
* `_read_case` parses the FEA CSV with `csv.reader` + a column index instead of
  `csv.DictReader` (same `float()` parsing, no dict per node).
* `case_detail` keeps its `lru_cache(maxsize=32)` keyed by case id only (that is
  the whole input), and its docstring now spells out that the cached dict is
  shared/read-only and why no defensive deepcopy is made.

### `desktop/gui.py`
* `_on_fea3d` no longer keeps the 3D render's base64 PNG in `self.last_fea3d`
  (only the stats/metadata are kept; the QPixmap is decoded and the bytes
  dropped).  Matplotlib was already imported lazily inside every rendering
  helper; the other pixmaps are Qt-owned and released when
  `ScaledImageLabel.set_source_pixmap` replaces them.

### `products/sensitivity/zernike.py`
Unchanged: it is a pure compatibility re-export, so it automatically points at
the optimised `products.optics.zernike` implementation.

## 3. Caching subtleties a reviewer must know

1. **Cached arrays are read-only, public results are copies.**  `_pupil_geometry_cached`,
   `_term_grids_cached` and `_low_order_grid_cached` `setflags(write=False)` so a
   caller cannot corrupt the cache; `pupil_geometry()`, `term_grids()` and
   `low_order_grid()` copy on the way out and keep returning writable arrays.
   `forward_psf_numpy` reads the cached arrays directly (no copy) — it never
   mutates them.
2. `_term_grids_cached` returns a *dict* and `term_grids` copies both the dict
   and the arrays; anything added to the public dict is per-call.
3. Cache keys are floats; `0.45` and `0.45000000000000001` are the same key, and
   a different `pupil_radius`/`obscuration` always misses.  `maxsize=8` bounds
   memory (PD uses n=64/128/256, CCD n=128).
4. `case_detail` returns the cached object itself (as before).  All in-tree
   callers only read it.  If a future caller needs to mutate it, copy there.
5. The normal-equation solve changes the last ~1e-12 nm of the coefficients
   (verified against SVD on all 22 bundled cases: same residual RMS to every
   printed digit, max coefficient delta 2e-12 nm).  The SVD path is still used
   for degenerate inputs.  Public `fit_surface` output keys/dtypes are
   unchanged.
6. `forward_psf_numpy` default 3-term output is **bit-identical** to the old
   implementation (the cached grids are produced by the same `zernike()` code);
   `products/tests/test_pd_terms.py`'s stored golden pixels pass with
   `rtol=0, atol=1e-12`.  The whole CCD `simulate()` output was compared
   bit-for-bit against the preserved pre-change bytecode for 5 configurations
   (default, fast path, 30 frames, noisy, small) — all arrays and metadata
   dicts equal.

## 4. Deliberately NOT optimised (and why)

* **`_to_grid` (SciPy `griddata` scattered→grid).**  With SciPy 1.18 the
  Delaunay construction *plus the one-off search-structure build on the first
  query* costs 20 ms (720 nodes) … 140 + 300 ms (18k nodes), i.e. 85–95 % of the
  cold `case_detail` path.  A faster equivalent would have to change the
  interpolation method/simplex search, which changes the displayed surface
  values — not allowed in an optimisation pass.  `case_detail`'s `lru_cache`
  already makes this a once-per-case cost.  (An inside-the-aperture-only query
  was measured too: the search-structure cost is per triangulation, not per
  query point, so it does not help.)
* **`float32` in the CCD model.**  The frames are 192x192 (~0.3 MB), the measured
  hot spots are Poisson sampling, the FFT PSF and libpng/base64, and the model
  round-trips integer ADU; `float32` would risk the byte-exact detector/
  determinism contracts for no measurable win here.
* **`fftshift` "single shift" strategy.**  The forward already shifts exactly
  once (`fftshift(fft2)`) and the PSF is then binned; shifting after the
  `abs()²` is a pure permutation.  Removing the shift entirely would change the
  2x2 binning phase, so it stays.
* **PNG data-URL encoding.**  ~15 ms of the 45 ms default `run_ccd_pipeline`
  is Pillow PNG + base64 for 15 images (frames + PSF thumbnails).  The bytes
  are part of what the GUI/web clients receive; changing the encoder or
  compression level would change them.  (`compress_level=6` is kept.)
* **Lazy `matplotlib` import in `products/ccd/io.py`.**  Deferring the
  module-level `matplotlib`/`pyplot` import (only the slow `png_data_url`
  fallback needs it) was measured at 1.385 s / 36.5 MB of Python allocations on
  a cold interpreter and cut the Qt worker start-up of a CCD GUI task from
  ~0.7 s to ~0.33 s.  It was reverted because `desktop/e2e_smoke.py` assumes a
  CCD worker task is still running after its 0.3 s sleep (its cancel check
  fails once the worker is that fast), and that smoke script is outside the
  allowed edit scope.  Recommended follow-up: lazy import **plus** a
  deterministic "long task" cancel fixture in the E2E smoke (or a
  `TaskRunner.cancel()` that reports `False` for a just-finished task without
  failing the scenario).
* **`products/ccd/service.py`.**  Profiled: after the model/OPTICS changes it is
  a thin wrapper — PNG encoding, `tolist()` of the PSF cutouts and the metrics
  are API payloads, no duplicated computation left.  Left unchanged.
* **`desktop/gui.py` layout/pixmap behaviour.**  Only the retained base64 copy
  in `_on_fea3d` was dropped; the `ScaledImageLabel`/`QPixmap` ownership is
  correct as-is (Qt owns the scaled pixmap, the Python `_source` reference is
  reassigned on replace).

## 5. Behaviour-equivalence evidence

* `py -3.13 -m unittest discover -s products/tests` → **96 tests, OK** before
  the change and **124 tests, OK** at final validation (another workstream added
  28 pipeline/acquisition/preprocess tests to this shared checkout meanwhile).
* `py -3.13 -m unittest discover -s desktop/tests` (offscreen) → **19 tests, OK**
  (14 at the start; 5 GUI-helper tests were added by another workstream).
* `py -3.13 desktop/e2e_smoke.py` (offscreen) → prints **E2E_OK**.
* Exact-value guards that stayed green: stored 3-term PSF golden pixels
  (`rtol=0, atol=1e-12`), `term_grids` vs `low_order_grid` (`atol=1e-12`),
  FEA Z4 golden `-380.527304` (places=4), sensitivity-matrix exact equality.
* `simulate()` output compared bit-for-bit against the pre-change bytecode
  (loaded from the `__pycache__` copies preserved before the edits) for default /
  fast / 30-frame / noisy / small configurations.
* `fit_surface` coefficients compared against the old SVD path on all 22 FEA
  cases: max |Δ| = 2.0e-12 nm, residual RMS identical to every printed digit.
