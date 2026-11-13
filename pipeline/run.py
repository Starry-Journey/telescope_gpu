"""Command-line entry point for the CCD pipeline.

Examples
--------
    python3 -m pipeline.run --config configs/pipeline.json --force-sim
    python3 -m pipeline.run --config configs/pipeline.json --skip-sim --log-level DEBUG
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from astropy.io import fits
from astropy.wcs import WCS

from .astrometry import solve_astrometry
from .config import PipelineConfig
from .detect import detect_moving, detect_stars
from .evaluate import evaluate
from .preprocess import preprocess_dataset
from .simulate import simulate_dataset
from . import visualize

log = logging.getLogger("pipeline")


def load_stack(path: str) -> np.ndarray:
    with np.load(path) as data:
        return np.asarray(data["frames"], dtype=float)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="CCD star-field / moving-target pipeline")
    parser.add_argument("--config", default="configs/pipeline.json")
    parser.add_argument("--force-sim", action="store_true", help="regenerate the synthetic dataset")
    parser.add_argument("--skip-sim", action="store_true", help="use an existing dataset")
    parser.add_argument("--no-figures", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)-5s | %(message)s",
        datefmt="%H:%M:%S",
    )

    cfg = PipelineConfig.load(args.config)
    data_dir = Path(cfg.data_dir)
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("config=%s data=%s output=%s", args.config, data_dir, out_dir)

    if args.force_sim or not (data_dir / "dataset.json").exists():
        log.info("[1/6] simulate dataset -> %s", data_dir)
        simulate_dataset(str(data_dir), cfg.sim)
    else:
        log.info("[1/6] use existing dataset -> %s", data_dir)

    log.info("[2/6] preprocess (bias/dark/flat, background, denoise)")
    pre = preprocess_dataset(str(data_dir), str(out_dir), cfg.preprocess)
    stack = load_stack(pre["stack"])
    log.info("      frames=%s shape=%s", stack.shape[0], stack.shape[1:])

    log.info("[3/6] star extraction / centroid")
    stars = detect_stars(stack[0], cfg.detect)
    log.info("      %d stars detected", len(stars))

    log.info("[4/6] moving-target detection (registration + difference + morphology + Hough)")
    moving = detect_moving(stack, cfg.detect)
    log.info("      %d candidates, %d tracks",
             len(moving["detections"]), len(moving["tracks"]))

    log.info("[5/6] astrometry (WCS + distortion + catalogue matching)")
    with fits.open(data_dir / "raw" / "frame_000.fits") as hdul:
        wcs = WCS(hdul[0].header)
    from .astrometry import load_catalog
    catalog = load_catalog(str(data_dir / "catalog.csv"))
    solution = solve_astrometry(stars, catalog, wcs, cfg.astrometry)
    log.info("      matched %d stars, RMS=%.3f arcsec",
             solution.n_matches, solution.rms_arcsec)

    log.info("[6/6] evaluate + visualize")
    metrics = evaluate(str(data_dir), str(out_dir), stars, moving, solution, cfg)
    paths = []
    if not args.no_figures:
        paths = visualize.make_all(str(data_dir), str(out_dir), stars, moving, solution, metrics)

    summary = {
        "data_dir": str(data_dir),
        "output_dir": str(out_dir),
        "metrics": metrics,
        "figures": paths,
    }
    with open(out_dir / "run_summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=2, ensure_ascii=False)

    log.info("summary written to %s", out_dir / "run_summary.json")
    print("\n=== summary ===")
    print("star completeness :", f"{metrics['stars']['completeness']:.3f}")
    print("star centroid RMSE:", f"{metrics['stars']['centroid_rmse_pix']:.3f} pix")
    print("target det. rate  :", f"{metrics['target']['detection_rate']:.3f}")
    print("false alarm rate  :", f"{metrics['target']['false_alarm_rate']:.3f}")
    print("target pos. RMSE  :", f"{metrics['target']['position_rmse_pix']:.3f} pix")
    print("astrometry RMS    :", f"{metrics['astrometry']['target_rmse_arcsec']:.3f} arcsec")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
