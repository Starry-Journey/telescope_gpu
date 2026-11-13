"""Detection / false-alarm / astrometric accuracy evaluation."""
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .astrometry import AstrometricSolution, angular_sep_arcsec, target_sky, load_catalog
from .config import PipelineConfig


def read_csv(path: str) -> List[Dict[str, str]]:
    with open(path, "r", newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: str, rows: Sequence[Dict[str, object]], fieldnames: Sequence[str]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=fieldnames)
        wr.writeheader()
        for row in rows:
            wr.writerow({k: row.get(k, "") for k in fieldnames})


def match_by_frame(detections: Sequence[Dict[str, float]], truth: Sequence[Dict[str, float]],
                   radius: float = 3.0) -> Dict[str, object]:
    """Greedy nearest-neighbour match of detections to truth target positions."""
    n_truth = len(truth)
    used = set()
    matches = []
    for t in truth:
        best = None
        best_d = 1e9
        for i, d in enumerate(detections):
            if i in used or int(d["frame"]) != int(t["frame"]):
                continue
            dist = math.hypot(float(d["x"]) - float(t["x_det"]),
                              float(d["y"]) - float(t["y_det"]))
            if dist < best_d and dist <= radius:
                best_d = dist
                best = (i, d)
        if best is not None:
            i, d = best
            used.add(i)
            matches.append({"frame": int(t["frame"]), "truth_x": float(t["x_det"]),
                            "truth_y": float(t["y_det"]), "x": float(d["x"]),
                            "y": float(d["y"]), "dist": best_d})
    n_false = len(detections) - len(used)
    n_det = len(detections)
    return {
        "n_truth": n_truth,
        "n_detections": n_det,
        "n_matched": len(matches),
        "n_false": n_false,
        "detection_rate": len(matches) / n_truth if n_truth else 0.0,
        "false_alarm_rate": n_false / n_det if n_det else 0.0,
        "position_rmse_pix": float(np.sqrt(np.mean([m["dist"] ** 2 for m in matches]))) if matches else float("nan"),
        "matches": matches,
    }


def star_metrics(stars: Sequence[Dict[str, float]], catalog: Dict[str, np.ndarray],
                 radius: float = 2.0) -> Dict[str, object]:
    """Completeness and centroid error using the detector-position catalogue."""
    cx = catalog["x_det"]
    cy = catalog["y_det"]
    used = set()
    errs = []
    for i in range(len(cx)):
        best = None
        best_d = 1e9
        for j, s in enumerate(stars):
            if j in used:
                continue
            d = math.hypot(s["x"] - cx[i], s["y"] - cy[i])
            if d < best_d and d <= radius:
                best_d = d
                best = j
        if best is not None:
            used.add(best)
            errs.append(best_d)
    return {
        "n_catalog": int(len(cx)),
        "n_detected": len(stars),
        "n_matched": len(errs),
        "completeness": len(errs) / len(cx) if len(cx) else 0.0,
        "centroid_rmse_pix": float(np.sqrt(np.mean(np.square(errs)))) if errs else float("nan"),
    }


def evaluate(data_dir: str, output_dir: str, stars: Sequence[Dict[str, float]],
             moving: Dict[str, object], solution: AstrometricSolution,
             cfg: PipelineConfig) -> Dict[str, object]:
    """Compute all metrics and write CSV/JSON tables."""
    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    truth = read_csv(str(Path(data_dir) / "truth.csv"))
    catalog = load_catalog(str(Path(data_dir) / "catalog.csv"))

    dets = moving["detections"]
    m = match_by_frame(dets, truth, radius=3.0)
    s = star_metrics(stars, catalog, radius=2.0)

    # target astrometry: apply the solution to the matched detections
    if m["matches"]:
        xs = np.array([x["x"] for x in m["matches"]])
        ys = np.array([x["y"] for x in m["matches"]])
        ra, dec = target_sky(solution, xs, ys)
        tra = np.array([float(truth[x["frame"]]["ra"]) for x in m["matches"]])
        tdec = np.array([float(truth[x["frame"]]["dec"]) for x in m["matches"]])
        astro_err = angular_sep_arcsec(ra, dec, tra, tdec)
        astro_rmse = float(np.sqrt(np.mean(astro_err ** 2)))
        astro_max = float(np.max(astro_err))
    else:
        astro_err = np.array([])
        astro_rmse = float("nan")
        astro_max = float("nan")

    longest = max((len(t["detections"]) for t in moving["tracks"]), default=0)
    n_tracks = len(moving["tracks"])
    metrics = {
        "target": m,
        "stars": s,
        "astrometry": {
            "n_matches": solution.n_matches,
            "star_residual_rms_arcsec": solution.rms_arcsec,
            "target_rmse_arcsec": astro_rmse,
            "target_max_arcsec": astro_max,
        },
        "tracks": {"n_tracks": n_tracks, "longest_track": longest},
        "config": {
            "n_frames": cfg.sim.n_frames,
            "n_stars": cfg.sim.n_stars,
            "target_flux": cfg.sim.target_flux,
        },
    }
    with open(root / "metrics.json", "w", encoding="utf-8") as fh:
        json.dump(metrics, fh, indent=2, ensure_ascii=False)

    write_csv(str(root / "detections.csv"), dets,
              ["frame", "x", "y", "flux", "area", "major", "minor", "elongation",
               "angle_deg", "hough_score", "hough_theta_deg", "snr"])
    write_csv(str(root / "stars.csv"), stars,
              ["id", "x", "y", "flux", "peak", "snr", "area"])
    astro_rows = []
    for k, e in enumerate(astro_err):
        astro_rows.append({"index": k, "error_arcsec": float(e)})
    write_csv(str(root / "astrometry_errors.csv"), astro_rows, ["index", "error_arcsec"])
    return metrics
