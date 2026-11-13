"""WCS + optical-distortion astrometry and catalogue matching.

Coordinate chain
----------------
    detector pixel --(distortion)--> ideal pixel --(TAN WCS)--> RA/Dec

The FITS header carries the TAN WCS.  The distortion polynomial is fitted
from star matches, then applied to the moving-target detections.  This is the
same decomposition used by SIP / SCAMP-style astrometric solutions, but the
polynomial is kept explicit so the least-squares step is visible.
"""
from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from astropy.wcs import WCS
from scipy.spatial import cKDTree

from .config import AstrometryConfig


def _unit_vectors(ra_deg: np.ndarray, dec_deg: np.ndarray) -> np.ndarray:
    ra = np.radians(np.asarray(ra_deg, dtype=float))
    dec = np.radians(np.asarray(dec_deg, dtype=float))
    return np.column_stack([np.cos(dec) * np.cos(ra),
                            np.cos(dec) * np.sin(ra),
                            np.sin(dec)])


def angular_sep_arcsec(ra1, dec1, ra2, dec2) -> np.ndarray:
    v1 = _unit_vectors(np.atleast_1d(ra1), np.atleast_1d(dec1))
    v2 = _unit_vectors(np.atleast_1d(ra2), np.atleast_1d(dec2))
    dot = np.clip(np.sum(v1 * v2, axis=1), -1.0, 1.0)
    return np.degrees(np.arccos(dot)) * 3600.0


@dataclass
class DistortionModel:
    cx: float
    cy: float
    scale: float
    degree: int = 2
    coeff_x: Optional[np.ndarray] = None
    coeff_y: Optional[np.ndarray] = None

    def _design(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        u = (np.asarray(x, dtype=float) - self.cx) / self.scale
        v = (np.asarray(y, dtype=float) - self.cy) / self.scale
        cols = [np.ones_like(u), u, v]
        if self.degree >= 2:
            cols += [u * u, u * v, v * v]
        if self.degree >= 3:
            cols += [u ** 3, u * u * v, u * v * v, v ** 3]
        return np.column_stack(cols).reshape(-1, len(cols))

    def fit(self, x_det: np.ndarray, y_det: np.ndarray,
            x_ideal: np.ndarray, y_ideal: np.ndarray) -> "DistortionModel":
        A = self._design(x_det, y_det)
        self.coeff_x, *_ = np.linalg.lstsq(A, np.asarray(x_ideal, dtype=float), rcond=None)
        self.coeff_y, *_ = np.linalg.lstsq(A, np.asarray(y_ideal, dtype=float), rcond=None)
        return self

    def apply(self, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if self.coeff_x is None or self.coeff_y is None:
            return np.asarray(x, dtype=float), np.asarray(y, dtype=float)
        A = self._design(x, y)
        return A @ self.coeff_x, A @ self.coeff_y

    def invert(self, x_ideal: np.ndarray, y_ideal: np.ndarray,
               iterations: int = 6) -> Tuple[np.ndarray, np.ndarray]:
        xi = np.asarray(x_ideal, dtype=float)
        yi = np.asarray(y_ideal, dtype=float)
        xd, yd = xi.copy(), yi.copy()
        for _ in range(iterations):
            fx, fy = self.apply(xd, yd)
            xd += xi - fx
            yd += yi - fy
        return xd, yd

    def to_dict(self) -> Dict[str, object]:
        return {
            "cx": self.cx, "cy": self.cy, "scale": self.scale,
            "degree": self.degree,
            "coeff_x": None if self.coeff_x is None else self.coeff_x.tolist(),
            "coeff_y": None if self.coeff_y is None else self.coeff_y.tolist(),
        }


@dataclass
class AstrometricSolution:
    wcs: WCS
    distortion: DistortionModel
    matched_det: List[int]
    matched_cat: List[int]
    residuals_arcsec: np.ndarray
    rms_arcsec: float
    n_matches: int

    def pixel_to_sky(self, x: np.ndarray, y: np.ndarray):
        xi, yi = self.distortion.apply(np.asarray(x), np.asarray(y))
        return self.wcs.pixel_to_world(xi, yi)


def load_catalog(path: str) -> Dict[str, np.ndarray]:
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append(row)
    return {
        "ra": np.array([float(r["ra"]) for r in rows]),
        "dec": np.array([float(r["dec"]) for r in rows]),
        "x_det": np.array([float(r["x_det"]) for r in rows]),
        "y_det": np.array([float(r["y_det"]) for r in rows]),
        "x_ideal": np.array([float(r["x_ideal"]) for r in rows]),
        "y_ideal": np.array([float(r["y_ideal"]) for r in rows]),
        "flux": np.array([float(r["flux"]) for r in rows]),
    }


def match_sky(det_ra, det_dec, cat_ra, cat_dec, radius_arcsec: float) -> Tuple[List[int], List[int]]:
    """Nearest-neighbour match on the unit sphere with a radius gate."""
    det = _unit_vectors(np.asarray(det_ra), np.asarray(det_dec))
    cat = _unit_vectors(np.asarray(cat_ra), np.asarray(cat_dec))
    tree = cKDTree(cat)
    dist, idx = tree.query(det, k=1)
    # chord distance -> angle
    ang = 2.0 * np.arcsin(np.clip(dist / 2.0, 0.0, 1.0))
    md, mc = [], []
    used = set()
    for i, (a, j) in enumerate(zip(ang, idx)):
        if a * 206264.806 <= radius_arcsec and int(j) not in used:
            md.append(i)
            mc.append(int(j))
            used.add(int(j))
    return md, mc


def solve_astrometry(stars: Sequence[Dict[str, float]], catalog: Dict[str, np.ndarray],
                     wcs: WCS, cfg: AstrometryConfig) -> AstrometricSolution:
    """Iterative distortion fit + catalogue matching.

    Steps
    -----
    1. initial WCS prediction for each detected star;
    2. match to catalogue;
    3. fit detector -> ideal-pixel polynomial using matched catalogue stars;
    4. apply, re-match, repeat.
    """
    x = np.array([s["x"] for s in stars], dtype=float)
    y = np.array([s["y"] for s in stars], dtype=float)
    if x.size < cfg.min_matches:
        raise RuntimeError(f"only {x.size} stars detected; need {cfg.min_matches}")

    sky = wcs.pixel_to_world(x, y)
    ra = np.asarray(sky.ra.deg, dtype=float)
    dec = np.asarray(sky.dec.deg, dtype=float)
    md, mc = match_sky(ra, dec, catalog["ra"], catalog["dec"], cfg.match_radius_arcsec)
    if len(md) < cfg.min_matches:
        raise RuntimeError(f"only {len(md)} catalogue matches; need {cfg.min_matches}")

    cx = float(np.mean(x))
    cy = float(np.mean(y))
    scale = float(max(x.max() - x.min(), y.max() - y.min(), 1.0))
    dist = DistortionModel(cx=cx, cy=cy, scale=scale, degree=cfg.distortion_degree)

    best = None
    for _ in range(cfg.max_iterations):
        md, mc = match_sky(ra, dec, catalog["ra"], catalog["dec"], cfg.match_radius_arcsec)
        if len(md) < cfg.min_matches:
            break
        # catalogue sky -> ideal pixel through the TAN WCS
        xi, yi = wcs.world_to_pixel_values(catalog["ra"][mc], catalog["dec"][mc])
        dist.fit(x[md], y[md], xi, yi)
        xc, yc = dist.apply(x, y)
        s2 = wcs.pixel_to_world(xc, yc)
        ra = np.asarray(s2.ra.deg, dtype=float)
        dec = np.asarray(s2.dec.deg, dtype=float)
        md2, mc2 = match_sky(ra, dec, catalog["ra"], catalog["dec"], cfg.match_radius_arcsec)
        res = angular_sep_arcsec(ra[md2], dec[md2], catalog["ra"][mc2], catalog["dec"][mc2])
        best = (md2, mc2, res)
        if len(md2) == len(md) and len(res) and float(np.sqrt(np.mean(res ** 2))) < 0.05:
            break

    if best is None:
        raise RuntimeError("astrometric fit failed")
    md2, mc2, res = best
    rms = float(np.sqrt(np.mean(res ** 2))) if len(res) else float("inf")
    return AstrometricSolution(wcs=wcs, distortion=dist, matched_det=md2, matched_cat=mc2,
                               residuals_arcsec=res, rms_arcsec=rms, n_matches=len(md2))


def target_sky(solution: AstrometricSolution, x: np.ndarray, y: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    sky = solution.pixel_to_sky(np.asarray(x), np.asarray(y))
    return np.asarray(sky.ra.deg), np.asarray(sky.dec.deg)
