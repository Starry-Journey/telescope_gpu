"""Self-contained conic ray tracer (no rayoptics/Zemax at runtime).

Reduced on-axis model of 2.5m ZMX:
  object -> M1 (conic) -> M2 (conic, decenter/tilt/focus) -> image
The real ZMX has fold flats/coordinate breaks; for on-axis OPD the folded path
is equivalent to the summed gaps, so this reduced model keeps the powered
surfaces and exposes the M2 perturbations.

Coordinate breaks are handled by a local frame on M2 (translation + Rx/Ry).
The main PD/control algorithms remain ours; rayoptics is only a comparator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

# M1/M2 from 2.5m主系统.ZMX (surface 2/4); gaps sum to 4928.494 mm.
M1 = {"z": 0.0, "c": -1.3640703860319192e-4, "k": -1.02347, "R": 1250.0}
M2 = {"z": -2783.7895666527347, "c": -4.6562955443907935e-4, "k": -2.23765, "R": 304.91}
IMAGE_Z = -2783.7895666527347 + 2083.7895666527347 + 1250.0 + 1594.70466518518
OBJ_Z = -3100.0


@dataclass
class Ray:
    o: np.ndarray
    d: np.ndarray


def sag(r, c, k):
    q = 1.0 - (1.0 + k) * c * c * r * r
    return c * r * r / (1.0 + np.sqrt(np.maximum(q, 1e-12)))


def _normal_local(x, y, c, k):
    r = math.hypot(x, y)
    # d(sag)/dr = c*r / sqrt(1-(1+k)c^2 r^2)
    q = max(1.0 - (1.0 + k) * c * c * r * r, 1e-12)
    dsdr = c * r / math.sqrt(q)
    if r > 1e-12:
        dzdx, dzdy = dsdr * x / r, dsdr * y / r
    else:
        dzdx = dzdy = 0.0
    n = np.array([-dzdx, -dzdy, 1.0])
    return n / np.linalg.norm(n)


def intersect(ray, surf, dx=0.0, dy=0.0, tx=0.0, ty=0.0):
    """Intersect ray with conic; local frame = M2 vertex + small Rx/Ry."""
    c, k, z0 = surf["c"], surf["k"], surf["z"]
    # rotation: local = R^T (global - origin)
    cx, sx = math.cos(tx), math.sin(tx)
    cy, sy = math.cos(ty), math.sin(ty)
    R = np.array([[cy, 0, sy], [sx * sy, cx, -sx * cy], [-cx * sy, sx, cx * cy]])
    origin = np.array([dx, dy, z0])
    o = R.T @ (ray.o - origin)
    d = R.T @ ray.d
    t = -o[2] / d[2]
    for _ in range(12):
        p = o + t * d
        r = math.hypot(p[0], p[1])
        zs = sag(r, c, k)
        f = p[2] - zs
        # numeric derivative of F along ray
        eps = 1e-5
        p2 = o + (t + eps) * d
        r2 = math.hypot(p2[0], p2[1])
        f2 = p2[2] - sag(r2, c, k)
        dt = f / ((f2 - f) / eps) if abs(f2 - f) > 1e-18 else 0.0
        t -= dt
        if abs(dt) < 1e-10:
            break
    p = o + t * d
    n = _normal_local(p[0], p[1], c, k)
    p_global = origin + R @ p
    n_global = R @ n
    # reflect; force normal to face incoming ray
    if np.dot(ray.d, n_global) > 0:
        n_global = -n_global
    d_out = ray.d - 2.0 * np.dot(ray.d, n_global) * n_global
    return Ray(p_global, d_out), t


def trace_exit(h, theta, dx=0.0, dy=0.0, tx=0.0, ty=0.0, focus=0.0):
    """Return exit-pupil point, direction, and object->exit path length."""
    x, y = h * math.cos(theta), h * math.sin(theta)
    ray = Ray(np.array([x, y, OBJ_Z]), np.array([0.0, 0.0, 1.0]))
    start = ray.o.copy()
    ray, _ = intersect(ray, M1)
    p1 = ray.o.copy()
    m2 = dict(M2); m2["z"] += focus
    ray, _ = intersect(ray, m2, dx, dy, tx, ty)
    p2 = ray.o.copy()
    path = np.linalg.norm(p1 - start) + np.linalg.norm(p2 - p1)
    return p2, ray.d.copy(), path


def trace_path(h, theta, dx=0.0, dy=0.0, tx=0.0, ty=0.0, focus=0.0):
    """Return image point and total geometric path length (no ref sphere)."""
    p2, d, path = trace_exit(h, theta, dx, dy, tx, ty, focus)
    t = (IMAGE_Z - p2[2]) / d[2]
    p3 = p2 + t * d
    return p3, path + t


def opd_reference_sphere(h, theta, dx=0.0, dy=0.0, tx=0.0, ty=0.0, focus=0.0,
                         chief_h=None):
    """OPD to a sphere centered at the chief image point.

    This is the missing piece versus the previous fixed-image-plane OPD.
    """
    if chief_h is None:
        chief_h = 0.30 * M1["R"]
    cp2, cd, cpath = trace_exit(chief_h, 0.0, dx, dy, tx, ty, focus)
    ct = (IMAGE_Z - cp2[2]) / cd[2]
    center = cp2 + ct * cd
    radius = np.linalg.norm(cp2 - center)

    p2, d, path = trace_exit(h, theta, dx, dy, tx, ty, focus)
    w = p2 - center
    b = 2.0 * np.dot(d, w)
    c = float(np.dot(w, w) - radius * radius)
    disc = b * b - 4.0 * c
    if disc < 0:
        # ray misses the reference sphere; use closest approach
        t = -np.dot(d, w)
    else:
        sq = math.sqrt(disc)
        roots = [(-b - sq) / 2.0, (-b + sq) / 2.0]
        roots = [r for r in roots if r > 0]
        t = min(roots) if roots else radius
    return (path + t) - (cpath + radius), center, radius
