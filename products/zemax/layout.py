"""2-D/3-D optical-layout renderer for a parsed ZMX prescription.

``products.zemax.zmx_parser`` returns the sequential surface table (``curv``,
``thickness``, ``conic``, ``diameter``, ``glass``, ``stop``).  This module turns
that table into a JSON-safe display model and draws the *real* optical layout:
a rotationally-symmetric conic patch per surface on a folded optical axis.

Conventions
-----------
* Units are millimetres throughout.
* ``curv`` is 1/R, so ``radius = 1/curv``.  ``curv == 0`` is a plane (infinite
  radius); it is reported as ``radius = None`` so the model stays JSON-safe.
* A surface vertex sits at ``z_i = sum(thickness[:i])``.  Negative thickness is
  a mirror fold: the folded axis runs backwards, exactly as in the sequential
  model, so the folded layout is the physical beam path.
* Sag is ``z(r) = c*r^2 / (1 + sqrt(1 - (1+k)*c^2*r^2))`` with ``c = curv`` and
  ``k = conic``.  A negative radius gives a negative sag (the surface curves
  toward -z).  If the radical goes negative it is clamped to zero so an edge
  sample always plots (that region is outside the conic's real domain anyway).
* ZMX ``DIAM`` is the *semi*-diameter (Zemax convention), hence
  ``clear_aperture = 2 * DIAM``.
* Zernike phase surfaces (``SZERNPHA``) are drawn as planes.  A phase profile
  is not part of a geometric layout.

The 3-D renderer uses matplotlib's Agg canvas directly (no pyplot, no global
backend switch) and writes deterministic PNGs below 400 KB.
"""
from __future__ import annotations

import base64
import io
import math
from pathlib import Path

import numpy as np

__all__ = ["model_geometry", "render_layout_3d", "render_layout_2d", "sag_z"]

_PNG_LIMIT_BYTES = 400 * 1024

# Display palette / transparency per optical element kind.
_COLORS = {
    "mirror": "#d95f02",     # orange
    "refractor": "#1f77b4",  # blue
    "dummy": "#7f7f7f",      # grey
    "phase": "#6a3d9a",      # purple (Zernike phase plate)
}
_ALPHA = {"mirror": 0.32, "refractor": 0.24, "phase": 0.22, "dummy": 0.12}


def _finite(value):
    """Return ``float(value)`` when it is finite, else ``None``."""
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _positive(value, default):
    v = _finite(value)
    return v if v is not None and v > 0.0 else default


def sag_z(r, curv, conic=0.0):
    """Sag of a rotationally-symmetric conic, ``z(r)`` in mm.

    ``curv`` is 1/R in 1/mm, ``conic`` is k.  A plane (``curv == 0``) returns
    zero.  Scalars come back as ``float``; arrays come back as ``ndarray``.
    """
    scalar = np.isscalar(r)
    rr = np.asarray(r, dtype=float)
    c = _finite(curv)
    if c is None or c == 0.0:
        out = np.zeros_like(rr)
    else:
        k = _finite(conic)
        k = 0.0 if k is None else k
        q = 1.0 - (1.0 + k) * c * c * rr * rr
        q = np.maximum(q, 0.0)  # outside the conic domain: keep the edge finite
        out = c * rr * rr / (1.0 + np.sqrt(q))
    return float(out) if scalar else out


def model_geometry(parsed: dict) -> dict:
    """Convert a ``parse_zmx_*`` result into a JSON-safe display model.

    Returns ``{"surfaces", "surface_count", "track_mm", "stop", "stop_z",
    "aperture_mm", "wavelengths_um", "extent", "folded_span_mm", "units",
    "notes"}``.  ``track_mm`` is the *unfolded* optical path length, i.e. the
    sum of the absolute finite thicknesses; ``folded_span_mm`` is the spatial
    ``z_max - z_min`` of the folded layout (they differ when mirrors fold).
    """
    parsed = parsed if isinstance(parsed, dict) else {}
    raw = list(parsed.get("surfaces") or [])
    aperture = _finite(parsed.get("aperture_mm"))
    stop = parsed.get("stop")
    try:
        stop = int(stop) if stop is not None else None
    except (TypeError, ValueError):
        stop = None

    notes = []
    surfaces = []
    index_by_id = {}
    z_cursor = 0.0
    track = 0.0
    mirror_count = 0
    stop_z = None

    for pos, src in enumerate(raw):
        src = src if isinstance(src, dict) else {}
        idx = src.get("index")
        try:
            idx = int(idx) if idx is not None else pos
        except (TypeError, ValueError):
            idx = pos

        # Vertex is the running sum of the *previous* thicknesses.
        vertex = z_cursor
        thickness = _finite(src.get("thickness"))
        if thickness is not None:
            track += abs(thickness)
            z_cursor += thickness
        elif pos > 0:
            notes.append(
                "Surface %s: non-finite/missing thickness treated as 0 for layout."
                % idx
            )

        stype = str(src.get("type") or "STANDARD").strip() or "STANDARD"
        type_u = stype.upper()
        glass = str(src.get("glass") or "").strip()
        glass_u = glass.upper()

        is_phase = ("SZERN" in type_u) or ("ZERN" in type_u) or ("PHASE" in type_u)
        is_mirror = (glass_u == "MIRROR") or ("MIRROR" in type_u)
        is_fold = thickness is not None and thickness < 0.0

        curv = _finite(src.get("curv"))
        curv = 0.0 if curv is None else curv
        if is_phase:
            notes.append(
                "Surface %s (%s) is a Zernike phase surface; it is drawn as a "
                "plane and the phase terms are not applied." % (idx, stype)
            )
            if curv != 0.0:
                notes.append(
                    "Surface %s: phase-surface curvature ignored for the layout." % idx
                )
            curv = 0.0

        radius = None
        if curv != 0.0:
            r_try = 1.0 / curv
            radius = r_try if math.isfinite(r_try) else None

        conic = _finite(src.get("conic"))
        conic = 0.0 if conic is None else conic

        # ZMX DIAM is the semi-diameter already.
        semi_diameter = abs(_finite(src.get("diameter")) or 0.0)
        clear_aperture = 2.0 * semi_diameter

        if is_mirror:
            kind = "mirror"
            mirror_count += 1
            label = "M%d" % mirror_count
        elif is_phase:
            kind = "dummy"
            label = "PHASE"
        elif glass_u and glass_u != "MIRROR":
            kind = "refractor"
            label = "S%s" % idx
        else:
            kind = "dummy"
            label = "S%s" % idx

        is_object = (pos == 0 and thickness is None and semi_diameter == 0.0)
        is_last = (pos == len(raw) - 1) and len(raw) > 1
        if is_mirror:
            pass
        elif is_phase:
            pass
        elif is_object:
            label = "OBJ"
        elif is_last:
            label = "IMAGE"
        if stop is not None and idx == stop:
            label = "%s (STOP)" % label
            stop_z = vertex

        note = None
        if is_phase:
            note = "Zernike phase surface drawn as a plane; phase terms not applied."
        elif is_fold:
            note = "Negative thickness: mirror fold along the optical axis."

        surface = {
            "index": idx,
            "type": stype,
            "glass": glass or None,
            "kind": kind,
            "label": label,
            "is_mirror": is_mirror,
            "is_phase": is_phase,
            "is_fold": is_fold,
            "vertex_z": float(vertex),
            "thickness": thickness,
            "curv": float(curv),
            "radius": float(radius) if radius is not None else None,
            "conic": float(conic),
            "semi_diameter": float(semi_diameter),
            "clear_aperture": float(clear_aperture),
            "sag": {
                "curv": float(curv),
                "radius": float(radius) if radius is not None else None,
                "conic": float(conic),
                "semi_diameter": float(semi_diameter),
                "max_sag": float(sag_z(semi_diameter, curv, conic)),
            },
            "note": note,
        }
        surfaces.append(surface)
        index_by_id[idx] = surface

    if raw:
        notes.append(
            "ZMX DIAM is the semi-diameter (Zemax convention); clear aperture = "
            "2 x DIAM."
        )
    else:
        notes.append("Empty surface table: nothing to render.")

    folds = [s["index"] for s in surfaces if s["is_fold"]]
    if folds:
        notes.append(
            "Mirror fold(s) at surface(s) %s: vertex z runs backwards along the "
            "folded axis." % ", ".join(str(i) for i in folds)
        )

    if stop is None:
        notes.append("No STOP surface in the prescription.")
    elif stop not in index_by_id:
        notes.append("STOP surface %s is not present in the surface table." % stop)

    zs = [s["vertex_z"] for s in surfaces]
    sag_lo = min(
        [s["vertex_z"] + min(0.0, s["sag"]["max_sag"]) for s in surfaces] or [0.0]
    )
    sag_hi = max(
        [s["vertex_z"] + max(0.0, s["sag"]["max_sag"]) for s in surfaces] or [0.0]
    )
    radius_max = max([s["semi_diameter"] for s in surfaces] or [0.0])
    wavelengths = []
    for w in parsed.get("wavelengths_um") or []:
        wv = _finite(w)
        if wv is not None:
            wavelengths.append(wv)

    return {
        "surfaces": surfaces,
        "surface_count": len(surfaces),
        "track_mm": float(track),
        "stop": stop,
        "stop_z": float(stop_z) if stop_z is not None else None,
        "aperture_mm": aperture,
        "wavelengths_um": wavelengths,
        "extent": {
            "z_min": float(min(zs) if zs else 0.0),
            "z_max": float(max(zs) if zs else 0.0),
            "z_min_with_sag": float(sag_lo),
            "z_max_with_sag": float(sag_hi),
            "radius": float(radius_max),
        },
        "folded_span_mm": float((max(zs) - min(zs)) if zs else 0.0),
        "units": "mm",
        "notes": notes,
    }


def _surface_color(surface: dict) -> str:
    if surface.get("is_phase"):
        return _COLORS["phase"]
    return _COLORS.get(surface.get("kind"), _COLORS["dummy"])


def _surface_alpha(surface: dict) -> float:
    if surface.get("is_phase"):
        return _ALPHA["phase"]
    return _ALPHA.get(surface.get("kind"), _ALPHA["dummy"])


def _label_visible(surface: dict) -> bool:
    """M1/M2/M3, STOP, IMAGE, PHASE and OBJ labels; skip generic S# clutter."""
    label = str(surface.get("label") or "")
    if "STOP" in label or label in ("IMAGE", "PHASE", "OBJ"):
        return True
    return len(label) > 1 and label[0] == "M" and label[1:2].isdigit()


def _limits(geom: dict):
    """Return ``(rmax, zmin, zmax)`` display limits including sag and padding."""
    surfaces = geom["surfaces"]
    radius = max(
        [s["semi_diameter"] for s in surfaces] + [(geom.get("aperture_mm") or 0.0) / 2.0, 1.0]
    )
    ext = geom["extent"]
    zmin = ext["z_min_with_sag"]
    zmax = ext["z_max_with_sag"]
    span = zmax - zmin
    if span <= 1e-9:
        pad = 1.0
    else:
        pad = 0.04 * span + 0.5
    return float(radius), float(zmin - pad), float(zmax + pad)


def _save_png(fig, png_path=None) -> bytes:
    """Save ``fig`` to PNG bytes, shrinking DPI until below the size limit."""
    dpi = float(fig.dpi)
    data = b""
    while True:
        buf = io.BytesIO()
        fig.savefig(
            buf,
            format="png",
            dpi=dpi,
            facecolor="white",
            metadata={"Software": "products.zemax.layout"},
        )
        data = buf.getvalue()
        if len(data) <= _PNG_LIMIT_BYTES or dpi <= 45.0:
            break
        dpi *= 0.8
    if png_path:
        path = Path(png_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(data)
    return data


def _annotation(geom: dict) -> str:
    stop = geom.get("stop")
    if stop is None:
        stop_line = "STOP: n/a"
    elif geom.get("stop_z") is not None:
        stop_line = "STOP: surface %s @ z=%.1f mm" % (stop, geom["stop_z"])
    else:
        stop_line = "STOP: surface %s" % stop
    lines = [
        "ZMX optical layout (folded axis)",
        "surfaces: %d" % geom["surface_count"],
        "track: %.1f mm" % geom["track_mm"],
    ]
    folded = geom.get("folded_span_mm")
    if folded is not None and abs(folded - geom["track_mm"]) > 1e-6:
        lines.append("folded span: %.1f mm" % folded)
    lines.append(stop_line)
    return "\n".join(lines)


def render_layout_3d(parsed, png_path=None, azim=-60, elev=18, width=6, height=4) -> dict:
    """Render the 3-D optical layout to PNG.

    Each surface is a rotationally-symmetric conic patch over its
    semi-diameter; mirrors are semi-transparent.  The optical axis, the surface
    labels (M1/M2/M3/STOP/IMAGE/PHASE) and the track/stop annotations are drawn.

    Returns ``{"png_b64", "surfaces", "track_mm", "stop", "notes",
    "png_size_bytes"}``.  Calculations use mm.
    """
    geom = model_geometry(parsed)
    surfaces = geom["surfaces"]
    width = _positive(width, 6.0)
    height = _positive(height, 4.0)
    azim = _finite(azim)
    elev = _finite(elev)
    azim = -60.0 if azim is None else azim
    elev = 18.0 if elev is None else elev

    import mpl_toolkits.mplot3d  # noqa: F401  (registers the "3d" projection)
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    fig = Figure(figsize=(width, height), dpi=120, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("white")
    ax.set_position([0.0, 0.0, 1.0, 1.0])

    # Layout convention: optical z runs along the matplotlib x axis so the tube
    # is horizontal; transverse coordinates use the y/z axes.  At azim=-60,
    # elev=18 this gives the familiar oblique optical-layout view.
    rmax, zlim_lo, zlim_hi = _limits(geom)
    ax.set_xlim(zlim_lo, zlim_hi)
    ax.set_ylim(-1.15 * rmax, 1.15 * rmax)
    ax.set_zlim(-1.15 * rmax, 1.15 * rmax)
    try:
        ax.set_box_aspect((zlim_hi - zlim_lo, 2.3 * rmax, 2.3 * rmax), zoom=1.35)
    except TypeError:  # pragma: no cover - matplotlib without the zoom kwarg
        try:
            ax.set_box_aspect((zlim_hi - zlim_lo, 2.3 * rmax, 2.3 * rmax))
        except AttributeError:
            pass
    except AttributeError:  # pragma: no cover - matplotlib < 3.3
        pass

    # Folded optical axis.
    ax.plot(
        [zlim_lo, zlim_hi],
        [0.0, 0.0],
        [0.0, 0.0],
        color="0.25",
        linewidth=0.8,
        linestyle="--",
        zorder=0,
    )

    n_r = 17
    n_t = 49
    theta = np.linspace(0.0, 2.0 * np.pi, n_t)
    ring = np.linspace(0.0, 2.0 * np.pi, 97)
    for surface in surfaces:
        sd = surface["semi_diameter"]
        if sd <= 1e-9:
            continue
        r = np.linspace(0.0, sd, n_r)
        rg, tg = np.meshgrid(r, theta)
        xg = surface["vertex_z"] + sag_z(rg, surface["curv"], surface["conic"])
        yg = rg * np.cos(tg)
        zg = rg * np.sin(tg)
        color = _surface_color(surface)
        ax.plot_surface(
            xg,
            yg,
            zg,
            color=color,
            alpha=_surface_alpha(surface),
            linewidth=0.0,
            antialiased=True,
            shade=False,
            rstride=1,
            cstride=1,
        )
        x_rim = surface["vertex_z"] + float(
            sag_z(sd, surface["curv"], surface["conic"])
        )
        ax.plot(
            np.full_like(ring, x_rim),
            sd * np.cos(ring),
            sd * np.sin(ring),
            color=color,
            linewidth=1.2,
            alpha=0.95,
        )

    for surface in surfaces:
        if not _label_visible(surface):
            continue
        sd = surface["semi_diameter"]
        label_y = min(max(sd * 1.04, 0.06 * rmax), 1.08 * rmax)
        ax.text(
            surface["vertex_z"],
            label_y,
            0.0,
            surface["label"],
            fontsize=7,
            color=_surface_color(surface),
            ha="center",
            va="bottom",
            zorder=10,
        )

    ax.text2D(
        0.02,
        0.975,
        _annotation(geom),
        transform=ax.transAxes,
        fontsize=7.5,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
              "edgecolor": "0.8", "alpha": 0.85},
    )
    ax.view_init(elev=elev, azim=azim, roll=0)
    ax.set_axis_off()

    data = _save_png(fig, png_path)
    return {
        "png_b64": base64.b64encode(data).decode("ascii"),
        "surfaces": surfaces,
        "track_mm": geom["track_mm"],
        "stop": geom["stop"],
        "notes": list(geom["notes"]),
        "png_size_bytes": len(data),
    }


def render_layout_2d(parsed, png_path=None) -> dict:
    """Render a y-z cross-section of the optical layout (fallback / docs).

    The sag curve of every surface is drawn over its semi-diameter, together
    with the folded optical axis, the M1/M2/M3/STOP/IMAGE/PHASE labels and the
    track/stop annotation.  Returns the same contract as
    :func:`render_layout_3d`.
    """
    geom = model_geometry(parsed)
    surfaces = geom["surfaces"]

    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure
    from matplotlib.lines import Line2D

    fig = Figure(figsize=(7.2, 3.2), dpi=110, facecolor="white")
    FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.set_facecolor("white")
    # Leave room for the x label / tick labels below the equal-aspect box.
    fig.subplots_adjust(left=0.08, right=0.99, top=0.96, bottom=0.20)

    rmax, zlim_lo, zlim_hi = _limits(geom)
    ax.set_xlim(zlim_lo, zlim_hi)
    ax.set_ylim(-1.12 * rmax, 1.18 * rmax)

    ax.plot(
        [zlim_lo, zlim_hi],
        [0.0, 0.0],
        color="0.25",
        linewidth=0.8,
        linestyle="--",
        zorder=0,
    )

    for surface in surfaces:
        sd = surface["semi_diameter"]
        color = _surface_color(surface)
        if sd > 1e-9:
            y = np.linspace(-sd, sd, 121)
            z = surface["vertex_z"] + sag_z(np.abs(y), surface["curv"], surface["conic"])
            ax.plot(z, y, color=color, linewidth=1.3, alpha=0.95)
        else:
            ax.plot(
                [surface["vertex_z"]],
                [0.0],
                marker="|",
                markersize=6,
                color=color,
            )

    placed = {}
    for surface in surfaces:
        if not _label_visible(surface):
            continue
        key = round(surface["vertex_z"], 6)
        stack = placed.get(key, 0)
        placed[key] = stack + 1
        sd = surface["semi_diameter"]
        label_y = max(sd + 0.055 * rmax, 0.09 * rmax) + stack * 0.11 * rmax
        ax.text(
            surface["vertex_z"],
            label_y,
            surface["label"],
            fontsize=7.5,
            color=_surface_color(surface),
            ha="center",
            va="bottom",
        )

    kinds = []
    handles = []
    legend_rows = (
        ("mirror", "mirror", any(s.get("kind") == "mirror" for s in surfaces)),
        ("refractor", "refractor", any(s.get("kind") == "refractor" for s in surfaces)),
        ("phase", "Zernike phase", any(s.get("is_phase") for s in surfaces)),
        (
            "dummy",
            "dummy/plane",
            any(s.get("kind") == "dummy" and not s.get("is_phase") for s in surfaces),
        ),
    )
    for kind, text, present in legend_rows:
        if present:
            kinds.append(text)
            handles.append(Line2D([0], [0], color=_COLORS[kind], linewidth=2.0))
    if handles:
        ax.legend(handles, kinds, fontsize=6.5, loc="lower right", frameon=False)

    ax.text(
        0.012,
        0.982,
        _annotation(geom),
        transform=ax.transAxes,
        fontsize=7,
        va="top",
        ha="left",
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white",
              "edgecolor": "0.8", "alpha": 0.85},
    )
    ax.set_xlabel("z along folded optical axis (mm)", fontsize=8)
    ax.set_ylabel("y / semi-diameter (mm)", fontsize=8)
    ax.tick_params(labelsize=7)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.grid(True, color="0.92", linewidth=0.6)
    ax.set_axisbelow(True)
    try:
        ax.set_aspect("equal", adjustable="box")
    except Exception:  # pragma: no cover - very old matplotlib
        pass

    data = _save_png(fig, png_path)
    return {
        "png_b64": base64.b64encode(data).decode("ascii"),
        "surfaces": surfaces,
        "track_mm": geom["track_mm"],
        "stop": geom["stop"],
        "notes": list(geom["notes"]),
        "png_size_bytes": len(data),
    }
