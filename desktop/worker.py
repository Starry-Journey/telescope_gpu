"""Service worker: no PySide6/Qt imports.

Called as: EXE --worker <task> --job job.json --result result.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path


def _fea_viz(payload):
    """Render uploaded/solved FEA surface data as a 3D PNG (matplotlib Agg).

    payload: {"x": [...], "y": [...], "z": [...], "label": "...",
              "azim": -60, "elev": 25,
              "obscuration": 0.28, "aperture_radius_mm": 1250}

    The x/y node arrays are in mm, z in nm.  A Delaunay triangulated surface
    coloured by dz_nm (with colourbar) is overlaid with a semi-transparent
    node cloud so the 3D structure reads at a glance.  No Qt imports here:
    the PNG, the regular interpolation grid and RMS/PV/n statistics travel
    back to the GUI through the ordinary worker result JSON.
    """
    import base64
    import io

    import numpy as np
    from scipy.interpolate import griddata

    x = np.asarray(payload.get("x") or [], dtype=float)
    y = np.asarray(payload.get("y") or [], dtype=float)
    z = np.asarray(payload.get("z") or [], dtype=float)
    label = str(payload.get("label") or "uploaded")
    if z.ndim == 2 and z.size and x.size * y.size == z.size:
        xx, yy = np.meshgrid(x.ravel(), y.ravel())
        x, y, z = xx.ravel(), yy.ravel(), z.ravel()
    else:
        x, y, z = x.ravel(), y.ravel(), z.ravel()
    if not (x.size == y.size == z.size):
        raise ValueError("fea_viz: x/y/z must have the same length")
    ok = np.isfinite(x) & np.isfinite(y) & np.isfinite(z)
    x, y, z = x[ok], y[ok], z[ok]
    if x.size < 4:
        raise ValueError("fea_viz: need at least 4 finite nodes")
    if not (np.ptp(x) > 0.0 and np.ptp(y) > 0.0):
        raise ValueError("fea_viz: x/y must span a 2D area")

    # central obscuration / aperture mask: the 2D heatmap already excludes the
    # hole; without masking here the triangulation bridges it in 3D.
    obsc = float(payload.get("obscuration") or 0.0)
    obsc = min(max(obsc, 0.0), 0.95)
    r_max = float(payload.get("aperture_radius_mm") or 0.0)
    r = np.sqrt(x * x + y * y)
    if r_max <= 0.0:
        r_max = float(r.max())
    masked = 0
    if obsc > 0.0 and r_max > 0.0:
        inside = r >= obsc * r_max
        masked = int((~inside).sum())
        x, y, z, r = x[inside], y[inside], z[inside], r[inside]
        if x.size < 4:
            raise ValueError("fea_viz: obscuration masks every node")
    # Delaunay rejects duplicate nodes; keep the first of each (x, y) pair.
    _, first = np.unique(np.column_stack([x, y]), axis=0, return_index=True)
    if first.size != x.size:
        x, y, z = x[first], y[first], z[first]
    if x.size < 4:
        raise ValueError("fea_viz: need at least 4 distinct nodes")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.tri as mtri
    from matplotlib import colors

    try:
        tri = mtri.Triangulation(x, y)
    except Exception as exc:  # degenerate layouts (collinear nodes, ...)
        raise ValueError("fea_viz: cannot triangulate surface (%s)" % exc)

    zmin, zmax = float(z.min()), float(z.max())
    if zmax <= zmin:  # perfectly flat surface: keep a readable colour scale
        zmin, zmax = zmin - 1.0, zmax + 1.0
    norm = colors.Normalize(vmin=zmin, vmax=zmax)
    fig = plt.figure(figsize=(4.0, 3.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_trisurf(x, y, z, triangles=tri.triangles, cmap="viridis",
                           norm=norm, linewidth=0.1, antialiased=True, alpha=0.95)
    ax.scatter(x, y, z, s=3, c="#101820", alpha=0.22, linewidths=0,
               depthshade=True)
    ax.set_xlabel("x [mm]", fontsize=6.5, labelpad=-4)
    ax.set_ylabel("y [mm]", fontsize=6.5, labelpad=-4)
    ax.set_zlabel("dz [nm]", fontsize=6.5, labelpad=-4)
    ax.tick_params(labelsize=5.5, pad=-2)
    ax.set_title(label if len(label) <= 42 else label[:41] + "\u2026",
                 fontsize=6.5)
    ax.view_init(elev=float(payload.get("elev", 25)),
                 azim=float(payload.get("azim", -60)))
    cb = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=14, pad=0.0)
    cb.set_label("dz [nm]", fontsize=6)
    cb.ax.tick_params(labelsize=5)
    fig.subplots_adjust(left=0.0, right=0.97, bottom=0.03, top=0.90)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)

    # regular grid for the GUI/export path; None marks outside-the-hull cells
    gx = np.linspace(float(x.min()), float(x.max()), 48)
    gy = np.linspace(float(y.min()), float(y.max()), 48)
    gxx, gyy = np.meshgrid(gx, gy)
    gz = griddata((x, y), z, (gxx, gyy), method="linear")
    gr = np.sqrt(gxx * gxx + gyy * gyy)
    gz[(gr < obsc * r_max) | (gr > r_max * 1.002)] = np.nan
    grid = {
        "x": [round(float(v), 6) for v in gx],
        "y": [round(float(v), 6) for v in gy],
        "z": [[None if not np.isfinite(v) else round(float(v), 6) for v in row]
              for row in gz],
    }
    rms = float(np.sqrt(np.mean((z - z.mean()) ** 2)))
    return {
        "label": label,
        "png3d_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "grid": grid,
        "stats": {"rms_nm": round(rms, 6),
                  "pv_nm": round(float(z.max() - z.min()), 6),
                  "n": int(x.size),
                  "masked_obscuration": masked,
                  "obscuration": obsc},
    }




def _fea_structure(payload):
    """Render the FEA *model* (annular mirror body + whiffletree supports).

    payload: {"outer_radius_mm": 1250, "inner_radius_mm": 350,
              "thickness_mm": 260, "support_rings": [[r, count], ...],
              "side_supports": 24, "label": "M1", "obscuration": 0.28}

    Draws the annular solid in 3D (top/bottom faces + inner/outer walls as
    wireframe), the axial support pads on the bottom face and the lateral pads
    at mid-thickness.  This is the documented geometry / idealized support
    layout, NOT the solved mesh; the deformed surface stays in `fea_viz`.
    """
    import base64
    import io
    import math

    import numpy as np

    r_out = float(payload.get("outer_radius_mm") or 1250.0)
    r_in = float(payload.get("inner_radius_mm") or 350.0)
    thick = float(payload.get("thickness_mm") or 260.0)
    rings = payload.get("support_rings") or [[375.0, 6], [625.0, 12],
                                             [875.0, 12], [1125.0, 6]]
    n_side = int(payload.get("side_supports") or 24)
    label = str(payload.get("label") or "M1")

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    th = np.linspace(0.0, 2.0 * math.pi, 121)
    fig = plt.figure(figsize=(4.0, 3.0), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    def ring(r, z):
        return np.column_stack([r * np.cos(th), r * np.sin(th),
                                np.full_like(th, z)])

    for r in (r_in, r_out):
        for z in (0.0, thick):
            pts = ring(r, z)
            ax.plot(pts[:, 0], pts[:, 1], pts[:, 2], color="#7fb3d5",
                    lw=0.6, alpha=0.9)
    # meridian walls (every 30 deg) to convey the solid
    for k in range(12):
        a = 2.0 * math.pi * k / 12.0
        ax.plot([r_in * math.cos(a), r_out * math.cos(a)],
                [r_in * math.sin(a), r_out * math.sin(a)],
                [0.0, 0.0], color="#7fb3d5", lw=0.4, alpha=0.5)
        ax.plot([r_in * math.cos(a), r_out * math.cos(a)],
                [r_in * math.sin(a), r_out * math.sin(a)],
                [thick, thick], color="#7fb3d5", lw=0.4, alpha=0.5)
        ax.plot([r_out * math.cos(a)] * 2, [r_out * math.sin(a)] * 2,
                [0.0, thick], color="#4a7fb5", lw=0.4, alpha=0.5)

    # axial whiffletree pads on the bottom face
    pads = []
    for ring_r, count in rings:
        ring_r, count = float(ring_r), int(count)
        for k in range(count):
            a = 2.0 * math.pi * k / count
            pads.append((ring_r * math.cos(a), ring_r * math.sin(a), 0.0))
    if pads:
        p = np.asarray(pads)
        ax.scatter(p[:, 0], p[:, 1], p[:, 2], s=14, c="#f5b041",
                   marker="o", depthshade=False, label="axial pads")
    # lateral pads at mid-thickness
    side = []
    for k in range(n_side):
        a = 2.0 * math.pi * k / n_side
        side.append((r_out * math.cos(a), r_out * math.sin(a), thick / 2.0))
    if side:
        sp = np.asarray(side)
        ax.scatter(sp[:, 0], sp[:, 1], sp[:, 2], s=8, c="#58d68d",
                   marker="^", depthshade=False, label="lateral pads")

    lim = r_out * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_zlim(0, thick * 1.4)
    ax.set_xlabel("x [mm]", fontsize=6.5, labelpad=-4)
    ax.set_ylabel("y [mm]", fontsize=6.5, labelpad=-4)
    ax.set_zlabel("z [mm]", fontsize=6.5, labelpad=-4)
    ax.tick_params(labelsize=5.5, pad=-2)
    ax.set_title("%s model: OD %.0f / ID %.0f / t %.0f mm" % (label, r_out * 2,
                                                              2 * r_in, thick),
                 fontsize=6.5)
    ax.view_init(elev=22, azim=-62)
    try:
        ax.legend(fontsize=5, loc="upper left", frameon=False)
    except Exception:
        pass
    fig.subplots_adjust(left=0.0, right=0.99, bottom=0.02, top=0.90)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    plt.close(fig)
    return {
        "label": label,
        "png3d_b64": base64.b64encode(buf.getvalue()).decode("ascii"),
        "geometry": {"outer_radius_mm": r_out, "inner_radius_mm": r_in,
                     "thickness_mm": thick,
                     "axial_pads": len(pads), "lateral_pads": len(side),
                     "support_rings": rings},
        "note": "documented geometry + idealized support layout (not the solved mesh)",
    }



def _zmx_layout(payload):
    """Render the parsed ZMX optical layout (2D cross-section + 3D)."""
    from products.zemax.layout import render_layout_2d, render_layout_3d

    parsed = payload.get("parsed")
    if not isinstance(parsed, dict) or not parsed.get("surfaces"):
        raise ValueError("zmx_layout: payload.parsed with surfaces is required")
    r3 = render_layout_3d(parsed, azim=float(payload.get("azim", -60)),
                          elev=float(payload.get("elev", 18)))
    r2 = render_layout_2d(parsed)
    return {
        "png3d_b64": r3["png_b64"],
        "png2d_b64": r2["png_b64"],
        "surfaces": r3.get("surfaces", []),
        "track_mm": r3.get("track_mm"),
        "stop": r3.get("stop"),
        "notes": r3.get("notes", []),
    }

def run_worker(task, job_path=None, result_path=None):
    if job_path:
        doc = json.loads(Path(job_path).read_text(encoding="utf-8-sig"))
    else:
        doc = json.loads(sys.stdin.read() or "{}")
    from desktop.protocol import parse_job
    payload = parse_job(doc, task)
    if task == "ccd":
        from products.ccd.service import run_ccd_pipeline
        kwargs = {k: payload[k] for k in (
            "exposure_s", "adu_gain_e_per_adu", "read_noise_e",
            "dark_current_e_per_s", "sky_background_e_per_pix_s",
            "plate_scale_arcsec_per_pix", "seeing_arcsec",
            "star_density_per_arcmin2", "mag_zero", "mag_limit",
            "saturation_adu", "bit_depth", "readout_bias_adu",
            "bad_pixel_fraction", "cosmic_ray_rate", "jitter_arcsec",
            "aperture_m", "obscuration", "fov_arcmin", "seed", "fast")
            if k in payload}
        r = run_ccd_pipeline(payload["c4"], payload["c8"], payload["c9"],
                             payload.get("noise", 5.0), payload.get("stars", 25),
                             payload.get("defocus"), **kwargs)
    elif task in ("fits", "fits_paths"):
        from products.ccd.fits_service import run_fits_pipeline
        if task == "fits_paths":
            import base64
            files = [{"name": Path(p).name,
                      "data_b64": base64.b64encode(Path(p).read_bytes()).decode()}
                     for p in payload["paths"]]
        else:
            files = payload["files"]
        r = run_fits_pipeline(files, payload.get("defocus"))
    elif task == "pd":
        from products.pd.service import invert_psf
        r = invert_psf(payload["psf"], payload["defocus"], None)
    elif task == "fea_case":
        from products.sensitivity import service as sens
        r = sens.case_detail(payload["cid"])
    elif task == "fea_upload":
        from products.sensitivity import service as sens
        from products.instrument import M1_OBSCURATION, M1_RADIUS_MM
        r = sens.custom_surface(payload["x"], payload["y"], payload["z"],
                                M1_RADIUS_MM, M1_OBSCURATION, payload["label"])
    elif task == "zmx_layout":
        r = _zmx_layout(payload)
    elif task == "fea_structure":
        r = _fea_structure(payload)
    elif task == "fea_viz":
        r = _fea_viz(payload)
    elif task == "s":
        from products.control import service as fw
        r = fw.sensitivity(source=payload.get("source", "auto"))
    elif task == "u":
        from products.control import service as fw
        r = fw.control(payload["target"],
                       source=payload.get("source", "auto"))
    elif task == "sens_measure":
        from products.sensitivity.measured import (poses_from_payload,
                                                   measure_sensitivity)
        kw = {k: payload[k] for k in ("maxeval", "n_starts", "seed")
              if k in payload}
        terms = payload.get("terms") or None
        r = measure_sensitivity(poses_from_payload(payload["poses"]),
                                terms=terms, invert_kwargs=kw or None)
    elif task == "sens_demo":
        from products.sensitivity.measured import demo_synthetic
        r = demo_synthetic(noise=float(payload.get("noise", 3e-4)),
                           seed=int(payload.get("seed", 11)),
                           n=int(payload.get("n", 32)),
                           amplitude=float(payload.get("amplitude", 0.05)),
                           maxeval=int(payload.get("maxeval", 150)),
                           n_starts=int(payload.get("n_starts", 2)))
    else:
        raise SystemExit("unknown task " + task)
    out = json.dumps(r, ensure_ascii=False, default=str)
    if result_path:
        Path(result_path).write_text(out, encoding="utf-8")
    else:
        print(out)


