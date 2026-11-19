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
              "azim": -60, "elev": 25}

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
                  "n": int(x.size)},
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
        r = run_ccd_pipeline(payload["c4"], payload["c8"], payload["c9"],
                             payload["noise"], payload["stars"], payload["defocus"])
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


