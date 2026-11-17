"""Service worker: no PySide6/Qt imports.

Called as: EXE --worker <task> --job job.json --result result.json
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

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
    elif task == "s":
        from products.control import service as fw
        r = fw.sensitivity()
    elif task == "u":
        from products.control import service as fw
        r = fw.control(payload["target"])
    else:
        raise SystemExit("unknown task " + task)
    out = json.dumps(r, ensure_ascii=False, default=str)
    if result_path:
        Path(result_path).write_text(out, encoding="utf-8")
    else:
        print(out)


