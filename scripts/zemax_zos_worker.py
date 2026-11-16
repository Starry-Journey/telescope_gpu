"""ZOS-API worker (Windows/OpticStudio).

Run on a Windows machine with OpticStudio + ZOS-API:

    python scripts/zemax_zos_worker.py --api http://<linux-host>:8000

For a pipeline test without OpticStudio, use --mock (clearly marked mock).

The worker claims one `zemax` job, runs the 3A bridge, and posts the result
(S matrix CSV/JSON) back to the FastAPI job queue.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
import urllib.parse
import urllib.request


def _post(url: str, obj: dict):
    req = urllib.request.Request(url, data=json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=120) as r:
        return json.loads(r.read())


def _mock_result(job: dict) -> dict:
    # A labelled mock so the queue/UI can be tested on Linux; never claim as Zemax.
    S = [[4.04, 0.083, -0.083, 0.976, -0.976],
         [0.0, -4.633, 0.0, 0.0, 0.0],
         [0.0, 0.0, -4.633, 0.0, 0.0]]
    return {"mock": True,
            "targets": ["Z4", "Z8", "Z9"],
            "controls": ["m2_focus", "m2_decenter_x", "m2_decenter_y", "m2_tilt_x", "m2_tilt_y"],
            "matrix": S,
            "note": "MOCK — not Zemax; replace with ZOS-API output"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://127.0.0.1:8000")
    ap.add_argument("--worker", default="zos-worker")
    ap.add_argument("--mock", action="store_true")
    ap.add_argument("--once", action="store_true", default=True)
    args = ap.parse_args()

    if not args.mock and platform.system() != "Windows":
        raise SystemExit("ZOS-API requires Windows/OpticStudio; use --mock for queue testing")

    claimed = _post(args.api + "/api/jobs/claim", {"kind": "zemax", "worker": args.worker})
    job = claimed.get("job")
    if not job:
        print("no queued zemax job")
        return
    jid = job["id"]
    try:
        if args.mock:
            result = _mock_result(job)
        else:
            from products.zemax.zos_bridge import run_job
            result = run_job(job["payload"])
        _post(args.api + f"/api/jobs/{jid}/complete", {"result": result})
        print("done", jid)
    except Exception as e:
        _post(args.api + f"/api/jobs/{jid}/fail?error=" + urllib.parse.quote(str(e)), {})
        print("failed", jid, e)


if __name__ == "__main__":
    main()
