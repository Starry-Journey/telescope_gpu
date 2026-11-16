"""ZOS-API bridge placeholder for Windows/OpticStudio.

This module deliberately does not fake Zemax values.  On Windows with
OpticStudio installed, implement `run_job` using the ZOS-API:

  1. write payload["data_b64"] to a temp .ZMX;
  2. open it with ZOSAPI_NetHelper / ZOSAPI;
  3. for each control in perturb.yaml:
       set M2 surface decenter/tilt/focus += delta;
       run Zernike/OPD analysis;
       restore nominal;
  4. S[:,j] = (z_plus - z_minus) / (2*delta);
  5. return {targets, controls, matrix, units, ...}.

The Web/job queue contract is stable even though this bridge is environment-
specific.  On Linux, use the Windows worker or upload a Zemax-exported CSV.
"""
from __future__ import annotations

import platform


def available() -> bool:
    return platform.system() == "Windows"


def run_job(payload: dict) -> dict:
    if not available():
        raise RuntimeError(
            "ZOS-API is Windows/OpticStudio-only. Run scripts/zemax_zos_worker.py "
            "on a Windows worker with OpticStudio, or upload the exported S CSV."
        )
    # Intentionally not implemented without OpticStudio; fill with ZOS-API calls.
    raise NotImplementedError(
        "ZOS-API bridge not configured. See docs/Zemax3A.md; do not substitute mock values."
    )
