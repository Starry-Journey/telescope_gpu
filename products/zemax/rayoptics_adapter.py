"""Optional rayoptics adapter for ZMX (open-source path).

Requires rayoptics (and its opticalglass/zmxread deps) in the Python runtime.
On this machine the Windows Python 3.13 environment has rayoptics 0.9.10.
The Linux WSL Python 3.8 env does not; the adapter fails explicitly there.

Important: `read_lens_file` requires a pathlib.Path, not a str.
"""
from __future__ import annotations

from pathlib import Path


def available() -> bool:
    try:
        import rayoptics  # noqa: F401
        from rayoptics.zemax import zmxread  # noqa: F401
        return True
    except Exception:
        return False


def inspect_zmx(path: str) -> dict:
    """Read a ZMX with rayoptics and return a JSON-safe prescription."""
    from rayoptics.zemax import zmxread

    model, info = zmxread.read_lens_file(Path(path))
    seq = model["seq_model"]
    surfaces = []
    for i, s in enumerate(seq.ifcs):
        prof = s.profile
        surfaces.append({
            "index": i,
            "label": getattr(s, "lbl", ""),
            "interact_mode": getattr(s, "interact_mode", ""),
            "curv": getattr(prof, "cv", None),
            "conic": getattr(prof, "cc", None),
        })
    return {
        "source": "rayoptics.zemax.zmxread",
        "n_surfaces": len(surfaces),
        "stop": seq.stop_surface,
        "wavelengths_nm": list(seq.wvlns),
        "surfaces": surfaces,
        "info": info[0] if isinstance(info, tuple) and info else str(info),
    }
