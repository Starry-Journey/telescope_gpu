"""Minimal ZMX reader: surfaces, stop, aperture and wavelengths.

This is not a Zemax ray tracer.  It extracts the prescription so the web app can
show what was uploaded and the ZOS-API bridge (Windows/OpticStudio) can be
called with the same file.  Full optical sensitivity must come from ZOS-API or a
validated ray tracer.
"""
from __future__ import annotations

import math
import re
from pathlib import Path


def _number(line: str, index: int = 0):
    parts = line.split()
    try:
        v = float(parts[index + 1])
        return v if math.isfinite(v) else None
    except (IndexError, ValueError):
        return None


def parse_zmx_text(text: str) -> dict:
    # ZMX comments are often GBK; latin-1 preserves bytes without crashing.
    lines = text.splitlines()
    model = {"aperture_mm": None, "wavelengths_um": [], "surfaces": [], "stop": None}
    current = None
    for raw in lines:
        line = raw.rstrip()
        if line.startswith("ENPD"):
            v = _number(line)
            if v is not None:
                model["aperture_mm"] = v
        elif line.startswith("WAVM"):
            parts = line.split()
            if len(parts) >= 3:
                try:
                    model["wavelengths_um"].append(float(parts[2]))
                except ValueError:
                    pass
        elif line.startswith("SURF"):
            parts = line.split()
            current = {"index": int(parts[1]) if len(parts) > 1 else None}
            model["surfaces"].append(current)
        elif current is not None:
            key = line.strip().split()[0] if line.strip() else ""
            if key == "STOP":
                model["stop"] = current["index"]
            elif key == "TYPE":
                current["type"] = line.split(None, 1)[1].strip() if len(line.split(None, 1)) > 1 else ""
            elif key == "CURV":
                current["curv"] = _number(line)
            elif key == "CONI":
                current["conic"] = _number(line)
            elif key == "DISZ":
                current["thickness"] = _number(line)
            elif key == "GLAS":
                parts = line.split()
                if len(parts) >= 2:
                    current["glass"] = parts[1]
            elif key == "DIAM":
                current["diameter"] = _number(line)
    return model


def parse_zmx_bytes(raw: bytes) -> dict:
    """ZMX is often UTF-16LE with BOM; fall back to latin-1."""
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    else:
        text = raw.decode("latin-1")
    return parse_zmx_text(text)


def parse_zmx_file(path: str) -> dict:
    return parse_zmx_bytes(Path(path).read_bytes())
