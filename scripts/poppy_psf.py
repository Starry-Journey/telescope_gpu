"""Generate physical (POPPY) defocus PSFs for the PD batch.

This is a diffraction calculation, not an observation.  It is useful for
replacing the purely analytic forward PSF with a POPPY pupil model:
  - 2.5 m aperture, 0.7 m secondary
  - 5 defocus values in waves (Noll Z4)
  - 500 nm, detector 0.01 arcsec/pixel, 2.56 arcsec field -> 256x256

Run with a Python that has poppy installed (Windows is fine):
  python scripts/poppy_psf.py --out data/psf/poppy_defocus_256.npy --n 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import poppy

WAVE = 500e-9
DEFOCUS = [-2.0, -1.0, 0.0, 1.0, 2.0]


def make_psf(defocus: float, n: int) -> np.ndarray:
    osys = poppy.OpticalSystem()
    osys.add_pupil(poppy.CircularAperture(radius=1.25))
    osys.add_pupil(poppy.SecondaryObscuration(secondary_radius=0.35))
    # coefficients are in meters; Noll j=4 is defocus.
    coeffs = [0.0, 0.0, 0.0, defocus * WAVE, 0.0, 0.0, 0.0, 0.0, 0.0]
    osys.add_pupil(poppy.ZernikeWFE(radius=1.25, coefficients=coeffs))
    fov = n * 0.01  # 256 -> 2.56 arcsec
    osys.add_detector(pixelscale=0.01, fov_arcsec=fov)
    hdul = osys.calc_psf(wavelength=WAVE)
    img = np.asarray(hdul[0].data, dtype=float)
    img = np.clip(img, 0, None)
    img /= max(float(img.sum()), 1e-30)
    # detector may produce n+1 pixels; crop/pad to n
    if img.shape[0] != n or img.shape[1] != n:
        out = np.zeros((n, n), dtype=float)
        h = min(n, img.shape[0]); w = min(n, img.shape[1])
        out[:h, :w] = img[:h, :w]
        img = out / max(out.sum(), 1e-30)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=256)
    ap.add_argument("--out", default="data/psf/poppy_defocus_256.npy")
    args = ap.parse_args()
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    stack = np.stack([make_psf(d, args.n) for d in DEFOCUS])
    np.save(args.out, stack)
    meta = {"source": "POPPY diffraction, not observation",
            "n": args.n, "defocus_waves": DEFOCUS,
            "wavelength_nm": WAVE * 1e9,
            "aperture_m": 2.5, "secondary_m": 0.7,
            "pixelscale_arcsec": 0.01, "synthetic": True,
            "truth_z4_z8_z9": [0.0, 0.0, 0.0]}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", args.out, stack.shape, "synthetic POPPY PSF")


if __name__ == "__main__":
    main()
