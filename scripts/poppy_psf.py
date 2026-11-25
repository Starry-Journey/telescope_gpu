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

from products.optics.forward import center_psf

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
    # P2-11: POPPY oversamples the detector by 2x, so `fov_arcsec=2.56`` with
    # 0.01 arcsec/px returns 512x512, not 256x256!  The old export cropped the
    # TOP-LEFT n x n block, which kept only the corner of the PSF (the PSF centre
    # sits at (255.5, 255.5) of the 512 grid) and silently destroyed the frame.
    # Center-crop instead: the PSF stays centred at ((n-1)/2, (n-1)/2).
    img = center_psf(img, n)
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
            "center_convention": "center-cropped from POPPY's 2x-oversampled "
                                 "detector grid; centroid at (n//2, n//2), "
                                 "sub-pixel residual <= 0.5 px (P2-11)",
            "truth_z4_z8_z9": [0.0, 0.0, 0.0]}
    Path(args.out).with_suffix(".json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print("wrote", args.out, stack.shape, "synthetic POPPY PSF")


if __name__ == "__main__":
    main()
