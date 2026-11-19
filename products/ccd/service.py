"""Module 1: realistic CCD simulation + calibration + PSF cutouts.

The synthetic frames come from :mod:`products.ccd.model`, a physically
motivated star-field + detector forward model (Fourier-optics PSF from
``Z4/Z8/Z9`` plus per-frame defocus, Gaussian seeing, Poisson/read noise,
bias/dark/bit-depth/saturation, bad pixels and cosmic rays).  Everything is
deterministic for a fixed ``seed``.

The public entry point keeps the historical positional signature
``run_ccd_pipeline(c4, c8, c9, noise, stars, defocus, seed)``; all new
physical parameters are keyword-only with documented defaults.
"""
from __future__ import annotations

from typing import List, Optional

import numpy as np

# `_png` is re-exported for the historical `web.ccd_pipeline_service` wrapper;
# the pipeline itself uses the faster Pillow encoder.
from products.ccd.io import png_data_url as _png  # noqa: F401
from products.ccd.io import png_data_url_fast as _png_fast
from products.ccd.model import (CcdModelConfig, REFERENCE_FOV_ARCMIN,
                                expected_star_count, simulate)
from products.instrument import DEFAULT_PSF_SIZE

#: Human-readable provenance label for the returned dataset.
_SYNTHETIC_LABEL = "SYNTHETIC-VALIDATED"

#: Default read noise.  ``read_noise_e`` is a new keyword; the historical
#: positional ``noise`` argument still wins when it is given a non-default
#: value, so old callers keep their requested noise level.
_DEFAULT_READ_NOISE_E = 5.0

_UNITS = {
    "frames.raw": "ADU (integer, readout bias included, saturating at the ADC ceiling)",
    "frames.cal": "ADU after bias + mean dark subtraction (unit flat field); sky remains",
    "psf": "normalised PSF cutout (dimensionless, unit sum, background plane removed)",
    "exposure_s": "s",
    "adu_gain_e_per_adu": "electron/ADU",
    "read_noise_e": "electron RMS",
    "dark_current_e_per_s": "electron/pixel/s",
    "sky_background_e_per_pix_s": "electron/pixel/s",
    "background_e_per_pix": "electron/pixel (mean sky + dark, before gain)",
    "total_electrons": "electron (expected star + sky + dark signal, all frames)",
    "plate_scale_arcsec_per_pix": "arcsec/pixel",
    "seeing_arcsec": "arcsec (FWHM); sigma_pix = seeing / (2.355 * plate_scale)",
    "fov_arcmin": "arcmin (nominal design field used to normalise star counts)",
    "star_density_per_arcmin2": "stars/arcmin^2",
    "mag_zero": "mag (bright limit of the truncated counts law)",
    "mag_limit": "mag (faint limit of the truncated counts law)",
    "saturation_count": "pixels across all frames at the ADC/full-well ceiling",
    "bad_pixel_count": "detector pixels (dead + hot), present in every frame",
    "cosmic_ray_count": "single-pixel cosmic-ray hits across all frames",
    "cosmic_ray_rate": "hits/pixel/frame",
    "bad_pixel_fraction": "fraction of detector pixels",
    "saturation_adu": "ADU",
    "readout_bias_adu": "ADU",
    "jitter_arcsec": "arcsec (per-frame field jitter)",
    "aperture_m": "m (entrance aperture, scales collected flux)",
    "obscuration": "fraction of the pupil radius blocked by the secondary",
}


def run_ccd_pipeline(c4: float = 0.3, c8: float = 0.1, c9: float = -0.05,
                     noise: float = 5.0, stars: int = 25,
                     defocus: Optional[List[float]] = None,
                     seed: int = 7,
                     *,
                     exposure_s: float = 1.0,
                     adu_gain_e_per_adu: float = 1.0,
                     read_noise_e: float = _DEFAULT_READ_NOISE_E,
                     dark_current_e_per_s: float = 0.01,
                     sky_background_e_per_pix_s: float = 1.0,
                     plate_scale_arcsec_per_pix: float = 0.05,
                     seeing_arcsec: float = 0.7,
                     star_density_per_arcmin2: float = 5.0,
                     mag_zero: float = 10.0,
                     mag_limit: float = 16.0,
                     saturation_adu: int = 65535,
                     bit_depth: int = 16,
                     readout_bias_adu: float = 500.0,
                     bad_pixel_fraction: float = 0.0005,
                     cosmic_ray_rate: float = 1e-4,
                     jitter_arcsec: float = 0.02,
                     aperture_m: float = 2.5,
                     obscuration: float = 0.28,
                     fov_arcmin: float = 2.56,
                     return_frames: bool = True,
                     fast: Optional[bool] = None) -> dict:
    """Simulate a raw CCD star field, calibrate it, extract PSF cutouts.

    The first seven positional arguments are byte-for-byte compatible with the
    historical signature.  New physical parameters are keyword-only.  The
    ``noise`` alias keeps working: when it is non-default it is used as the
    read noise unless an explicit non-default ``read_noise_e`` is also given.

    ``return_frames=False`` skips the (expensive) per-frame PNG rendering but
    still returns the PSF cutouts, metadata and manifest; the ``frames`` key is
    always present (an empty list in that case).

    ``fast`` forces the Gaussian PSF path; by default it is selected
    automatically when the per-star optical PSF rendering would be too slow.
    The chosen path is recorded in ``meta["model"]``.
    """
    if defocus is None or len(defocus) < 2:
        defocus = [0.0, 1.0, -1.0, 2.0, -2.0]
    defocus = [float(d) for d in defocus]

    # Historical positional `noise` argument vs. the new `read_noise_e`.
    resolved_read_noise = (float(read_noise_e)
                           if float(read_noise_e) != _DEFAULT_READ_NOISE_E
                           else float(noise))

    n_stars = expected_star_count(int(stars), float(star_density_per_arcmin2),
                                  float(fov_arcmin))
    cfg = CcdModelConfig(
        coeffs=(float(c4), float(c8), float(c9)),
        defocus=tuple(defocus),
        n_stars=n_stars,
        exposure_s=float(exposure_s),
        adu_gain_e_per_adu=float(adu_gain_e_per_adu),
        read_noise_e=resolved_read_noise,
        dark_current_e_per_s=float(dark_current_e_per_s),
        sky_background_e_per_pix_s=float(sky_background_e_per_pix_s),
        plate_scale_arcsec_per_pix=float(plate_scale_arcsec_per_pix),
        seeing_arcsec=float(seeing_arcsec),
        mag_zero=float(mag_zero),
        mag_limit=float(mag_limit),
        saturation_adu=int(saturation_adu),
        bit_depth=int(bit_depth),
        readout_bias_adu=float(readout_bias_adu),
        bad_pixel_fraction=float(bad_pixel_fraction),
        cosmic_ray_rate=float(cosmic_ray_rate),
        jitter_arcsec=float(jitter_arcsec),
        aperture_m=float(aperture_m),
        obscuration=float(obscuration),
        psf_size=int(DEFAULT_PSF_SIZE),
        seed=int(seed),
        fast=fast,
    )
    sim = simulate(cfg)

    frames_out: List[dict] = []
    psf_png: List[str] = []
    if return_frames:
        for i, d in enumerate(defocus):
            raw = sim["raw"][i]
            cal = sim["cal"][i]
            frames_out.append({
                "defocus": float(d),
                "raw": _png_fast(raw, f"raw d={d:+.1f}"),
                "cal": _png_fast(cal, f"cal d={d:+.1f}"),
            })
    for i, d in enumerate(defocus):
        cut = sim["psf"][i]
        psf_png.append(_png_fast(cut, f"PSF d={d:+.1f}"))

    psf_lists = [np.asarray(cut, dtype=float).tolist() for cut in sim["psf"]]
    cal0 = np.asarray(sim["cal"][0], dtype=float)
    cal_med = float(np.median(cal0))
    # Background RMS is measured before the non-negative clip: clipping the
    # calibrated frame at zero would turn the lower half of the read-noise
    # distribution into a point mass at 0 and hide the read-noise level.
    dark_adu = (float(dark_current_e_per_s) * float(exposure_s)
                / float(adu_gain_e_per_adu))
    bg_residual = (np.asarray(sim["raw"][0], dtype=float)
                   - float(readout_bias_adu) - dark_adu)
    bg_med = float(np.median(bg_residual))
    cal_rms = float(1.4826 * np.median(np.abs(bg_residual - bg_med)))
    counts = sim["counts"]
    electrons = sim["electrons"]

    metrics = {
        # Historical keys (GUI + older tests read these).
        "n_frames": len(defocus),
        "shape": [int(cfg.height), int(cfg.width)],
        "read_noise": float(resolved_read_noise),
        "background_median": cal_med,
        "psf_peak": float(sim["psf_peak"]),
        # Extended physical metrics.
        "n_stars": int(counts["n_stars_rendered"]),
        "total_electrons": float(electrons["total_expected"]),
        "total_electrons_measured": float(electrons["total_measured"]),
        "background_e_per_pix": float(electrons["background_e_per_pix"]),
        "background_rms_adu": cal_rms,
        "saturation_count": int(counts["saturation_count"]),
        "bad_pixel_count": int(counts["bad_pixel_count"]),
        "cosmic_ray_count": int(counts["cosmic_ray_count"]),
        "saturation_adu": int(sim["adu"]["max_adu"]),
        "bit_depth": int(cfg.bit_depth),
        "exposure_s": float(cfg.exposure_s),
        "adu_gain_e_per_adu": float(cfg.adu_gain_e_per_adu),
        "plate_scale_arcsec_per_pix": float(cfg.plate_scale_arcsec_per_pix),
        "seeing_arcsec": float(cfg.seeing_arcsec),
        "seeing_sigma_pix": float(sim["seeing_sigma_pix"]),
    }

    params = {
        "c4": float(c4), "c8": float(c8), "c9": float(c9),
        "defocus": [float(d) for d in defocus],
        "stars": int(stars),
        "star_density_per_arcmin2": float(star_density_per_arcmin2),
        "fov_arcmin": float(fov_arcmin),
        "exposure_s": float(cfg.exposure_s),
        "adu_gain_e_per_adu": float(cfg.adu_gain_e_per_adu),
        "read_noise_e": float(resolved_read_noise),
        "noise_legacy": float(noise),
        "dark_current_e_per_s": float(cfg.dark_current_e_per_s),
        "sky_background_e_per_pix_s": float(cfg.sky_background_e_per_pix_s),
        "plate_scale_arcsec_per_pix": float(cfg.plate_scale_arcsec_per_pix),
        "seeing_arcsec": float(cfg.seeing_arcsec),
        "mag_zero": float(cfg.mag_zero),
        "mag_limit": float(cfg.mag_limit),
        "saturation_adu": int(cfg.saturation_adu),
        "bit_depth": int(cfg.bit_depth),
        "readout_bias_adu": float(cfg.readout_bias_adu),
        "bad_pixel_fraction": float(cfg.bad_pixel_fraction),
        "cosmic_ray_rate": float(cfg.cosmic_ray_rate),
        "jitter_arcsec": float(cfg.jitter_arcsec),
        "aperture_m": float(cfg.aperture_m),
        "obscuration": float(cfg.obscuration),
        "psf_size": int(DEFAULT_PSF_SIZE),
        "seed": int(cfg.seed),
        "fast": sim["fast_mode"],
    }

    meta = {
        "model": sim["model"],
        "label": _SYNTHETIC_LABEL,
        "mode": "synthetic",
        "source": "products.ccd forward model (Fourier optics + detector chain)",
        "params": params,
        "n_stars_rendered": int(counts["n_stars_rendered"]),
        "fov_arcmin": float(fov_arcmin),
        "plate_scale_arcsec_per_pix": float(cfg.plate_scale_arcsec_per_pix),
        "seeing_arcsec": float(cfg.seeing_arcsec),
        "seeing_sigma_pix": float(sim["seeing_sigma_pix"]),
        "exposure_s": float(cfg.exposure_s),
        "saturation_count": int(counts["saturation_count"]),
        "bad_pixel_count": int(counts["bad_pixel_count"]),
        "cosmic_ray_count": int(counts["cosmic_ray_count"]),
        "background_e_per_pix": float(electrons["background_e_per_pix"]),
        "read_noise_e": float(resolved_read_noise),
        "seed": int(cfg.seed),
        "frame_shape": [int(cfg.height), int(cfg.width)],
        "psf_size": int(DEFAULT_PSF_SIZE),
        "psf_kernel_sizes": [int(k) for k in sim["psf_kernel_sizes"]],
        "fast_mode": bool(sim["fast_mode"]),
        "n_frames": len(defocus),
        "reference_stars": int(stars),
        "nominal_fov_arcmin": float(REFERENCE_FOV_ARCMIN),
        "detector_fov_arcmin": float(
            cfg.width * cfg.plate_scale_arcsec_per_pix / 60.0),
        "star_count_rule": ("n = round(stars * (density/5.0) * "
                            "(fov_arcmin/2.56)**2); one anchor star is pinned "
                            "at mag_zero"),
        "units": dict(_UNITS),
    }

    manifest = {
        "schema_version": 2,
        "generated_by": "products/ccd/service.py:run_ccd_pipeline",
        "mode": "synthetic",
        "label": _SYNTHETIC_LABEL,
        "source": meta["source"],
        "model": sim["model"],
        "params": params,
        "counts": dict(counts),
        "adu": sim["adu"],
        "truth": {"Z4": float(c4), "Z8": float(c8), "Z9": float(c9)},
        "defocus": [float(d) for d in defocus],
        "frames": [dict(m) for m in sim["frames"]],
        "catalog": [dict(s) for s in sim["catalog"]],
        "psf_star": dict(sim["star"]),
        "units": dict(_UNITS),
    }

    return {
        # Historical keys (GUI / PD / tests depend on all of them).
        "frames": frames_out,
        "psf": psf_lists,
        "psf_png": psf_png,
        "defocus": [float(d) for d in defocus],
        "truth": [float(c4), float(c8), float(c9)],
        "star": dict(sim["star"]),
        "metrics": metrics,
        "note": ("phase-optics + seeing + detector-chain simulation "
                 "(SYNTHETIC-VALIDATED); not real CCD data"),
        "manifest": manifest,
        # Machine-readable provenance.
        "mode": "synthetic",
        "source": meta["source"],
        "label": _SYNTHETIC_LABEL,
        "meta": meta,
    }
