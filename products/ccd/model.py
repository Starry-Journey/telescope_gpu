"""Physical star-field + CCD detector forward model.

This module replaces the old toy star field used by :mod:`products.ccd.service`
with a small but physically motivated simulation:

star field
    Star positions are uniform (no clustering) over the detector.  Star
    magnitudes are drawn from a power-law differential counts law
    ``N(m) ~ 10**(0.35 * m)`` truncated to ``[mag_zero, mag_limit]`` and the
    flux follows ``F ~ 10**(-0.4 * m)`` around a fixed m=10 zero point.  Every
    star is rendered from the Fourier-optics PSF returned by
    :func:`products.optics.forward.forward_psf_numpy` for the requested
    ``Z4/Z8/Z9`` state and per-frame defocus.

optics / seeing
    The optical PSF is convolved with a Gaussian seeing kernel.  Seeing is
    conventionally quoted as a FWHM, so the converter is
    ``sigma_pix = seeing_arcsec / (2.355 * plate_scale_arcsec_per_pix)``; for
    Gaussians convolving is the same as adding the widths in quadrature.  The
    combined kernel is placed with sub-pixel bilinear splatting (pure NumPy,
    four shifted slice adds per star).

detector chain
    ``electrons = stars * exposure + sky * exposure + dark * exposure`` is
    Poisson sampled, read noise is added as a Gaussian in electrons, the
    result is divided by the gain, offset by the readout bias, clipped to the
    ADC/full-well ceiling, quantised to integer ADU (bit depth), then dead/hot
    pixels and single-pixel cosmic-ray hits are injected.

Everything is deterministic for a fixed ``seed``; NumPy's PCG64 generator is
used throughout.  A ``fast`` Gaussian path is selected automatically when the
full per-star PSF rendering would be too expensive (very large star counts).
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from products.ccd.io import clean_cutout as _clean_cutout, cutout as _cutout
from products.instrument import DEFAULT_PSF_SIZE
from products.optics.forward import forward_psf_numpy

#: FWHM -> sigma for a Gaussian profile.
FWHM_PER_SIGMA = 2.0 * math.sqrt(2.0 * math.log(2.0))

#: Flux zero point: a 10th-magnitude star through a 2.5 m aperture deposits
#: 1e5 electrons per second at the detector (broad-band, post-optics; a
#: deliberately simple but documented photometric calibration).
MAG_ZERO_POINT_E_PER_S = 1.0e5
MAG_ZERO_POINT_REFERENCE_MAG = 10.0
MAG_ZERO_POINT_REFERENCE_APERTURE_M = 2.5

# Reference configuration for the legacy ``stars`` knob.  The GUI slider still
# means "stars at the reference density/FOV"; density and FOV scale it.
REFERENCE_STARS = 25
REFERENCE_DENSITY_PER_ARCMIN2 = 5.0
REFERENCE_FOV_ARCMIN = 2.56

#: Auto-select the Gaussian path above this many kernel taps per run.
FAST_PATH_TAP_BUDGET = 3.0e8
#: Absolute star count above which the per-star PSF path is not worth it.
FAST_PATH_MAX_STARS = 800
#: Safety cap so a silly density request cannot exhaust memory.
MAX_STARS = 5000


@dataclass(frozen=True)
class CcdModelConfig:
    """All inputs of one synthetic CCD exposure sequence (SI-ish units)."""

    width: int = 192
    height: int = 192
    coeffs: Tuple[float, float, float] = (0.3, 0.1, -0.05)
    defocus: Tuple[float, ...] = (0.0, 1.0, -1.0, 2.0, -2.0)
    n_stars: int = REFERENCE_STARS

    exposure_s: float = 1.0
    adu_gain_e_per_adu: float = 1.0
    read_noise_e: float = 5.0
    dark_current_e_per_s: float = 0.01
    sky_background_e_per_pix_s: float = 1.0
    plate_scale_arcsec_per_pix: float = 0.05
    seeing_arcsec: float = 0.7
    mag_zero: float = 10.0
    mag_limit: float = 16.0
    saturation_adu: int = 65535
    bit_depth: int = 16
    readout_bias_adu: float = 500.0
    bad_pixel_fraction: float = 0.0005
    cosmic_ray_rate: float = 1.0e-4
    jitter_arcsec: float = 0.02
    aperture_m: float = 2.5
    obscuration: float = 0.28
    psf_size: int = DEFAULT_PSF_SIZE
    seed: int = 7
    fast: Optional[bool] = None


# ---------------------------------------------------------------------------
# star catalogue
# ---------------------------------------------------------------------------

def expected_star_count(stars: int, density_per_arcmin2: float,
                        fov_arcmin: float) -> int:
    """Number of stars for the requested density and field.

    The legacy ``stars`` argument is the count at the reference density
    (5 / arcmin^2) and reference FOV (2.56 arcmin); the physical density and
    field scale it.  This keeps the existing GUI slider meaningful while
    making ``star_density_per_arcmin2`` and ``fov_arcmin`` physical knobs.
    """
    density_scale = (max(0.0, float(density_per_arcmin2))
                     / REFERENCE_DENSITY_PER_ARCMIN2)
    fov_scale = (max(0.0, float(fov_arcmin)) / REFERENCE_FOV_ARCMIN) ** 2
    n = int(round(max(1, int(stars)) * density_scale * fov_scale))
    return int(max(1, min(n, MAX_STARS)))


def power_law_magnitudes(rng: np.random.Generator, n: int,
                         mag_zero: float, mag_limit: float) -> np.ndarray:
    """Sample ``N(m) ~ 10**(0.35 m)`` truncated to ``[mag_zero, mag_limit]``."""
    m1, m2 = float(min(mag_zero, mag_limit)), float(max(mag_zero, mag_limit))
    a = 0.35 * math.log(10.0)
    lo, hi = math.exp(a * m1), math.exp(a * m2)
    if hi <= lo:
        return np.full(int(n), m1, dtype=float)
    u = rng.uniform(0.0, 1.0, int(n))
    return np.log(lo + u * (hi - lo)) / a


def magnitude_to_flux_e_per_s(mag: float, aperture_m: float = 2.5,
                              zero_point: float = MAG_ZERO_POINT_E_PER_S,
                              reference_mag: float = MAG_ZERO_POINT_REFERENCE_MAG
                              ) -> float:
    """Broad-band flux in electrons/s for a star of the given magnitude."""
    aperture_scale = (float(aperture_m) / MAG_ZERO_POINT_REFERENCE_APERTURE_M) ** 2
    return float(zero_point * aperture_scale
                 * 10.0 ** (-0.4 * (float(mag) - float(reference_mag))))


def generate_catalog(n_stars: int, width: int, height: int, margin: int,
                     rng: np.random.Generator, exposure_s: float,
                     aperture_m: float, mag_zero: float, mag_limit: float
                     ) -> List[dict]:
    """Uniform star positions + power-law magnitudes + fluxes.

    ``N-1`` stars follow the truncated counts law; one star is pinned at the
    bright truncation limit ``mag_zero``.  That bright-end anchor is the
    variance-reduction equivalent of the fixed calibration star used by the
    previous simulator: with a steep power law a small field would otherwise
    almost never contain a star near the bright limit (and never a saturated
    one), while the count normalisation is unchanged.
    """
    n_stars = max(1, int(n_stars))
    x = rng.uniform(float(margin), float(max(margin + 1, width - margin)),
                    n_stars)
    y = rng.uniform(float(margin), float(max(margin + 1, height - margin)),
                    n_stars)
    m_bright = float(min(mag_zero, mag_limit))
    m_faint = float(max(mag_zero, mag_limit))
    mags = np.empty(n_stars, dtype=float)
    mags[0] = m_bright
    if n_stars > 1:
        mags[1:] = power_law_magnitudes(rng, n_stars - 1, m_bright, m_faint)
    catalog: List[dict] = []
    for i in range(n_stars):
        mag = float(mags[i])
        flux_rate = magnitude_to_flux_e_per_s(mag, aperture_m)
        catalog.append({
            "id": int(i + 1),
            "x": float(x[i]),
            "y": float(y[i]),
            "mag": mag,
            "flux_e_per_s": float(flux_rate),
            "flux_e": float(flux_rate * float(exposure_s)),
        })
    return catalog


# ---------------------------------------------------------------------------
# PSF / seeing kernels
# ---------------------------------------------------------------------------

def seeing_sigma_pix(seeing_arcsec: float, plate_scale_arcsec_per_pix: float) -> float:
    """Seeing FWHM [arcsec] -> Gaussian sigma [detector pixels]."""
    plate = float(plate_scale_arcsec_per_pix)
    if plate <= 0:
        raise ValueError("plate_scale_arcsec_per_pix must be > 0")
    return max(0.0, float(seeing_arcsec)) / FWHM_PER_SIGMA / plate


def optical_psf(coeffs: Sequence[float], defocus: float,
                psf_size: int = DEFAULT_PSF_SIZE,
                obscuration: float = 0.0) -> np.ndarray:
    """Fourier-optics PSF for the historical 3-term Z4/Z8/Z9 state."""
    if len(coeffs) != 3:
        raise ValueError("coeffs must be the 3-term (Z4, Z8, Z9) state")
    return forward_psf_numpy([float(v) for v in coeffs], float(defocus),
                             n=int(psf_size), obscuration=float(obscuration))


def psf_sigma(psf: np.ndarray) -> float:
    """Radial Gaussian-equivalent sigma of a normalised PSF, in pixels."""
    a = np.asarray(psf, dtype=float)
    total = float(a.sum())
    if total <= 0:
        return 0.0
    w = a / total
    n0, n1 = w.shape
    yy, xx = np.indices(w.shape, dtype=float)
    cy, cx = (n0 - 1) / 2.0, (n1 - 1) / 2.0
    var = float(np.sum(w * ((xx - cx) ** 2 + (yy - cy) ** 2)) / 2.0)
    return math.sqrt(max(var, 0.0))


def _embed_optical_psf(psf: np.ndarray, size: int) -> np.ndarray:
    """Place an even-sized PSF in the centre of an odd ``size`` canvas."""
    n = int(psf.shape[0])
    canvas = np.zeros((size, size), dtype=float)
    start = (size - n) // 2
    canvas[start:start + n, start:start + n] = psf
    return canvas


def combine_with_seeing(psf: np.ndarray, sigma_pix: float) -> np.ndarray:
    """Convolve an optical PSF with a Gaussian seeing kernel (NumPy FFT).

    The canvas is padded by 3 sigma on each side, so the pair (optical PSF,
    seeing Gaussian) is added *in quadrature* in the usual Gaussian sense: the
    resulting profile is the convolution, whose second moment is the sum of the
    input second moments.
    """
    optical = np.asarray(psf, dtype=float)
    optical = np.clip(optical, 0.0, None)
    total = float(optical.sum())
    if total <= 0:
        raise ValueError("optical PSF has no flux")
    optical = optical / total
    sigma = float(max(0.0, sigma_pix))
    pad = int(math.ceil(3.0 * sigma)) if sigma > 0 else 0
    size = int(optical.shape[0]) + 2 * pad + 1
    kernel = _embed_optical_psf(optical, size)
    if sigma > 0:
        fy = np.fft.fftfreq(size)[:, None]
        fx = np.fft.fftfreq(size)[None, :]
        otf = np.exp(-2.0 * math.pi ** 2 * sigma ** 2 * (fx * fx + fy * fy))
        kernel = np.fft.ifft2(np.fft.fft2(kernel) * otf).real
        kernel = np.clip(kernel, 0.0, None)
    kernel /= max(float(kernel.sum()), 1e-30)
    return kernel


def gaussian_kernel(sigma_pix: float) -> np.ndarray:
    """Normalised 2-D Gaussian kernel (odd size, 3-sigma support)."""
    sigma = float(max(0.0, sigma_pix))
    if sigma <= 1e-6:
        return np.ones((1, 1), dtype=float)
    half = max(1, int(math.ceil(3.0 * sigma)))
    yy, xx = np.mgrid[-half:half + 1, -half:half + 1].astype(float)
    k = np.exp(-(xx * xx + yy * yy) / (2.0 * sigma * sigma))
    return k / float(k.sum())


def splat_star(image: np.ndarray, x: float, y: float, flux: float,
               kernel: np.ndarray) -> None:
    """Add ``flux * kernel`` at sub-pixel position (x, y), bilinear splatting.

    Instead of quantising the star to the nearest pixel the *whole* kernel is
    distributed over the four neighbouring integer placements with bilinear
    weights.  That preserves the fractional centroid while keeping the hot
    loop vectorised (four shifted slice adds).
    """
    kh, kw = kernel.shape
    # Odd kernels have their centre at kh // 2; even kernels only arise for
    # the caller-supplied case, and the same convention keeps them usable.
    cy, cx = kh // 2, kw // 2
    base_x = float(x) - cx
    base_y = float(y) - cy
    ix = int(math.floor(base_x))
    iy = int(math.floor(base_y))
    fx = base_x - ix
    fy = base_y - iy
    f = float(flux)
    weights = ((1.0 - fx) * (1.0 - fy), fx * (1.0 - fy),
               (1.0 - fx) * fy, fx * fy)
    offsets = ((0, 0), (1, 0), (0, 1), (1, 1))
    h, w = image.shape
    for weight, (ox, oy) in zip(weights, offsets):
        if weight == 0.0:
            continue
        x0 = ix + ox
        y0 = iy + oy
        x_lo = max(0, -x0)
        x_hi = min(kw, w - x0)
        y_lo = max(0, -y0)
        y_hi = min(kh, h - y0)
        if x_lo >= x_hi or y_lo >= y_hi:
            continue
        image[y0 + y_lo:y0 + y_hi, x0 + x_lo:x0 + x_hi] += (
            f * weight * kernel[y_lo:y_hi, x_lo:x_hi])


# ---------------------------------------------------------------------------
# detector artefacts
# ---------------------------------------------------------------------------

def _defect_mask(width: int, height: int, fraction: float,
                 rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return (bad, dead, hot) pixel masks for the detector."""
    n_pix = int(width) * int(height)
    n_bad = int(round(max(0.0, min(1.0, float(fraction))) * n_pix))
    bad = np.zeros(n_pix, dtype=bool)
    dead = np.zeros(n_pix, dtype=bool)
    hot = np.zeros(n_pix, dtype=bool)
    if n_bad > 0:
        idx = rng.choice(n_pix, size=min(n_bad, n_pix), replace=False)
        bad[idx] = True
        # Roughly half dead (dark) and half hot (saturating) pixels.
        is_dead = rng.random(idx.size) < 0.5
        dead[idx[is_dead]] = True
        hot[idx[~is_dead]] = True
    return (bad.reshape(height, width), dead.reshape(height, width),
            hot.reshape(height, width))


def _repair_defects(image: np.ndarray, mask: np.ndarray,
                    iterations: int = 2) -> np.ndarray:
    """Replace flagged pixels by the mean of their good 8-neighbours."""
    if not bool(np.any(mask)):
        return image
    out = np.array(image, dtype=float, copy=True)
    remaining = np.array(mask, dtype=bool, copy=True)
    for _ in range(int(max(1, iterations))):
        acc = np.zeros_like(out)
        cnt = np.zeros_like(out)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                if dy == 0 and dx == 0:
                    continue
                shifted = np.roll(out, (dy, dx), axis=(0, 1))
                good = ~np.roll(remaining, (dy, dx), axis=(0, 1))
                ys = remaining & good
                acc[ys] += shifted[ys]
                cnt[ys] += 1.0
        update = remaining & (cnt > 0)
        out[update] = acc[update] / cnt[update]
        remaining = remaining & ~update
        if not bool(np.any(remaining)):
            break
    return out


def _cosmic_ray_map(width: int, height: int, rate: float,
                    rng: np.random.Generator) -> np.ndarray:
    """Single-pixel cosmic-ray hit mask for one frame."""
    rate = max(0.0, float(rate))
    mask = np.zeros((int(height), int(width)), dtype=bool)
    if rate > 0:
        n_hits = int(rng.poisson(rate * int(width) * int(height)))
        if n_hits > 0:
            flat = rng.integers(0, int(width) * int(height), size=n_hits)
            mask.ravel()[flat] = True
    return mask


# ---------------------------------------------------------------------------
# PSF-star selection
# ---------------------------------------------------------------------------

def choose_psf_star(catalog: List[dict], full_well_e: float,
                    peak_fraction: float = 0.0) -> int:
    """Pick a usable PSF star: bright enough, not clipped, well isolated.

    ``peak_fraction`` is the largest kernel value; a star is considered
    unsaturated when ``flux * peak_fraction`` stays below half the full well
    (i.e. the per-pixel peak, not the total charge, is what saturates).
    """
    if not catalog:
        raise ValueError("empty star catalogue")
    fluxes = np.asarray([s["flux_e"] for s in catalog], dtype=float)
    xs = np.asarray([s["x"] for s in catalog], dtype=float)
    ys = np.asarray([s["y"] for s in catalog], dtype=float)

    peak = float(max(0.0, peak_fraction))
    if peak > 0:
        usable = np.nonzero((fluxes * peak <= 0.5 * full_well_e)
                            & (fluxes > 0.0))[0]
    else:
        usable = np.nonzero(fluxes <= 0.5 * full_well_e)[0]
    if usable.size == 0:
        usable = np.arange(len(catalog))

    best_idx = int(usable[0])
    best_score = -1.0
    for idx in usable:
        # Isolation from stars that could noticeably contaminate the cutout.
        neighbours = (fluxes > 0.05 * fluxes[idx])
        neighbours[idx] = False
        if not bool(np.any(neighbours)):
            isolation = float("inf")
        else:
            dx = xs[neighbours] - xs[idx]
            dy = ys[neighbours] - ys[idx]
            isolation = float(np.min(np.hypot(dx, dy)))
        # Prefer isolated stars with a healthy but unsaturated signal.
        signal = float(fluxes[idx]) / max(full_well_e, 1e-9)
        score = isolation + 8.0 * min(signal, 0.2)
        if score > best_score:
            best_score = score
            best_idx = int(idx)
    return best_idx


# ---------------------------------------------------------------------------
# main simulation
# ---------------------------------------------------------------------------

def simulate(config: CcdModelConfig) -> dict:
    """Run the full star-field + detector simulation.

    Returns NumPy frames and PSF cutouts plus all bookkeeping needed by the
    service layer (catalogue, per-frame counters, model description).
    """
    width, height = int(config.width), int(config.height)
    psf_size = int(config.psf_size)
    if width <= psf_size or height <= psf_size:
        raise ValueError("detector must be larger than the PSF cutout")
    gain = float(config.adu_gain_e_per_adu)
    if gain <= 0:
        raise ValueError("adu_gain_e_per_adu must be > 0")
    sigma_see = seeing_sigma_pix(config.seeing_arcsec,
                                 config.plate_scale_arcsec_per_pix)
    jitter_pix = float(max(0.0, config.jitter_arcsec)) / float(
        config.plate_scale_arcsec_per_pix)

    defocus = tuple(float(d) for d in config.defocus)
    if len(defocus) < 1:
        raise ValueError("at least one defocus value is required")

    max_adu = int(min(int(config.saturation_adu), 2 ** int(config.bit_depth) - 1))
    if max_adu < 1:
        raise ValueError("saturation_adu / bit_depth leave no ADC range")
    bias_adu = float(config.readout_bias_adu)
    full_well_e = float(max_adu) * gain

    # Independent streams for the structural random draws (catalogue, defects,
    # jitter, detector noise, cosmic rays).  Keeping them separate means the
    # geometry stays fixed when a purely photometric parameter (e.g. exposure)
    # changes: NumPy's Poisson sampler consumes a parameter-dependent number of
    # raw variates, which would otherwise shift every later draw.
    seq = np.random.SeedSequence(int(config.seed) & 0xFFFFFFFFFFFFFFFF)
    cat_seq, defect_seq, jitter_seq, noise_seq, cosmic_seq = seq.spawn(5)
    rng_catalog = np.random.default_rng(cat_seq)
    rng_defect = np.random.default_rng(defect_seq)
    rng_jitter = np.random.default_rng(jitter_seq)
    rng_noise = np.random.default_rng(noise_seq)
    rng_cosmic = np.random.default_rng(cosmic_seq)

    # Star catalogue -------------------------------------------------------
    margin = 32 + int(math.ceil(3.0 * jitter_pix)) + 2
    margin = int(max(psf_size // 2 + 2, min(margin, (min(width, height) - 2) // 2)))
    catalog = generate_catalog(int(config.n_stars), width, height, margin,
                               rng_catalog, float(config.exposure_s),
                               float(config.aperture_m), float(config.mag_zero),
                               float(config.mag_limit))
    n_stars = len(catalog)

    # Model path: full per-star optical PSF, or fast Gaussian ---------------
    est_kernel = psf_size + 2 * int(math.ceil(3.0 * sigma_see)) + 1
    taps = float(n_stars) * float(len(defocus)) * est_kernel ** 2 * 4.0
    if config.fast is None:
        fast_mode = bool(n_stars > FAST_PATH_MAX_STARS or taps > FAST_PATH_TAP_BUDGET)
    else:
        fast_mode = bool(config.fast)

    optical_cache: Dict[float, np.ndarray] = {}

    def _optical(d: float) -> np.ndarray:
        if d not in optical_cache:
            optical_cache[d] = optical_psf(config.coeffs, d, psf_size,
                                           config.obscuration)
        return optical_cache[d]

    if fast_mode:
        kernels = []
        for d in defocus:
            sigma_eff = math.sqrt(psf_sigma(_optical(d)) ** 2 + sigma_see ** 2)
            kernels.append(gaussian_kernel(sigma_eff))
        model_name = (
            "fast-gaussian: per-star Gaussian approximation "
            "(optical/defocus second moment + seeing added in quadrature); "
            "selected because the full per-star PSF rendering was too expensive")
    else:
        kernels = [combine_with_seeing(_optical(d), sigma_see) for d in defocus]
        model_name = (
            "full-psf: per-star Fourier-optics PSF (Z4/Z8/Z9, per-frame "
            "defocus) convolved with a Gaussian seeing kernel; sub-pixel "
            "bilinear splatting")
    model_name += ("; catalogue: power-law counts + one bright-end anchor "
                   "star pinned at mag_zero")

    # Detector artefacts (fixed pattern across frames) ---------------------
    bad_mask, dead_mask, hot_mask = _defect_mask(
        width, height, config.bad_pixel_fraction, rng_defect)
    bad_pixel_count = int(bad_mask.sum())
    peak_fraction = float(max(k.max() for k in kernels))
    psf_star_index = choose_psf_star(catalog, full_well_e, peak_fraction)

    jitters = (rng_jitter.normal(0.0, jitter_pix, size=(len(defocus), 2))
               if jitter_pix > 0 else np.zeros((len(defocus), 2), dtype=float))

    sky_e = float(config.sky_background_e_per_pix_s) * float(config.exposure_s)
    dark_e = float(config.dark_current_e_per_s) * float(config.exposure_s)
    background_e = float(max(0.0, sky_e) + max(0.0, dark_e))

    raw_frames: List[np.ndarray] = []
    cal_frames: List[np.ndarray] = []
    repaired_frames: List[np.ndarray] = []
    frame_meta: List[dict] = []
    psf_cutouts: List[np.ndarray] = []
    total_saturation = 0
    total_cosmic = 0
    total_electrons = 0.0
    total_electrons_measured = 0.0

    for frame_i, d in enumerate(defocus):
        kernel = kernels[frame_i]
        jx, jy = float(jitters[frame_i, 0]), float(jitters[frame_i, 1])
        scene = np.zeros((height, width), dtype=float)
        for s in catalog:
            splat_star(scene, s["x"] + jx, s["y"] + jy,
                       s["flux_e"], kernel)
        expected_e = np.clip(scene + background_e, 0.0, None)
        expected_total = float(expected_e.sum())

        electrons = rng_noise.poisson(expected_e).astype(float)
        measured_total = float(electrons.sum())
        if float(config.read_noise_e) > 0:
            electrons += rng_noise.normal(0.0, float(config.read_noise_e),
                                          electrons.shape)

        adu = electrons / gain + bias_adu
        saturated = adu >= float(max_adu)
        sat_count = int(np.count_nonzero(saturated))
        total_saturation += sat_count

        adu = np.clip(np.rint(adu), 0.0, float(max_adu))
        raw = adu.copy()

        cosmic_mask = _cosmic_ray_map(width, height, config.cosmic_ray_rate,
                                      rng_cosmic)
        cr_count = int(cosmic_mask.sum())
        total_cosmic += cr_count
        raw[dead_mask] = 0.0
        raw[hot_mask] = float(max_adu)
        raw[cosmic_mask] = float(max_adu)

        # Simple calibration: bias + mean dark subtraction, unit flat field.
        cal = np.clip(raw - bias_adu - dark_e / gain, 0.0, None)
        defect = bad_mask | cosmic_mask
        repaired = _repair_defects(cal, defect)

        raw_frames.append(raw)
        cal_frames.append(cal)
        repaired_frames.append(repaired)
        total_electrons += expected_total
        total_electrons_measured += measured_total
        frame_meta.append({
            "defocus": float(d),
            "jitter_pix": [float(jx), float(jy)],
            "saturation_count": sat_count,
            "cosmic_ray_count": cr_count,
            "total_electrons": expected_total,
            "total_electrons_measured": measured_total,
            "background_e_per_pix": background_e,
            "psf_kernel_size": int(kernel.shape[0]),
        })

        # PSF cutout from the cleaned calibrated frame (defects / CRs
        # interpolated, as a calibration pipeline would flag them).
        half = psf_size // 2
        cx = float(catalog[psf_star_index]["x"] + float(jx))
        cy = float(catalog[psf_star_index]["y"] + float(jy))
        cx = float(np.clip(int(round(cx)), half, width - half - 1))
        cy = float(np.clip(int(round(cy)), half, height - half - 1))
        cut = _cutout(repaired, cx, cy, psf_size)
        psf_cutouts.append(_clean_cutout(cut))

    star = catalog[psf_star_index]
    jitter0 = frame_meta[0]["jitter_pix"] if frame_meta else [0.0, 0.0]
    star_public = {
        "x": float(star["x"] + jitter0[0]),
        "y": float(star["y"] + jitter0[1]),
        "flux": float(star["flux_e"]),
        "mag": float(star["mag"]),
        "id": int(star["id"]),
    }

    psf_peak = float(np.max(psf_cutouts[0])) if psf_cutouts else 0.0
    result = {
        "raw": raw_frames,
        "cal": cal_frames,
        "psf": psf_cutouts,
        "psf_peak": psf_peak,
        "star": star_public,
        "catalog": [{"id": int(s["id"]), "x": float(s["x"]), "y": float(s["y"]),
                     "mag": float(s["mag"]), "flux_e_per_s": float(s["flux_e_per_s"]),
                     "flux_e": float(s["flux_e"])} for s in catalog],
        "frames": frame_meta,
        "counts": {
            "saturation_count": int(total_saturation),
            "bad_pixel_count": bad_pixel_count,
            "cosmic_ray_count": int(total_cosmic),
            "n_stars_rendered": int(n_stars),
        },
        "electrons": {
            "total_expected": float(total_electrons),
            "total_measured": float(total_electrons_measured),
            "background_e_per_pix": background_e,
        },
        "model": model_name,
        "fast_mode": bool(fast_mode),
        "seeing_sigma_pix": float(sigma_see),
        "psf_kernel_sizes": sorted({int(m["psf_kernel_size"]) for m in frame_meta}),
        "adu": {
            "max_adu": int(max_adu),
            "bit_depth": int(config.bit_depth),
            "saturation_adu": int(config.saturation_adu),
            "readout_bias_adu": bias_adu,
            "full_well_e": full_well_e,
        },
        "seed": int(config.seed),
    }
    return result
