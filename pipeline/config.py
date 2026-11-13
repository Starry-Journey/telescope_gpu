"""Configuration objects for the CCD star-field / moving-target pipeline.

The pipeline is deliberately dependency-light: numpy + scipy + astropy +
matplotlib.  SEP / photutils / OpenCV are named in the report as production
alternatives, but this repository implements the same operations with
numpy/scipy so that it runs in the existing WSL environment without extra
system packages.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class SimConfig:
    # detector / sequence
    width: int = 512
    height: int = 512
    n_frames: int = 20
    n_stars: int = 20
    seed: int = 42

    # optics / PSF
    psf_sigma: float = 1.6          # pixels, Gaussian approximation
    pixel_scale_arcsec: float = 1.0
    rotation_deg: float = 30.0
    crval: Tuple[float, float] = (150.0, 20.0)   # RA/Dec at reference pixel
    crpix: Optional[Tuple[float, float]] = None  # default image centre

    # star and target fluxes (electrons)
    star_flux_min: float = 2000.0
    star_flux_max: float = 20000.0
    target_flux: float = 1200.0
    target_start: Tuple[float, float] = (80.0, 90.0)
    target_velocity: Tuple[float, float] = (1.6, 0.9)
    target_streak: float = 0.0       # >0: draw a trailed source of this length (pix)

    # detector / calibration
    exptime: float = 10.0
    bias_level: float = 1000.0
    dark_current: float = 0.02      # e-/s/pix
    background: float = 300.0       # e-/pix
    read_noise: float = 5.0         # e- RMS
    flat_vignette: float = 0.22     # relative vignetting at the corner
    gain_variation: float = 0.02    # pixel-to-pixel flat-field variation
    saturation: float = 60000.0
    cosmic_rays: int = 4
    jitter: float = 0.6             # frame-to-frame pointing jitter (pix RMS)

    # lens distortion (normalised coordinates, applied detector <-> ideal)
    distortion_k1: float = 0.018
    distortion_p1: float = 0.004
    distortion_p2: float = -0.003


@dataclass
class PreprocessConfig:
    background_box: int = 64        # mesh size for background map
    background_sigma: float = 3.0   # sigma clipping
    denoise_sigma: float = 0.8      # Gaussian sigma; 0 disables
    bad_pixel_sigma: float = 6.0
    saturation_margin: float = 0.98


@dataclass
class DetectConfig:
    star_threshold_sigma: float = 5.0
    min_star_area: int = 3
    max_star_area: int = 400
    star_centroid_box: int = 5
    moving_threshold_sigma: float = 3.0
    moving_smooth_sigma: float = 1.0
    moving_min_area: int = 4
    moving_max_area: int = 500
    morphology_size: int = 3
    hough_theta_bins: int = 180
    hough_rho_bin: float = 1.0
    hough_threshold: int = 18
    track_max_step: float = 5.0
    track_min_length: int = 5
    track_max_gap: int = 1


@dataclass
class AstrometryConfig:
    match_radius_arcsec: float = 3.0
    distortion_degree: int = 3
    max_iterations: int = 5
    min_matches: int = 6


@dataclass
class PipelineConfig:
    data_dir: str = "data/ccd_pipeline"
    output_dir: str = "results/ccd_pipeline"
    sim: SimConfig = field(default_factory=SimConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    detect: DetectConfig = field(default_factory=DetectConfig)
    astrometry: AstrometryConfig = field(default_factory=AstrometryConfig)

    @classmethod
    def from_dict(cls, obj: Dict[str, Any]) -> "PipelineConfig":
        return cls(
            data_dir=obj.get("data_dir", cls.data_dir),
            output_dir=obj.get("output_dir", cls.output_dir),
            sim=SimConfig(**obj.get("sim", {})),
            preprocess=PreprocessConfig(**obj.get("preprocess", {})),
            detect=DetectConfig(**obj.get("detect", {})),
            astrometry=AstrometryConfig(**obj.get("astrometry", {})),
        )

    @classmethod
    def load(cls, path: Optional[str]) -> "PipelineConfig":
        if not path:
            return cls()
        with open(path, "r", encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(asdict(self), fh, indent=2, ensure_ascii=False)

