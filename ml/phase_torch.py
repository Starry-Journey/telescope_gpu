"""Differentiable Fourier-optics forward model for phase diversity.

This is deliberately a *small* model: a circular pupil, three Zernike modes
(Z4 defocus, Z8/Z9 coma), five known defocus frames, Fraunhofer propagation and
a 2x2 detector binning.  Every operation is a torch op, so the same code can

  * generate training data,
  * be used as the forward model in an unrolled/optimization loop,
  * be exported to ONNX (see ``export_onnx.py``).

The C++/CUDA project uses a more complete 8-slot Zernike model; this Python
model is an independent, differentiable surrogate for learning experiments.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn.functional as F


@dataclass
class ForwardConfig:
    n: int = 64                 # detector pixels after binning
    oversample: int = 2         # FFT grid = n * oversample
    pupil_radius: float = 0.45  # normalised pupil radius
    defocus: tuple = (-2.0, -1.0, 0.0, 1.0, 2.0)  # waves
    noise_sigma: float = 0.01   # relative Gaussian noise, 0 disables


def zernike_basis(n: int, pupil_radius: float = 0.45, device=None):
    """Return (mask, Z4, Z8, Z9) on an n x n grid.

    Coordinates are normalised by the pupil radius so the polynomials live on
    the unit disk; the mask keeps only the physical pupil.

    Z4 = sqrt(3) (2 rho^2 - 1)
    Z8 = sqrt(8) (3 rho^3 - 2 rho) cos(theta)   (coma x)
    Z9 = sqrt(8) (3 rho^3 - 2 rho) sin(theta)   (coma y)
    """
    x = torch.linspace(-1.0, 1.0, n, device=device)
    yy, xx = torch.meshgrid(x, x, indexing="ij")
    r = torch.sqrt(xx * xx + yy * yy)
    rho = r / pupil_radius
    theta = torch.atan2(yy, xx)
    mask = (r <= pupil_radius).to(torch.float32)
    z4 = torch.sqrt(torch.tensor(3.0, device=device)) * (2.0 * rho ** 2 - 1.0)
    radial = torch.sqrt(torch.tensor(8.0, device=device)) * (3.0 * rho ** 3 - 2.0 * rho)
    z8 = radial * torch.cos(theta)
    z9 = radial * torch.sin(theta)
    return mask, z4, z8, z9


def psf_from_coeff(coeff: torch.Tensor, defocus: float,
                   cfg: ForwardConfig = ForwardConfig()) -> torch.Tensor:
    """One defocused PSF, normalised to unit sum.

    coeff: tensor (..., 3) = [c4, c8, c9] in waves.
    Returns (..., n, n).
    """
    if coeff.shape[-1] != 3:
        raise ValueError("coeff must have last dimension 3 (Z4, Z8, Z9)")
    single = coeff.dim() == 1
    if single:
        coeff = coeff.view(1, 3)
    n = cfg.n
    fft_n = n * cfg.oversample
    device = coeff.device
    # build the pupil on the FFT grid, then pad; this avoids interpolation
    mask, z4, z8, z9 = zernike_basis(fft_n, cfg.pupil_radius, device)
    c4 = coeff[..., 0].view(-1, 1, 1)
    c8 = coeff[..., 1].view(-1, 1, 1)
    c9 = coeff[..., 2].view(-1, 1, 1)
    phase = 2.0 * torch.pi * (c4 * z4 + c8 * z8 + c9 * z9 + defocus * z4)
    pupil = mask * torch.exp(1j * phase)
    field = torch.fft.fftshift(torch.fft.fft2(pupil, norm="ortho"), dim=(-2, -1))
    psf = (field.real ** 2 + field.imag ** 2)
    psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    if cfg.oversample > 1:
        b = psf.shape[0]
        psf = F.avg_pool2d(psf.view(b, 1, fft_n, fft_n),
                           kernel_size=cfg.oversample).view(b, n, n)
        psf = psf / psf.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return psf[0] if single else psf


def forward_stack(coeff: torch.Tensor,
                  cfg: ForwardConfig = ForwardConfig(),
                  add_noise: bool = False) -> torch.Tensor:
    """Return (..., n_frames, n, n) phase-diversity stack.

    The five frames share the same Zernike coefficients but have known defocus
    offsets.  Noise is relative Gaussian, clipped at zero.
    """
    frames = [psf_from_coeff(coeff, d, cfg) for d in cfg.defocus]
    out = torch.stack(frames, dim=-3)
    if add_noise and cfg.noise_sigma > 0:
        # noise is relative to the frame peak, not to 1.0: a normalised PSF
        # has peak ~1/n^2, so an absolute 1% would swamp the signal.
        peak = out.amax(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
        out = out + cfg.noise_sigma * peak * torch.randn_like(out)
        out = out.clamp_min(0.0)
        out = out / out.sum(dim=(-2, -1), keepdim=True).clamp_min(1e-12)
    return out


def wavefront_rms(coeff: torch.Tensor) -> torch.Tensor:
    """RMS of the Zernike expansion (orthonormal basis -> sqrt(sum c^2))."""
    return torch.sqrt((coeff ** 2).sum(dim=-1).clamp_min(0.0))
