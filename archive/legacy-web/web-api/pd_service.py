"""Compatibility wrapper: PD service lives in products.pd."""
from products.pd.service import invert_psf

__all__ = ['invert_psf']
