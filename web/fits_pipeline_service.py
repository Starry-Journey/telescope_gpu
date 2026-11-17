"""Compatibility wrapper: FITS service lives in products.ccd."""
from products.ccd.fits_service import run_fits_pipeline

__all__ = ['run_fits_pipeline']
