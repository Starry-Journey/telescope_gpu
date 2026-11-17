"""Compatibility wrapper: CCD service lives in products.ccd."""
from products.ccd.service import run_ccd_pipeline, _png

__all__ = ['run_ccd_pipeline']
