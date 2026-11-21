"""Compatibility wrapper: control service lives in products.control."""
from products.control.service import metadata, sensitivity, control

__all__ = ['metadata', 'sensitivity', 'control']
