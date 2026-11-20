"""Compatibility wrapper: matrix service lives in products.sensitivity."""
from products.sensitivity.matrix_service import analyze, control

__all__ = ["analyze", "control"]
