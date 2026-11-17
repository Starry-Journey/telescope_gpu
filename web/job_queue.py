"""Compatibility wrapper: jobs live in products.jobs."""
from products.jobs.store import JobStore, store

__all__ = ["JobStore", "store"]
