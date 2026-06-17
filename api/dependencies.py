"""Shared FastAPI dependencies (dependency injection)."""

from __future__ import annotations

import time

from cachetools import TTLCache

_app_start_time: float = time.time()


def get_app_start_time() -> float:
    """Return the Unix timestamp when the application started."""
    return _app_start_time


def get_cache_instance() -> TTLCache:
    """Return the shared TTLCache singleton."""
    from api.services.cache_service import get_cache
    return get_cache()
