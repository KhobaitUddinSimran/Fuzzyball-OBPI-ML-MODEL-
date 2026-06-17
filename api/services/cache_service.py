"""In-memory TTL cache service using cachetools."""

from __future__ import annotations

import json
import logging
from typing import Any

from cachetools import TTLCache

logger = logging.getLogger("obpi.api.cache")

_DEFAULT_MAXSIZE = 256
_DEFAULT_TTL = 3600  # seconds

_cache_instance: TTLCache | None = None


def get_cache(maxsize: int = _DEFAULT_MAXSIZE, ttl: int = _DEFAULT_TTL) -> TTLCache:
    """Return the singleton TTLCache instance (lazily created)."""
    global _cache_instance
    if _cache_instance is None:
        _cache_instance = TTLCache(maxsize=maxsize, ttl=ttl)
        logger.info("TTLCache initialised (maxsize=%d, ttl=%ds)", maxsize, ttl)
    return _cache_instance


def get_cached(key: str) -> Any | None:
    """Retrieve cached JSON data by key."""
    cache = get_cache()
    raw = cache.get(key)
    if raw is None:
        return None
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        logger.warning("Cache key %r contained invalid JSON; evicting", key)
        cache.pop(key, None)
        return None


def set_cached(key: str, data: Any, ttl: int | None = None) -> None:
    """Store JSON-serialisable data under key."""
    cache = get_cache()
    raw = json.dumps(data, default=str)
    # TTLCache does not support per-key TTL overrides, so we just set it.
    cache[key] = raw
    logger.debug("Cache set: %r", key)


def clear_cache() -> None:
    """Clear the entire cache."""
    cache = get_cache()
    cache.clear()
    logger.info("Cache cleared")
