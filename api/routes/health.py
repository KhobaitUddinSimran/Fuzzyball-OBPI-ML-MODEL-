"""Health check endpoint."""

from __future__ import annotations

import time

from fastapi import APIRouter

from api.dependencies import get_app_start_time, get_cache_instance
from api.models import HealthResponse

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Return service health, uptime, and cache connectivity."""
    uptime = round(time.time() - get_app_start_time(), 2)
    cache = get_cache_instance()
    return HealthResponse(
        status="ok",
        model_version="1.0.0",
        schema_version=2,
        uptime_seconds=uptime,
        cache_connected=cache is not None,
    )
