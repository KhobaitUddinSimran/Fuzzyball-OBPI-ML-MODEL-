"""FastAPI application entrypoint for the OBPI dashboard API."""

from __future__ import annotations

import logging
import time
import uuid

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse

from api.routes import analyze, compare, health, leaderboard, players

logger = logging.getLogger("obpi.api")

app = FastAPI(
    title="OBPI Dashboard API",
    description="Off-Ball Positional Intelligence engine REST API",
    version="1.0.0",
)

# ─── CORS ─────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "https://obpi-dashboard.onrender.com"],
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

# ─── GZip ─────────────────────────────────────────────────────────────────
app.add_middleware(GZipMiddleware, minimum_size=1024)

# ─── Request Logging ──────────────────────────────────────────────────────
@app.middleware("http")
async def log_requests(request: Request, call_next):
    """Log method, path, status, and latency for every request."""
    start = time.time()
    response = await call_next(request)
    latency = (time.time() - start) * 1000
    logger.info(
        "%s %s — %d (%.2f ms)",
        request.method,
        request.url.path,
        response.status_code,
        latency,
    )
    return response


# ─── Global Exception Handler ───────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """Catch unhandled exceptions, log with trace_id, return 500."""
    trace_id = str(uuid.uuid4())[:8]
    logger.error("Unhandled exception [trace_id=%s]: %s", trace_id, exc, exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal pipeline error", "trace_id": trace_id},
    )


# ─── Routers ──────────────────────────────────────────────────────────────
app.include_router(health.router)
app.include_router(players.router)
app.include_router(analyze.router)
app.include_router(compare.router)
app.include_router(leaderboard.router)
