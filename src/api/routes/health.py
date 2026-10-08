"""Health check — reports service status and live mode availability."""
from __future__ import annotations

import os
import shutil
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter
from pydantic import BaseModel

from src.engine.mode_manager import manager, EPISODES

router = APIRouter(prefix="/health", tags=["health"])


class LiveAvailability(BaseModel):
    available: bool
    reason: str


class HealthResponse(BaseModel):
    status:      str
    ts:          str
    version:     str
    mode:        str
    live:        LiveAvailability
    episodes:    list[str]
    uptime_s:    float | None


def _check_live() -> LiveAvailability:
    """Determine whether Live mode can actually ingest real data."""
    news_key = os.getenv("NEWS_API_KEY", "").strip()
    if news_key:
        return LiveAvailability(available=True, reason="NEWS_API_KEY is set")

    # Synthetic feed is always available
    return LiveAvailability(
        available=True,
        reason="Synthetic feed active (set NEWS_API_KEY in .env for real data)",
    )


@router.get("", response_model=HealthResponse, summary="Health check")
async def health() -> HealthResponse:
    """Returns 200 OK with detailed status:
    - mode: current engine mode (idle / live / replay)
    - live.available: whether live ingestion is possible and why not
    - episodes: list of replay episodes available
    """
    status = manager.status()
    live   = _check_live()

    return HealthResponse(
        status="ok",
        ts=datetime.now(tz=timezone.utc).isoformat(),
        version="0.2.0",
        mode=status["mode"],
        live=live,
        episodes=list(EPISODES.keys()),
        uptime_s=status.get("uptime_s"),
    )
