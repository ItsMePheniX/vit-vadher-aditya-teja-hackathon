"""Stress-test API routes.

Endpoints
---------
  GET  /stress/latest          — most recent StressResult (or 404)
  GET  /stress/history         — last N stress results
  POST /stress/run             — manually trigger a stress test on a given signal
  WS   /ws/stress              — real-time stream of StressResult objects
"""
from __future__ import annotations

import asyncio
import dataclasses
import json
import logging
from datetime import datetime, timezone
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.modules.stress_test import get_trigger, StressResult
from src.modules.stress_test.valuator import AssetClassResult

logger = logging.getLogger(__name__)

router     = APIRouter(prefix="/stress", tags=["stress"])
ws_router  = APIRouter(tags=["stress-ws"])


# ── Pydantic response models ──────────────────────────────────────────────────

class AssetClassOut(BaseModel):
    asset_class:  str
    n_positions:  int
    value_before: float
    value_after:  float
    loss:         float
    loss_pct:     float


class StressResultOut(BaseModel):
    signal_id:     str
    signal_entity: str
    event_type:    str
    scenario_name: str
    sentiment:     float
    impact:        int
    confidence:    float
    evidence:      list[str]
    shock_multiplier: float

    value_before:  float
    value_after:   float
    loss:          float
    loss_pct:      float

    breakdown:     list[AssetClassOut]
    triggered_at:  str

    equity_pct_applied:       float
    rate_bp_applied:          float
    credit_spread_bp_applied: float
    hy_spread_bp_applied:     float
    fx_pct_applied:           float


def _to_out(r: StressResult) -> StressResultOut:
    return StressResultOut(
        signal_id=r.signal_id,
        signal_entity=r.signal_entity,
        event_type=r.event_type,
        scenario_name=r.scenario_name,
        sentiment=r.sentiment,
        impact=r.impact,
        confidence=r.confidence,
        evidence=r.evidence,
        shock_multiplier=r.shock_multiplier,
        value_before=r.value_before,
        value_after=r.value_after,
        loss=r.loss,
        loss_pct=r.loss_pct,
        breakdown=[
            AssetClassOut(
                asset_class=b.asset_class,
                n_positions=b.n_positions,
                value_before=b.value_before,
                value_after=b.value_after,
                loss=b.loss,
                loss_pct=b.loss_pct,
            )
            for b in r.breakdown
        ],
        triggered_at=r.triggered_at,
        equity_pct_applied=r.equity_pct_applied,
        rate_bp_applied=r.rate_bp_applied,
        credit_spread_bp_applied=r.credit_spread_bp_applied,
        hy_spread_bp_applied=r.hy_spread_bp_applied,
        fx_pct_applied=r.fx_pct_applied,
    )


# ── REST endpoints ────────────────────────────────────────────────────────────

@router.get("/latest", response_model=StressResultOut, summary="Most recent stress result")
async def latest_stress() -> StressResultOut:
    """Return the most recent stress-test result.

    Returns 404 if no stress test has been triggered yet.
    Start a replay or live session and let high-impact signals flow in.
    """
    r = get_trigger().latest()
    if r is None:
        raise HTTPException(
            status_code=404,
            detail="No stress test results yet. Start a replay with POST /replay/start.",
        )
    return _to_out(r)


@router.get("/history", response_model=list[StressResultOut], summary="Stress result history")
async def stress_history(
    n: Annotated[int, Query(ge=1, le=50)] = 10,
) -> list[StressResultOut]:
    """Return the last N stress-test results, newest first."""
    results = get_trigger().history(n=n)
    return [_to_out(r) for r in reversed(results)]


class ManualRunRequest(BaseModel):
    entity:     str   = "JPM"
    event_type: str   = "Credit Event"
    sentiment:  float = -0.8
    impact:     int   = 8
    confidence: float = 0.85
    evidence:   list[str] = ["Manual stress test trigger"]


@router.post("/run", response_model=StressResultOut, status_code=201,
             summary="Manually trigger a stress test")
async def run_stress(req: ManualRunRequest) -> StressResultOut:
    """Run a stress test immediately with synthetic signal values.

    Useful for demo and dashboard testing without waiting for a qualifying signal.
    """
    from src.engine.schema import Signal, EventType   # noqa: PLC0415
    from ulid import ULID                             # noqa: PLC0415
    import typing                                     # noqa: PLC0415

    _valid = typing.get_args(EventType)
    if req.event_type not in _valid:
        raise HTTPException(
            status_code=422,
            detail=f"event_type must be one of {list(_valid)}, got '{req.event_type}'",
        )

    sig = Signal(
        id=str(ULID()),
        ts=datetime.now(tz=timezone.utc),
        entity=req.entity,
        sentiment=req.sentiment,
        event_type=req.event_type,  # type: ignore[arg-type]
        impact=req.impact,
        confidence=req.confidence,
        half_life_hours=24.0,
        evidence=req.evidence,
        sources=["manual"],
    )
    result = await asyncio.to_thread(get_trigger().run_now, sig)
    if result is None:
        raise HTTPException(
            status_code=422,
            detail=f"No scenario defined for event_type='{req.event_type}'",
        )
    return _to_out(result)


# ── WebSocket /ws/stress ──────────────────────────────────────────────────────

@ws_router.websocket("/ws/stress")
async def ws_stress_stream(websocket: WebSocket) -> None:
    """Real-time stream of StressResult objects.

    Each message is a JSON-serialised StressResultOut.
    Connection stays open until client disconnects.
    """
    await websocket.accept()
    logger.info("Stress WS client connected: %s", websocket.client)

    q: asyncio.Queue[StressResult] = asyncio.Queue(maxsize=20)
    trigger = get_trigger()
    trigger.add_ws_client(q)

    try:
        while True:
            result = await q.get()
            out    = _to_out(result)
            await websocket.send_text(out.model_dump_json())
    except WebSocketDisconnect:
        logger.info("Stress WS client disconnected")
    except Exception as exc:
        logger.warning("Stress WS error: %s", exc)
    finally:
        trigger.remove_ws_client(q)
