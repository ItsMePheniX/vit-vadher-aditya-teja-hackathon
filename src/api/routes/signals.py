"""Signals API routes — complete specification.

Endpoints
---------
REST:
  GET  /signals                  — list signals (entity, since, min_impact filters)
  GET  /signals/{id}             — single signal by ULID
  POST /signals/ingest           — ad-hoc text → signals
  POST /signals/risk             — portfolio VaR (dashboard widget)

State:
  GET  /state                    — latest decayed sentiment per entity
  GET  /state/{entity}           — single entity state

Control:
  POST /replay/start             — start replay episode
  POST /replay/stop              — stop replay
  POST /live/start               — start live (synthetic) ingestor
  POST /live/stop                — stop live

WebSocket:
  WS   /ws/signals               — real-time signal stream (JSON per message)
"""
from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src import config
from src.engine.schema import Signal
from src.engine.state import get_store, StateEntry
from src.engine.mode_manager import manager, EPISODES, Mode

logger = logging.getLogger(__name__)

# ── Sub-routers ────────────────────────────────────────────────────────────────
signals_router = APIRouter(prefix="/signals", tags=["signals"])
state_router   = APIRouter(prefix="/state",   tags=["state"])
replay_router  = APIRouter(prefix="/replay",  tags=["replay"])
live_router    = APIRouter(prefix="/live",    tags=["live"])
ws_router      = APIRouter(tags=["websocket"])

_JSONL_PATH = Path(config.get("nlp", "signals_path", default="data/signals.jsonl"))

# ── WebSocket fan-out (module-level, shared with mode_manager) ─────────────────
_ws_clients: set[asyncio.Queue[Signal]] = set()

# Shared ingestor queue — populated by broadcaster
from src.engine import ingestor as _ingestor_mod
signal_queue: asyncio.Queue[Signal] = _ingestor_mod.signal_queue


# ── Helpers ────────────────────────────────────────────────────────────────────

def _read_signals_sync(
    limit: int = 100,
    offset: int = 0,
    entity: str | None = None,
    since: datetime | None = None,
    min_impact: int | None = None,
) -> list[Signal]:
    """Read and filter signals from JSONL (synchronous, called via to_thread)."""
    if not _JSONL_PATH.exists():
        return []
    signals: list[Signal] = []
    with _JSONL_PATH.open(encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                signals.append(Signal.model_validate_json(line))
            except Exception:
                continue

    # Newest first
    signals.reverse()

    # Filters
    if entity:
        signals = [s for s in signals if s.entity.upper() == entity.upper()]
    if since:
        since_utc = since if since.tzinfo else since.replace(tzinfo=timezone.utc)
        signals = [
            s for s in signals
            if (s.ts if s.ts.tzinfo else s.ts.replace(tzinfo=timezone.utc)) >= since_utc
        ]
    if min_impact is not None:
        signals = [s for s in signals if s.impact >= min_impact]

    return signals[offset: offset + limit]


# ── /signals REST ─────────────────────────────────────────────────────────────

@signals_router.get("", response_model=list[Signal], summary="List recent signals")
async def list_signals(
    limit:      Annotated[int, Query(ge=1, le=500)] = 50,
    offset:     Annotated[int, Query(ge=0)]         = 0,
    entity:     str | None     = Query(default=None, description="Filter by ticker"),
    since:      datetime | None = Query(default=None, description="ISO8601 lower bound"),
    min_impact: int | None     = Query(default=None, ge=1, le=10, description="Minimum impact"),
) -> list[Signal]:
    """Return signals from JSONL, newest-first, with optional filters."""
    return await asyncio.to_thread(
        _read_signals_sync, limit, offset, entity, since, min_impact
    )


@signals_router.get("/{signal_id}", response_model=Signal, summary="Get signal by ULID")
async def get_signal(signal_id: str) -> Signal:
    signals = await asyncio.to_thread(_read_signals_sync, 10_000, 0, None, None, None)
    for s in signals:
        if s.id == signal_id:
            return s
    raise HTTPException(status_code=404, detail=f"Signal '{signal_id}' not found")


class IngestRequest(BaseModel):
    text: str
    source: str = "manual"
    entity_hint: str | None = None


@signals_router.post(
    "/ingest", response_model=list[Signal], status_code=201,
    summary="Manually ingest text",
)
async def ingest_text(req: IngestRequest) -> list[Signal]:
    """Process raw text through the NLP pipeline → returns one Signal per entity."""
    from src.engine.ingest.base   import RawItem            # noqa: PLC0415
    from src.engine.nlp.pipeline  import process, write_signals  # noqa: PLC0415
    from src.engine.corroboration import corroborate_one    # noqa: PLC0415

    item = RawItem(
        id=f"manual:{hash(req.text) & 0xFFFFFF:06x}",
        ts=datetime.now(tz=timezone.utc),
        text=req.text,
        source=req.source,
        url="",
    )
    signals: list[Signal] = await asyncio.to_thread(process, item)
    store = get_store()
    for sig in signals:
        store.ingest(sig)
        _broadcast(sig)
    await asyncio.to_thread(write_signals, signals)
    return signals


# ── /signals/risk (dashboard widget) ─────────────────────────────────────────

class RiskRequest(BaseModel):
    signals: list[Signal] = []
    tickers: list[str] = []


@signals_router.post("/risk", summary="Compute portfolio VaR from recent signals")
async def compute_risk(req: RiskRequest) -> dict:
    try:
        from src.modules.portfolio_risk import compute  # noqa: PLC0415
        risk = await asyncio.to_thread(compute, req.signals, req.tickers or None)
        return {
            "var_95": risk.var_95, "cvar_95": risk.cvar_95,
            "signal_adjusted_risk": risk.signal_adjusted_risk,
            "ticker_exposures": risk.ticker_exposures,
            "as_of": risk.as_of,
        }
    except Exception as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


# ── /state ────────────────────────────────────────────────────────────────────

class StateEntryOut(BaseModel):
    entity:            str
    decayed_sentiment: float
    signal_count:      int
    strongest_event:   str
    last_updated:      datetime
    half_life_hours:   float


def _entry_to_out(e: StateEntry) -> StateEntryOut:
    return StateEntryOut(
        entity=e.entity,
        decayed_sentiment=e.decayed_sentiment,
        signal_count=e.signal_count,
        strongest_event=e.strongest_event,
        last_updated=e.last_updated,
        half_life_hours=e.half_life_hours,
    )


@state_router.get("", response_model=list[StateEntryOut], summary="Latest decayed sentiment per entity")
async def get_state() -> list[StateEntryOut]:
    """Return the current decayed sentiment for all known entities, sorted by |sentiment|."""
    snap = get_store().snapshot()
    entries = [_entry_to_out(e) for e in snap.values()]
    entries.sort(key=lambda e: -abs(e.decayed_sentiment))
    return entries


@state_router.get("/{entity}", response_model=StateEntryOut, summary="Single entity state")
async def get_entity_state(entity: str) -> StateEntryOut:
    entry = get_store().get(entity.upper())
    if entry is None:
        raise HTTPException(status_code=404, detail=f"No state for entity '{entity}'")
    return _entry_to_out(entry)


# ── /replay ───────────────────────────────────────────────────────────────────

class ReplayStartRequest(BaseModel):
    episode: str = "svb_march_2023"
    speed: float = 60.0


@replay_router.post("/start", summary="Start replay episode")
async def replay_start(req: ReplayStartRequest) -> dict:
    """Start replaying a historical episode at `speed`× real-time.

    - episode: one of the registered episode names (see GET /health)
    - speed: time compression factor (60 = 1 hour of history in 1 minute)
    """
    if req.episode not in EPISODES:
        raise HTTPException(
            status_code=422,
            detail=f"Unknown episode '{req.episode}'. Available: {list(EPISODES)}",
        )
    if req.speed <= 0:
        raise HTTPException(status_code=422, detail="speed must be > 0")

    manager.set_queue(signal_queue)
    try:
        await manager.start_replay(req.episode, req.speed)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    return {
        "status": "ok",
        "message": f"Replay started: episode={req.episode} speed={req.speed}x",
        **manager.status(),
    }


@replay_router.post("/stop", summary="Stop replay")
async def replay_stop() -> dict:
    if manager.mode != Mode.REPLAY:
        raise HTTPException(status_code=409, detail="Not in replay mode")
    await manager.stop()
    return {"status": "ok", "message": "Replay stopped", **manager.status()}


# ── /live ─────────────────────────────────────────────────────────────────────

@live_router.post("/start", summary="Start live (synthetic) ingestor")
async def live_start() -> dict:
    """Start the live ingestor.

    In this environment the ingestor generates synthetic news unless
    NEWS_API_KEY is set in .env.
    """
    manager.set_queue(signal_queue)
    await manager.start_live()
    return {"status": "ok", "message": "Live mode started", **manager.status()}


@live_router.post("/stop", summary="Stop live ingestor")
async def live_stop() -> dict:
    if manager.mode != Mode.LIVE:
        raise HTTPException(status_code=409, detail="Not in live mode")
    await manager.stop()
    return {"status": "ok", "message": "Live stopped", **manager.status()}


# ── /ws/signals WebSocket ─────────────────────────────────────────────────────

@ws_router.websocket("/ws/signals")
async def ws_signal_stream(websocket: WebSocket) -> None:
    """Real-time signal stream.

    Each message is a JSON-serialised Signal object.
    The connection stays open until the client disconnects.
    """
    await websocket.accept()
    logger.info("WS client connected: %s", websocket.client)

    client_q: asyncio.Queue[Signal] = asyncio.Queue(maxsize=200)
    _ws_clients.add(client_q)
    try:
        while True:
            signal = await client_q.get()
            await websocket.send_text(signal.model_dump_json())
    except WebSocketDisconnect:
        logger.info("WS client disconnected")
    except Exception as exc:
        logger.warning("WS error: %s", exc)
    finally:
        _ws_clients.discard(client_q)


def _broadcast(signal: Signal) -> None:
    """Fan-out a signal to all connected WebSocket queues."""
    for q in list(_ws_clients):
        try:
            q.put_nowait(signal)
        except asyncio.QueueFull:
            pass


# ── Broadcaster background task ───────────────────────────────────────────────

async def broadcast_from_ingestor() -> None:
    """Background task: drain the shared signal_queue → WS clients + state store."""
    from src.modules.rebalancer import get_rebalancer
    store = get_store()
    rebalancer = get_rebalancer()
    while True:
        sig: Signal = await signal_queue.get()
        store.ingest(sig)
        _broadcast(sig)
        rebalancer.notify_signal_ts(sig.ts)
        signal_queue.task_done()
