"""weights.py — Rebalancer API routes (GET and WebSocket)."""
import asyncio
import logging
from typing import Any

from fastapi import APIRouter, WebSocket, WebSocketDisconnect
from pydantic import BaseModel

from src.modules.rebalancer import get_rebalancer

logger = logging.getLogger(__name__)

router = APIRouter(tags=["rebalancer"])
ws_router = APIRouter(tags=["rebalancer"])


class WeightSnapshotOut(BaseModel):
    ts: str
    weights: dict[str, float]
    turnover: float
    trigger: str
    sentiment_map: dict[str, float]


@router.get("/weights", response_model=list[WeightSnapshotOut])
def get_weight_history(limit: int = 50) -> Any:
    """Get the recent rebalancing history."""
    rb = get_rebalancer()
    history = rb.history(n=limit)
    out = []
    for snap in history:
        out.append(
            WeightSnapshotOut(
                ts=snap.ts.isoformat(),
                weights=snap.weights,
                turnover=snap.turnover,
                trigger=snap.trigger,
                sentiment_map=snap.sentiment_map,
            )
        )
    return out


@router.get("/weights/latest", response_model=WeightSnapshotOut | dict[str, Any])
def get_latest_weights() -> Any:
    """Get the most recent weights."""
    rb = get_rebalancer()
    snap = rb.latest()
    if snap:
        return WeightSnapshotOut(
            ts=snap.ts.isoformat(),
            weights=snap.weights,
            turnover=snap.turnover,
            trigger=snap.trigger,
            sentiment_map=snap.sentiment_map,
        )
    return {}


@ws_router.websocket("/ws/weights")
async def ws_weights(websocket: WebSocket) -> None:
    """Real-time stream of rebalancer weight snapshots."""
    await websocket.accept()
    rb = get_rebalancer()
    
    # Send history first
    history = rb.history(n=50)
    for snap in history:
        out = WeightSnapshotOut(
            ts=snap.ts.isoformat(),
            weights=snap.weights,
            turnover=snap.turnover,
            trigger=snap.trigger,
            sentiment_map=snap.sentiment_map,
        )
        await websocket.send_json(out.model_dump())

    # Stream new snapshots
    q: asyncio.Queue[Any] = asyncio.Queue()
    rb.add_ws_client(q)
    try:
        while True:
            snap = await q.get()
            out = WeightSnapshotOut(
                ts=snap.ts.isoformat(),
                weights=snap.weights,
                turnover=snap.turnover,
                trigger=snap.trigger,
                sentiment_map=snap.sentiment_map,
            )
            await websocket.send_json(out.model_dump())
    except WebSocketDisconnect:
        logger.debug("WS client disconnected from /ws/weights")
    except Exception as exc:
        logger.error("Error in /ws/weights: %s", exc)
    finally:
        rb.remove_ws_client(q)
