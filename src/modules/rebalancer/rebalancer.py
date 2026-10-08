"""rebalancer.py — background rebalancing task that watches the StateStore.

Lifecycle
---------
The Rebalancer runs as an asyncio background task (started in FastAPI lifespan).
Every `rebalance_interval_minutes` of *simulated* time (or real time in live mode)
it:
    1. Reads the current StateStore snapshot (decayed sentiment per entity)
    2. Feeds the sentiment map into RebalancerEngine.compute()
    3. Stores a WeightSnapshot in history
    4. Broadcasts the snapshot to connected WebSocket clients

Batching
--------
Signals may arrive in rapid bursts during replay.  We do NOT rebalance on every
signal — instead we track the simulated (or real) clock and rebalance at most
once per `rebalance_interval_minutes`.  The interval is based on the *timestamp*
of the most-recently ingested signal, so replay at 60× still produces a realistic
number of rebalances.

Design note on sentiment_map
-----------------------------
StateStore.snapshot() returns StateEntry objects with a `sentiment` float.
We compute s_c = entry.sentiment (which is already sentiment × confidence decayed)
divided by the number of active contributions (approximation of confidence ≈ 1).
Since the formula uses k * s_c, this is directionally correct and the k parameter
can absorb calibration.
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone, timedelta
from typing import Callable

from src import config as cfg_mod
from src.engine.state import get_store
from src.modules.rebalancer.engine import RebalancerEngine, WeightSnapshot, UNIVERSE

logger = logging.getLogger(__name__)


def _cfg(key: str, default):
    return cfg_mod.get("rebalancer", key, default=default)


class Rebalancer:
    """Runs periodic rebalances and maintains weight history."""

    def __init__(self) -> None:
        self._engine     = RebalancerEngine()
        self._history:   deque[WeightSnapshot] = deque(maxlen=int(_cfg("max_history", 500)))
        self._ws_clients: set[asyncio.Queue]   = set()
        self._prev_weights: dict[str, float] | None = None
        self._last_rebal_sim_time: datetime | None  = None
        self._last_signal_ts:     datetime | None   = None   # track sim time via signal ts
        self._interval_minutes = float(_cfg("rebalance_interval_minutes", 5))

    # ── Public helpers ────────────────────────────────────────────────────────

    def notify_signal_ts(self, ts: datetime) -> None:
        """Called by the signal pipeline with the signal's timestamp.

        This advances the simulated clock so we know when to rebalance.
        """
        self._last_signal_ts = ts

    def should_rebalance(self) -> bool:
        """True if enough simulated time has passed since the last rebalance."""
        if self._last_rebal_sim_time is None:
            return True
        ref = self._last_signal_ts or datetime.now(tz=timezone.utc)
        elapsed_min = (ref - self._last_rebal_sim_time).total_seconds() / 60
        return elapsed_min >= self._interval_minutes

    def rebalance_now(self, trigger: str = "scheduled") -> WeightSnapshot:
        """Execute a rebalance immediately and store/broadcast the result."""
        store    = get_store()
        snapshot = store.snapshot()

        # Build sentiment map: entity → decayed_sentiment (already s × c in state)
        sentiment_map: dict[str, float] = {}
        for entity, entry in snapshot.items():
            if entity in set(UNIVERSE):
                sentiment_map[entity] = float(entry.sentiment)

        new_weights, turnover = self._engine.compute(sentiment_map, self._prev_weights)

        ts = self._last_signal_ts or datetime.now(tz=timezone.utc)
        snap = WeightSnapshot(
            ts=ts,
            weights=new_weights,
            turnover=turnover,
            trigger=trigger,
            sentiment_map=dict(sentiment_map),
        )
        self._history.append(snap)
        self._prev_weights      = dict(new_weights)
        self._last_rebal_sim_time = ts
        self._broadcast(snap)
        logger.info(
            "Rebalanced (%s): turnover=%.3f top=%s",
            trigger, turnover,
            sorted(new_weights, key=new_weights.get, reverse=True)[:3],
        )
        return snap

    # ── Background loop ───────────────────────────────────────────────────────

    async def run(self) -> None:
        """Periodic rebalance loop (checks every 2 real seconds)."""
        logger.info("Rebalancer started (interval=%g min)", self._interval_minutes)
        # Initial rebalance at startup with equal weights
        self.rebalance_now(trigger="initial")
        while True:
            await asyncio.sleep(2)
            try:
                if self.should_rebalance():
                    self.rebalance_now(trigger="scheduled")
            except Exception as exc:
                logger.error("Rebalancer error: %s", exc)

    # ── WebSocket ─────────────────────────────────────────────────────────────

    def add_ws_client(self, q: asyncio.Queue) -> None:
        self._ws_clients.add(q)

    def remove_ws_client(self, q: asyncio.Queue) -> None:
        self._ws_clients.discard(q)

    def _broadcast(self, snap: WeightSnapshot) -> None:
        for q in list(self._ws_clients):
            try:
                q.put_nowait(snap)
            except asyncio.QueueFull:
                pass

    # ── Query ─────────────────────────────────────────────────────────────────

    def latest(self) -> WeightSnapshot | None:
        return self._history[-1] if self._history else None

    def history(self, n: int = 100) -> list[WeightSnapshot]:
        snaps = list(self._history)
        return snaps[-n:]

    def current_weights(self) -> dict[str, float]:
        snap = self.latest()
        if snap:
            return snap.weights
        return RebalancerEngine.equal_weights()


# ── Module singleton ──────────────────────────────────────────────────────────
_rebalancer: Rebalancer | None = None


def get_rebalancer() -> Rebalancer:
    global _rebalancer
    if _rebalancer is None:
        _rebalancer = Rebalancer()
    return _rebalancer
