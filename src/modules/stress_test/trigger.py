"""trigger.py — subscribe to the signal stream and run stress tests on high-impact signals.

Trigger Logic
-------------
A stress test is fired when BOTH:
    signal.impact     >= cfg.min_impact      (default: 7)
    signal.confidence >= cfg.min_confidence  (default: 0.55)

The trigger subscribes to the module-level ingestor.signal_queue and runs
asynchronously as a background task started by the FastAPI lifespan.

Results are stored in a thread-safe deque (latest N results).
WebSocket clients are notified via a fan-out queue set.

Usage
-----
    # In FastAPI lifespan:
    from src.modules.stress_test.trigger import StressTrigger
    trigger = StressTrigger()
    asyncio.create_task(trigger.run())

    # In API route:
    result = trigger.latest()         # most recent StressResult or None
    results = trigger.history(n=10)   # last N results
"""
from __future__ import annotations

import asyncio
import logging
from collections import deque
from datetime import datetime, timezone

import pandas as pd

from src.engine import ingestor as _ingestor
from src.engine.schema import Signal
from src.modules.stress_test import scenario_loader, valuator

logger = logging.getLogger(__name__)

_MAX_HISTORY = 50    # keep last N stress results in memory


class StressTrigger:
    """Background task: listen to signal_queue, fire stress tests on threshold signals."""

    def __init__(self) -> None:
        self._history:    deque[valuator.StressResult] = deque(maxlen=_MAX_HISTORY)
        self._ws_clients: set[asyncio.Queue] = set()
        self._portfolio:  pd.DataFrame | None = None
        self._cfg = scenario_loader.get_config()

    # ── Portfolio (lazy load) ──────────────────────────────────────────────────

    def _get_portfolio(self) -> pd.DataFrame:
        if self._portfolio is None:
            from src.modules.stress_test.portfolio_builder import load_portfolio  # noqa: PLC0415
            try:
                self._portfolio = load_portfolio()
                logger.info("Portfolio loaded: %d positions", len(self._portfolio))
            except FileNotFoundError:
                logger.warning(
                    "Portfolio file not found — building synthetic portfolio now…"
                )
                from src.modules.stress_test.portfolio_builder import build_portfolio  # noqa: PLC0415
                self._portfolio = build_portfolio()
        return self._portfolio

    # ── Trigger check ─────────────────────────────────────────────────────────

    def _should_trigger(self, sig: Signal) -> bool:
        return (
            sig.impact     >= self._cfg.min_impact
            and sig.confidence >= self._cfg.min_confidence
        )

    # ── Run loop ──────────────────────────────────────────────────────────────

    async def run(self) -> None:
        """Background task: drain ingestor queue, fire stress tests on qualifying signals."""
        logger.info(
            "StressTrigger started (min_impact=%d, min_confidence=%.2f)",
            self._cfg.min_impact, self._cfg.min_confidence,
        )
        while True:
            # Tap into the shared ingestor signal_queue via a secondary queue
            # The broadcaster in signals.py fans out to ws_clients — we use a
            # dedicated trigger queue registered separately.
            sig: Signal = await _ingestor.signal_queue.get()
            _ingestor.signal_queue.task_done()

            if not self._should_trigger(sig):
                continue

            scenario = scenario_loader.get_scenario(sig.event_type)
            if scenario is None:
                logger.debug("No scenario for event_type=%s — skipping", sig.event_type)
                continue

            try:
                portfolio = self._get_portfolio()
                result = await asyncio.to_thread(
                    valuator.apply_stress,
                    portfolio,
                    scenario,
                    sig.id,
                    sig.entity,
                    sig.event_type,
                    sig.sentiment,
                    sig.impact,
                    sig.confidence,
                    sig.evidence,
                    datetime.now(tz=timezone.utc).isoformat(),
                    self._cfg.sentiment_scale_factor,
                )
                self._history.append(result)
                self._broadcast(result)
                logger.info(
                    "Stress result: loss=%.2f (%.2f%%) signal=%s",
                    result.loss, result.loss_pct * 100, sig.id,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error("Stress test failed for signal %s: %s", sig.id, exc)

    # ── WebSocket fan-out ─────────────────────────────────────────────────────

    def add_ws_client(self, q: asyncio.Queue) -> None:
        self._ws_clients.add(q)

    def remove_ws_client(self, q: asyncio.Queue) -> None:
        self._ws_clients.discard(q)

    def _broadcast(self, result: valuator.StressResult) -> None:
        for q in list(self._ws_clients):
            try:
                q.put_nowait(result)
            except asyncio.QueueFull:
                pass

    # ── Results API ───────────────────────────────────────────────────────────

    def latest(self) -> valuator.StressResult | None:
        return self._history[-1] if self._history else None

    def history(self, n: int = 10) -> list[valuator.StressResult]:
        results = list(self._history)
        return results[-n:]

    def run_now(self, sig: Signal) -> valuator.StressResult | None:
        """Synchronously run a stress test for a signal (for testing/manual use)."""
        scenario = scenario_loader.get_scenario(sig.event_type)
        if scenario is None:
            return None
        portfolio = self._get_portfolio()
        result = valuator.apply_stress(
            portfolio, scenario, sig.id, sig.entity, sig.event_type,
            sig.sentiment, sig.impact, sig.confidence, sig.evidence,
            datetime.now(tz=timezone.utc).isoformat(),
            self._cfg.sentiment_scale_factor,
        )
        self._history.append(result)
        self._broadcast(result)
        return result


# Module-level singleton (shared with API routes)
_trigger: StressTrigger | None = None


def get_trigger() -> StressTrigger:
    global _trigger
    if _trigger is None:
        _trigger = StressTrigger()
    return _trigger
