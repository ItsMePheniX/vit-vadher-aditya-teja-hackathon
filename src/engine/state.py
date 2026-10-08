"""state.py — per-entity decayed sentiment state store.

Formula
-------
Each signal contributes a weighted sentiment component that decays as:

    S_i(t) = signal.sentiment * signal.confidence
             * exp( -ln(2) * Δt_hours / signal.half_life_hours )

The entity-level state is the sum of all active contributions:

    state[entity] = Σ S_i(t)   for all non-expired signals

A signal is pruned when |S_i(t)| < min_active_sentiment (config).

Thread-safety
-------------
All mutations are protected by a threading.Lock so the StateStore can be
used safely from both asyncio tasks (via to_thread) and direct sync calls.

Public API
----------
    store = StateStore()
    store.ingest(signal)                       # add a new signal
    snap  = store.snapshot()                   # dict[entity → StateEntry]
    entry = store.get(entity)                  # single entity, or None
    store.prune()                              # remove expired contributions
"""
from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src import config as cfg_mod
from src.engine.schema import Signal

logger = logging.getLogger(__name__)

_LN2 = math.log(2)


def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)


def _cfg(*keys, default=None):
    return cfg_mod.get(*keys, default=default)


# ── Data types ────────────────────────────────────────────────────────────────

@dataclass
class _Contribution:
    """A single signal's contribution to an entity's state."""
    signal_id:       str
    added_at:        datetime          # when the signal was ingested
    initial_weight:  float             # sentiment * confidence at t=0
    half_life_hours: float

    def current_weight(self, now: datetime | None = None) -> float:
        """S_i(t) = initial_weight * exp(-ln2 * Δt_hours / half_life_hours)."""
        t = now or _now_utc()
        added = self.added_at if self.added_at.tzinfo else self.added_at.replace(tzinfo=timezone.utc)
        dt_hours = (t - added).total_seconds() / 3600.0
        return self.initial_weight * math.exp(-_LN2 * dt_hours / max(self.half_life_hours, 1e-6))


@dataclass
class StateEntry:
    """Public state for a single entity at a given point in time."""
    entity:            str
    decayed_sentiment: float   # current sum of decayed contributions
    signal_count:      int     # active contributing signals
    strongest_event:   str     # event_type of the highest |weight| contribution
    last_updated:      datetime
    half_life_hours:   float   # weighted average half-life of contributions


# ── State store ───────────────────────────────────────────────────────────────

class StateStore:
    """Thread-safe in-memory store for decayed per-entity sentiment state."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        # entity → list of contributions
        self._contributions: dict[str, list[_Contribution]] = {}
        # entity → event_type of most recent signal (for strongest_event fallback)
        self._last_event: dict[str, str] = {}
        self._last_signal: dict[str, Signal] = {}

    # ── Write ─────────────────────────────────────────────────────────────────

    def ingest(self, signal: Signal) -> None:
        """Add a signal's contribution to the entity state.

        The weight = sentiment * confidence, decays over signal.half_life_hours.
        """
        weight = signal.sentiment * signal.confidence
        contrib = _Contribution(
            signal_id=signal.id,
            added_at=signal.ts if signal.ts.tzinfo else signal.ts.replace(tzinfo=timezone.utc),
            initial_weight=weight,
            half_life_hours=signal.half_life_hours,
        )
        with self._lock:
            if signal.entity not in self._contributions:
                self._contributions[signal.entity] = []
            self._contributions[signal.entity].append(contrib)
            self._last_event[signal.entity]  = signal.event_type
            self._last_signal[signal.entity] = signal

    # ── Read ──────────────────────────────────────────────────────────────────

    def get(self, entity: str, now: datetime | None = None) -> StateEntry | None:
        """Return decayed state for a single entity, or None if unknown."""
        with self._lock:
            contribs = self._contributions.get(entity)
            if not contribs:
                return None
            return self._compute(entity, contribs, now or _now_utc())

    def snapshot(self, now: datetime | None = None) -> dict[str, StateEntry]:
        """Return decayed state for all known entities."""
        t = now or _now_utc()
        result: dict[str, StateEntry] = {}
        with self._lock:
            for entity, contribs in self._contributions.items():
                if contribs:
                    result[entity] = self._compute(entity, contribs, t)
        return result

    # ── Maintenance ───────────────────────────────────────────────────────────

    def prune(self, now: datetime | None = None) -> int:
        """Remove contributions whose |weight| has fallen below min_active_sentiment.

        Returns number of pruned contributions.
        """
        min_w = float(_cfg("state", "min_active_sentiment", default=0.01))
        t = now or _now_utc()
        total_pruned = 0
        with self._lock:
            for entity in list(self._contributions):
                before = len(self._contributions[entity])
                self._contributions[entity] = [
                    c for c in self._contributions[entity]
                    if abs(c.current_weight(t)) >= min_w
                ]
                pruned = before - len(self._contributions[entity])
                total_pruned += pruned
                if pruned:
                    logger.debug("Pruned %d expired contributions for %s", pruned, entity)
        return total_pruned

    def clear(self) -> None:
        """Reset entire state (useful for tests)."""
        with self._lock:
            self._contributions.clear()
            self._last_event.clear()
            self._last_signal.clear()

    # ── Internal ──────────────────────────────────────────────────────────────

    def _compute(self, entity: str, contribs: list[_Contribution], now: datetime) -> StateEntry:
        weights = [c.current_weight(now) for c in contribs]
        total   = sum(weights)
        n_active = sum(1 for w in weights if abs(w) >= 0.001)

        # Dominant event = contribution with largest |weight|
        if contribs:
            max_idx = max(range(len(weights)), key=lambda i: abs(weights[i]))
            # Retrieve event type from last signal for this entity (approx.)
            dom_event = self._last_event.get(entity, "Other")
        else:
            dom_event = "Other"

        # Weighted average half-life
        abs_weights = [abs(w) for w in weights]
        total_abs = sum(abs_weights) or 1.0
        avg_hl = sum(
            c.half_life_hours * abs_weights[i] / total_abs
            for i, c in enumerate(contribs)
        )

        last_sig = self._last_signal.get(entity)
        last_ts  = last_sig.ts if last_sig else now

        return StateEntry(
            entity=entity,
            decayed_sentiment=round(total, 5),
            signal_count=n_active,
            strongest_event=dom_event,
            last_updated=last_ts if last_ts.tzinfo else last_ts.replace(tzinfo=timezone.utc),
            half_life_hours=round(avg_hl, 2),
        )


# ── Module-level singleton shared by the API ──────────────────────────────────
_store = StateStore()


def get_store() -> StateStore:
    """Return the module-level singleton StateStore."""
    return _store
