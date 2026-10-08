"""engine.py — core rebalancing logic for the NLP-driven index.

Weight Formula
--------------
    raw_i = base_i * exp(k * s_i * c_i)
    w_i   = clip(raw_i / Σraw, min_weight, max_weight)   [re-normalize after clip]

where:
    base_i  = base weight for ticker i (equal by default: 1/N)
    k       = sensitivity parameter (config: rebalancer.k)
    s_i     = decayed_sentiment for entity i from StateStore
    c_i     = confidence score associated with the entity's current state
              (approximated as: |Σ contributions| / Σ |initial_weights|,
               i.e. how much of the original signal strength remains active)
    min/max = weight bounds from config

Turnover Cap
------------
After computing new weights, the delta from previous weights is checked:
    turnover = Σ |w_new_i - w_old_i|   (0→2 scale, 1 = full turnover)
If turnover > cap, we blend: w_final = α * w_new + (1-α) * w_old
where α = cap / turnover.  This preserves direction but limits churn.

All parameters come from config.yaml under the `rebalancer` key.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import NamedTuple

import numpy as np
import yaml
from pathlib import Path

from src import config as cfg_mod

logger = logging.getLogger(__name__)

# ── 15 tickers from tickers.yaml (fallback hardcoded if file missing) ─────────
_TICKERS_YAML = Path("src/engine/tickers.yaml")
_FALLBACK_TICKERS = ["JPM", "BAC", "C", "GS", "WFC", "AAPL", "MSFT",
                     "AMZN", "GOOGL", "META", "NVDA", "XOM", "JNJ", "BRK-B", "V"]


def _load_tickers() -> list[str]:
    try:
        raw = yaml.safe_load(_TICKERS_YAML.read_text())
        tickers = [e["ticker"] for e in raw.get("tickers", [])]
        return tickers if tickers else _FALLBACK_TICKERS
    except Exception:
        return _FALLBACK_TICKERS


UNIVERSE: list[str] = _load_tickers()
N = len(UNIVERSE)


def _cfg(key: str, default):
    return cfg_mod.get("rebalancer", key, default=default)


# ── Result types ──────────────────────────────────────────────────────────────

@dataclass
class WeightSnapshot:
    """One rebalancing event."""
    ts:            datetime
    weights:       dict[str, float]    # ticker → weight (sum ≈ 1.0)
    turnover:      float               # Σ|Δw| from previous snapshot
    trigger:       str                 # 'scheduled' | 'state_change' | 'manual'
    sentiment_map: dict[str, float]    # ticker → effective decayed sentiment used


# ── Core engine ───────────────────────────────────────────────────────────────

class RebalancerEngine:
    """Stateless weight calculator.  All state lives in the caller (Rebalancer)."""

    def __init__(self) -> None:
        self._reload_config()

    def _reload_config(self) -> None:
        self.k           = float(_cfg("k",            3.0))
        self.min_w       = float(_cfg("min_weight",   0.02))
        self.max_w       = float(_cfg("max_weight",   0.20))
        self.turnover_cap = float(_cfg("turnover_cap", 0.30))
        raw_base: dict   = _cfg("base_weights",       {})
        # Build base_weights: equal if empty, else override specific tickers
        eq = 1.0 / N
        self.base_weights: dict[str, float] = {t: eq for t in UNIVERSE}
        for t, v in raw_base.items():
            if t in self.base_weights:
                self.base_weights[t] = float(v)
        # Normalize base weights to sum to 1
        total = sum(self.base_weights.values())
        self.base_weights = {t: v / total for t, v in self.base_weights.items()}

    def compute(
        self,
        sentiment_map: dict[str, float],       # ticker → decayed_sentiment * confidence
        prev_weights:  dict[str, float] | None,
    ) -> tuple[dict[str, float], float]:
        """Compute new weights and turnover.

        Args:
            sentiment_map:  entity → (decayed_sentiment × confidence) score.
                            Missing tickers treated as 0.
            prev_weights:   weights from last rebalance (or None for first run).

        Returns:
            (new_weights, turnover)  where turnover = Σ|Δw| ∈ [0, 2]
        """
        self._reload_config()

        # Step 1: raw scores
        raw: dict[str, float] = {}
        for t in UNIVERSE:
            s_c = sentiment_map.get(t, 0.0)   # already sentiment × confidence
            raw[t] = self.base_weights[t] * math.exp(self.k * s_c)

        total_raw = sum(raw.values())
        if total_raw == 0:
            total_raw = 1.0

        # Step 2: normalize
        w = {t: raw[t] / total_raw for t in UNIVERSE}

        # Step 3: clip [min, max] and renormalize (iterative convergence)
        w = self._clip_and_renormalize(w)

        # Step 4: turnover cap — blend with previous weights
        turnover = 0.0
        if prev_weights:
            turnover = sum(abs(w[t] - prev_weights.get(t, 1.0 / N)) for t in UNIVERSE)
            if turnover > self.turnover_cap and turnover > 0:
                alpha = self.turnover_cap / turnover
                w = {t: alpha * w[t] + (1 - alpha) * prev_weights.get(t, 1.0 / N)
                     for t in UNIVERSE}
                w = self._clip_and_renormalize(w)
                turnover = sum(abs(w[t] - prev_weights.get(t, 1.0 / N)) for t in UNIVERSE)

        return w, round(turnover, 6)

    def _clip_and_renormalize(self, w: dict[str, float]) -> dict[str, float]:
        """Iteratively clip weights to [min, max] and renormalize until stable."""
        for _ in range(20):  # converges in < 5 iterations typically
            clipped = {t: max(self.min_w, min(self.max_w, v)) for t, v in w.items()}
            total = sum(clipped.values())
            w = {t: v / total for t, v in clipped.items()}
            # Check stability: all weights within [min, max]
            if all(self.min_w - 1e-9 <= v <= self.max_w + 1e-9 for v in w.values()):
                break
        return w

    @staticmethod
    def equal_weights() -> dict[str, float]:
        """Return the equal-weight benchmark."""
        eq = 1.0 / N
        return {t: eq for t in UNIVERSE}
