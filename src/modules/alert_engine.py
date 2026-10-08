"""Alert engine: emits structured alerts when signals cross thresholds.

Thresholds are configured in config.yaml under alerts.*.
An Alert is not a schema-frozen object — it wraps a Signal with metadata.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone

from src import config
from src.engine.schema import Signal

logger = logging.getLogger(__name__)

_IMPACT_THRESHOLD: int = config.get("alerts", "impact_threshold", default=7)
_SENTIMENT_THRESHOLD: float = config.get("alerts", "sentiment_threshold", default=-0.5)
_CONFIDENCE_THRESHOLD: float = config.get("alerts", "confidence_threshold", default=0.6)


@dataclass
class Alert:
    signal_id: str
    entity: str
    event_type: str
    reason: str
    severity: str      # "HIGH" | "MEDIUM" | "LOW"
    ts: str = field(default_factory=lambda: datetime.now(tz=timezone.utc).isoformat())


def evaluate(signal: Signal) -> Alert | None:
    """Check signal against alert thresholds; returns Alert or None.

    Severity:
    - HIGH  : impact >= threshold AND sentiment <= threshold
    - MEDIUM: impact >= threshold OR  sentiment <= threshold (with confidence)
    - LOW   : confidence low but impact still elevated
    """
    high_impact = signal.impact >= _IMPACT_THRESHOLD
    negative_sentiment = signal.sentiment <= _SENTIMENT_THRESHOLD
    sufficient_confidence = signal.confidence >= _CONFIDENCE_THRESHOLD

    if not sufficient_confidence:
        return None

    reasons: list[str] = []
    if high_impact:
        reasons.append(f"impact={signal.impact}>={_IMPACT_THRESHOLD}")
    if negative_sentiment:
        reasons.append(f"sentiment={signal.sentiment:.2f}<={_SENTIMENT_THRESHOLD}")

    if not reasons:
        return None

    severity = "HIGH" if (high_impact and negative_sentiment) else "MEDIUM"

    alert = Alert(
        signal_id=signal.id,
        entity=signal.entity,
        event_type=signal.event_type,
        reason="; ".join(reasons),
        severity=severity,
    )
    logger.warning("ALERT [%s] %s — %s", severity, signal.entity, alert.reason)
    return alert


def evaluate_batch(signals: list[Signal]) -> list[Alert]:
    """Evaluate a list of signals and return all triggered alerts."""
    alerts: list[Alert] = []
    for signal in signals:
        alert = evaluate(signal)
        if alert:
            alerts.append(alert)
    return alerts
