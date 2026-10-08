"""Tests for the frozen Signal schema."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.engine.schema import Signal


VALID_PAYLOAD = {
    "id": "01JBXYZ1234567890ABCDEF",
    "ts": "2026-10-01T10:00:00+00:00",
    "entity": "AAPL",
    "sentiment": -0.75,
    "event_type": "Earnings",
    "impact": 7,
    "confidence": 0.85,
    "half_life_hours": 12.0,
    "evidence": ["AAPL missed Q3 earnings"],
    "sources": ["newsapi"],
}


def test_valid_signal() -> None:
    s = Signal(**VALID_PAYLOAD)
    assert s.entity == "AAPL"
    assert s.impact == 7
    assert s.event_type == "Earnings"


def test_sentiment_bounds() -> None:
    with pytest.raises(ValidationError):
        Signal(**{**VALID_PAYLOAD, "sentiment": 1.5})
    with pytest.raises(ValidationError):
        Signal(**{**VALID_PAYLOAD, "sentiment": -1.1})


def test_impact_bounds() -> None:
    with pytest.raises(ValidationError):
        Signal(**{**VALID_PAYLOAD, "impact": 0})
    with pytest.raises(ValidationError):
        Signal(**{**VALID_PAYLOAD, "impact": 11})


def test_confidence_bounds() -> None:
    with pytest.raises(ValidationError):
        Signal(**{**VALID_PAYLOAD, "confidence": -0.1})
    with pytest.raises(ValidationError):
        Signal(**{**VALID_PAYLOAD, "confidence": 1.1})


def test_invalid_event_type() -> None:
    with pytest.raises(ValidationError):
        Signal(**{**VALID_PAYLOAD, "event_type": "Unknown"})


def test_valid_event_types() -> None:
    valid_types = [
        "Geopolitical", "Macroeconomic", "Credit Event",
        "Merger/Acquisition", "Earnings", "Regulatory",
        "Product Launch", "Other",
    ]
    for et in valid_types:
        s = Signal(**{**VALID_PAYLOAD, "event_type": et})
        assert s.event_type == et


def test_json_roundtrip() -> None:
    s = Signal(**VALID_PAYLOAD)
    json_str = s.model_dump_json()
    s2 = Signal.model_validate_json(json_str)
    assert s.id == s2.id
    assert s.sentiment == s2.sentiment
