"""Tests for the signal scorer."""
from __future__ import annotations

import pytest

from src.engine.scorer import score


@pytest.mark.parametrize("sentiment,confidence,event_type,expected_impact_range", [
    (-0.9, 0.95, "Geopolitical", (8, 10)),     # strong negative + high weight
    (0.3, 0.5, "Other", (1, 3)),               # weak positive, low weight
    (0.0, 1.0, "Earnings", (1, 1)),            # neutral sentiment → minimal impact
    (-1.0, 1.0, "Credit Event", (10, 10)),     # max negative = max impact
    (0.5, 0.8, "Macroeconomic", (3, 5)),       # moderate
])
def test_impact_range(
    sentiment: float,
    confidence: float,
    event_type: str,
    expected_impact_range: tuple[int, int],
) -> None:
    result = score(sentiment, confidence, event_type)  # type: ignore[arg-type]
    lo, hi = expected_impact_range
    assert lo <= result.impact <= hi, (
        f"impact={result.impact} not in [{lo},{hi}] "
        f"for sentiment={sentiment}, confidence={confidence}, event={event_type}"
    )


def test_impact_clamped() -> None:
    result = score(-1.0, 1.0, "Geopolitical")
    assert 1 <= result.impact <= 10


def test_half_life_positive() -> None:
    result = score(-0.5, 0.8, "Earnings")
    assert result.half_life_hours > 0


def test_half_life_by_event_type() -> None:
    """Geopolitical events should have longer half-life than Earnings."""
    geo = score(-0.5, 0.8, "Geopolitical")
    earn = score(-0.5, 0.8, "Earnings")
    assert geo.half_life_hours > earn.half_life_hours
