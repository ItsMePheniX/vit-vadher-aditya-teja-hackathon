"""Tests for the event type classifier."""
from __future__ import annotations

import pytest

from src.engine.classifier import classify


@pytest.mark.parametrize("text,expected", [
    ("Federal Reserve raises interest rates by 25bps", "Macroeconomic"),
    ("AAPL reported record earnings, beating estimates", "Earnings"),
    ("Geopolitical tensions and trade war tariffs hit markets", "Geopolitical"),
    ("Company files for bankruptcy after debt restructuring fails", "Credit Event"),
    ("MSFT acquires AI startup in $2.5B merger deal", "Merger/Acquisition"),
    ("SEC launches investigation into accounting irregularities", "Regulatory"),
    ("Apple unveils new product line of AI hardware devices", "Product Launch"),
    ("Nothing notable happening in markets today", "Other"),
])
def test_classify(text: str, expected: str) -> None:
    result = classify(text)
    assert result == expected, f"Expected '{expected}', got '{result}' for: {text!r}"
