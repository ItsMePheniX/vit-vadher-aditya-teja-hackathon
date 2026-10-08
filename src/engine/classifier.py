"""Event type classifier.

Uses a simple keyword-based rule set as the primary classifier.
A future improvement (documented in ASSUMPTIONS.md) would be to train
a fine-tuned model; the keyword approach is transparent and auditable,
which is important for a jury demo.
"""
from __future__ import annotations

import re

from src.engine.schema import EventType

# Each tuple: (event_type, compiled regex pattern)
# Order matters — first match wins.
_RULES: list[tuple[EventType, re.Pattern[str]]] = [
    (
        "Merger/Acquisition",
        re.compile(
            r"\b(acqui|merger|takeover|buyout|m&a|deal|bid for|offer for)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Credit Event",
        re.compile(
            r"\b(default|downgrad|credit rating|bankrupt|insolvenc|debt restructur|junk|cds spread|files for bankruptcy)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Geopolitical",
        re.compile(
            r"\b(sanction|war|conflict|geopolit|tariff|trade war|embargo|military|invasion|coup)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Macroeconomic",
        re.compile(
            r"\b(inflation|interest rates?|federal reserve|central bank|gdp|recession|unemployment|cpi|ppi|monetary policy)\b"
            r"|\bfed\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Earnings",
        re.compile(
            r"\b(earnings?|profit|revenue|eps|quarter|guidance|beat|miss|forecast|outlook|beating)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Regulatory",
        re.compile(
            r"\b(regulat|sec |fca |compliance|penalty|fine|antitrust|lawsuit|litigation|investigation)\b",
            re.IGNORECASE,
        ),
    ),
    (
        "Product Launch",
        re.compile(
            r"\b(launch|product|release|unveil|introduc|announce new|debut)\b",
            re.IGNORECASE,
        ),
    ),
]


def classify(text: str) -> EventType:
    """Classify event type from free text using rule-based matching.

    Returns the first matching category or 'Other'.
    """
    for event_type, pattern in _RULES:
        if pattern.search(text):
            return event_type
    return "Other"
