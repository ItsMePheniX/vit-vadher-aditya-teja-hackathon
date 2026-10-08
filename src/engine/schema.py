"""Frozen Signal schema — field names must never change."""
from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

EventType = Literal[
    "Geopolitical",
    "Macroeconomic",
    "Credit Event",
    "Merger/Acquisition",
    "Earnings",
    "Regulatory",
    "Product Launch",
    "Other",
]


class Signal(BaseModel):
    """Canonical output of the NLP pipeline.

    Field names are **frozen** — changing them breaks downstream consumers.
    See README.md §Signal Schema.
    """

    id: str = Field(..., description="Unique signal ID (ULID)")
    ts: datetime = Field(..., description="ISO8601 timestamp of signal creation")
    entity: str = Field(..., description="Ticker symbol or 'MARKET'")
    sentiment: float = Field(..., ge=-1.0, le=1.0, description="FinBERT sentiment score")
    event_type: EventType = Field(..., description="Classified event category")
    impact: int = Field(..., ge=1, le=10, description="Estimated market impact 1-10")
    confidence: float = Field(..., ge=0.0, le=1.0, description="Model confidence [0,1]")
    half_life_hours: float = Field(..., gt=0, description="Expected signal decay half-life in hours")
    evidence: list[str] = Field(default_factory=list, description="Key phrases driving classification")
    sources: list[str] = Field(default_factory=list, description="Source URLs or feed names")

    model_config = ConfigDict()
