"""Base types for the ingest subsystem.

RawItem is the lingua franca between sources and the NLP pipeline.
Source is an abstract async generator that yields RawItems.

Field contract (frozen — never rename):
    id     : str              deterministic item identifier (source:hash or row index)
    ts     : datetime         publication/post timestamp (UTC-aware)
    text   : str              raw text (headline + body snippet, or tweet body)
    source : str              feed name, e.g. "kaggle_financial_news", "gdelt", "kaggle_tweets"
    url    : str              canonical URL or empty string if unavailable
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from typing import AsyncIterator


@dataclass(frozen=True)
class RawItem:
    """Canonical input type for the NLP pipeline.

    Immutable: use dataclasses.replace() if you need a modified copy.
    """

    id: str
    ts: datetime       # must be UTC-aware
    text: str
    source: str        # feed / publication identifier
    url: str           # article URL or empty string


class Source(ABC):
    """Abstract async source that yields :class:`RawItem` objects.

    Subclasses implement :meth:`items` as an async generator.
    The caller is responsible for closing the generator on cancellation.

    Example::

        class MySource(Source):
            async def items(self) -> AsyncIterator[RawItem]:
                for row in some_data:
                    yield RawItem(...)
    """

    @abstractmethod
    def items(self) -> AsyncIterator[RawItem]:
        """Yield RawItem objects (async generator)."""
        ...

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}>"
