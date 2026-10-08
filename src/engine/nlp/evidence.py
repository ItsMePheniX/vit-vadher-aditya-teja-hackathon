"""Evidence extraction: identify the 1-2 sentences that drive the sentiment.

Strategy (per entity)
---------------------
1. Split the full text into sentences.
2. Collect "entity-relevant" sentences — those that mention the entity's
   ticker symbol, company name, or any of its known aliases.
3. If no entity-relevant sentences exist, fall back to all sentences.
4. Score each candidate sentence using FinBERT (sentiment magnitude).
5. Return up to `max_sentences` in their original order, ranked by
   |sentiment| (strongest sentiment = most evidential).

The call to sentiment.analyse() inside this module is cheap because results
are cached by the sentiment module's LRU cache.

Public API
----------
    evidence = extract(text, entity_ticker)
    # → ["SVB stock collapsed 60%.", "Depositors withdrew $42B overnight."]
"""
from __future__ import annotations

import re

from src.config import get as cfg
from src.engine.linker import EntityLinker

_MAX_SENTS   = cfg("nlp", "evidence_max_sentences", default=2)
_linker      = EntityLinker()

# Sentence boundary splitter (handles common abbreviations poorly — acceptable
# for financial headlines which tend to be grammatically well-formed)
_SENT_RE = re.compile(r"(?<=[.!?])\s+")


def _split_sentences(text: str) -> list[str]:
    parts = _SENT_RE.split(text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 10]


def _entity_patterns(ticker: str) -> list[re.Pattern[str]]:
    """Compile case-insensitive patterns for ticker and all its known aliases."""
    entry = _linker.get_entry(ticker)
    terms = [re.escape(ticker)]
    if entry:
        terms += [re.escape(a) for a in entry.aliases]
    return [re.compile(t, re.IGNORECASE) for t in terms]


def extract(text: str, entity: str = "MARKET", max_sentences: int | None = None) -> list[str]:
    """Return up to max_sentences evidential sentences for the given entity.

    Args:
        text:          Full article / tweet text.
        entity:        Ticker symbol or 'MARKET'.
        max_sentences: Override config default.

    Returns:
        List of sentence strings, in original text order.
    """
    max_n = max_sentences or _MAX_SENTS
    sentences = _split_sentences(text)

    if not sentences:
        # Single-sentence or unsplittable text
        return [text[:300].strip()] if text.strip() else []

    if len(sentences) <= max_n:
        return sentences

    # ── Entity-relevant filtering ─────────────────────────────────────────────
    if entity != "MARKET":
        patterns = _entity_patterns(entity)
        relevant = [
            s for s in sentences
            if any(p.search(s) for p in patterns)
        ]
        if relevant:
            sentences = relevant

    # ── Score by sentiment magnitude ──────────────────────────────────────────
    # Import here to avoid circular dependency at module load time
    from src.engine.nlp import sentiment  # noqa: PLC0415

    scored: list[tuple[float, int, str]] = []
    for i, sent in enumerate(sentences):
        try:
            result = sentiment.analyse(sent)
            magnitude = abs(result.score)
        except Exception:  # noqa: BLE001
            magnitude = 0.0
        scored.append((magnitude, i, sent))

    # Sort by magnitude descending, stable by original position for ties
    scored.sort(key=lambda x: (-x[0], x[1]))
    top = scored[:max_n]

    # Return in original text order
    top_by_pos = sorted(top, key=lambda x: x[1])
    return [t[2] for t in top_by_pos]
