"""src/engine/pipeline.py — backward-compat shim.

The canonical pipeline implementation now lives in src/engine/nlp/pipeline.py.
All imports from this module continue to work unchanged.

Change from previous API
------------------------
  Old: process(item) -> Signal          (single signal)
  New: process(item) -> list[Signal]    (one per entity; minimum one)

Callers that expected a single Signal should take results[0] or iterate.
"""
from src.engine.nlp.pipeline import process, write_signals

__all__ = ["process", "write_signals"]
