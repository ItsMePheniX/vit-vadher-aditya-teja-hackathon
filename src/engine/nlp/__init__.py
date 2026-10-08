"""NLP sub-package for the Financial Risk Engine.

Public API
----------
    from src.engine.nlp import process, write_signals
    from src.engine.nlp.sentiment import analyse, analyse_batch
    from src.engine.nlp.classifier import classify
    from src.engine.nlp.evidence import extract
"""
from src.engine.nlp.pipeline import process, write_signals

__all__ = ["process", "write_signals"]
