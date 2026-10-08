"""src/modules/rebalancer — NLP-driven index rebalancer.

Public API
----------
    from src.modules.rebalancer import get_rebalancer, Rebalancer, WeightSnapshot

    rb = get_rebalancer()           # module singleton
    rb.rebalance_now("manual")      # trigger immediately
    snap = rb.latest()              # most recent WeightSnapshot
    history = rb.history(n=50)      # list of WeightSnapshots
"""
from src.modules.rebalancer.rebalancer import get_rebalancer, Rebalancer
from src.modules.rebalancer.engine import WeightSnapshot, UNIVERSE

__all__ = ["get_rebalancer", "Rebalancer", "WeightSnapshot", "UNIVERSE"]
