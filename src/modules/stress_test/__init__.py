"""src/modules/stress_test — hero stress-test module.

Public API
----------
    from src.modules.stress_test import get_trigger, StressResult

    trigger = get_trigger()          # module singleton
    result  = trigger.run_now(sig)   # sync, for testing
    latest  = trigger.latest()       # most recent StressResult
    history = trigger.history(n=10)
"""
from src.modules.stress_test.trigger import get_trigger, StressTrigger
from src.modules.stress_test.valuator import StressResult, AssetClassResult

__all__ = [
    "get_trigger",
    "StressTrigger",
    "StressResult",
    "AssetClassResult",
]
