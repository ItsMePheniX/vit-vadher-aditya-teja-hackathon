"""scenario_loader.py — load and validate scenarios.yaml."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import NamedTuple

import yaml

logger = logging.getLogger(__name__)

_SCENARIOS_PATH = Path(__file__).parent / "scenarios.yaml"


class Scenario(NamedTuple):
    name:                str
    label:               str
    historical_analog:   str
    equity_pct:          float
    rate_bp:             float
    credit_spread_bp:    float
    hy_spread_bp:        float
    fx_pct:              float
    sentiment_scale_factor: float = 0.5


class ScenarioConfig(NamedTuple):
    scenarios:              dict[str, Scenario]
    min_impact:             int
    min_confidence:         float
    sentiment_scale_factor: float


def load(path: Path = _SCENARIOS_PATH) -> ScenarioConfig:
    """Parse scenarios.yaml and return a ScenarioConfig."""
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))

    sens_scale = float(raw.get("sentiment_scale_factor", 0.5))
    trigger    = raw.get("trigger", {})
    min_impact = int(trigger.get("min_impact", 7))
    min_conf   = float(trigger.get("min_confidence", 0.55))

    scenarios: dict[str, Scenario] = {}
    for name, sc in raw.get("scenarios", {}).items():
        scenarios[name] = Scenario(
            name=name,
            label=sc.get("label", name),
            historical_analog=sc.get("historical_analog", ""),
            equity_pct=float(sc.get("equity_pct", 0.0)),
            rate_bp=float(sc.get("rate_bp", 0.0)),
            credit_spread_bp=float(sc.get("credit_spread_bp", 0.0)),
            hy_spread_bp=float(sc.get("hy_spread_bp", 0.0)),
            fx_pct=float(sc.get("fx_pct", 0.0)),
            sentiment_scale_factor=sens_scale,
        )

    logger.debug("Loaded %d scenarios from %s", len(scenarios), path)
    return ScenarioConfig(
        scenarios=scenarios,
        min_impact=min_impact,
        min_confidence=min_conf,
        sentiment_scale_factor=sens_scale,
    )


_cfg: ScenarioConfig | None = None


def get_config() -> ScenarioConfig:
    global _cfg
    if _cfg is None:
        _cfg = load()
    return _cfg


def get_scenario(event_type: str) -> Scenario | None:
    """Return the Scenario for a given EventType label, or None."""
    cfg = get_config()
    return cfg.scenarios.get(event_type)
