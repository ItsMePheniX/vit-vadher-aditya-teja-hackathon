"""Report generator: produces a markdown summary of recent signals and risk.

Outputs a plain markdown string suitable for display in the dashboard
or saving to file. No external dependencies beyond stdlib + pydantic.
"""
from __future__ import annotations

from datetime import datetime, timezone

from src.engine.schema import Signal
from src.modules.alert_engine import Alert
from src.modules.portfolio_risk import PortfolioRisk


def _signal_table(signals: list[Signal]) -> str:
    if not signals:
        return "_No signals yet._\n"
    header = "| # | Entity | Event | Sentiment | Impact | Confidence | Half-life |\n"
    sep    = "|---|--------|-------|-----------|--------|------------|-----------|\n"
    rows = []
    for i, s in enumerate(signals[:20], 1):
        rows.append(
            f"| {i} | `{s.entity}` | {s.event_type} "
            f"| {s.sentiment:+.3f} | {s.impact} | {s.confidence:.2f} "
            f"| {s.half_life_hours:.1f}h |"
        )
    return header + sep + "\n".join(rows) + "\n"


def _alert_section(alerts: list[Alert]) -> str:
    if not alerts:
        return "_No alerts triggered._\n"
    lines = []
    for a in alerts:
        emoji = "🔴" if a.severity == "HIGH" else "🟡"
        lines.append(f"- {emoji} **[{a.severity}]** `{a.entity}` — {a.reason} ({a.event_type})")
    return "\n".join(lines) + "\n"


def generate(
    signals: list[Signal],
    alerts: list[Alert],
    risk: PortfolioRisk | None,
    as_of: datetime | None = None,
) -> str:
    """Generate a markdown risk report string."""
    as_of = as_of or datetime.now(tz=timezone.utc)

    sections: list[str] = [
        f"# Financial Risk Report\n\n_Generated: {as_of.isoformat()}_\n",
        "## 📊 Recent Signals\n",
        _signal_table(signals),
        "## 🚨 Active Alerts\n",
        _alert_section(alerts),
    ]

    if risk:
        sections += [
            "## 📉 Portfolio Risk (VaR)\n",
            f"- **VaR 95% (daily)**: {risk.var_95:.4%}",
            f"- **CVaR 95% (daily)**: {risk.cvar_95:.4%}",
            f"- **Signal-adjusted Risk**: {risk.signal_adjusted_risk:.4%}",
            "",
            "### Ticker Exposures",
            "| Ticker | Risk Weight |",
            "|--------|-------------|",
        ]
        for ticker, weight in risk.ticker_exposures.items():
            sections.append(f"| `{ticker}` | {weight:.2%} |")
        sections.append("")

    return "\n".join(sections)
