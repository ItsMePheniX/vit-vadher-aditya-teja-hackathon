/** KPI stat cards across the top of the dashboard */
import { impactBg, impactFg } from '../utils/helpers.js'

function KpiCard({ label, value, sub, icon, accent }) {
  return (
    <div className="kpi-card" style={{ '--kpi-accent': accent }}>
      <div className="kpi-label">{label}</div>
      <div className="kpi-value">{value}</div>
      {sub && <div className="kpi-sub">{sub}</div>}
    </div>
  )
}

export function KpiGrid({ signals, risk }) {
  const total = signals.length
  const positive = signals.filter(s => s.sentiment >  0.1).length
  const negative = signals.filter(s => s.sentiment < -0.1).length
  const highImpact = signals.filter(s => s.impact >= 7).length
  const avgConf = total
    ? (signals.reduce((a,s) => a + s.confidence, 0) / total * 100).toFixed(0)
    : '—'

  const varDisplay = risk
    ? `${(risk.var_95 * 100).toFixed(2)}%`
    : '—'

  const sentRatio = total
    ? `${((positive / total) * 100).toFixed(0)}% pos / ${((negative / total) * 100).toFixed(0)}% neg`
    : 'awaiting data'

  return (
    <div className="kpi-grid">
      <KpiCard
        label="Total Signals"
        value={total}
        sub="processed this session"
        accent="var(--accent-blue)"
      />
      <KpiCard
        label="Sentiment Split"
        value={positive > negative ? 'Bullish' : negative > positive ? 'Bearish' : '—'}
        sub={sentRatio}
        accent={positive >= negative ? 'var(--positive)' : 'var(--negative)'}
      />
      <KpiCard
        label="High-Impact Alerts"
        value={highImpact}
        sub="impact ≥ 7 / 10"
        accent={highImpact > 0 ? 'var(--accent-red)' : 'var(--accent-green)'}
      />
      <KpiCard
        label="Avg Confidence"
        value={avgConf === '—' ? '—' : `${avgConf}%`}
        sub="FinBERT + NLI blend"
        accent="var(--accent-violet)"
      />
      <KpiCard
        label="Portfolio VaR 95%"
        value={varDisplay}
        sub={risk ? `CVaR: ${(risk.cvar_95 * 100).toFixed(2)}%` : 'connect API'}
        accent="var(--accent-amber)"
      />
    </div>
  )
}
