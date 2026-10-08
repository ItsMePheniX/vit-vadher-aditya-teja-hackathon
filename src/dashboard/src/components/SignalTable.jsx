/** SignalTable — scrollable table of recent signals with animated new-row flash */
import { sentClass, sentColor, impactBg, impactFg, fmtDatetime } from '../utils/helpers.js'

function SentimentCell({ score }) {
  const pct = Math.abs(score) * 100
  const cls = sentClass(score)
  return (
    <div className="sentiment-cell">
      <div className="sentiment-bar-track">
        <div
          className="sentiment-bar-fill"
          style={{
            width: `${pct}%`,
            background: sentColor(score),
          }}
        />
      </div>
      <span className={`sentiment-value ${cls}`}>{score >= 0 ? '+' : ''}{score.toFixed(3)}</span>
    </div>
  )
}

function ImpactPill({ impact }) {
  return (
    <span
      className="impact-pill"
      style={{ background: impactBg(impact), color: impactFg(impact) }}
    >
      {impact}
    </span>
  )
}

function EventBadge({ label }) {
  const palette = {
    'Earnings':           ['rgba(16,185,129,0.15)', '#6ee7b7'],
    'Geopolitical':       ['rgba(239,68,68,0.15)',  '#fca5a5'],
    'Macroeconomic':      ['rgba(59,130,246,0.15)', '#93c5fd'],
    'Credit Event':       ['rgba(245,158,11,0.15)', '#fcd34d'],
    'Merger/Acquisition': ['rgba(124,58,237,0.15)', '#c4b5fd'],
    'Regulatory':         ['rgba(239,68,68,0.12)',  '#fca5a5'],
    'Product Launch':     ['rgba(6,182,212,0.15)',  '#67e8f9'],
    'Other':              ['rgba(148,163,184,0.1)', '#94a3b8'],
  }
  const [bg, fg] = palette[label] || palette['Other']
  return (
    <span className="badge" style={{ background: bg, color: fg }}>
      {label}
    </span>
  )
}

export function SignalTable({ signals, latestId }) {
  if (!signals.length) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">📡</div>
        <div className="empty-state-text">Waiting for signals from the NLP engine…</div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          Run <code style={{ fontFamily: 'var(--font-mono)' }}>make run</code> to start the API
        </div>
      </div>
    )
  }

  return (
    <div className="signal-table-wrap">
      <table className="signal-table">
        <thead>
          <tr>
            <th>Time</th>
            <th>Entity</th>
            <th>Event Type</th>
            <th>Sentiment</th>
            <th>Impact</th>
            <th>Conf.</th>
            <th>Evidence</th>
          </tr>
        </thead>
        <tbody>
          {signals.map(s => (
            <tr key={s.id} className={s.id === latestId ? 'new-signal' : ''}>
              <td style={{ fontFamily: 'var(--font-mono)', fontSize: 10, color: 'var(--text-muted)' }}>
                {fmtDatetime(s.ts)}
              </td>
              <td>
                <span className={`badge ${s.entity === 'MARKET' ? 'badge-market' : 'badge-entity'}`}>
                  {s.entity}
                </span>
              </td>
              <td><EventBadge label={s.event_type} /></td>
              <td><SentimentCell score={s.sentiment} /></td>
              <td><ImpactPill impact={s.impact} /></td>
              <td style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--text-secondary)' }}>
                {(s.confidence * 100).toFixed(0)}%
              </td>
              <td>
                <div className="evidence-text" title={(s.evidence || []).join(' | ')}>
                  {(s.evidence || []).join(' ').slice(0, 90) || '—'}
                </div>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}
