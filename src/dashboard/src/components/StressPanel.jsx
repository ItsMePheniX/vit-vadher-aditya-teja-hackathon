/**
 * StressPanel — hero stress-test results display.
 *
 * Shows:
 *  - Before/after portfolio value bars with loss highlighted
 *  - Per-asset-class breakdown (Bond / Loan / Derivative)
 *  - Triggering signal entity, sentiment, event type, evidence
 *  - Historical analog from scenario
 *  - Empty state when no stress result available yet
 */
import { TrendingDown, ShieldAlert, Info } from 'lucide-react'
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

function fmt$(n) {
  if (Math.abs(n) >= 1e9) return `$${(n / 1e9).toFixed(2)}B`
  if (Math.abs(n) >= 1e6) return `$${(n / 1e6).toFixed(1)}M`
  return `$${n.toLocaleString(undefined, { maximumFractionDigits: 0 })}`
}

function pct(v) { return (v * 100).toFixed(3) + '%' }

function SentimentPill({ score }) {
  const col = score > 0.1 ? '#10b981' : score < -0.1 ? '#ef4444' : '#94a3b8'
  return (
    <span style={{
      fontFamily: 'var(--font-mono)', fontSize: 13, fontWeight: 700,
      color: col, marginLeft: 4,
    }}>
      {score >= 0 ? '+' : ''}{score.toFixed(3)}
    </span>
  )
}

function PortfolioBars({ before, after, loss, lossPct }) {
  const data = [
    { name: 'Before', value: before, fill: '#3b82f6' },
    { name: 'After',  value: after,  fill: after < before ? '#ef4444' : '#10b981' },
  ]
  return (
    <div>
      <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>Portfolio Value</span>
        <span style={{
          fontSize: 12, fontWeight: 700,
          color: loss > 0 ? '#ef4444' : '#10b981',
        }}>
          {loss > 0 ? '▼' : '▲'} {fmt$(Math.abs(loss))} ({pct(lossPct)})
        </span>
      </div>
      <ResponsiveContainer width="100%" height={72}>
        <BarChart data={data} layout="vertical" margin={{ left: 0, right: 0, top: 0, bottom: 0 }}>
          <XAxis type="number" hide domain={['dataMin - 1000000', 'dataMax + 1000000']} />
          <YAxis type="category" dataKey="name" width={46}
            tick={{ fill: 'var(--text-secondary)', fontSize: 11 }} />
          <Tooltip
            formatter={(v) => [fmt$(v), 'Value']}
            contentStyle={{
              background: 'var(--bg-elevated)', border: '1px solid var(--border)',
              borderRadius: 6, fontSize: 11, color: 'var(--text-primary)',
            }}
          />
          <Bar dataKey="value" radius={[0, 4, 4, 0]}>
            {data.map((d, i) => <Cell key={i} fill={d.fill} fillOpacity={0.85} />)}
          </Bar>
        </BarChart>
      </ResponsiveContainer>
      <div style={{ display: 'flex', gap: 20, marginTop: 4 }}>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Before</div>
          <div style={{ fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-mono)' }}>{fmt$(before)}</div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>After</div>
          <div style={{ fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-mono)', color: loss > 0 ? '#ef4444' : '#10b981' }}>
            {fmt$(after)}
          </div>
        </div>
        <div>
          <div style={{ fontSize: 10, color: 'var(--text-muted)' }}>Loss</div>
          <div style={{ fontSize: 13, fontWeight: 700, fontFamily: 'var(--font-mono)', color: '#ef4444' }}>
            {fmt$(loss)}
          </div>
        </div>
      </div>
    </div>
  )
}

const AC_COLORS = { Bond: '#3b82f6', Loan: '#f59e0b', Derivative: '#7c3aed' }

function BreakdownBars({ breakdown }) {
  const data = breakdown.map(b => ({
    name: b.asset_class,
    loss: b.loss,
    pct:  b.loss_pct,
  }))

  return (
    <div>
      <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 8 }}>
        Loss by Asset Class
      </div>
      {data.map(d => {
        const maxLoss = Math.max(...data.map(x => Math.abs(x.loss)), 1)
        const barW    = Math.abs(d.loss) / maxLoss * 100
        const col     = AC_COLORS[d.name] || '#94a3b8'
        return (
          <div key={d.name} style={{ marginBottom: 8 }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 3 }}>
              <span style={{ fontSize: 12, fontWeight: 600 }}>{d.name}</span>
              <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: d.loss > 0 ? '#fca5a5' : '#6ee7b7' }}>
                {d.loss > 0 ? '−' : '+'}{fmt$(Math.abs(d.loss))}
                <span style={{ color: 'var(--text-muted)', marginLeft: 6 }}>({(d.pct * 100).toFixed(2)}%)</span>
              </span>
            </div>
            <div style={{ height: 8, background: 'var(--bg-elevated)', borderRadius: 4, overflow: 'hidden' }}>
              <div style={{
                height: '100%', width: `${barW}%`,
                background: col, borderRadius: 4,
                opacity: d.loss > 0 ? 0.8 : 0.4,
                transition: 'width 0.6s cubic-bezier(0.4,0,0.2,1)',
              }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}

const EVENT_COLORS = {
  'Earnings':           '#6ee7b7',
  'Geopolitical':       '#fca5a5',
  'Macroeconomic':      '#93c5fd',
  'Credit Event':       '#fcd34d',
  'Merger/Acquisition': '#c4b5fd',
  'Regulatory':         '#fca5a5',
  'Product Launch':     '#67e8f9',
  'Other':              '#94a3b8',
}

function TriggerInfo({ result }) {
  const col = EVENT_COLORS[result.event_type] || '#94a3b8'
  return (
    <div style={{
      padding: 12, borderRadius: 8,
      background: 'var(--bg-elevated)', border: '1px solid var(--border)',
    }}>
      <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8, flexWrap: 'wrap' }}>
        <span style={{ fontSize: 14, fontWeight: 800, fontFamily: 'var(--font-mono)' }}>
          {result.signal_entity}
        </span>
        <span style={{
          padding: '2px 8px', borderRadius: 10, fontSize: 10, fontWeight: 700,
          background: `${col}20`, color: col, border: `1px solid ${col}40`,
        }}>
          {result.event_type}
        </span>
        <SentimentPill score={result.sentiment} />
        <span style={{
          marginLeft: 'auto', fontSize: 11, fontFamily: 'var(--font-mono)',
          color: 'var(--text-muted)',
        }}>
          impact <span style={{ color: 'var(--accent-amber)', fontWeight: 700 }}>{result.impact}</span>
          {' / '}conf <span style={{ color: 'var(--accent-cyan)' }}>{(result.confidence * 100).toFixed(0)}%</span>
        </span>
      </div>

      {/* Evidence */}
      {result.evidence?.length > 0 && (
        <div style={{ marginBottom: 8 }}>
          <div style={{ fontSize: 10, color: 'var(--text-muted)', marginBottom: 4, textTransform: 'uppercase', letterSpacing: '0.06em' }}>
            Evidence
          </div>
          {result.evidence.map((ev, i) => (
            <div key={i} style={{
              fontSize: 12, color: 'var(--text-secondary)', padding: '4px 8px',
              borderLeft: '2px solid var(--accent-blue)', marginBottom: 3,
              background: 'rgba(59,130,246,0.04)', borderRadius: '0 4px 4px 0',
            }}>
              "{ev}"
            </div>
          ))}
        </div>
      )}

      {/* Shock parameters */}
      <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
        {[
          ['Equity', `${result.equity_pct_applied?.toFixed(1)}%`],
          ['Rate', `${result.rate_bp_applied?.toFixed(0)}bp`],
          ['Credit', `${result.credit_spread_bp_applied?.toFixed(0)}bp`],
          ['FX', `${result.fx_pct_applied?.toFixed(1)}%`],
          ['Mult', `${result.shock_multiplier?.toFixed(2)}×`],
        ].map(([k, v]) => (
          <div key={k} style={{
            padding: '3px 8px', borderRadius: 5,
            background: 'var(--bg-card)', border: '1px solid var(--border)',
          }}>
            <span style={{ fontSize: 9, color: 'var(--text-muted)', marginRight: 4 }}>{k}</span>
            <span style={{ fontSize: 11, fontFamily: 'var(--font-mono)', color: 'var(--text-primary)' }}>{v}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

export function StressPanel({ latest, history }) {
  if (!latest) {
    return (
      <div className="empty-state">
        <div className="empty-state-icon">🛡️</div>
        <div className="empty-state-text">No stress tests triggered yet</div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', textAlign: 'center', maxWidth: 260 }}>
          Start a <strong>Replay</strong> or <strong>Live</strong> session. A stress test fires automatically
          when a signal reaches <strong>impact ≥ 7</strong> and <strong>confidence ≥ 55%</strong>.
          <br /><br />
          Or use <code style={{ fontFamily: 'var(--font-mono)', fontSize: 10 }}>POST /stress/run</code> for an instant demo.
        </div>
      </div>
    )
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>
      {/* Scenario header */}
      <div style={{
        display: 'flex', alignItems: 'flex-start', gap: 8,
        padding: '8px 12px', borderRadius: 8,
        background: 'rgba(239,68,68,0.07)', border: '1px solid rgba(239,68,68,0.2)',
      }}>
        <ShieldAlert size={15} color="#ef4444" style={{ marginTop: 1, flexShrink: 0 }} />
        <div>
          <div style={{ fontSize: 12, fontWeight: 700, color: '#fca5a5' }}>
            {latest.scenario_name}
          </div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginTop: 2 }}>
            📚 {latest.scenario_name} — historical analog available in scenarios.yaml
          </div>
        </div>
        <div style={{ marginLeft: 'auto', fontSize: 10, color: 'var(--text-muted)', textAlign: 'right', whiteSpace: 'nowrap' }}>
          {new Date(latest.triggered_at).toLocaleTimeString()}
        </div>
      </div>

      {/* Before/after bars */}
      <PortfolioBars
        before={latest.value_before}
        after={latest.value_after}
        loss={latest.loss}
        lossPct={latest.loss_pct}
      />

      {/* Asset class breakdown */}
      {latest.breakdown?.length > 0 && <BreakdownBars breakdown={latest.breakdown} />}

      {/* Triggering signal */}
      <div>
        <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
          Triggering Signal
        </div>
        <TriggerInfo result={latest} />
      </div>

      {/* History mini-list */}
      {history.length > 1 && (
        <div>
          <div style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 6 }}>
            Recent Stress Events ({history.length})
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
            {history.slice(1, 5).map((r, i) => (
              <div key={i} style={{
                display: 'flex', alignItems: 'center', gap: 8,
                padding: '5px 8px', borderRadius: 5,
                background: 'var(--bg-elevated)', fontSize: 11,
              }}>
                <span style={{ fontWeight: 700, fontFamily: 'var(--font-mono)', width: 36 }}>
                  {r.signal_entity}
                </span>
                <span style={{ color: 'var(--text-muted)', flex: 1 }}>{r.event_type}</span>
                <span style={{ color: '#fca5a5', fontFamily: 'var(--font-mono)' }}>
                  −{fmt$(r.loss)}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}
