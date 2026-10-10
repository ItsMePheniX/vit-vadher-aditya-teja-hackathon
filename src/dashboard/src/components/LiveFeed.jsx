/** LiveFeed — scrolling event ticker styled like a Bloomberg terminal feed */
import { sentClass, fmtTime } from '../utils/helpers.js'

function FeedItem({ signal }) {
  const sentDir = signal.sentiment > 0.05 ? 'sent-positive' : signal.sentiment < -0.05 ? 'sent-negative' : ''
  return (
    <div className={`feed-item ${sentDir}`}>
      <div className="feed-ts">{fmtTime(signal.ts)}</div>
      <div className="feed-body">
        <div className="feed-header">
          <span className={`badge ${signal.entity === 'MARKET' ? 'badge-market' : 'badge-entity'}`}>
            {signal.entity}
          </span>
          <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>{signal.event_type}</span>
          <span style={{
            fontSize: 10,
            fontFamily: 'var(--font-mono)',
            color: signal.sentiment > 0 ? 'var(--positive)' : 'var(--negative)',
          }}>
            {signal.sentiment >= 0 ? '+' : ''}{signal.sentiment.toFixed(3)}
          </span>
        </div>
        <div className="feed-text">
          {(signal.evidence || []).join(' ') || signal.sources?.join(' ') || '—'}
        </div>
      </div>
    </div>
  )
}

export function LiveFeed({ signals }) {
  if (!signals.length) {
    return (
      <div className="empty-state">
        <div className="empty-state-text">Live event feed will appear here</div>
      </div>
    )
  }

  return (
    <div className="feed-list">
      {signals.slice(0, 40).map(s => (
        <FeedItem key={s.id} signal={s} />
      ))}
    </div>
  )
}
