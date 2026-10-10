/** AlertPanel — shows derived alerts from high-impact / strongly-negative signals */
import { deriveAlerts, fmtTime } from '../utils/helpers.js'

export function AlertPanel({ signals }) {
  const alerts = deriveAlerts(signals)

  if (!alerts.length) {
    return (
      <div className="no-alerts">
        <div>No high-impact alerts</div>
        <div style={{ fontSize: 10 }}>Alerts trigger at impact ≥ 7 or sentiment ≤ −0.40</div>
      </div>
    )
  }

  return (
    <div className="alert-list">
      {alerts.map(a => (
        <div key={a.id} className={`alert-item ${a.severity}`}>
          <div className="alert-icon" style={{ fontSize: 10, fontWeight: 700 }}>
            {a.severity === 'HIGH' ? 'HIGH' : 'MED'}
          </div>
          <div className="alert-body">
            <div className="alert-entity">
              {a.entity}
              <span style={{ fontWeight: 400, marginLeft: 6, fontSize: 11, color: 'var(--text-secondary)' }}>
                {a.event_type}
              </span>
            </div>
            <div className="alert-reason">{a.reason}</div>
          </div>
          <div className="alert-time">{fmtTime(a.ts)}</div>
        </div>
      ))}
    </div>
  )
}
