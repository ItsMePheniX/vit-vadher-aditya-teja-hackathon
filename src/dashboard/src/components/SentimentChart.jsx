/** SentimentChart — hourly bar chart of positive/negative/neutral signal counts */
import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, Legend, ResponsiveContainer
} from 'recharts'
import { bucketByHour } from '../utils/helpers.js'

const CustomTooltip = ({ active, payload, label }) => {
  if (!active || !payload?.length) return null
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '10px 14px',
      fontSize: 12,
    }}>
      <div style={{ color: 'var(--text-muted)', marginBottom: 6 }}>{label}</div>
      {payload.map(p => (
        <div key={p.name} style={{ color: p.fill, marginBottom: 2 }}>
          {p.name}: {p.value}
        </div>
      ))}
    </div>
  )
}

export function SentimentChart({ signals }) {
  const data = bucketByHour(signals, 12)

  if (!signals.length) {
    return (
      <div className="empty-state" style={{ height: 200 }}>
        <div className="empty-state-icon">📊</div>
        <div className="empty-state-text">Chart populates as signals arrive</div>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={220}>
      <BarChart data={data} barCategoryGap="30%" margin={{ top: 4, right: 4, bottom: 0, left: -16 }}>
        <CartesianGrid strokeDasharray="3 3" vertical={false} />
        <XAxis dataKey="label" tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
        <YAxis allowDecimals={false} tick={{ fontSize: 10, fill: 'var(--text-muted)' }} />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.04)' }} />
        <Legend iconType="circle" iconSize={8} wrapperStyle={{ fontSize: 11 }} />
        <Bar dataKey="positive" name="Positive" fill="var(--positive)" radius={[3,3,0,0]} />
        <Bar dataKey="neutral"  name="Neutral"  fill="var(--neutral)"  radius={[3,3,0,0]} />
        <Bar dataKey="negative" name="Negative" fill="var(--negative)" radius={[3,3,0,0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
