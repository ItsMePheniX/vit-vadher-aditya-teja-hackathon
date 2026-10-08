/** EntityHeatmap — mini bar chart showing avg sentiment per entity */
import {
  BarChart, Bar, XAxis, YAxis, Cell,
  Tooltip, ResponsiveContainer
} from 'recharts'
import { entitySummary } from '../utils/helpers.js'

const CustomTooltip = ({ active, payload }) => {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div style={{
      background: 'var(--bg-elevated)',
      border: '1px solid var(--border)',
      borderRadius: 8,
      padding: '8px 12px',
      fontSize: 11,
    }}>
      <div style={{ fontWeight: 600, marginBottom: 4 }}>{d.entity}</div>
      <div style={{ color: 'var(--text-secondary)' }}>Avg sentiment: {d.avgSent.toFixed(3)}</div>
      <div style={{ color: 'var(--text-secondary)' }}>Avg impact: {d.avgImpact.toFixed(1)}</div>
      <div style={{ color: 'var(--text-muted)' }}>{d.count} signal(s)</div>
    </div>
  )
}

export function EntityHeatmap({ signals }) {
  const data = entitySummary(signals).slice(0, 10)

  if (!data.length) {
    return (
      <div className="empty-state" style={{ height: 180 }}>
        <div className="empty-state-icon">🏢</div>
        <div className="empty-state-text">Entity breakdown will appear here</div>
      </div>
    )
  }

  return (
    <ResponsiveContainer width="100%" height={200}>
      <BarChart
        data={data}
        layout="vertical"
        margin={{ top: 0, right: 8, bottom: 0, left: 0 }}
        barCategoryGap="20%"
      >
        <XAxis
          type="number"
          domain={[-1, 1]}
          tickCount={5}
          tick={{ fontSize: 9, fill: 'var(--text-muted)' }}
          tickFormatter={v => v.toFixed(1)}
        />
        <YAxis
          type="category"
          dataKey="entity"
          width={46}
          tick={{ fontSize: 10, fill: 'var(--text-secondary)', fontFamily: 'var(--font-mono)' }}
        />
        <Tooltip content={<CustomTooltip />} cursor={{ fill: 'rgba(255,255,255,0.03)' }} />
        <Bar dataKey="avgSent" name="Avg Sentiment" radius={[0,3,3,0]}>
          {data.map((entry, i) => (
            <Cell
              key={i}
              fill={entry.avgSent > 0.05 ? 'var(--positive)' : entry.avgSent < -0.05 ? 'var(--negative)' : 'var(--neutral)'}
              fillOpacity={0.8}
            />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
