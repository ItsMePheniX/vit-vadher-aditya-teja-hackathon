/**
 * WeightChart.jsx — Stacked area chart showing rebalancer weights over time.
 */
import { useMemo } from 'react'
import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer
} from 'recharts'

// Distinct colors for 15 tickers
const COLORS = [
  '#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6',
  '#ec4899', '#06b6d4', '#f97316', '#6366f1', '#14b8a6',
  '#f43f5e', '#a855f7', '#84cc16', '#0ea5e9', '#d946ef'
]

export function WeightChart({ history = [] }) {
  const data = useMemo(() => {
    return history.map(snap => {
      const point = { time: new Date(snap.ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' }) }
      for (const [ticker, w] of Object.entries(snap.weights)) {
        point[ticker] = w * 100 // convert to percentage
      }
      return point
    })
  }, [history])

  // Get unique tickers from the latest snapshot
  const tickers = useMemo(() => {
    if (history.length === 0) return []
    return Object.keys(history[history.length - 1].weights).sort()
  }, [history])

  if (data.length === 0) {
    return (
      <div className="empty-state" style={{ height: 180, opacity: 0.5 }}>
        <div style={{ fontSize: 24, marginBottom: 8 }}>⚖️</div>
        <div>Awaiting weight snapshots...</div>
      </div>
    )
  }

  return (
    <div style={{ height: 200, width: '100%' }}>
      <ResponsiveContainer width="100%" height="100%">
        <AreaChart
          data={data}
          margin={{ top: 5, right: 0, left: -20, bottom: 0 }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.05)" vertical={false} />
          <XAxis 
            dataKey="time" 
            stroke="rgba(255,255,255,0.2)" 
            tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }}
            tickMargin={8}
            minTickGap={20}
          />
          <YAxis 
            stroke="rgba(255,255,255,0.2)" 
            tick={{ fill: 'rgba(255,255,255,0.4)', fontSize: 10 }}
            tickFormatter={(v) => `${v}%`}
            domain={[0, 100]}
          />
          <Tooltip 
            contentStyle={{ 
              backgroundColor: 'var(--bg-card)', 
              borderColor: 'rgba(255,255,255,0.1)',
              borderRadius: 8,
              boxShadow: '0 4px 12px rgba(0,0,0,0.5)'
            }}
            itemStyle={{ fontSize: 11 }}
            labelStyle={{ color: 'var(--text-muted)', marginBottom: 4, fontSize: 11 }}
            formatter={(val) => [`${val.toFixed(1)}%`]}
          />
          {tickers.map((ticker, i) => (
            <Area 
              key={ticker}
              type="monotone" 
              dataKey={ticker} 
              stackId="1" 
              stroke={COLORS[i % COLORS.length]} 
              fill={COLORS[i % COLORS.length]} 
              fillOpacity={0.6}
              isAnimationActive={false}
            />
          ))}
        </AreaChart>
      </ResponsiveContainer>
    </div>
  )
}
