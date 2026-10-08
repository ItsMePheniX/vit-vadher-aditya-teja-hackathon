/** Shared utility helpers */

/** Format a UTC ISO timestamp as HH:MM:SS */
export function fmtTime(iso) {
  try {
    return new Date(iso).toLocaleTimeString('en-US', {
      hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false,
    })
  } catch { return '—' }
}

/** Format as MM-DD HH:MM */
export function fmtDatetime(iso) {
  try {
    const d = new Date(iso)
    return `${String(d.getMonth()+1).padStart(2,'0')}-${String(d.getDate()).padStart(2,'0')} ${String(d.getHours()).padStart(2,'0')}:${String(d.getMinutes()).padStart(2,'0')}`
  } catch { return '—' }
}

/** Return CSS class for sentiment direction */
export function sentClass(score) {
  if (score >  0.05) return 'sent-pos'
  if (score < -0.05) return 'sent-neg'
  return 'sent-neu'
}

/** Return CSS color var for sentiment */
export function sentColor(score) {
  if (score >  0.05) return 'var(--positive)'
  if (score < -0.05) return 'var(--negative)'
  return 'var(--neutral)'
}

/** Return background CSS for impact level 1-10 */
export function impactBg(impact) {
  if (impact >= 8) return 'rgba(239,68,68,0.25)'
  if (impact >= 5) return 'rgba(245,158,11,0.25)'
  return 'rgba(59,130,246,0.15)'
}
export function impactFg(impact) {
  if (impact >= 8) return '#fca5a5'
  if (impact >= 5) return '#fcd34d'
  return '#93c5fd'
}

/** Derive per-entity sentiment summary from signals array */
export function entitySummary(signals) {
  const map = {}
  for (const s of signals) {
    if (!map[s.entity]) map[s.entity] = { entity: s.entity, scores: [], impacts: [], count: 0 }
    map[s.entity].scores.push(s.sentiment)
    map[s.entity].impacts.push(s.impact)
    map[s.entity].count++
  }
  return Object.values(map)
    .map(e => ({
      entity: e.entity,
      avgSent: e.scores.reduce((a,b)=>a+b,0) / e.scores.length,
      avgImpact: e.impacts.reduce((a,b)=>a+b,0) / e.impacts.length,
      count: e.count,
    }))
    .sort((a,b) => b.avgImpact - a.avgImpact)
}

/** Aggregate signals into time buckets for the chart (hourly buckets) */
export function bucketByHour(signals, n = 24) {
  if (!signals.length) return []
  const now = Date.now()
  const hourMs = 3_600_000
  const buckets = Array.from({ length: n }, (_, i) => ({
    label: fmtTime(new Date(now - (n-1-i) * hourMs).toISOString()).slice(0,5),
    positive: 0, negative: 0, neutral: 0, count: 0,
  }))
  for (const s of signals) {
    const age = now - new Date(s.ts).getTime()
    const idx = Math.floor((n * hourMs - age) / hourMs)
    if (idx < 0 || idx >= n) continue
    if (s.sentiment >  0.1) buckets[idx].positive++
    else if (s.sentiment < -0.1) buckets[idx].negative++
    else buckets[idx].neutral++
    buckets[idx].count++
  }
  return buckets
}

/** Derive alert-like rows from recent high-impact signals */
export function deriveAlerts(signals, impactThresh=7, sentThresh=-0.4) {
  return signals
    .filter(s => s.impact >= impactThresh || s.sentiment <= sentThresh)
    .slice(0, 15)
    .map(s => ({
      id: s.id,
      entity: s.entity,
      severity: s.impact >= 8 ? 'HIGH' : 'MEDIUM',
      reason: `impact=${s.impact}, sentiment=${s.sentiment.toFixed(3)}`,
      event_type: s.event_type,
      ts: s.ts,
    }))
}
