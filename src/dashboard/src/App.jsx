/**
 * App.jsx — Financial Risk NLP Engine Dashboard
 *
 * Layout (projector-friendly, high contrast, large fonts):
 *   TopBar
 *   ┌─────────────────────────────────┬──────────────────────────────┐
 *   │ Signal Stream (live feed table) │ Mode Controls (Live/Replay)  │
 *   │                                 │ Stress-Test Panel            │
 *   ├─────────────────────────────────┼──────────────────────────────┤
 *   │ Sentiment chart + Entity grid   │ Alerts + KPI grid            │
 *   └─────────────────────────────────┴──────────────────────────────┘
 *
 * All data comes from the API/WebSocket — no fake data is ever shown;
 * disconnected state shows a clear empty state per section.
 */
import { useState, useCallback } from 'react'
import { TopBar }        from './components/TopBar.jsx'
import { KpiGrid }       from './components/KpiGrid.jsx'
import { SignalTable }   from './components/SignalTable.jsx'
import { SentimentChart } from './components/SentimentChart.jsx'
import { EntityHeatmap } from './components/EntityHeatmap.jsx'
import { LiveFeed }      from './components/LiveFeed.jsx'
import { AlertPanel }    from './components/AlertPanel.jsx'
import { ModeToggle }    from './components/ModeToggle.jsx'
import { StressPanel }   from './components/StressPanel.jsx'
import { WeightChart }   from './components/WeightChart.jsx'
import { useSignalStream } from './hooks/useSignalStream.js'
import { usePortfolioRisk } from './hooks/usePortfolioRisk.js'
import { useStressStream }  from './hooks/useStressStream.js'
import { useModeControl }   from './hooks/useModeControl.js'
import { useWeightStream }  from './hooks/useWeightStream.js'

export default function App() {
  // ── Data hooks ────────────────────────────────────────────────────────────
  const { signals, status: sigStatus, latestId } = useSignalStream()
  const risk = usePortfolioRisk(signals)
  const { latest: stressLatest, history: stressHistory } = useStressStream()
  const { history: weightHistory } = useWeightStream()
  const {
    health, mode, liveAvailable, episodes,
    startLive, stopLive, startReplay, stopReplay,
    modeLoading, modeError,
  } = useModeControl()

  // ── Ingest form ───────────────────────────────────────────────────────────
  const [ingestText,   setIngestText]   = useState('')
  const [ingestStatus, setIngestStatus] = useState('')
  const [submitting,   setSubmitting]   = useState(false)

  const handleIngest = useCallback(async (e) => {
    e.preventDefault()
    if (!ingestText.trim()) return
    setSubmitting(true)
    setIngestStatus('')
    try {
      const resp = await fetch(`http://${window.location.hostname}:8000/signals/ingest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: ingestText, source: 'dashboard' }),
      })
      if (resp.ok) {
        const result = await resp.json()
        setIngestStatus(`Success: ${result.length} signal(s) produced`)
        setIngestText('')
      } else {
        setIngestStatus('Error: API error — is the server running?')
      }
    } catch {
      setIngestStatus('Error: Cannot reach API (run `make run`)')
    }
    setSubmitting(false)
  }, [ingestText])

  // Connection state
  const apiOnline = health !== null

  return (
    <div className="app-layout">
      <TopBar
        status={sigStatus}
        signalCount={signals.length}
        mode={mode}
        apiOnline={apiOnline}
      />

      <main className="main-content">
        {/* ── Row 1: KPIs ──────────────────────────────────────────────── */}
        <KpiGrid signals={signals} risk={risk} />

        {/* ── Row 2: Signal stream + right column ──────────────────────── */}
        <div className="dashboard-row two-col">

          {/* Left: Signal table */}
          <div className="card signal-stream-card">
            <div className="card-header">
              <div className="card-title">
                Signal Stream
                <span style={{
                  marginLeft: 8, fontSize: 10, padding: '2px 8px',
                  background: sigStatus === 'connected'
                    ? 'rgba(16,185,129,0.12)' : 'rgba(239,68,68,0.12)',
                  color: sigStatus === 'connected' ? '#6ee7b7' : '#fca5a5',
                  borderRadius: 4,
                  border: `1px solid ${sigStatus === 'connected'
                    ? 'rgba(16,185,129,0.3)' : 'rgba(239,68,68,0.3)'}`,
                }}>
                  {sigStatus === 'connected' ? 'WS LIVE' : sigStatus === 'connecting' ? 'CONNECTING…' : 'OFFLINE'}
                </span>
              </div>
              <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
                {signals.length} signals
              </span>
            </div>
            <SignalTable signals={signals.slice(0, 50)} latestId={latestId} />
          </div>

          {/* Right column */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {/* Mode controls */}
            <div className="card">
              <div className="card-header">
                <div className="card-title">Engine Mode</div>
              </div>
              <ModeToggle
                mode={mode}
                episodes={episodes}
                liveAvailable={liveAvailable}
                modeLoading={modeLoading}
                modeError={modeError}
                startLive={startLive}
                stopLive={stopLive}
                startReplay={startReplay}
                stopReplay={stopReplay}
              />
            </div>

            {/* Stress-test panel */}
            <div className="card" style={{ flex: 1 }}>
              <div className="card-header">
                <div className="card-title">
                  Stress Test
                  {stressLatest && (
                    <span style={{
                      marginLeft: 8, fontSize: 10, padding: '2px 8px',
                      background: 'rgba(239,68,68,0.12)', color: '#fca5a5',
                      borderRadius: 4, border: '1px solid rgba(239,68,68,0.3)',
                    }}>
                      TRIGGERED
                    </span>
                  )}
                </div>
                {stressLatest && (
                  <span style={{ fontSize: 11, color: '#fca5a5', fontFamily: 'var(--font-mono)' }}>
                    −{stressLatest.loss >= 1e6
                      ? `$${(stressLatest.loss / 1e6).toFixed(1)}M`
                      : `$${stressLatest.loss.toLocaleString(undefined, { maximumFractionDigits: 0 })}`}
                  </span>
                )}
              </div>
              <StressPanel latest={stressLatest} history={stressHistory} />
            </div>

          </div>
        </div>

        {/* ── Row 3: Charts + Feed + Alerts ────────────────────────────── */}
        <div className="dashboard-row three-col">

          {/* Sentiment chart + Entity heatmap */}
          <div className="card chart-card" style={{ display: 'flex', flexDirection: 'column', gap: 20 }}>
            <div>
              <div className="card-header">
                <div className="card-title">Sentiment Over Time</div>
              </div>
              <SentimentChart signals={signals} />
            </div>
            <div>
              <div className="card-header" style={{ marginTop: 24 }}>
                <div className="card-title">Entity Sentiment</div>
              </div>
              <EntityHeatmap signals={signals} />
            </div>
            <div>
              <div className="card-header" style={{ marginTop: 24 }}>
                <div className="card-title">Index Weights</div>
              </div>
              <WeightChart history={weightHistory} />
            </div>
          </div>

          {/* Live feed + ingest */}
          <div className="card feed-card">
            <div className="card-header">
              <div className="card-title">Live Event Feed</div>
            </div>

            <form onSubmit={handleIngest} style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
              <input
                id="ingest-input"
                type="text"
                value={ingestText}
                onChange={e => setIngestText(e.target.value)}
                placeholder="Paste a headline to analyse…"
                style={{
                  flex: 1, background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                  borderRadius: 6, padding: '7px 12px', color: 'var(--text-primary)',
                  fontSize: 12, fontFamily: 'var(--font-sans)', outline: 'none',
                }}
              />
              <button
                type="submit"
                disabled={submitting || !ingestText.trim()}
                style={{
                  padding: '7px 16px', borderRadius: 6, border: 'none',
                  background: submitting ? 'var(--bg-elevated)' : 'var(--accent-blue)',
                  color: 'white', fontSize: 12, fontWeight: 600, cursor: 'pointer',
                  opacity: submitting ? 0.6 : 1,
                }}
              >
                {submitting ? '…' : 'Analyse'}
              </button>
            </form>
            {ingestStatus && (
              <div style={{ fontSize: 11, marginBottom: 8, color: 'var(--text-secondary)' }}>
                {ingestStatus}
              </div>
            )}
            <LiveFeed signals={signals} />
          </div>

          {/* Alerts panel */}
          <div className="card alerts-card">
            <div className="card-header">
              <div className="card-title">Risk Alerts</div>
              <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>impact ≥ 7 or sent ≤ −0.40</span>
            </div>
            <AlertPanel signals={signals} />
          </div>

        </div>
      </main>
    </div>
  )
}
