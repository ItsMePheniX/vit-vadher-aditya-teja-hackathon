/**
 * ModeToggle — Live / Replay mode selector with status and controls.
 *
 * Shows:
 *  - Current mode badge (IDLE / LIVE / REPLAY)
 *  - Live toggle with clear disabled message when unavailable
 *  - Replay panel: episode picker, speed slider, start/stop
 *  - Simulated time clock during replay
 */
import { useState, useEffect, useRef } from 'react'
import { Radio, Play, Square, Tv2, Wifi, WifiOff, Clock } from 'lucide-react'

const SPEED_MARKS = [1, 10, 30, 60, 120, 300]

function fmt(v) { return v >= 60 ? `${v / 60}m` : `${v}s` }

function SimClock({ active, speed }) {
  const [simTime, setSimTime] = useState(null)
  const startWall = useRef(null)
  const startSim  = useRef(null)

  useEffect(() => {
    if (!active) { setSimTime(null); startWall.current = null; return }
    if (!startWall.current) {
      startWall.current = Date.now()
      // Replay always starts from SVB episode start (2023-03-08)
      startSim.current  = new Date('2023-03-08T09:00:00Z').getTime()
    }
    const id = setInterval(() => {
      const elapsed = Date.now() - startWall.current   // ms
      const simMs   = elapsed * speed                  // compressed ms
      setSimTime(new Date(startSim.current + simMs))
    }, 250)
    return () => clearInterval(id)
  }, [active, speed])

  if (!simTime) return null
  return (
    <div style={{
      display: 'flex', alignItems: 'center', gap: 6,
      padding: '4px 10px', borderRadius: 6,
      background: 'rgba(6,182,212,0.1)', border: '1px solid rgba(6,182,212,0.25)',
    }}>
      <Clock size={13} color="var(--accent-cyan)" />
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent-cyan)' }}>
        {simTime.toISOString().replace('T', ' ').slice(0, 19)} UTC
      </span>
      <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>sim</span>
    </div>
  )
}

export function ModeToggle({
  mode, episodes, liveAvailable, modeLoading, modeError,
  startLive, stopLive, startReplay, stopReplay,
}) {
  const [tab,     setTab]     = useState('live')      // 'live' | 'replay'
  const [episode, setEpisode] = useState('')
  const [speed,   setSpeed]   = useState(60)

  useEffect(() => {
    if (episodes.length && !episode) setEpisode(episodes[0])
  }, [episodes, episode])

  const isIdle   = mode === 'idle'
  const isLive   = mode === 'live'
  const isReplay = mode === 'replay'

  const modeBadge = {
    idle:   { label: 'IDLE',   bg: 'rgba(148,163,184,0.1)', fg: '#94a3b8' },
    live:   { label: 'LIVE',   bg: 'rgba(16,185,129,0.15)', fg: '#6ee7b7' },
    replay: { label: 'REPLAY', bg: 'rgba(59,130,246,0.15)', fg: '#93c5fd' },
  }[mode] || { label: mode.toUpperCase(), bg: 'rgba(148,163,184,0.1)', fg: '#94a3b8' }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      {/* Header row */}
      <div style={{ display: 'flex', alignItems: 'center', gap: 10, flexWrap: 'wrap' }}>
        <span style={{
          padding: '3px 10px', borderRadius: 20, fontSize: 11, fontWeight: 700,
          letterSpacing: '0.08em', background: modeBadge.bg, color: modeBadge.fg,
          border: `1px solid ${modeBadge.fg}33`,
        }}>
          {modeBadge.label}
        </span>
        {isReplay && <SimClock active speed={speed} />}
        {modeLoading && (
          <span style={{ fontSize: 11, color: 'var(--text-muted)', fontStyle: 'italic' }}>
            Applying…
          </span>
        )}
      </div>

      {/* Error */}
      {modeError && (
        <div style={{
          fontSize: 11, color: '#fca5a5', padding: '6px 10px', borderRadius: 6,
          background: 'rgba(239,68,68,0.1)', border: '1px solid rgba(239,68,68,0.2)',
        }}>
          Error: {modeError}
        </div>
      )}

      {/* Tab bar */}
      <div style={{ display: 'flex', gap: 4, background: 'var(--bg-elevated)', borderRadius: 8, padding: 3 }}>
        {['live', 'replay'].map(t => (
          <button key={t} onClick={() => setTab(t)} style={{
            flex: 1, padding: '6px 0', border: 'none', borderRadius: 6, cursor: 'pointer',
            fontSize: 12, fontWeight: 600, letterSpacing: '0.04em',
            background: tab === t ? 'var(--bg-card)' : 'transparent',
            color: tab === t ? 'var(--text-primary)' : 'var(--text-muted)',
            transition: 'all 0.15s',
          }}>
            {t === 'live' ? 'Live' : 'Replay'}
          </button>
        ))}
      </div>

      {/* Live tab */}
      {tab === 'live' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
          {/* Availability message */}
          <div style={{
            display: 'flex', alignItems: 'flex-start', gap: 8,
            padding: '8px 10px', borderRadius: 6,
            background: liveAvailable.available
              ? 'rgba(16,185,129,0.07)' : 'rgba(245,158,11,0.07)',
            border: `1px solid ${liveAvailable.available
              ? 'rgba(16,185,129,0.2)' : 'rgba(245,158,11,0.2)'}`,
          }}>
            {liveAvailable.available
              ? <Wifi size={13} color="var(--accent-green)" style={{ marginTop: 2, flexShrink: 0 }} />
              : <WifiOff size={13} color="var(--accent-amber)" style={{ marginTop: 2, flexShrink: 0 }} />}
            <span style={{ fontSize: 11, color: 'var(--text-secondary)', lineHeight: 1.5 }}>
              {liveAvailable.reason}
            </span>
          </div>

          {/* Live controls */}
          {isLive ? (
            <button onClick={stopLive} disabled={modeLoading} style={btnStyle('#ef4444', modeLoading)}>
              <Square size={13} /> Stop Live
            </button>
          ) : (
            <button
              onClick={startLive}
              disabled={modeLoading || !liveAvailable.available || isReplay}
              style={btnStyle('var(--accent-green)', modeLoading || !liveAvailable.available || isReplay)}
            >
              <Radio size={13} /> Start Live
            </button>
          )}
          {isReplay && (
            <p style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: -4 }}>
              Stop the active replay first.
            </p>
          )}
        </div>
      )}

      {/* Replay tab */}
      {tab === 'replay' && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {/* Episode picker */}
          <div>
            <label style={{ fontSize: 11, color: 'var(--text-muted)', marginBottom: 4, display: 'block' }}>
              Episode
            </label>
            <select
              value={episode}
              onChange={e => setEpisode(e.target.value)}
              disabled={isReplay || isLive}
              style={{
                width: '100%', padding: '7px 10px', borderRadius: 6,
                background: 'var(--bg-elevated)', border: '1px solid var(--border)',
                color: 'var(--text-primary)', fontSize: 12, outline: 'none',
                cursor: isReplay || isLive ? 'not-allowed' : 'pointer',
                fontFamily: 'var(--font-sans)',
              }}
            >
              {episodes.length === 0 && <option value="">No episodes available</option>}
              {episodes.map(ep => (
                <option key={ep} value={ep}>{ep.replace(/_/g, ' ')}</option>
              ))}
            </select>
          </div>

          {/* Speed slider */}
          <div>
            <div style={{ display: 'flex', justifyContent: 'space-between', marginBottom: 4 }}>
              <label style={{ fontSize: 11, color: 'var(--text-muted)' }}>Playback speed</label>
              <span style={{ fontSize: 12, fontWeight: 700, color: 'var(--accent-cyan)', fontFamily: 'var(--font-mono)' }}>
                {speed}×
              </span>
            </div>
            <input
              type="range" min={0} max={SPEED_MARKS.length - 1} step={1}
              value={SPEED_MARKS.indexOf(speed) === -1 ? 3 : SPEED_MARKS.indexOf(speed)}
              onChange={e => setSpeed(SPEED_MARKS[+e.target.value])}
              disabled={isReplay || isLive}
              style={{ width: '100%', accentColor: 'var(--accent-cyan)' }}
            />
            <div style={{ display: 'flex', justifyContent: 'space-between', marginTop: 2 }}>
              {SPEED_MARKS.map(s => (
                <span key={s} style={{ fontSize: 9, color: 'var(--text-muted)' }}>{s}×</span>
              ))}
            </div>
          </div>

          {/* Controls */}
          {isReplay ? (
            <button onClick={stopReplay} disabled={modeLoading} style={btnStyle('#ef4444', modeLoading)}>
              <Square size={13} /> Stop Replay
            </button>
          ) : (
            <button
              onClick={() => startReplay(episode, speed)}
              disabled={modeLoading || !episode || isLive}
              style={btnStyle('var(--accent-blue)', modeLoading || !episode || isLive)}
            >
              <Play size={13} /> Start Replay
            </button>
          )}
          {isLive && (
            <p style={{ fontSize: 10, color: 'var(--text-muted)', marginTop: -4 }}>
              Stop the active live session first.
            </p>
          )}
        </div>
      )}
    </div>
  )
}

function btnStyle(bg, disabled) {
  return {
    display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 6,
    width: '100%', padding: '9px 0', border: 'none', borderRadius: 7,
    background: disabled ? 'var(--bg-elevated)' : bg,
    color: disabled ? 'var(--text-muted)' : 'white',
    fontSize: 13, fontWeight: 700, cursor: disabled ? 'not-allowed' : 'pointer',
    transition: 'all 0.15s',
    opacity: disabled ? 0.6 : 1,
  }
}
