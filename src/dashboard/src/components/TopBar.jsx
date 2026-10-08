/** TopBar — sticky header with brand, mode badge, signal count, and WS status pill */

const MODE_STYLE = {
  idle:   { label: 'IDLE',   bg: 'rgba(148,163,184,0.1)', fg: '#94a3b8' },
  live:   { label: 'LIVE',   bg: 'rgba(16,185,129,0.15)', fg: '#6ee7b7' },
  replay: { label: 'REPLAY', bg: 'rgba(59,130,246,0.15)', fg: '#93c5fd' },
}

export function TopBar({ status, signalCount, mode = 'idle', apiOnline = false }) {
  const modeStyle = MODE_STYLE[mode] || MODE_STYLE.idle
  const wsConnected = status === 'connected'

  return (
    <header className="topbar">
      <div className="topbar-brand">
        <div className="topbar-logo">📡</div>
        <div>
          <div className="topbar-title">FinRisk NLP Engine</div>
          <div className="topbar-subtitle">S&amp;P Global &amp; Crisil Campus Hackathon 2026</div>
        </div>
      </div>

      <div className="topbar-right">
        {/* Mode badge */}
        <span style={{
          padding: '3px 10px', borderRadius: 20,
          fontSize: 11, fontWeight: 700, letterSpacing: '0.08em',
          background: modeStyle.bg, color: modeStyle.fg,
          border: `1px solid ${modeStyle.fg}33`,
        }}>
          {modeStyle.label}
        </span>

        {/* Signal count */}
        <span style={{ fontSize: 11, color: 'var(--text-muted)' }}>
          {signalCount} signals
        </span>

        {/* API status */}
        <span style={{ fontSize: 11, color: apiOnline ? 'var(--text-muted)' : '#fca5a5' }}>
          API {apiOnline ? '✓' : '✗'}
        </span>

        {/* WS status */}
        <div className={`status-pill ${wsConnected ? 'connected' : 'disconnected'}`}>
          <div className="status-dot" />
          {wsConnected ? 'WS Live' : status === 'connecting' ? 'Connecting…' : 'Offline'}
        </div>
      </div>
    </header>
  )
}
