/**
 * useSignalStream — WebSocket hook for live signal updates.
 *
 * Connects to ws://localhost:8000/signals/stream (proxied via Vite as /api/signals/stream).
 * Reconnects automatically with exponential backoff on disconnect.
 *
 * Returns:
 *   signals   – array of Signal objects (newest first, max MAX_SIGNALS)
 *   status    – 'connecting' | 'connected' | 'disconnected'
 *   latestId  – id of the most recently received signal (for animations)
 */
import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL = `ws://${window.location.hostname}:8000/ws/signals`
const MAX_SIGNALS = 200
const BACKOFF_BASE = 1500
const BACKOFF_MAX  = 30000

export function useSignalStream() {
  const [signals,  setSignals]  = useState([])
  const [status,   setStatus]   = useState('connecting')
  const [latestId, setLatestId] = useState(null)

  const wsRef      = useRef(null)
  const retryCount = useRef(0)
  const retryTimer = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return

    setStatus('connecting')
    const ws = new WebSocket(WS_URL)
    wsRef.current = ws

    ws.onopen = () => {
      setStatus('connected')
      retryCount.current = 0
    }

    ws.onmessage = (ev) => {
      try {
        const signal = JSON.parse(ev.data)
        setLatestId(signal.id)
        setSignals(prev => [signal, ...prev].slice(0, MAX_SIGNALS))
      } catch (_) {}
    }

    ws.onclose = () => {
      setStatus('disconnected')
      const delay = Math.min(BACKOFF_BASE * 2 ** retryCount.current, BACKOFF_MAX)
      retryCount.current++
      retryTimer.current = setTimeout(connect, delay)
    }

    ws.onerror = () => ws.close()
  }, [])

  useEffect(() => {
    fetch(`http://${window.location.hostname}:8000/signals?limit=50`)
      .then(res => res.ok ? res.json() : [])
      .then(data => {
        if (Array.isArray(data) && data.length > 0) {
          setSignals(prev => {
            const existingIds = new Set(prev.map(s => s.id))
            const newItems = data.filter(s => !existingIds.has(s.id))
            return [...prev, ...newItems].slice(0, MAX_SIGNALS)
          })
        }
      })
      .catch(() => {})
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(retryTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return { signals, status, latestId }
}
