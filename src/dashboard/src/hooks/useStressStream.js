/**
 * useStressStream — WebSocket hook for real-time StressResult updates.
 *
 * Connects to ws://localhost:8000/ws/stress
 * Reconnects automatically with exponential backoff.
 *
 * Returns:
 *   latest   – most recent StressResult or null
 *   history  – array of StressResults (newest first, max 20)
 *   status   – 'connecting' | 'connected' | 'disconnected'
 */
import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL     = `ws://${window.location.hostname}:8000/ws/stress`
const MAX_HISTORY = 20
const BACKOFF_BASE = 1500
const BACKOFF_MAX  = 30000

export function useStressStream() {
  const [history, setHistory] = useState([])
  const [status,  setStatus]  = useState('connecting')

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
        const result = JSON.parse(ev.data)
        setHistory(prev => [result, ...prev].slice(0, MAX_HISTORY))
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
    connect()
    return () => {
      clearTimeout(retryTimer.current)
      wsRef.current?.close()
    }
  }, [connect])

  return {
    latest:  history[0] ?? null,
    history,
    status,
  }
}
