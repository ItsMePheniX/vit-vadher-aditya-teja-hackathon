/**
 * useWeightStream — WebSocket hook for real-time Rebalancer weight snapshots.
 *
 * Connects to ws://localhost:8000/ws/weights
 * Reconnects automatically with exponential backoff.
 *
 * Returns:
 *   history  – array of WeightSnapshots (oldest first for charting)
 *   status   – 'connecting' | 'connected' | 'disconnected'
 */
import { useState, useEffect, useRef, useCallback } from 'react'

const WS_URL     = `ws://${window.location.hostname}:8000/ws/weights`
const MAX_HISTORY = 500
const BACKOFF_BASE = 1500
const BACKOFF_MAX  = 30000

export function useWeightStream() {
  const [history, setHistory] = useState([])
  const [status,  setStatus]  = useState('connecting')

  const wsRef      = useRef(null)
  const retryCount = useRef(0)
  const retryTimer = useRef(null)

  const connect = useCallback(() => {
    if (wsRef.current?.readyState === WebSocket.OPEN) return
    setStatus('connecting')
    const ws = new WebSocket(WS_URL)

    ws.onopen = () => {
      setStatus('connected')
      retryCount.current = 0
    }

    ws.onmessage = (event) => {
      try {
        const snap = JSON.parse(event.data)
        setHistory(prev => {
          // Keep it sorted by time, oldest first, for charting
          const updated = [...prev, snap]
          if (updated.length > MAX_HISTORY) {
            return updated.slice(updated.length - MAX_HISTORY)
          }
          return updated
        })
      } catch (err) {
        console.error("Failed to parse weight message:", err)
      }
    }

    ws.onclose = () => {
      setStatus('disconnected')
      // Exponential backoff reconnect
      const delay = Math.min(BACKOFF_MAX, BACKOFF_BASE * (1.5 ** retryCount.current))
      retryCount.current++
      clearTimeout(retryTimer.current)
      retryTimer.current = setTimeout(connect, delay)
    }

    ws.onerror = () => {
      ws.close()
    }

    wsRef.current = ws
  }, [])

  useEffect(() => {
    connect()
    return () => {
      clearTimeout(retryTimer.current)
      if (wsRef.current) {
        wsRef.current.onclose = null // prevent reconnect on unmount
        wsRef.current.close()
      }
    }
  }, [connect])

  return { history, status }
}
