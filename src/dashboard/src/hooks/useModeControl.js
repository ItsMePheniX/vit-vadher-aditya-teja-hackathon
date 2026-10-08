/**
 * useModeControl — manages Live / Replay engine mode via the REST API.
 *
 * Polls GET /health every 5 s for mode status and live availability.
 * Exposes helpers to start/stop live and replay.
 *
 * Returns:
 *   health         – latest health payload or null
 *   mode           – 'idle' | 'live' | 'replay'
 *   liveAvailable  – { available, reason }
 *   episodes       – string[]
 *   startLive()
 *   stopLive()
 *   startReplay(episode, speed)
 *   stopReplay()
 *   modeLoading    – bool
 *   modeError      – string or null
 */
import { useState, useEffect, useRef, useCallback } from 'react'

const API = `http://${window.location.hostname}:8000`
const POLL_MS = 5000

async function apiCall(path, method = 'GET', body = null) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } }
  if (body) opts.body = JSON.stringify(body)
  const resp = await fetch(`${API}${path}`, opts)
  if (!resp.ok) {
    const err = await resp.json().catch(() => ({ detail: resp.statusText }))
    throw new Error(err.detail || resp.statusText)
  }
  return resp.json()
}

export function useModeControl() {
  const [health,    setHealth]    = useState(null)
  const [loading,   setLoading]   = useState(false)
  const [modeError, setModeError] = useState(null)
  const pollTimer = useRef(null)

  const fetchHealth = useCallback(async () => {
    try {
      const h = await apiCall('/health')
      setHealth(h)
    } catch (_) {
      // API offline — leave health as null
    }
  }, [])

  useEffect(() => {
    fetchHealth()
    pollTimer.current = setInterval(fetchHealth, POLL_MS)
    return () => clearInterval(pollTimer.current)
  }, [fetchHealth])

  const withLoading = useCallback(async (fn) => {
    setLoading(true)
    setModeError(null)
    try {
      await fn()
      await fetchHealth()
    } catch (e) {
      setModeError(e.message)
    } finally {
      setLoading(false)
    }
  }, [fetchHealth])

  const startLive    = useCallback(() => withLoading(() => apiCall('/live/start',    'POST')), [withLoading])
  const stopLive     = useCallback(() => withLoading(() => apiCall('/live/stop',     'POST')), [withLoading])
  const stopReplay   = useCallback(() => withLoading(() => apiCall('/replay/stop',   'POST')), [withLoading])
  const startReplay  = useCallback((episode, speed) =>
    withLoading(() => apiCall('/replay/start', 'POST', { episode, speed })),
  [withLoading])

  return {
    health,
    mode:          health?.mode          ?? 'idle',
    liveAvailable: health?.live          ?? { available: false, reason: 'API offline' },
    episodes:      health?.episodes      ?? [],
    startLive,
    stopLive,
    startReplay,
    stopReplay,
    modeLoading:   loading,
    modeError,
  }
}
