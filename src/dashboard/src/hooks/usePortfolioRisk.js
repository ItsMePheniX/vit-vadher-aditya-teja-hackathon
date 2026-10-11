/**
 * usePortfolioRisk — fetches VaR/CVaR from the REST API, refreshes every 60 s.
 */
import { useState, useEffect } from 'react'

export function usePortfolioRisk(signals) {
  const [risk, setRisk] = useState(null)

  useEffect(() => {
    async function fetch_risk() {
      try {
        const resp = await fetch(`http://${window.location.hostname}:8000/signals/risk`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ signals: signals.slice(0, 20) }),
        })
        if (resp.ok) setRisk(await resp.json())
      } catch (_) {}
    }
    fetch_risk()
    const id = setInterval(fetch_risk, 60_000)
    return () => clearInterval(id)
  }, [signals.length])   // re-fetch when signal count changes

  return risk
}
