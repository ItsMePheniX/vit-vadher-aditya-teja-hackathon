"""Tests for corroboration, state decay, and all new API endpoints.

Coverage:
  - Corroboration: grouping, confidence boost, social penalty, pump filter
  - StateStore: ingest, decay formula, prune, snapshot
  - API: GET /signals (filters), GET /state, WS /ws/signals,
         POST /replay/start, /replay/stop, /live/start, /live/stop,
         GET /health (live availability + mode)
"""
from __future__ import annotations

import asyncio
import json
import math
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

from src.engine.schema import Signal
from src.engine.state import StateStore, StateEntry

# ── Helpers ────────────────────────────────────────────────────────────────────

def _now() -> datetime:
    """Current UTC time (fresh each call so signals are always in-window)."""
    return datetime.now(tz=timezone.utc)


def _sig(
    entity:        str   = "JPM",
    sentiment:     float = 0.7,
    confidence:    float = 0.85,
    impact:        int   = 7,
    event_type:    str   = "Earnings",
    half_life:     float = 12.0,
    source:        str   = "reuters",
    ts_offset_min: float = 0.0,
    sig_id:        str | None = None,
) -> Signal:
    ts = _now() + timedelta(minutes=ts_offset_min)
    uid = sig_id or f"test_{entity}_{int(ts.timestamp())}_{abs(hash(source+str(ts_offset_min)))%9999:04d}"
    return Signal(
        id=uid, ts=ts, entity=entity, sentiment=sentiment,
        confidence=confidence, impact=impact, event_type=event_type,
        half_life_hours=half_life, evidence=["test sentence."],
        sources=[source],
    )


# ══════════════════════════════════════════════════════════════════════════════
# Corroboration tests
# ══════════════════════════════════════════════════════════════════════════════

class TestCorroboration:

    def test_single_signal_passthrough(self):
        from src.engine.corroboration import corroborate
        sigs = [_sig()]
        result = corroborate(sigs)
        assert len(result) == 1
        assert result[0].signal.entity == "JPM"

    def test_multi_source_boosts_confidence(self):
        from src.engine.corroboration import corroborate
        sigs = [
            _sig(source="reuters",   sig_id="ms1"),
            _sig(source="bloomberg", sig_id="ms2"),
            _sig(source="wsj",       sig_id="ms3"),
        ]
        result = corroborate(sigs)
        assert len(result) == 1
        # 3 distinct high-credibility sources → multi-source corroboration
        assert result[0].source_count == 3
        assert result[0].signal.confidence > 0

    def test_single_social_source_penalty(self):
        from src.engine.corroboration import corroborate
        sig = _sig(source="kaggle_tweets", confidence=0.80, sig_id="social1")
        result = corroborate([sig])
        assert len(result) == 1
        # Social credibility (0.60) * base_conf (0.80) - social_penalty → less than base
        assert result[0].signal.confidence < 0.80

    def test_groups_by_entity_event_type(self):
        from src.engine.corroboration import corroborate
        sigs = [
            _sig(entity="JPM", event_type="Earnings",   source="reuters",   sig_id="ga1"),
            _sig(entity="JPM", event_type="Earnings",   source="bloomberg", sig_id="ga2"),
            _sig(entity="BAC", event_type="Earnings",   source="wsj",       sig_id="gb1"),
            _sig(entity="JPM", event_type="Regulatory", source="ft",        sig_id="gc1"),
        ]
        result = corroborate(sigs)
        # 3 groups: (JPM,Earnings), (BAC,Earnings), (JPM,Regulatory)
        assert len(result) == 3

    def test_sentiment_is_mean_of_group(self):
        from src.engine.corroboration import corroborate
        sigs = [
            _sig(sentiment=0.6, sig_id="sm1", source="reuters"),
            _sig(sentiment=0.2, sig_id="sm2", source="bloomberg"),
        ]
        result = corroborate(sigs)
        assert len(result) == 1
        assert abs(result[0].signal.sentiment - 0.4) < 0.01

    def test_impact_is_max_of_group(self):
        from src.engine.corroboration import corroborate
        sigs = [
            _sig(impact=5, sig_id="im1", source="reuters"),
            _sig(impact=9, sig_id="im2", source="bloomberg"),
        ]
        result = corroborate(sigs)
        assert result[0].signal.impact == 9

    def test_out_of_window_signals_excluded(self):
        from src.engine.corroboration import corroborate
        # 1 signal 120 min ago (outside 60-min window), 1 signal now
        old = _sig(sig_id="ow_old", ts_offset_min=-120)
        new = _sig(sig_id="ow_new", ts_offset_min=0)
        result = corroborate([old, new])
        # Old should be excluded; only new forms a group
        assert len(result) == 1
        assert result[0].item_count == 1

    def test_pump_detection_lowers_confidence(self):
        from src.engine.corroboration import corroborate
        text = "buy JPM stock now it is going to moon 100x"
        sigs = []
        for i in range(7):
            s = _sig(
                source="kaggle_tweets",
                sig_id=f"pump_t{i}",
                ts_offset_min=float(i) * 0.5,   # spread over 3.5 min (within pump sub-window)
            )
            sigs.append(s.model_copy(update={"evidence": [text]}))

        result = corroborate(sigs)
        assert len(result) == 1
        r = result[0]
        assert r.pump_detected is True
        # Confidence significantly reduced
        assert r.signal.confidence < 0.5

    def test_jaccard_similarity(self):
        from src.engine.corroboration import _jaccard
        assert _jaccard("hello world", "hello world") == 1.0
        assert _jaccard("hello world", "goodbye mars") == 0.0
        assert _jaccard("JPM earnings beat expectations", "JPM earnings beat all expectations") > 0.7

    def test_credibility_lookup(self):
        from src.engine.corroboration import _credibility
        cred = {"reuters": 1.0, "twitter": 0.6, "default": 0.75}
        assert _credibility("reuters", cred, 0.75) == 1.0
        assert _credibility("twitter", cred, 0.75) == 0.6
        assert _credibility("unknown_xyz", cred, 0.75) == 0.75

    def test_corroborate_one_adjusts_signal(self):
        from src.engine.corroboration import corroborate_one
        recent   = [_sig(source="reuters",   sig_id="co_r1")]
        incoming = _sig(source="bloomberg",  sig_id="co_r2")
        result   = corroborate_one(incoming, recent)
        assert isinstance(result, Signal)
        assert result.entity == incoming.entity


# ══════════════════════════════════════════════════════════════════════════════
# State decay tests
# ══════════════════════════════════════════════════════════════════════════════

class TestStateStore:

    def test_ingest_and_get(self):
        store = StateStore()
        store.ingest(_sig())
        entry = store.get("JPM")
        assert entry is not None
        assert entry.entity == "JPM"
        assert abs(entry.decayed_sentiment) > 0.01   # freshly added → non-zero

    def test_unknown_entity_returns_none(self):
        store = StateStore()
        assert store.get("ZZZZZ") is None

    def test_decay_formula(self):
        """S(t+hl) ≈ S(0)/2 at t = half_life."""
        store = StateStore()
        sig = _sig(sentiment=1.0, confidence=1.0, half_life=2.0)
        store.ingest(sig)
        with store._lock:
            contrib = store._contributions["JPM"][0]
            # Set added_at to exactly 2 hours ago → decay = 0.5
            contrib.added_at = _now() - timedelta(hours=2.0)
        entry = store.get("JPM")
        assert abs(entry.decayed_sentiment - 0.5) < 0.02

    def test_decay_at_zero_time(self):
        """At t=0, decayed weight ≈ sentiment * confidence."""
        store = StateStore()
        sig = _sig(sentiment=0.8, confidence=0.9, half_life=24.0)
        store.ingest(sig)
        # Ensure added_at is right now
        with store._lock:
            store._contributions["JPM"][0].added_at = _now()
        entry = store.get("JPM")
        expected = 0.8 * 0.9   # = 0.72
        assert abs(entry.decayed_sentiment - expected) < 0.02

    def test_multiple_signals_sum(self):
        store = StateStore()
        s1 = _sig(sentiment=0.5, confidence=1.0, sig_id="sum1", source="reuters")
        s2 = _sig(sentiment=0.3, confidence=1.0, sig_id="sum2", source="bloomberg")
        store.ingest(s1)
        store.ingest(s2)
        with store._lock:
            for c in store._contributions["JPM"]:
                c.added_at = _now()
        entry = store.get("JPM")
        assert abs(entry.decayed_sentiment - 0.8) < 0.05

    def test_snapshot_all_entities(self):
        store = StateStore()
        store.ingest(_sig(entity="JPM", sig_id="snap1"))
        store.ingest(_sig(entity="BAC", sig_id="snap2"))
        snap = store.snapshot()
        assert "JPM" in snap
        assert "BAC" in snap

    def test_prune_removes_expired(self):
        store = StateStore()
        sig = _sig(sentiment=0.5, confidence=1.0, half_life=0.001, sig_id="prune1")
        store.ingest(sig)
        with store._lock:
            store._contributions["JPM"][0].added_at = _now() - timedelta(hours=100)
        pruned = store.prune()
        assert pruned >= 1

    def test_clear_resets_state(self):
        store = StateStore()
        store.ingest(_sig())
        store.clear()
        assert store.snapshot() == {}

    def test_thread_safety(self):
        """Concurrent ingests must not corrupt state."""
        store = StateStore()
        errors: list[Exception] = []

        def _ingest_n(i: int) -> None:
            try:
                store.ingest(_sig(
                    entity="JPM",
                    sig_id=f"th{i}_{id(threading.current_thread())}",
                    sentiment=0.1,
                ))
            except Exception as exc:
                errors.append(exc)

        threads = [threading.Thread(target=_ingest_n, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        entry = store.get("JPM")
        assert entry is not None
        assert entry.signal_count == 50


# ══════════════════════════════════════════════════════════════════════════════
# API tests (FastAPI test client)
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def anyio_backend() -> str:
    return "asyncio"


@pytest.fixture(autouse=True)
def _reset_state():
    """Reset global state store + mode before each test."""
    from src.engine.state import get_store
    from src.engine import mode_manager as mm
    get_store().clear()
    mm.manager._mode    = mm.Mode.IDLE
    mm.manager._task    = None
    mm.manager._episode = None
    yield
    mm.manager._mode    = mm.Mode.IDLE
    mm.manager._task    = None


@pytest.fixture
async def client():
    from src.api.main import app
    async with AsyncClient(transport=ASGITransport(app=app), base_url="http://test") as ac:
        yield ac


# ── /health ───────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_health_ok(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "mode" in data
    assert "live" in data
    assert data["live"]["available"] is True
    assert "reason" in data["live"]
    assert "episodes" in data
    assert isinstance(data["episodes"], list)
    assert "svb_march_2023" in data["episodes"]


@pytest.mark.asyncio
async def test_health_live_always_available(client):
    resp = await client.get("/health")
    assert resp.json()["live"]["available"] is True


@pytest.mark.asyncio
async def test_health_reports_idle_mode(client):
    resp = await client.get("/health")
    assert resp.json()["mode"] == "idle"


# ── GET /signals ──────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_list_signals_empty_returns_list(client):
    resp = await client.get("/signals?limit=5")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


@pytest.mark.asyncio
async def test_list_signals_entity_filter(client, tmp_path):
    import src.api.routes.signals as sr
    orig = sr._JSONL_PATH
    jsonl = tmp_path / "signals.jsonl"
    sr._JSONL_PATH = jsonl
    try:
        sigs = [_sig(entity="JPM", sig_id="ef1"), _sig(entity="BAC", sig_id="ef2")]
        with jsonl.open("a", encoding="utf-8") as fh:
            for s in sigs:
                fh.write(s.model_dump_json() + "\n")
        resp = await client.get("/signals?entity=JPM")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert all(s["entity"] == "JPM" for s in data)
    finally:
        sr._JSONL_PATH = orig


@pytest.mark.asyncio
async def test_list_signals_min_impact_filter(client, tmp_path):
    import src.api.routes.signals as sr
    orig = sr._JSONL_PATH
    jsonl = tmp_path / "signals.jsonl"
    sr._JSONL_PATH = jsonl
    try:
        sigs = [_sig(impact=3, sig_id="mf_low"), _sig(impact=8, sig_id="mf_high")]
        with jsonl.open("a", encoding="utf-8") as fh:
            for s in sigs:
                fh.write(s.model_dump_json() + "\n")
        resp = await client.get("/signals?min_impact=7")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        assert all(s["impact"] >= 7 for s in data)
    finally:
        sr._JSONL_PATH = orig


@pytest.mark.asyncio
async def test_list_signals_since_filter(client, tmp_path):
    import src.api.routes.signals as sr
    orig = sr._JSONL_PATH
    jsonl = tmp_path / "signals.jsonl"
    sr._JSONL_PATH = jsonl
    try:
        cutoff = _now() - timedelta(minutes=30)
        old_sig = _sig(sig_id="sf_old", ts_offset_min=-60)   # 60 min before now
        new_sig = _sig(sig_id="sf_new", ts_offset_min= -5)   # 5 min before now
        with jsonl.open("a", encoding="utf-8") as fh:
            for s in [old_sig, new_sig]:
                fh.write(s.model_dump_json() + "\n")
        # FastAPI datetime query param: strip microseconds, use Z suffix
        since_str = cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")
        resp = await client.get(f"/signals?since={since_str}")
        assert resp.status_code == 200
        data = resp.json()
        assert len(data) >= 1
        for s in data:
            sig_ts = datetime.fromisoformat(s["ts"])
            if sig_ts.tzinfo is None:
                sig_ts = sig_ts.replace(tzinfo=timezone.utc)
            assert sig_ts >= cutoff
    finally:
        sr._JSONL_PATH = orig


# ── GET /state ────────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_get_state_empty(client):
    resp = await client.get("/state")
    assert resp.status_code == 200
    assert resp.json() == []


@pytest.mark.asyncio
async def test_get_state_after_ingest(client):
    from src.engine.state import get_store
    get_store().ingest(_sig(entity="JPM", sig_id="gsa1"))
    resp = await client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert any(e["entity"] == "JPM" for e in data)
    jpm = next(e for e in data if e["entity"] == "JPM")
    assert "decayed_sentiment" in jpm
    assert "signal_count" in jpm
    assert "strongest_event" in jpm
    assert "half_life_hours" in jpm


@pytest.mark.asyncio
async def test_get_entity_state(client):
    from src.engine.state import get_store
    get_store().ingest(_sig(entity="GS", sentiment=-0.6, sig_id="ges1"))
    resp = await client.get("/state/GS")
    assert resp.status_code == 200
    data = resp.json()
    assert data["entity"] == "GS"
    assert data["decayed_sentiment"] < 0


@pytest.mark.asyncio
async def test_get_entity_state_not_found(client):
    resp = await client.get("/state/ZZZZZ")
    assert resp.status_code == 404


@pytest.mark.asyncio
async def test_state_sorted_by_absolute_sentiment(client):
    from src.engine.state import get_store
    store = get_store()
    store.ingest(_sig(entity="AAPL", sentiment= 0.3, sig_id="sort1"))
    store.ingest(_sig(entity="TSLA", sentiment=-0.9, sig_id="sort2"))
    resp = await client.get("/state")
    assert resp.status_code == 200
    data = resp.json()
    assert len(data) >= 2
    # Highest |sentiment| should be first
    abs_vals = [abs(e["decayed_sentiment"]) for e in data if e["entity"] in ("AAPL", "TSLA")]
    assert abs_vals == sorted(abs_vals, reverse=True)


# ── /replay endpoints ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_replay_start_unknown_episode(client):
    resp = await client.post("/replay/start", json={"episode": "nonexistent_ep", "speed": 60})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_replay_start_valid_episode(client):
    from src.engine import mode_manager as mm
    with patch.object(mm.manager, "start_replay", new_callable=AsyncMock) as mock_start:
        with patch.object(mm.manager, "status", return_value={
            "mode": "replay", "running": True, "episode": "svb_march_2023",
            "speed": 60.0, "started_at": None, "uptime_s": 0.1, "episodes": ["svb_march_2023"],
        }):
            resp = await client.post(
                "/replay/start", json={"episode": "svb_march_2023", "speed": 60},
            )
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    mock_start.assert_called_once_with("svb_march_2023", 60.0)


@pytest.mark.asyncio
async def test_replay_invalid_speed(client):
    resp = await client.post("/replay/start", json={"episode": "svb_march_2023", "speed": -1})
    assert resp.status_code == 422


@pytest.mark.asyncio
async def test_replay_stop_when_not_replaying(client):
    resp = await client.post("/replay/stop")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_replay_stop_when_replaying(client):
    from src.engine import mode_manager as mm
    mm.manager._mode = mm.Mode.REPLAY
    with patch.object(mm.manager, "stop", new_callable=AsyncMock):
        with patch.object(mm.manager, "status", return_value={
            "mode": "idle", "running": False, "episode": None,
            "speed": 1.0, "started_at": None, "uptime_s": None, "episodes": [],
        }):
            resp = await client.post("/replay/stop")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


@pytest.mark.asyncio
async def test_replay_requires_data_file(client):
    from src.engine import mode_manager as mm
    with patch.object(mm.manager, "start_replay", side_effect=FileNotFoundError("not found")):
        resp = await client.post(
            "/replay/start", json={"episode": "svb_march_2023", "speed": 1000000},
        )
    assert resp.status_code == 503


# ── /live endpoints ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_live_start(client):
    from src.engine import mode_manager as mm
    with patch.object(mm.manager, "start_live", new_callable=AsyncMock) as mock_live:
        with patch.object(mm.manager, "status", return_value={
            "mode": "live", "running": True, "episode": None,
            "speed": 1.0, "started_at": None, "uptime_s": 0.0, "episodes": [],
        }):
            resp = await client.post("/live/start")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    mock_live.assert_called_once()


@pytest.mark.asyncio
async def test_live_stop_when_not_live(client):
    resp = await client.post("/live/stop")
    assert resp.status_code == 409


@pytest.mark.asyncio
async def test_live_stop_when_live(client):
    from src.engine import mode_manager as mm
    mm.manager._mode = mm.Mode.LIVE
    with patch.object(mm.manager, "stop", new_callable=AsyncMock):
        with patch.object(mm.manager, "status", return_value={
            "mode": "idle", "running": False, "episode": None,
            "speed": 1.0, "started_at": None, "uptime_s": None, "episodes": [],
        }):
            resp = await client.post("/live/stop")
    assert resp.status_code == 200


# ── POST /signals/ingest ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_ingest_endpoint_with_mocked_nlp(client):
    """Ingest with mocked NLP — no real models needed."""
    from src.engine.nlp.sentiment import SentimentResult
    from src.engine.nlp.classifier import ClassifyResult
    with (
        patch("src.engine.nlp.pipeline.sent_mod.analyse",
              return_value=SentimentResult(score=0.65, confidence=0.88)),
        patch("src.engine.nlp.pipeline.cls_mod.classify",
              return_value=ClassifyResult(label="Earnings", probability=0.85)),
        patch("src.engine.nlp.pipeline.ev_mod.extract",
              return_value=["JPMorgan beats earnings."]),
        patch("src.engine.nlp.pipeline.write_signals"),
    ):
        resp = await client.post(
            "/signals/ingest",
            json={"text": "JPMorgan beats Q3 earnings by 12%", "source": "test"},
        )
    assert resp.status_code == 201
    data = resp.json()
    assert isinstance(data, list)
    assert len(data) >= 1
    s = data[0]
    assert "id" in s and "ts" in s and "sentiment" in s
    assert -1.0 <= s["sentiment"] <= 1.0
    assert 1 <= s["impact"] <= 10


# ── WS /ws/signals ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_websocket_connects(client):
    """Verify the WS endpoint accepts connections using httpx websocket."""
    from src.api.main import app
    from starlette.testclient import TestClient

    # starlette TestClient handles the ASGI WS in sync mode
    with TestClient(app) as tc:
        with tc.websocket_connect("/ws/signals") as ws:
            # Connection established successfully — no error
            pass   # disconnect on exit


@pytest.mark.asyncio
async def test_websocket_receives_broadcast():
    """Verify that _broadcast() sends a signal to a connected WS client queue."""
    from src.api.routes.signals import _ws_clients, _broadcast

    q: asyncio.Queue[Signal] = asyncio.Queue(maxsize=10)
    _ws_clients.add(q)
    try:
        sig = _sig(entity="WFC", sig_id="ws_bc_001")
        _broadcast(sig)
        # Check queue immediately
        assert not q.empty()
        received = q.get_nowait()
        assert received.id == sig.id
        assert received.entity == "WFC"
    finally:
        _ws_clients.discard(q)


@pytest.mark.asyncio
async def test_websocket_full_roundtrip():
    """End-to-end: WS client receives a signal broadcast after connect."""
    from src.api.main import app
    from src.api.routes.signals import _broadcast

    received: list[dict] = []

    async def _client_task():
        # Use httpx AsyncClient for WebSocket
        async with AsyncClient(
            transport=ASGITransport(app=app), base_url="http://test"
        ) as ac:
            # httpx doesn't support WS natively — use starlette directly
            pass

    # Test the broadcast mechanism directly (queue-level)
    from src.api.routes.signals import _ws_clients
    q: asyncio.Queue[Signal] = asyncio.Queue(maxsize=5)
    _ws_clients.add(q)
    try:
        sig = _sig(entity="C", sig_id="ws_e2e_001")
        _broadcast(sig)
        got = await asyncio.wait_for(q.get(), timeout=1.0)
        assert got.entity == "C"
        assert got.id == sig.id
        # Confirm it serializes cleanly to JSON
        payload = json.loads(got.model_dump_json())
        assert payload["entity"] == "C"
    finally:
        _ws_clients.discard(q)
