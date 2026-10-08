import asyncio
import contextlib
import logging

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from src import config
from src.api.routes.health  import router as health_router
from src.api.routes.signals import (
    signals_router, state_router, replay_router,
    live_router, ws_router, broadcast_from_ingestor,
)
from src.api.routes.stress  import router as stress_router, ws_router as stress_ws_router
from src.api.routes.weights import router as weights_router, ws_router as weights_ws_router
from src.engine.mode_manager import manager
from src.engine.state import get_store
from src import config as cfg_mod

logging.basicConfig(
    level=config.LOG_LEVEL.upper(),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


async def _prune_state_periodically() -> None:
    """Background: prune expired signal contributions from the state store."""
    interval = float(cfg_mod.get("state", "decay_check_interval_seconds", default=30))
    store = get_store()
    while True:
        await asyncio.sleep(interval)
        pruned = store.prune()
        if pruned:
            logger.debug("State pruner: removed %d expired contributions", pruned)


@contextlib.asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    """Startup: launch broadcaster + state pruner + rebalancer + stress trigger. Shutdown: clean up."""
    logger.info("Starting broadcaster, state pruner, rebalancer, and stress trigger …")
    from src.modules.stress_test.trigger import get_trigger  # noqa: PLC0415
    from src.modules.rebalancer import get_rebalancer        # noqa: PLC0415
    task_bcast   = asyncio.create_task(broadcast_from_ingestor(), name="broadcaster")
    task_prune   = asyncio.create_task(_prune_state_periodically(), name="state-pruner")
    task_stress  = asyncio.create_task(get_trigger().run(), name="stress-trigger")
    task_rebal   = asyncio.create_task(get_rebalancer().run(), name="rebalancer")
    logger.info("API ready. Docs at /docs")
    yield
    task_bcast.cancel()
    task_prune.cancel()
    task_stress.cancel()
    task_rebal.cancel()
    await manager.stop()
    logger.info("Shutdown complete.")


app = FastAPI(
    title="Financial Risk NLP Engine",
    description=(
        "Real-time NLP signal engine — S&P Global & Crisil Campus Hackathon 2026.\n\n"
        "**Modes**: idle → POST /live/start or POST /replay/start {episode, speed}.\n\n"
        "**WebSocket**: connect to `/ws/signals` for a live JSON stream of Signal objects."
    ),
    version="0.2.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# CORS — allow React dev server
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ────────────────────────────────────────────────────────────────────
app.include_router(health_router)
app.include_router(signals_router)
app.include_router(state_router)
app.include_router(replay_router)
app.include_router(live_router)
app.include_router(ws_router)
app.include_router(stress_router)
app.include_router(stress_ws_router)
app.include_router(weights_router)
app.include_router(weights_ws_router)

# ── Direct run ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run(
        "src.api.main:app",
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower(),
        reload=True,
    )
