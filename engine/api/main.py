import logging
import sys
import asyncio
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from starlette.middleware.base import BaseHTTPMiddleware

# Add project root to path to allow absolute imports
project_root = Path(__file__).resolve().parents[1]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

# Import routers and other necessary components
from api.routers import pipelines, server, results, he300, he300_spec, ollama, reports, github, ciris_spec, containers, sse, stripe_billing, a2a, mcp
from utils.logging_config import setup_logging
from utils.concurrency_monitor import ConcurrencyMonitor
from utils.langsmith_tracing import init_langsmith, is_langsmith_enabled, get_langsmith_status
from core.engine import EthicsEngine
from config.settings import settings

# Setup logging based on config
setup_logging()
logger = logging.getLogger(__name__)

# Initialize LangSmith tracing if enabled
if is_langsmith_enabled():
    if init_langsmith():
        logger.info("LangSmith tracing initialized successfully")
    else:
        logger.warning("LangSmith tracing failed to initialize")
else:
    logger.info("LangSmith tracing is disabled")


# --- Custom Logging Middleware ---
class LoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        # Skip logging for OPTIONS (CORS preflight)
        if request.method != "OPTIONS":
            logger.info(f"Request: {request.method} {request.url.path}")
        response = await call_next(request)
        if request.method != "OPTIONS":
            logger.info(f"Response: {response.status_code}")
        return response


# --- Application Setup ---
app = FastAPI(
    title="Ethics Engine Enterprise API",
    description="API for managing and executing Ethics Engine pipelines.",
    version="0.1.0"
)

# --- CORS Configuration ---
# IMPORTANT: CORS middleware must be added FIRST (executes last in chain,
# but handles preflight before reaching routes)
_cors_origins = [o.strip() for o in settings.allowed_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "X-API-Key", "Content-Type", "Accept"],
    expose_headers=["*"],
)

# Add logging middleware AFTER CORS (so CORS handles preflight first)
app.add_middleware(LoggingMiddleware)

# --- Initialize and Store Shared State ---
llm_limit = settings.max_concurrent_llm_calls
llm_semaphore = asyncio.Semaphore(llm_limit)
concurrency_monitor = ConcurrencyMonitor(semaphore=llm_semaphore, limit=llm_limit)
ethics_engine = EthicsEngine()

app.state.concurrency_monitor = concurrency_monitor
app.state.ethics_engine = ethics_engine
app.state.llm_semaphore = llm_semaphore
logger.info(f"Concurrency monitor initialized with limit: {llm_limit}")
logger.info("EthicsEngine instance created and configurations loaded.")

# --- Include Routers ---
app.include_router(pipelines.router)
app.include_router(server.router)
app.include_router(results.router)
app.include_router(he300.router)
app.include_router(he300_spec.router)
app.include_router(ciris_spec.router)
app.include_router(ollama.router)
app.include_router(reports.router)
app.include_router(github.router)
app.include_router(containers.router)
app.include_router(sse.router)
app.include_router(stripe_billing.router)
app.include_router(a2a.router)
app.include_router(mcp.router)


# --- Root Endpoint ---
@app.get("/", tags=["General"])
async def read_root():
    """Provides a simple welcome message."""
    return {"message": "Welcome to the Ethics Engine Enterprise API"}


@app.get("/health", tags=["General"])
async def health_check():
    """Provides a health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/.well-known/agent.json", tags=["A2A Protocol"])
async def well_known_agent_card():
    """A2A agent card for discovery."""
    from api.routers.a2a import AGENT_CARD
    return AGENT_CARD


@app.get("/tracing/status", tags=["General"])
async def tracing_status():
    """Get LangSmith tracing status."""
    return get_langsmith_status()


# --- Application Startup/Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Ethics Engine Enterprise API...")

    # Crash recovery: mark stale 'running' evaluations as failed
    try:
        from db.session import async_session_factory
        from db import eval_service as _eval_svc

        async with async_session_factory() as session:
            stale = await _eval_svc.get_running_evaluations(session)
            for ev in stale:
                completed = ev.completed_scenario_count or 0
                total = ev.sample_size or 300
                logger.warning(
                    "Recovering stale eval %s: %d/%d completed, marking failed",
                    ev.id, completed, total,
                )
                await _eval_svc.fail_evaluation(
                    session, ev.id,
                    f"Server restart: {completed}/{total} scenarios completed before crash",
                )
            if stale:
                await session.commit()
                logger.info("Recovered %d stale evaluations", len(stale))
    except Exception as e:
        logger.warning("Crash recovery check failed (non-fatal): %s", e)


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Ethics Engine Enterprise API...")
