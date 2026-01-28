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
from api.routers import pipelines, server, results, he300, he300_spec, ollama, reports, github, ciris_spec
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
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=False,  # Must be False when using "*"
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS", "PATCH"],
    allow_headers=["*"],
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


# --- Root Endpoint ---
@app.get("/", tags=["General"])
async def read_root():
    """Provides a simple welcome message."""
    return {"message": "Welcome to the Ethics Engine Enterprise API"}


@app.get("/health", tags=["General"])
async def health_check():
    """Provides a health check endpoint."""
    return {"status": "healthy", "version": "0.1.0"}


@app.get("/tracing/status", tags=["General"])
async def tracing_status():
    """Get LangSmith tracing status."""
    return get_langsmith_status()


# --- Application Startup/Shutdown Events ---
@app.on_event("startup")
async def startup_event():
    logger.info("Starting Ethics Engine Enterprise API...")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Shutting down Ethics Engine Enterprise API...")
