from fastapi import APIRouter, Depends, Request, HTTPException, status
import logging
from pathlib import Path
import sys

# Add project root to path if not already added
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from utils.concurrency_monitor import ConcurrencyMonitor # Assuming singleton or accessible instance

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/server",
    tags=["server"],
    responses={404: {"description": "Not found"}},
)

# --- Dependency to get the shared ConcurrencyMonitor instance ---
async def get_concurrency_monitor(request: Request) -> ConcurrencyMonitor:
    """
    Dependency function to retrieve the ConcurrencyMonitor instance
    stored in the application state.
    """
    if hasattr(request.app.state, 'concurrency_monitor'):
        return request.app.state.concurrency_monitor
    else:
        # This should not happen if main.py initializes it correctly
        logger.error("ConcurrencyMonitor not found in application state!")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Server configuration error: Concurrency monitor not initialized."
        )

# --- Endpoint ---

@router.get("/concurrency", response_model=dict)
async def get_server_concurrency_status(
    monitor: ConcurrencyMonitor = Depends(get_concurrency_monitor)
):
    """
    Provides information about the current LLM call concurrency limits and usage.
    """
    limit = monitor._limit
    active = monitor._active_tasks

    # Attempt to get waiter count (similar to _log_status, acknowledging potential issues)
    waiters = 0
    if hasattr(monitor._semaphore, '_waiters') and monitor._semaphore._waiters is not None:
        try:
            waiters = len(monitor._semaphore._waiters)
        except TypeError:
            waiters = -1 # Indicate unknown waiter count
            logger.debug("Could not determine exact number of waiters for semaphore in API endpoint.")
    else:
         # If _waiters doesn't exist or is None
         waiters = 0 # Assume 0 if attribute not found

    status = {
        "limit": limit,
        "active": active,
        "waiting": waiters if waiters != -1 else "unknown" # Return 'unknown' if count failed
    }

    logger.info(f"Reporting concurrency status: {status}")
    return status
