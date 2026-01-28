# Removed asyncio import as loop runs in separate thread
import logging
import sys
import threading # Import threading
import time
from typing import Optional, TYPE_CHECKING

# Conditional import for type hinting asyncio.Semaphore without runtime dependency
if TYPE_CHECKING:
    import asyncio

logger = logging.getLogger(__name__)

class ConcurrencyMonitor:
    """
    Monitors an asyncio.Semaphore and prints its status periodically using a separate thread.
    Also tracks the number of tasks actively holding the semaphore.
    """
    def __init__(self, semaphore: 'asyncio.Semaphore', limit: int, name: str = "Semaphore"):
        self._semaphore = semaphore
        self._limit = limit
        self._name = name
        self._active_tasks = 0
        self._running = False
        self._monitor_thread: Optional[threading.Thread] = None
        # Lock for thread-safe access to _active_tasks if needed, though GIL might suffice for simple increments/decrements
        # self._lock = threading.Lock()

    def increment_active(self):
        """Increment the count of active tasks holding the semaphore."""
        # with self._lock: # Optional lock if more complex state is managed
        self._active_tasks += 1

    def decrement_active(self):
        """Decrement the count of active tasks holding the semaphore."""
        # with self._lock: # Optional lock
        self._active_tasks = max(0, self._active_tasks - 1)

    def _log_status(self):
        """Prints the current status of the semaphore to stderr."""
        limit = self._limit
        # Accessing semaphore._value might not be perfectly thread-safe,
        # but should be okay for monitoring purposes.
        available = self._semaphore._value
        waiters = 0
        # Accessing _waiters is also not guaranteed thread-safe
        if hasattr(self._semaphore, '_waiters') and self._semaphore._waiters is not None:
             try:
                 waiters = len(self._semaphore._waiters)
             except TypeError:
                 waiters = -1
                 logger.debug("Could not determine exact number of waiters for semaphore.")

        # Access _active_tasks (potentially with lock if needed)
        # with self._lock:
        active = self._active_tasks

        print(
            f"[{self._name} Monitor] Status: "
            f"Limit={limit}, "
            f"Available={available}, "
            f"Active(Tracked)={active}, "
            f"Waiting={waiters if waiters != -1 else '?'}",
            file=sys.stderr,
            flush=True
        )

    def run(self, interval_seconds: int):
        """Runs the monitoring loop (target for the thread)."""
        logger.info(f"[{self._name} Monitor] Starting monitoring thread (Interval: {interval_seconds}s)")
        try:
            while self._running:
                self._log_status()
                time.sleep(interval_seconds)
        except Exception as e:
            logger.error(f"[{self._name} Monitor] Error in monitoring thread: {e}", exc_info=True)
        finally:
            # Ensure running flag is False if loop exits unexpectedly
            self._running = False
            logger.info(f"[{self._name} Monitor] Monitoring thread finished.")

    def start(self, interval_seconds: int = 2):
        """Creates and starts the monitoring thread."""
        if self._monitor_thread and self._monitor_thread.is_alive():
             logger.warning(f"[{self._name} Monitor] Thread already exists and is running.")
             return

        self._running = True # Set flag before starting thread
        self._monitor_thread = threading.Thread(
            target=self.run,
            args=(interval_seconds,),
            daemon=True # Set as daemon so it doesn't block program exit
        )
        self._monitor_thread.start()

    def stop(self):
        """Signals the monitoring thread to stop and waits for it."""
        if not self._running or not self._monitor_thread or not self._monitor_thread.is_alive():
            return # Silently return if not running

        logger.info(f"[{self._name} Monitor] Signaling monitoring thread to stop...")
        self._running = False
        # Calculate join timeout based on the interval the thread was started with
        # Need to store interval_seconds if we want this, otherwise use a fixed timeout
        join_timeout = 5.0 # Use a fixed timeout for join
        self._monitor_thread.join(timeout=join_timeout)

        if self._monitor_thread.is_alive():
             logger.warning(f"[{self._name} Monitor] Monitoring thread did not stop within {join_timeout}s timeout.")
        else:
             logger.info(f"[{self._name} Monitor] Monitoring thread stopped cleanly.")
        self._monitor_thread = None
