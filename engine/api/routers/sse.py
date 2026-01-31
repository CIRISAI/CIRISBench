"""
Server-Sent Events (SSE) router for real-time status streaming.

Provides live status updates for the CIRISBench system including:
- System health and concurrency
- Benchmark execution progress
- Log messages and events
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import AsyncGenerator, Optional
from collections import deque

from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sse", tags=["SSE"])


# Global event bus for distributing events to SSE clients
class EventBus:
    """Simple event bus for broadcasting events to multiple SSE clients."""

    def __init__(self, max_history: int = 100):
        self._subscribers: list[asyncio.Queue] = []
        self._history: deque = deque(maxlen=max_history)
        self._lock = asyncio.Lock()

    async def subscribe(self) -> asyncio.Queue:
        """Subscribe to events. Returns a queue that receives events."""
        queue: asyncio.Queue = asyncio.Queue()
        async with self._lock:
            self._subscribers.append(queue)
        return queue

    async def unsubscribe(self, queue: asyncio.Queue):
        """Unsubscribe from events."""
        async with self._lock:
            if queue in self._subscribers:
                self._subscribers.remove(queue)

    async def publish(self, event_type: str, data: dict):
        """Publish an event to all subscribers."""
        event = {
            "type": event_type,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data
        }
        self._history.append(event)

        async with self._lock:
            for queue in self._subscribers:
                try:
                    queue.put_nowait(event)
                except asyncio.QueueFull:
                    pass  # Skip if queue is full

    def get_history(self, limit: int = 50) -> list:
        """Get recent event history."""
        return list(self._history)[-limit:]

    @property
    def subscriber_count(self) -> int:
        return len(self._subscribers)


# Global event bus instance
event_bus = EventBus()


# Custom logging handler to capture logs for SSE
class SSELogHandler(logging.Handler):
    """Logging handler that publishes log messages to the SSE event bus."""

    def __init__(self, bus: EventBus, min_level: int = logging.INFO):
        super().__init__()
        self.bus = bus
        self.min_level = min_level
        self._loop: Optional[asyncio.AbstractEventLoop] = None

    def emit(self, record: logging.LogRecord):
        if record.levelno < self.min_level:
            return

        try:
            # Get or create event loop
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                return  # No running loop, skip

            # Create the log event
            log_event = {
                "level": record.levelname,
                "logger": record.name,
                "message": record.getMessage(),
                "module": record.module,
                "line": record.lineno,
            }

            # Schedule the publish on the event loop
            loop.create_task(self.bus.publish("log", log_event))
        except Exception:
            pass  # Don't let logging errors break the app


# Install the SSE log handler
_sse_handler: Optional[SSELogHandler] = None


def install_sse_log_handler():
    """Install the SSE log handler on the root logger."""
    global _sse_handler
    if _sse_handler is None:
        _sse_handler = SSELogHandler(event_bus, min_level=logging.INFO)
        _sse_handler.setFormatter(logging.Formatter('%(message)s'))
        logging.getLogger().addHandler(_sse_handler)
        logger.info("SSE log handler installed")


# Helper to publish events from anywhere in the codebase
async def publish_event(event_type: str, data: dict):
    """Publish an event to SSE subscribers."""
    await event_bus.publish(event_type, data)


def publish_event_sync(event_type: str, data: dict):
    """Synchronous version - schedules event publish on the event loop."""
    try:
        loop = asyncio.get_running_loop()
        loop.create_task(event_bus.publish(event_type, data))
    except RuntimeError:
        pass  # No running loop


async def generate_sse_stream(request: Request) -> AsyncGenerator[str, None]:
    """Generate SSE stream with status updates and events."""
    queue = await event_bus.subscribe()

    try:
        # Send initial connection event
        yield format_sse_event("connected", {
            "message": "SSE stream connected",
            "subscribers": event_bus.subscriber_count
        })

        # Send recent history
        for event in event_bus.get_history(20):
            yield format_sse_event(event["type"], event["data"], event["timestamp"])

        # Periodic status updates
        last_status_time = 0
        status_interval = 5  # seconds

        while True:
            # Check if client disconnected
            if await request.is_disconnected():
                break

            # Check for new events (with timeout for status updates)
            try:
                event = await asyncio.wait_for(queue.get(), timeout=1.0)
                yield format_sse_event(event["type"], event["data"], event["timestamp"])
            except asyncio.TimeoutError:
                pass

            # Send periodic status update
            current_time = time.time()
            if current_time - last_status_time >= status_interval:
                last_status_time = current_time

                # Get system status from app state
                concurrency_status = get_concurrency_status(request)

                yield format_sse_event("status", {
                    "health": "healthy",
                    "concurrency": concurrency_status,
                    "subscribers": event_bus.subscriber_count,
                    "uptime_seconds": int(current_time - _start_time)
                })

    finally:
        await event_bus.unsubscribe(queue)


def format_sse_event(event_type: str, data: dict, timestamp: Optional[str] = None) -> str:
    """Format data as an SSE event."""
    if timestamp is None:
        timestamp = datetime.utcnow().isoformat()

    payload = {
        "type": event_type,
        "timestamp": timestamp,
        "data": data
    }

    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


def get_concurrency_status(request: Request) -> dict:
    """Get concurrency monitor status from app state."""
    try:
        monitor = request.app.state.concurrency_monitor
        return monitor.get_status()
    except Exception:
        return {"error": "unavailable"}


# Track server start time
_start_time = time.time()


@router.get("/stream")
async def sse_stream(request: Request):
    """
    Server-Sent Events stream for real-time status updates.

    Events include:
    - `connected`: Initial connection confirmation
    - `status`: Periodic system status (health, concurrency, uptime)
    - `log`: Log messages from the system
    - `benchmark_start`: Benchmark run started
    - `benchmark_progress`: Benchmark progress update
    - `benchmark_complete`: Benchmark run completed
    - `error`: Error events
    """
    # Install log handler on first connection
    install_sse_log_handler()

    return StreamingResponse(
        generate_sse_stream(request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",  # Disable nginx buffering
        }
    )


@router.get("/status")
async def get_status(request: Request):
    """Get current system status (non-streaming)."""
    concurrency_status = get_concurrency_status(request)

    return {
        "health": "healthy",
        "concurrency": concurrency_status,
        "subscribers": event_bus.subscriber_count,
        "uptime_seconds": int(time.time() - _start_time),
        "recent_events": len(event_bus.get_history())
    }


@router.get("/history")
async def get_history(limit: int = 50):
    """Get recent event history."""
    return {
        "events": event_bus.get_history(limit),
        "total": len(event_bus._history)
    }


@router.post("/test")
async def test_event(message: str = "Test event"):
    """Send a test event to all SSE subscribers (for debugging)."""
    await event_bus.publish("test", {"message": message})
    return {"sent": True, "subscribers": event_bus.subscriber_count}
