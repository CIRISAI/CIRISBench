"""A2A (Agent-to-Agent) protocol router.

Implements JSON-RPC based A2A protocol for agent communication.
Allows external agents to interact with CIRISBench for ethics evaluation.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/a2a", tags=["A2A Protocol"])


# ---------------------------------------------------------------------------
# JSON-RPC Models
# ---------------------------------------------------------------------------

class JSONRPCRequest(BaseModel):
    """JSON-RPC 2.0 request."""
    jsonrpc: str = "2.0"
    method: str
    params: Optional[Dict[str, Any]] = None
    id: Optional[str] = None


class JSONRPCError(BaseModel):
    """JSON-RPC 2.0 error."""
    code: int
    message: str
    data: Optional[Any] = None


class JSONRPCResponse(BaseModel):
    """JSON-RPC 2.0 response."""
    jsonrpc: str = "2.0"
    result: Optional[Any] = None
    error: Optional[JSONRPCError] = None
    id: Optional[str] = None


# ---------------------------------------------------------------------------
# A2A Agent Card (Well-Known)
# ---------------------------------------------------------------------------

AGENT_CARD = {
    "name": "CIRISBench Ethics Engine",
    "description": "AI Ethics Benchmark Platform - Evaluates agent responses against the Hendrycks Ethics dataset",
    "version": "0.1.0",
    "protocols": ["a2a", "mcp"],
    "capabilities": {
        "benchmark": {
            "categories": ["commonsense", "commonsense_hard", "deontology", "justice", "virtue"],
            "sample_sizes": [50, 100, 300, "full"],
            "concurrency_levels": [10, 50, 100],
        },
        "evaluation": {
            "modes": ["heuristic", "semantic", "hybrid"],
            "llm_providers": ["ollama", "openai", "anthropic", "openrouter"],
        },
    },
    "endpoints": {
        "a2a": "/a2a",
        "mcp": "/mcp",
        "benchmark": "/he300/agentbeats/run",
        "health": "/health",
    },
    "authentication": {
        "type": "bearer",
        "header": "Authorization",
    },
}


# ---------------------------------------------------------------------------
# A2A Methods
# ---------------------------------------------------------------------------

async def handle_tasks_send(params: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Handle tasks/send - receive a scenario and return evaluation response."""
    scenario_id = params.get("scenario_id", str(uuid.uuid4()))
    scenario = params.get("scenario", "")
    question = params.get("question", "Is this action ethical or unethical?")

    if not scenario:
        raise HTTPException(status_code=400, detail="Missing 'scenario' in params")

    # For now, return a placeholder response
    # In full implementation, this would run through the ethics engine
    return {
        "scenario_id": scenario_id,
        "status": "received",
        "message": "Scenario received for evaluation",
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


async def handle_benchmark_run(params: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Handle benchmark/run - start a benchmark evaluation against an agent."""
    agent_url = params.get("agent_url")
    if not agent_url:
        raise HTTPException(status_code=400, detail="Missing 'agent_url' in params")

    sample_size = params.get("sample_size", 50)
    categories = params.get("categories", ["commonsense"])
    concurrency = params.get("concurrency", 50)

    # Start benchmark via the existing HE300 infrastructure
    from api.routers.he300 import AgentBeatsBenchmarkRequest, run_agentbeats_benchmark

    bench_request = AgentBeatsBenchmarkRequest(
        agent_url=agent_url,
        sample_size=sample_size,
        concurrency=concurrency,
        categories=categories,
    )

    result = await run_agentbeats_benchmark(bench_request, request)
    return result


async def handle_agent_card(params: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Handle agent/card - return the agent card."""
    return AGENT_CARD


async def handle_status(params: Dict[str, Any], request: Request) -> Dict[str, Any]:
    """Handle status - return current engine status."""
    return {
        "status": "healthy",
        "version": "0.1.0",
        "capabilities": list(AGENT_CARD["capabilities"].keys()),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


# Method dispatcher
A2A_METHODS = {
    "tasks/send": handle_tasks_send,
    "benchmark/run": handle_benchmark_run,
    "agent/card": handle_agent_card,
    "status": handle_status,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.post("")
@router.post("/")
async def a2a_rpc(rpc_request: JSONRPCRequest, request: Request) -> JSONRPCResponse:
    """
    A2A JSON-RPC endpoint.

    Supported methods:
    - tasks/send: Send a scenario for evaluation
    - benchmark/run: Start a benchmark evaluation
    - agent/card: Get agent capabilities
    - status: Get engine status
    """
    try:
        handler = A2A_METHODS.get(rpc_request.method)
        if not handler:
            return JSONRPCResponse(
                id=rpc_request.id,
                error=JSONRPCError(
                    code=-32601,
                    message=f"Method not found: {rpc_request.method}",
                    data={"available_methods": list(A2A_METHODS.keys())},
                ),
            )

        result = await handler(rpc_request.params or {}, request)
        return JSONRPCResponse(id=rpc_request.id, result=result)

    except HTTPException as e:
        return JSONRPCResponse(
            id=rpc_request.id,
            error=JSONRPCError(code=-32602, message=str(e.detail)),
        )
    except Exception as e:
        logger.exception("A2A RPC error: %s", e)
        return JSONRPCResponse(
            id=rpc_request.id,
            error=JSONRPCError(code=-32603, message=f"Internal error: {str(e)}"),
        )


@router.get("/card")
async def get_agent_card() -> Dict[str, Any]:
    """Get the A2A agent card (also available at /.well-known/agent.json)."""
    return AGENT_CARD
