"""MCP (Model Context Protocol) router.

Implements MCP protocol for tool invocation.
Allows LLM agents to invoke CIRISBench tools via structured protocol.
"""

import logging
import uuid
from typing import Any, Dict, List, Optional
from datetime import datetime, timezone

from fastapi import APIRouter, Request, HTTPException
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/mcp", tags=["MCP Protocol"])


# ---------------------------------------------------------------------------
# MCP Models
# ---------------------------------------------------------------------------

class MCPToolInput(BaseModel):
    """MCP tool input schema."""
    type: str = "object"
    properties: Dict[str, Any] = {}
    required: List[str] = []


class MCPTool(BaseModel):
    """MCP tool definition."""
    name: str
    description: str
    inputSchema: MCPToolInput


class MCPToolCallRequest(BaseModel):
    """MCP tools/call request."""
    method: str = "tools/call"
    params: Dict[str, Any]


class MCPContent(BaseModel):
    """MCP response content."""
    type: str = "text"
    text: str


class MCPToolCallResponse(BaseModel):
    """MCP tools/call response."""
    content: List[MCPContent]
    isError: bool = False


# ---------------------------------------------------------------------------
# Tool Definitions
# ---------------------------------------------------------------------------

TOOLS: List[MCPTool] = [
    MCPTool(
        name="run_benchmark",
        description="Run HE-300 ethics benchmark against an agent endpoint",
        inputSchema=MCPToolInput(
            properties={
                "agent_url": {"type": "string", "description": "Agent endpoint URL to benchmark"},
                "sample_size": {"type": "integer", "description": "Number of scenarios (default: 50)", "default": 50},
                "concurrency": {"type": "integer", "description": "Concurrent calls (10, 50, 100)", "default": 50},
                "categories": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Categories to test: commonsense, commonsense_hard, deontology, justice, virtue",
                },
            },
            required=["agent_url"],
        ),
    ),
    MCPTool(
        name="get_scenario",
        description="Get a specific ethical scenario from the catalog",
        inputSchema=MCPToolInput(
            properties={
                "category": {
                    "type": "string",
                    "enum": ["commonsense", "commonsense_hard", "deontology", "justice", "virtue"],
                    "description": "Ethics category",
                },
                "scenario_id": {"type": "string", "description": "Specific scenario ID (optional)"},
                "random": {"type": "boolean", "description": "Get a random scenario", "default": True},
            },
            required=[],
        ),
    ),
    MCPTool(
        name="evaluate_response",
        description="Evaluate an agent response against expected ethical judgment",
        inputSchema=MCPToolInput(
            properties={
                "scenario": {"type": "string", "description": "The ethical scenario text"},
                "response": {"type": "string", "description": "Agent's response to evaluate"},
                "expected_label": {"type": "integer", "description": "Expected label (0=wrong, 1=right)", "enum": [0, 1]},
                "category": {"type": "string", "description": "Category for evaluation context"},
            },
            required=["scenario", "response", "expected_label"],
        ),
    ),
    MCPTool(
        name="get_catalog_stats",
        description="Get statistics about available scenarios in the catalog",
        inputSchema=MCPToolInput(
            properties={},
            required=[],
        ),
    ),
]


# ---------------------------------------------------------------------------
# Tool Handlers
# ---------------------------------------------------------------------------

async def tool_run_benchmark(args: Dict[str, Any], request: Request) -> str:
    """Run benchmark against an agent."""
    from api.routers.he300 import AgentBeatsBenchmarkRequest, run_agentbeats_benchmark

    agent_url = args.get("agent_url")
    if not agent_url:
        raise ValueError("agent_url is required")

    bench_request = AgentBeatsBenchmarkRequest(
        agent_url=agent_url,
        sample_size=args.get("sample_size", 50),
        concurrency=args.get("concurrency", 50),
        categories=args.get("categories"),
    )

    result = await run_agentbeats_benchmark(bench_request, request)
    return f"Benchmark started. Evaluation ID: {result.get('evaluation_id', 'unknown')}"


async def tool_get_scenario(args: Dict[str, Any], request: Request) -> str:
    """Get a scenario from the catalog."""
    import random
    from api.routers.he300 import get_all_scenarios, HE300Category

    category_str = args.get("category", "commonsense")
    scenario_id = args.get("scenario_id")

    # Map string to category enum
    category_map = {
        "commonsense": HE300Category.COMMONSENSE,
        "commonsense_hard": HE300Category.COMMONSENSE_HARD,
        "deontology": HE300Category.DEONTOLOGY,
        "justice": HE300Category.JUSTICE,
        "virtue": HE300Category.VIRTUE,
    }
    category = category_map.get(category_str.lower(), HE300Category.COMMONSENSE)

    all_scenarios = get_all_scenarios()
    scenarios = all_scenarios.get(category, [])

    if not scenarios:
        return f"No scenarios found for category: {category_str}"

    if scenario_id:
        for s in scenarios:
            if s.scenario_id == scenario_id:
                return f"Scenario ({category_str}): {s.input_text}\n\nExpected: {'right' if s.expected_label == 1 else 'wrong'}"
        return f"Scenario {scenario_id} not found in {category_str}"

    scenario = random.choice(scenarios)
    return f"Scenario ID: {scenario.scenario_id}\nCategory: {category_str}\n\n{scenario.input_text}\n\nQuestion: Is this action ethical or unethical?"


async def tool_evaluate_response(args: Dict[str, Any], request: Request) -> str:
    """Evaluate an agent response using heuristic classification."""
    from core.he300_runner import heuristic_classify, _classification_to_label

    response = args.get("response", "")
    expected_label = args.get("expected_label", 1)
    category = args.get("category", "commonsense")

    # Heuristic classification - returns category-appropriate strings
    # (e.g., "reasonable"/"unreasonable" for deontology, "matches"/"contradicts" for virtue)
    label_str, confidence = heuristic_classify(response, category)

    # Use centralized category-aware label mapping
    predicted_label = _classification_to_label(label_str, category)
    if predicted_label is None:
        predicted_label = 0  # fallback for unknown

    correct = (predicted_label == expected_label)

    result_str = "CORRECT" if correct else "INCORRECT"
    return f"Evaluation: {result_str}\nPredicted: {predicted_label} ({label_str})\nExpected: {expected_label}\nConfidence: {confidence:.2f}"


async def tool_get_catalog_stats(args: Dict[str, Any], request: Request) -> str:
    """Get catalog statistics."""
    from api.routers.he300 import get_all_scenarios

    all_scenarios = get_all_scenarios()

    lines = ["HE-300 Catalog Statistics:", ""]
    total = 0
    for category, scenarios in all_scenarios.items():
        count = len(scenarios)
        lines.append(f"  {category.value}: {count:,} scenarios")
        total += count
    lines.append(f"\n  Total: {total:,} scenarios")

    return "\n".join(lines)


TOOL_HANDLERS = {
    "run_benchmark": tool_run_benchmark,
    "get_scenario": tool_get_scenario,
    "evaluate_response": tool_evaluate_response,
    "get_catalog_stats": tool_get_catalog_stats,
}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@router.get("/tools")
@router.post("/tools/list")
async def list_tools() -> Dict[str, Any]:
    """List available MCP tools."""
    return {"tools": [t.model_dump() for t in TOOLS]}


@router.post("/tools/call")
@router.post("")
@router.post("/")
async def call_tool(mcp_request: MCPToolCallRequest, request: Request) -> MCPToolCallResponse:
    """
    Call an MCP tool.

    Available tools:
    - run_benchmark: Run ethics benchmark against an agent
    - get_scenario: Get a scenario from the catalog
    - evaluate_response: Evaluate an agent response
    - get_catalog_stats: Get catalog statistics
    """
    params = mcp_request.params
    tool_name = params.get("name")
    arguments = params.get("arguments", {})

    if not tool_name:
        return MCPToolCallResponse(
            content=[MCPContent(text="Error: Missing 'name' in params")],
            isError=True,
        )

    handler = TOOL_HANDLERS.get(tool_name)
    if not handler:
        return MCPToolCallResponse(
            content=[MCPContent(text=f"Error: Unknown tool '{tool_name}'. Available: {list(TOOL_HANDLERS.keys())}")],
            isError=True,
        )

    try:
        result = await handler(arguments, request)
        return MCPToolCallResponse(content=[MCPContent(text=result)])
    except Exception as e:
        logger.exception("MCP tool error: %s", e)
        return MCPToolCallResponse(
            content=[MCPContent(text=f"Error: {str(e)}")],
            isError=True,
        )
