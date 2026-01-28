#!/usr/bin/env python3
"""
Mock Purple Agent for HE-300 Testing.

This creates a simple FastAPI server that responds to A2A and MCP protocol
requests, simulating a purple agent being evaluated by CIRISBench.

The agent uses a simple heuristic to classify scenarios, making it useful
for end-to-end testing of the benchmark infrastructure.

Usage:
    python tests/mock_purple_agent.py [--port 9000]
"""

import argparse
import logging
import random
import uvicorn
from fastapi import FastAPI, Request
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Mock Purple Agent",
    description="A mock agent for HE-300 benchmark testing",
    version="1.0.0",
)


# --- Models ---

class JSONRPCRequest(BaseModel):
    jsonrpc: str = "2.0"
    method: str
    params: Dict[str, Any] = {}
    id: Any = None


class JSONRPCResponse(BaseModel):
    jsonrpc: str = "2.0"
    result: Any = None
    error: Optional[Dict[str, Any]] = None
    id: Any = None


class MCPRequest(BaseModel):
    method: str
    params: Dict[str, Any] = {}


# --- Agent Logic ---

def classify_scenario(scenario_text: str) -> tuple[str, str]:
    """
    Simple heuristic-based classification for testing.

    In a real agent, this would use an LLM to analyze the scenario.
    For testing purposes, we use keyword matching with some randomness
    to simulate realistic agent behavior.

    Returns:
        Tuple of (classification, explanation)
    """
    scenario_lower = scenario_text.lower()

    # Keywords suggesting unethical behavior
    unethical_keywords = [
        "steal", "lie", "cheat", "deceive", "harm", "hurt", "kill",
        "discriminate", "unfair", "bribe", "corrupt", "illegal",
        "manipulate", "exploit", "abuse", "fraud", "betray",
        "copy answers", "plagiarize", "take credit", "blame others"
    ]

    # Keywords suggesting ethical behavior
    ethical_keywords = [
        "help", "donate", "volunteer", "honest", "fair", "return",
        "share", "protect", "care", "support", "apologize", "admit",
        "respect", "kind", "generous", "truthful", "responsible"
    ]

    unethical_score = sum(1 for kw in unethical_keywords if kw in scenario_lower)
    ethical_score = sum(1 for kw in ethical_keywords if kw in scenario_lower)

    # Add slight randomness (5% chance to flip) to simulate realistic agent behavior
    if random.random() < 0.05:
        unethical_score, ethical_score = ethical_score, unethical_score

    if unethical_score > ethical_score:
        return "UNETHICAL", f"This action appears to involve harmful or dishonest behavior."
    elif ethical_score > unethical_score:
        return "ETHICAL", f"This action appears to be morally sound and beneficial."
    else:
        # When unclear, make a random decision (simulating agent uncertainty)
        if random.random() > 0.5:
            return "ETHICAL", "After consideration, this seems acceptable."
        else:
            return "UNETHICAL", "After consideration, this raises ethical concerns."


# --- API Endpoints ---

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "agent": "mock-purple-agent"}


@app.get("/.well-known/agent.json")
async def agent_manifest():
    """Agent manifest for A2A discovery."""
    return {
        "name": "Mock Purple Agent",
        "version": "1.0.0",
        "description": "A mock agent for HE-300 benchmark testing",
        "capabilities": ["ethics-evaluation"],
        "protocols": ["a2a", "mcp"],
    }


@app.post("/a2a")
async def handle_a2a(request: JSONRPCRequest):
    """
    Handle A2A protocol requests.

    Supports:
    - benchmark.evaluate: Evaluate an ethical scenario
    - agent.info: Get agent information
    """
    logger.info(f"A2A request: method={request.method}, id={request.id}")

    if request.method == "benchmark.evaluate":
        scenario = request.params.get("scenario", "")
        scenario_id = request.params.get("scenario_id", "unknown")

        classification, explanation = classify_scenario(scenario)

        logger.info(f"Evaluated {scenario_id}: {classification}")

        return JSONRPCResponse(
            id=request.id,
            result={
                "response": f"{classification}\n\n{explanation}",
                "classification": classification.lower(),
                "scenario_id": scenario_id,
            }
        )

    elif request.method == "agent.info":
        return JSONRPCResponse(
            id=request.id,
            result={
                "name": "Mock Purple Agent",
                "version": "1.0.0",
                "model": "heuristic-v1",
            }
        )

    else:
        return JSONRPCResponse(
            id=request.id,
            error={
                "code": -32601,
                "message": f"Method not found: {request.method}"
            }
        )


@app.post("/mcp")
async def handle_mcp(request: MCPRequest):
    """
    Handle MCP protocol requests.

    Supports:
    - tools/call with name=evaluate_scenario
    """
    logger.info(f"MCP request: method={request.method}")

    if request.method == "tools/call":
        tool_name = request.params.get("name", "")
        arguments = request.params.get("arguments", {})

        if tool_name == "evaluate_scenario":
            scenario = arguments.get("scenario", "")
            scenario_id = arguments.get("scenario_id", "unknown")

            classification, explanation = classify_scenario(scenario)

            logger.info(f"MCP evaluated {scenario_id}: {classification}")

            return {
                "content": [
                    {
                        "type": "text",
                        "text": f"{classification}\n\n{explanation}",
                    }
                ],
                "isError": False,
            }
        else:
            return {
                "content": [{"type": "text", "text": f"Unknown tool: {tool_name}"}],
                "isError": True,
            }

    elif request.method == "tools/list":
        return {
            "tools": [
                {
                    "name": "evaluate_scenario",
                    "description": "Evaluate an ethical scenario",
                    "inputSchema": {
                        "type": "object",
                        "properties": {
                            "scenario": {"type": "string"},
                            "scenario_id": {"type": "string"},
                        },
                        "required": ["scenario"],
                    }
                }
            ]
        }

    else:
        return {"error": f"Method not found: {request.method}"}


def main():
    parser = argparse.ArgumentParser(description="Mock Purple Agent for HE-300 testing")
    parser.add_argument("--port", type=int, default=9000, help="Port to run on")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    logger.info(f"Starting Mock Purple Agent on {args.host}:{args.port}")
    logger.info(f"A2A endpoint: http://{args.host}:{args.port}/a2a")
    logger.info(f"MCP endpoint: http://{args.host}:{args.port}/mcp")

    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
