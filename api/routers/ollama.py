"""
Ollama Model Management API Router

Provides endpoints to list, pull, delete, and manage Ollama models.
Supports configuration of model endpoints from the UI.
"""

import logging
import asyncio
import httpx
from typing import Optional, List
from pathlib import Path

from fastapi import APIRouter, HTTPException, status, BackgroundTasks
from pydantic import BaseModel, Field

# Add project root to path for imports
import sys
project_root = Path(__file__).resolve().parents[2]
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

import os

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/ollama",
    tags=["ollama", "models"],
    responses={
        404: {"description": "Not found"},
        500: {"description": "Internal server error"},
    },
)

# --- Configuration ---
def get_ollama_host() -> str:
    """Get Ollama host from environment or use default."""
    return os.environ.get("OLLAMA_HOST", "http://localhost:11434")


# --- Schemas ---
class OllamaModelInfo(BaseModel):
    """Information about an Ollama model."""
    name: str
    model: str = Field(default="")
    modified_at: Optional[str] = None
    size: Optional[int] = None
    digest: Optional[str] = None
    details: Optional[dict] = None
    
class OllamaModelList(BaseModel):
    """List of Ollama models."""
    models: List[OllamaModelInfo]
    ollama_host: str


class OllamaPullRequest(BaseModel):
    """Request to pull a model."""
    model_name: str = Field(..., description="Model name to pull (e.g., 'gemma3:4b-it-q8_0')")
    

class OllamaPullResponse(BaseModel):
    """Response for model pull operation."""
    status: str
    model_name: str
    message: str


class OllamaDeleteRequest(BaseModel):
    """Request to delete a model."""
    model_name: str = Field(..., description="Model name to delete")


class OllamaGenerateRequest(BaseModel):
    """Request for text generation."""
    model: str
    prompt: str
    stream: bool = False
    options: Optional[dict] = None


class OllamaGenerateResponse(BaseModel):
    """Response from text generation."""
    model: str
    response: str
    done: bool
    total_duration: Optional[int] = None
    eval_count: Optional[int] = None


class OllamaConfigUpdate(BaseModel):
    """Update Ollama configuration."""
    ollama_host: Optional[str] = None


# --- Pull Status Tracking ---
_pull_tasks: dict = {}  # model_name -> {"status": str, "progress": str}


# --- API Endpoints ---

@router.get("/health")
async def ollama_health():
    """Check Ollama connectivity."""
    ollama_host = get_ollama_host()
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(f"{ollama_host}/api/tags")
            if response.status_code == 200:
                return {
                    "status": "healthy",
                    "ollama_host": ollama_host,
                    "message": "Ollama is reachable"
                }
            else:
                return {
                    "status": "unhealthy",
                    "ollama_host": ollama_host,
                    "message": f"Ollama returned status {response.status_code}"
                }
    except Exception as e:
        return {
            "status": "unhealthy",
            "ollama_host": ollama_host,
            "message": str(e)
        }


@router.get("/models", response_model=OllamaModelList)
async def list_models():
    """List all available Ollama models."""
    ollama_host = get_ollama_host()
    try:
        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(f"{ollama_host}/api/tags")
            response.raise_for_status()
            data = response.json()
            
            models = []
            for m in data.get("models", []):
                models.append(OllamaModelInfo(
                    name=m.get("name", ""),
                    model=m.get("model", m.get("name", "")),
                    modified_at=m.get("modified_at"),
                    size=m.get("size"),
                    digest=m.get("digest"),
                    details=m.get("details"),
                ))
            
            return OllamaModelList(models=models, ollama_host=ollama_host)
            
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Ollama API error: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Error listing models: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to list models: {str(e)}"
        )


@router.post("/models/pull", response_model=OllamaPullResponse)
async def pull_model(request: OllamaPullRequest, background_tasks: BackgroundTasks):
    """
    Pull a model from Ollama registry.
    
    This operation runs in the background. Use /models/pull/status/{model_name} 
    to check progress.
    """
    model_name = request.model_name
    ollama_host = get_ollama_host()
    
    # Check if already pulling
    if model_name in _pull_tasks and _pull_tasks[model_name].get("status") == "pulling":
        return OllamaPullResponse(
            status="already_pulling",
            model_name=model_name,
            message="Model is already being pulled"
        )
    
    # Initialize tracking
    _pull_tasks[model_name] = {"status": "starting", "progress": "0%"}
    
    async def do_pull():
        try:
            _pull_tasks[model_name] = {"status": "pulling", "progress": "0%"}
            
            async with httpx.AsyncClient(timeout=3600.0) as client:  # 1 hour timeout
                async with client.stream(
                    "POST",
                    f"{ollama_host}/api/pull",
                    json={"name": model_name, "stream": True}
                ) as response:
                    async for line in response.aiter_lines():
                        if line:
                            import json
                            try:
                                data = json.loads(line)
                                if "completed" in data and "total" in data:
                                    pct = int((data["completed"] / data["total"]) * 100)
                                    _pull_tasks[model_name]["progress"] = f"{pct}%"
                                if data.get("status") == "success":
                                    _pull_tasks[model_name] = {"status": "completed", "progress": "100%"}
                            except json.JSONDecodeError:
                                pass
            
            _pull_tasks[model_name] = {"status": "completed", "progress": "100%"}
            
        except Exception as e:
            logger.error(f"Error pulling model {model_name}: {e}")
            _pull_tasks[model_name] = {"status": "error", "progress": str(e)}
    
    background_tasks.add_task(do_pull)
    
    return OllamaPullResponse(
        status="started",
        model_name=model_name,
        message=f"Started pulling {model_name}. Check /ollama/models/pull/status/{model_name}"
    )


@router.get("/models/pull/status/{model_name}")
async def pull_status(model_name: str):
    """Get the status of a model pull operation."""
    if model_name not in _pull_tasks:
        return {"status": "unknown", "model_name": model_name, "message": "No pull operation found"}
    
    return {
        "model_name": model_name,
        **_pull_tasks[model_name]
    }


@router.delete("/models/{model_name}")
async def delete_model(model_name: str):
    """Delete an Ollama model."""
    ollama_host = get_ollama_host()
    
    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            response = await client.delete(
                f"{ollama_host}/api/delete",
                json={"name": model_name}
            )
            
            if response.status_code == 200:
                return {
                    "status": "deleted",
                    "model_name": model_name,
                    "message": f"Model {model_name} deleted successfully"
                }
            elif response.status_code == 404:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail=f"Model {model_name} not found"
                )
            else:
                raise HTTPException(
                    status_code=response.status_code,
                    detail=f"Failed to delete model: {response.text}"
                )
                
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Ollama API error: {e.response.text}"
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting model {model_name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete model: {str(e)}"
        )


@router.post("/generate", response_model=OllamaGenerateResponse)
async def generate_text(request: OllamaGenerateRequest):
    """Generate text using an Ollama model (non-streaming)."""
    ollama_host = get_ollama_host()
    
    try:
        async with httpx.AsyncClient(timeout=300.0) as client:
            response = await client.post(
                f"{ollama_host}/api/generate",
                json={
                    "model": request.model,
                    "prompt": request.prompt,
                    "stream": False,
                    "options": request.options or {}
                }
            )
            response.raise_for_status()
            data = response.json()
            
            return OllamaGenerateResponse(
                model=data.get("model", request.model),
                response=data.get("response", ""),
                done=data.get("done", True),
                total_duration=data.get("total_duration"),
                eval_count=data.get("eval_count"),
            )
            
    except httpx.HTTPStatusError as e:
        raise HTTPException(
            status_code=e.response.status_code,
            detail=f"Generation failed: {e.response.text}"
        )
    except Exception as e:
        logger.error(f"Error generating text: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Generation failed: {str(e)}"
        )


@router.get("/config")
async def get_config():
    """Get current Ollama configuration."""
    return {
        "ollama_host": get_ollama_host(),
        "configured_via": "OLLAMA_HOST environment variable"
    }


@router.post("/config")
async def update_config(config: OllamaConfigUpdate):
    """
    Update Ollama configuration.
    
    Note: This updates the environment variable for the current process.
    For persistent changes, update the .env file or Docker environment.
    """
    if config.ollama_host:
        os.environ["OLLAMA_HOST"] = config.ollama_host
        
    return {
        "status": "updated",
        "ollama_host": get_ollama_host(),
        "message": "Configuration updated for current session"
    }
