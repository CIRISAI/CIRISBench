"""
Container Management Router for Purple Agents.

Provides endpoints to start, stop, and check status of purple agent containers.
"""

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import logging
import os

logger = logging.getLogger(__name__)

router = APIRouter(
    prefix="/api/containers",
    tags=["containers"],
    responses={404: {"description": "Not found"}},
)

# Container configurations
CONTAINER_CONFIGS = {
    "eee-purple-agent": {
        "image": "ghcr.io/cirisai/cirisbench:eee-purple",
        "port": 9000,
        "health_endpoint": "/health",
        "env_vars": ["OPENROUTER_API_KEY", "TOGETHER_API_KEY", "LLM_MODEL", "ETHICAL_GUIDANCE", "IDENTITY_PROFILE"],
        "description": "EEE Purple Agent - EthicsEngine reasoning pipeline",
    },
    "ciris-agent": {
        "image": "ghcr.io/cirisai/ciris-agent:purple",
        "port": 9001,
        "health_endpoint": "/health",
        "env_vars": ["TOGETHER_API_KEY", "OPENROUTER_API_KEY", "LLM_MODEL"],
        "description": "CIRIS Agent - Full H3ERE reasoning pipeline",
    },
}


class ContainerStatus(BaseModel):
    name: str
    status: str  # running, stopped, not_found, error
    port: Optional[int] = None
    health: Optional[str] = None  # healthy, unhealthy, unknown
    image: Optional[str] = None
    error: Optional[str] = None


class ContainerStartRequest(BaseModel):
    env_vars: Optional[Dict[str, str]] = None
    model: Optional[str] = None
    ethical_guidance: Optional[str] = None
    identity_profile: Optional[str] = None


class ContainerListResponse(BaseModel):
    containers: List[ContainerStatus]


def get_docker_client():
    """Get Docker client, handling import errors gracefully."""
    try:
        import docker
        return docker.from_env()
    except ImportError:
        logger.warning("Docker SDK not installed. Container management disabled.")
        return None
    except Exception as e:
        logger.error(f"Failed to connect to Docker: {e}")
        return None


def check_container_health(port: int, endpoint: str = "/health") -> str:
    """Check if container is responding on health endpoint."""
    import httpx
    try:
        response = httpx.get(f"http://localhost:{port}{endpoint}", timeout=5.0)
        if response.status_code == 200:
            return "healthy"
        return "unhealthy"
    except Exception:
        return "unknown"


@router.get("", response_model=ContainerListResponse)
async def list_containers():
    """List all configured purple agent containers and their status."""
    client = get_docker_client()
    containers = []

    for name, config in CONTAINER_CONFIGS.items():
        container_status = ContainerStatus(
            name=name,
            status="not_found",
            port=config["port"],
            image=config["image"],
        )

        if client:
            try:
                container = client.containers.get(name)
                container_status.status = container.status
                if container.status == "running":
                    container_status.health = check_container_health(
                        config["port"],
                        config["health_endpoint"]
                    )
            except Exception:
                container_status.status = "not_found"

        containers.append(container_status)

    return ContainerListResponse(containers=containers)


@router.get("/{name}", response_model=ContainerStatus)
async def get_container_status(name: str):
    """Get status of a specific container."""
    if name not in CONTAINER_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown container: {name}. Available: {list(CONTAINER_CONFIGS.keys())}"
        )

    config = CONTAINER_CONFIGS[name]
    container_status = ContainerStatus(
        name=name,
        status="not_found",
        port=config["port"],
        image=config["image"],
    )

    client = get_docker_client()
    if client:
        try:
            container = client.containers.get(name)
            container_status.status = container.status
            if container.status == "running":
                container_status.health = check_container_health(
                    config["port"],
                    config["health_endpoint"]
                )
        except Exception:
            container_status.status = "not_found"

    return container_status


@router.post("/{name}/start", response_model=ContainerStatus)
async def start_container(name: str, request: ContainerStartRequest = None):
    """Start a purple agent container."""
    if name not in CONTAINER_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown container: {name}"
        )

    client = get_docker_client()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Docker is not available"
        )

    config = CONTAINER_CONFIGS[name]

    # Check if already running
    try:
        existing = client.containers.get(name)
        if existing.status == "running":
            return ContainerStatus(
                name=name,
                status="running",
                port=config["port"],
                health=check_container_health(config["port"], config["health_endpoint"]),
                image=config["image"],
            )
        # Remove stopped container
        existing.remove()
    except Exception:
        pass  # Container doesn't exist

    # Build environment variables
    env = {}
    for var in config["env_vars"]:
        # First check request, then environment
        if request and request.env_vars and var in request.env_vars:
            env[var] = request.env_vars[var]
        elif os.environ.get(var):
            env[var] = os.environ.get(var)

    # Add model config if provided
    if request:
        if request.model:
            env["LLM_MODEL"] = request.model
        if request.ethical_guidance:
            env["ETHICAL_GUIDANCE"] = request.ethical_guidance
        if request.identity_profile:
            env["IDENTITY_PROFILE"] = request.identity_profile

    try:
        container = client.containers.run(
            config["image"],
            name=name,
            detach=True,
            network_mode="host",
            environment=env,
            remove=False,
        )

        # Wait a moment for startup
        import time
        time.sleep(2)

        container.reload()
        health = "unknown"
        if container.status == "running":
            health = check_container_health(config["port"], config["health_endpoint"])

        return ContainerStatus(
            name=name,
            status=container.status,
            port=config["port"],
            health=health,
            image=config["image"],
        )
    except Exception as e:
        logger.error(f"Failed to start container {name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to start container: {str(e)}"
        )


@router.post("/{name}/stop", response_model=ContainerStatus)
async def stop_container(name: str):
    """Stop a purple agent container."""
    if name not in CONTAINER_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown container: {name}"
        )

    client = get_docker_client()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Docker is not available"
        )

    config = CONTAINER_CONFIGS[name]

    try:
        container = client.containers.get(name)
        container.stop(timeout=10)
        container.remove()

        return ContainerStatus(
            name=name,
            status="stopped",
            port=config["port"],
            image=config["image"],
        )
    except Exception as e:
        if "No such container" in str(e) or "404" in str(e):
            return ContainerStatus(
                name=name,
                status="not_found",
                port=config["port"],
                image=config["image"],
            )
        logger.error(f"Failed to stop container {name}: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to stop container: {str(e)}"
        )


@router.get("/{name}/logs")
async def get_container_logs(name: str, tail: int = 100):
    """Get recent logs from a container."""
    if name not in CONTAINER_CONFIGS:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown container: {name}"
        )

    client = get_docker_client()
    if not client:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Docker is not available"
        )

    try:
        container = client.containers.get(name)
        logs = container.logs(tail=tail, timestamps=True).decode("utf-8")
        return {"name": name, "logs": logs}
    except Exception as e:
        if "No such container" in str(e) or "404" in str(e):
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Container {name} not found"
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to get logs: {str(e)}"
        )
