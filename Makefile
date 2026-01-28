.PHONY: help install install-dev install-all clean test lint format \
        engine-up engine-down engine-logs benchmark docker-build \
        check typecheck pre-commit \
        agentbeats agentbeats-gpu agentbeats-full agentbeats-down agentbeats-logs

PYTHON ?= python3
VENV ?= .venv
PIP := $(VENV)/bin/pip
PYTEST := $(VENV)/bin/pytest
RUFF := $(VENV)/bin/ruff
MYPY := $(VENV)/bin/mypy

help:
	@echo "CIRISBench - Benchmarking infrastructure for CIRIS AI agents"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@echo "Setup:"
	@echo "  install       Install base dependencies"
	@echo "  install-dev   Install with dev dependencies"
	@echo "  install-all   Install all dependencies (dev + llm)"
	@echo "  clean         Remove build artifacts and cache"
	@echo ""
	@echo "Development:"
	@echo "  test          Run all tests"
	@echo "  test-engine   Run engine tests only"
	@echo "  test-infra    Run infra tests only"
	@echo "  lint          Run linter (ruff)"
	@echo "  format        Format code (ruff)"
	@echo "  typecheck     Run type checker (mypy)"
	@echo "  check         Run all checks (lint + typecheck + test)"
	@echo ""
	@echo "Services:"
	@echo "  engine-up     Start EthicsEngine server"
	@echo "  engine-down   Stop EthicsEngine server"
	@echo "  engine-logs   View EthicsEngine logs"
	@echo ""
	@echo "Benchmarks:"
	@echo "  benchmark     Run HE-300 benchmark (mock)"
	@echo "  benchmark-ollama  Run HE-300 with Ollama"
	@echo ""
	@echo "Docker:"
	@echo "  docker-build  Build all Docker images"
	@echo "  docker-up     Start all services via Docker Compose"
	@echo "  docker-down   Stop all Docker services"
	@echo ""
	@echo "AgentBeats:"
	@echo "  agentbeats       Start unified CIRISBench container"
	@echo "  agentbeats-gpu   Start with Ollama GPU inference"
	@echo "  agentbeats-full  Start full stack (Postgres/Redis/Ollama)"
	@echo "  agentbeats-down  Stop AgentBeats services"
	@echo "  agentbeats-logs  View AgentBeats logs"

# ============================================================================
# Setup
# ============================================================================

$(VENV)/bin/activate:
	$(PYTHON) -m venv $(VENV)
	$(PIP) install --upgrade pip

install: $(VENV)/bin/activate
	$(PIP) install -e .

install-dev: $(VENV)/bin/activate
	$(PIP) install -e ".[dev]"

install-all: $(VENV)/bin/activate
	$(PIP) install -e ".[all]"

clean:
	rm -rf build/ dist/ *.egg-info/
	rm -rf .pytest_cache/ .mypy_cache/ .ruff_cache/
	rm -rf htmlcov/ .coverage coverage.xml
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type f -name "*.pyc" -delete 2>/dev/null || true

# ============================================================================
# Development
# ============================================================================

test: install-dev
	$(PYTEST) tests/ engine/tests/ -v

test-engine: install-dev
	$(PYTEST) engine/tests/ -v

test-infra: install-dev
	$(PYTEST) infra/tests/ -v

lint:
	$(RUFF) check .

format:
	$(RUFF) format .
	$(RUFF) check --fix .

typecheck:
	$(MYPY) cirisbench/ engine/

check: lint typecheck test

pre-commit:
	$(VENV)/bin/pre-commit run --all-files

# ============================================================================
# Services
# ============================================================================

engine-up:
	cd engine && $(PYTHON) -m uvicorn api.main:app --host 0.0.0.0 --port 8080 --reload

engine-down:
	@pkill -f "uvicorn api.main:app" || echo "No engine process found"

engine-logs:
	@tail -f engine/logs/*.log 2>/dev/null || echo "No log files found"

# ============================================================================
# Benchmarks
# ============================================================================

benchmark:
	@echo "Running HE-300 benchmark (mock mode)..."
	cd engine && $(PYTHON) run_pipeline.py --mock --sample-size 10

benchmark-ollama:
	@echo "Running HE-300 benchmark with Ollama..."
	cd engine && $(PYTHON) run_pipeline.py --model ollama/llama3.2 --sample-size 50

# ============================================================================
# Docker
# ============================================================================

docker-build:
	docker build -t cirisbench-engine:latest engine/
	@echo "Built cirisbench-engine:latest"

docker-up:
	docker compose -f infra/docker/docker-compose.yml up -d

docker-down:
	docker compose -f infra/docker/docker-compose.yml down

docker-logs:
	docker compose -f infra/docker/docker-compose.yml logs -f

# ============================================================================
# AgentBeats (Unified Deployment)
# ============================================================================

agentbeats:
	@echo "Starting CIRISBench for AgentBeats..."
	docker compose -f docker/agentbeats/docker-compose.yml up -d
	@echo ""
	@echo "CIRISNode (MCP/A2A): http://localhost:8000"
	@echo "EthicsEngine:        http://localhost:8080"

agentbeats-gpu:
	@echo "Starting CIRISBench with GPU inference..."
	docker compose -f docker/agentbeats/docker-compose.yml --profile gpu up -d
	@echo ""
	@echo "CIRISNode (MCP/A2A): http://localhost:8000"
	@echo "EthicsEngine:        http://localhost:8080"
	@echo "Ollama:              http://localhost:11434"

agentbeats-full:
	@echo "Starting CIRISBench full stack..."
	docker compose -f docker/agentbeats/docker-compose.yml --profile full up -d
	@echo ""
	@echo "CIRISNode (MCP/A2A): http://localhost:8000"
	@echo "EthicsEngine:        http://localhost:8080"
	@echo "Ollama:              http://localhost:11434"
	@echo "PostgreSQL:          localhost:5432"
	@echo "Redis:               localhost:6379"

agentbeats-down:
	docker compose -f docker/agentbeats/docker-compose.yml --profile full down

agentbeats-logs:
	docker compose -f docker/agentbeats/docker-compose.yml logs -f

agentbeats-build:
	docker compose -f docker/agentbeats/docker-compose.yml build
