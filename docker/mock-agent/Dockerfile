# =============================================================================
# Mock Purple Agent for HE-300 Testing
# =============================================================================
#
# A baseline purple agent that demonstrates A2A and MCP protocol compliance.
# Uses heuristic-based classification for testing the benchmark infrastructure.
#
# Build:
#   docker build -t cirisbench:mock-agent -f docker/mock-agent/Dockerfile .
#
# Run:
#   docker run -p 9000:9000 cirisbench:mock-agent
#
# =============================================================================

FROM python:3.11-slim

# Labels
LABEL org.opencontainers.image.title="CIRISBench Mock Purple Agent"
LABEL org.opencontainers.image.description="Baseline purple agent for HE-300 benchmark testing"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="CIRIS AI"
LABEL org.opencontainers.image.source="https://github.com/CIRISAI/CIRISBench"
LABEL ai.agentbeats.protocols="a2a,mcp"
LABEL ai.agentbeats.type="purple-agent"

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pydantic

# Copy agent code
COPY tests/mock_purple_agent.py /app/agent.py

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Expose port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health')" || exit 1

# Run agent
CMD ["python", "/app/agent.py", "--host", "0.0.0.0", "--port", "9000"]
