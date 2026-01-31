# =============================================================================
# CIRISBench Unified Image for AgentBeats
# =============================================================================
#
# Single container with CIRISNode + EthicsEngine pre-configured for AgentBeats.
# Supports MCP (Model Context Protocol) and A2A (Agent-to-Agent) connections.
#
# Build:
#   docker build -t cirisbench:agentbeats -f docker/agentbeats/Dockerfile .
#
# Run:
#   docker run -p 8000:8000 -p 8080:8080 cirisbench:agentbeats
#
# =============================================================================

FROM python:3.11-slim AS base

# Build arguments
ARG BUILD_DATE=unknown
ARG VERSION=0.1.0
ARG GIT_HASH=unknown

# Labels for AgentBeats discovery
LABEL org.opencontainers.image.title="CIRISBench"
LABEL org.opencontainers.image.description="CIRIS AI Agent Benchmarking - MCP & A2A Ready"
LABEL org.opencontainers.image.version="${VERSION}"
LABEL org.opencontainers.image.created="${BUILD_DATE}"
LABEL org.opencontainers.image.revision="${GIT_HASH}"
LABEL org.opencontainers.image.vendor="CIRIS AI"
LABEL org.opencontainers.image.source="https://github.com/CIRISAI/CIRISBench"
# AgentBeats specific labels
LABEL ai.agentbeats.protocols="mcp,a2a"
LABEL ai.agentbeats.type="benchmark"
LABEL ai.agentbeats.capabilities="ethics-evaluation,agent-testing,trace-validation"

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PIP_NO_CACHE_DIR=1
ENV PIP_DISABLE_PIP_VERSION_CHECK=1

# =============================================================================
# Builder stage - install dependencies
# =============================================================================
FROM base AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Copy and install CIRISBench core
COPY pyproject.toml README.md ./
COPY cirisbench/ ./cirisbench/
RUN pip install --prefix=/install .

# Install EthicsEngine dependencies
COPY engine/requirements.txt ./engine-requirements.txt
RUN pip install --prefix=/install -r engine-requirements.txt

# Install CIRISNode dependencies
COPY CIRISNode/requirements.txt ./cirisnode-requirements.txt
RUN pip install --prefix=/install -r cirisnode-requirements.txt

# =============================================================================
# Runtime stage
# =============================================================================
FROM base AS runtime

# Install runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libpq5 \
    supervisor \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Create app user
RUN useradd -m -u 1000 -s /bin/bash ciris \
    && mkdir -p /app /data /config \
    && chown -R ciris:ciris /app /data /config

WORKDIR /app

# Copy application code
COPY --chown=ciris:ciris cirisbench/ ./cirisbench/
COPY --chown=ciris:ciris engine/ ./engine/
COPY --chown=ciris:ciris CIRISNode/cirisnode/ ./cirisnode/

# Copy AgentBeats configuration
COPY --chown=ciris:ciris docker/agentbeats/supervisord.conf /etc/supervisor/conf.d/supervisord.conf
COPY --chown=ciris:ciris docker/agentbeats/entrypoint.sh /entrypoint.sh
COPY --chown=ciris:ciris docker/agentbeats/agentbeats.json /app/agentbeats.json

RUN chmod +x /entrypoint.sh

# Environment defaults for AgentBeats
ENV PYTHONPATH=/app
ENV LOG_LEVEL=info

# CIRISNode settings
ENV CIRISNODE_HOST=0.0.0.0
ENV CIRISNODE_PORT=8000
ENV MCP_ENABLED=true
ENV A2A_ENABLED=true

# EthicsEngine settings
ENV EEE_HOST=0.0.0.0
ENV EEE_PORT=8080
ENV HE300_ENABLED=true
ENV DATA_DIR=engine/data

# AgentBeats mode - runs both services
ENV AGENTBEATS_MODE=true

# Expose ports
# 8000 - CIRISNode API (MCP/A2A)
# 8080 - EthicsEngine API (Benchmarks)
EXPOSE 8000 8080

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health && curl -f http://localhost:8080/health || exit 1

# Volumes for persistence
VOLUME ["/data", "/config"]

USER ciris

ENTRYPOINT ["/entrypoint.sh"]
CMD ["supervisord", "-c", "/etc/supervisor/conf.d/supervisord.conf"]
