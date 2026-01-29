# =============================================================================
# Multi-Provider Purple Agent for HE-300 Benchmarking
# =============================================================================
#
# LLM-powered purple agent supporting multiple providers:
# - OpenRouter (Llama 4, Claude, GPT, Gemini, open-source)
# - Anthropic Direct (Claude)
# - OpenAI Direct (GPT)
# - Google Direct (Gemini)
# - Together.ai (Open-source models)
#
# Build:
#   docker build -t cirisbench:multi-provider -f docker/multi-provider-agent/Dockerfile .
#
# Run:
#   docker run -p 9000:9000 \
#     -e OPENROUTER_API_KEY=your-key \
#     cirisbench:multi-provider \
#     --provider openrouter --model meta-llama/llama-4-maverick
#
# =============================================================================

FROM python:3.11-slim

# Labels
LABEL org.opencontainers.image.title="CIRISBench Multi-Provider Purple Agent"
LABEL org.opencontainers.image.description="LLM-powered purple agent for HE-300 benchmark with multi-provider support"
LABEL org.opencontainers.image.version="2.0.0"
LABEL org.opencontainers.image.vendor="CIRIS AI"
LABEL org.opencontainers.image.source="https://github.com/CIRISAI/CIRISBench"
LABEL ai.agentbeats.protocols="a2a,mcp"
LABEL ai.agentbeats.type="purple-agent"

# Environment
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Default provider/model (can be overridden via args)
ENV LLM_PROVIDER=openrouter
ENV LLM_MODEL=meta-llama/llama-4-maverick

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic

# Copy agent code
COPY tests/multi_provider_agent.py /app/agent.py

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Expose port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health')" || exit 1

# Default entrypoint - uses env vars, can be overridden with docker run args
ENTRYPOINT ["python", "/app/agent.py", "--host", "0.0.0.0", "--port", "9000"]
CMD ["--provider", "openrouter", "--model", "meta-llama/llama-4-maverick"]
