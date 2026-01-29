# =============================================================================
# EEE (EthicsEngine Enterprise) Purple Agent
# =============================================================================
#
# Exposes the full EEE reasoning pipeline as a configurable purple agent
# for AgentBeats benchmarking.
#
# Build:
#   docker build -t cirisbench:eee-purple -f docker/eee-purple-agent/Dockerfile .
#
# Run with defaults:
#   docker run -p 9000:9000 -e OPENROUTER_API_KEY=xxx cirisbench:eee-purple
#
# Run with Utilitarian framework:
#   docker run -p 9000:9000 \
#     -e OPENROUTER_API_KEY=xxx \
#     -e ETHICAL_GUIDANCE=Utilitarian \
#     cirisbench:eee-purple
#
# Run with deep reasoning:
#   docker run -p 9000:9000 \
#     -e OPENROUTER_API_KEY=xxx \
#     -e REASONING_LEVEL=high \
#     -e REASONING_MAX_DEPTH=3 \
#     cirisbench:eee-purple
#
# =============================================================================

FROM python:3.11-slim

# Labels
LABEL org.opencontainers.image.title="CIRISBench EEE Purple Agent"
LABEL org.opencontainers.image.description="EthicsEngine Enterprise reasoning pipeline as a purple agent"
LABEL org.opencontainers.image.version="1.0.0"
LABEL org.opencontainers.image.vendor="CIRIS AI"
LABEL org.opencontainers.image.source="https://github.com/CIRISAI/CIRISBench"
LABEL ai.agentbeats.protocols="a2a"
LABEL ai.agentbeats.type="purple-agent"

# Environment defaults (all configurable at runtime)
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# LLM Configuration
ENV LLM_PROVIDER=openrouter
ENV LLM_MODEL=openai/gpt-4o-mini
ENV LLM_TEMPERATURE=0.1
ENV LLM_MAX_TOKENS=300
ENV LLM_TIMEOUT=60

# Ethical Guidance Framework (Utilitarian | Deontological | Virtue | Fairness | Agentic | Neutral)
ENV ETHICAL_GUIDANCE=Neutral

# Identity Profile (Neutral | Agentic_Identity | NIMHs | Jiminies | Megacricks)
ENV IDENTITY_PROFILE=Neutral

# AG2 Reasoning Configuration
ENV REASONING_LEVEL=basic
# ENV REASONING_MAX_DEPTH=0  # Uncomment to override level preset
ENV REASONING_METHOD=beam_search
ENV REASONING_BEAM_SIZE=3
ENV REASONING_ANSWER_APPROACH=pool

# System Prompt Configuration
ENV SYSTEM_PROMPT_BASE="You are an ethical reasoning assistant."
# ENV SYSTEM_PROMPT_APPEND=""  # Additional prompt text
ENV INCLUDE_GUIDANCE_PROMPT=true
ENV INCLUDE_IDENTITY_CONTEXT=true

# Response Configuration
ENV RESPONSE_FORMAT=plain
ENV REQUIRE_REASONING=true

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir \
    fastapi \
    uvicorn \
    httpx \
    pydantic

# Copy agent code
COPY tests/eee_purple_agent.py /app/agent.py

# Create non-root user
RUN useradd -m -u 1000 agent && chown -R agent:agent /app
USER agent

# Expose port
EXPOSE 9000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:9000/health')" || exit 1

# Run agent
ENTRYPOINT ["python", "/app/agent.py", "--host", "0.0.0.0", "--port", "9000"]
