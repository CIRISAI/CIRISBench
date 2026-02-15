#!/bin/bash
# Start CIRISBench with OpenAI configuration
cd /home/emoore/CIRISBench
source .venv/bin/activate

export LLM_PROVIDER=openai
export LLM_MODEL=gpt-4o-mini
export OPENAI_API_KEY=$(grep -E "^OPENAI_API_KEY=" /home/emoore/CIRISAgent/.env | cut -d= -f2-)
export AUTH_ENABLED=false

echo "Starting CIRISBench with:"
echo "  LLM_PROVIDER=$LLM_PROVIDER"
echo "  LLM_MODEL=$LLM_MODEL"
echo "  OPENAI_API_KEY=${OPENAI_API_KEY:0:15}..."

python -m uvicorn engine.api.main:app --host 127.0.0.1 --port 8080 --log-level info
