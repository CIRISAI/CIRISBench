#!/bin/bash
#
# setup-gpu-ollama.sh - Enable GPU and pull high-spec model
#
# GPU: A10 (24 GB PCIe)
# Server: 30 vCPUs, 200 GiB RAM, 1.4 TiB SSD
#
# HITL: This script enables GPU acceleration and downloads a large model
# Security: See docs/sec.md - model download from Ollama registry
#

set -e

HOST="ubuntu@163.192.58.165"
INSTALL_DIR="/opt/he300"

echo "=========================================="
echo "  HE-300 GPU Setup"
echo "  GPU: A10 (24 GB VRAM)"
echo "=========================================="
echo ""

# Step 1: Verify GPU access
echo "[1/5] Verifying GPU access..."
ssh $HOST "nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv"
echo ""

# Step 2: Update docker-compose with GPU support
echo "[2/5] Enabling GPU for Ollama..."
ssh $HOST "cat > $INSTALL_DIR/docker-compose.override.yml << 'EOF'
services:
  ethicsengine:
    command: [\"uvicorn\", \"api.main:app\", \"--host\", \"0.0.0.0\", \"--port\", \"8080\"]
    healthcheck:
      test: [\"CMD\", \"curl\", \"-f\", \"http://localhost:8080/health\"]
      interval: 30s
      timeout: 10s
      retries: 3
    environment:
      - OLLAMA_HOST=http://ollama:11434

  cirisnode:
    environment:
      - OLLAMA_HOST=http://ollama:11434

  ollama:
    image: ollama/ollama:latest
    container_name: he300-ollama
    ports:
      - \"11434:11434\"
    volumes:
      - ollama_data:/root/.ollama
    networks:
      - he300-net
    restart: unless-stopped
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: 1
              capabilities: [gpu]

volumes:
  ollama_data:
    name: he300-ollama-data
EOF"
echo "✅ GPU configuration written"

# Step 3: Restart Ollama with GPU
echo ""
echo "[3/5] Restarting Ollama with GPU support..."
ssh $HOST "cd $INSTALL_DIR && docker compose up -d ollama --force-recreate"
echo ""

# Step 4: Wait for Ollama to be ready
echo "[4/5] Waiting for Ollama to initialize..."
sleep 5
ssh $HOST "docker exec he300-ollama ollama --version"
echo ""

# Step 5: Pull high-spec model (Qwen2.5 7B - fits well on 24GB A10)
echo "[5/5] Pulling Qwen2.5:7B model (this will take a few minutes)..."
echo ""
echo "Model: Qwen2.5:7B-Instruct"
echo "Size: ~4.7GB download"
echo "VRAM: ~8GB loaded"
echo ""
ssh $HOST "docker exec he300-ollama ollama pull qwen2.5:7b"

echo ""
echo "=========================================="
echo "  GPU Setup Complete!"
echo "=========================================="
echo ""
echo "GPU Model: Qwen2.5:7B loaded on A10"
echo ""
echo "Verify with:"
echo "  ssh $HOST 'docker exec he300-ollama ollama list'"
echo ""
echo "Test inference:"
echo "  ssh $HOST 'docker exec he300-ollama ollama run qwen2.5:7b \"Hello\"'"
echo ""
