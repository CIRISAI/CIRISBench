#!/bin/bash
#
# Deploy HE-300 to remote server
# Target: ubuntu@163.192.58.165
#

set -e

HOST="ubuntu@163.192.58.165"
INSTALL_DIR="/opt/he300"

echo "=========================================="
echo "  HE-300 Remote Deployment"
echo "  Target: $HOST"
echo "=========================================="
echo ""

# Step 1: Install Docker if needed
echo "[1/6] Checking Docker installation..."
ssh $HOST "command -v docker" > /dev/null 2>&1 || {
    echo "Installing Docker..."
    ssh $HOST 'curl -fsSL https://get.docker.com | sudo sh'
    ssh $HOST 'sudo usermod -aG docker ubuntu'
}
echo "✅ Docker ready"

# Step 2: Create directories
echo ""
echo "[2/6] Creating directories..."
ssh $HOST "sudo mkdir -p $INSTALL_DIR/{data/models,data/results,logs} && sudo chown -R ubuntu:ubuntu $INSTALL_DIR"
echo "✅ Directories created"

# Step 3: Clone repositories
echo ""
echo "[3/6] Cloning repositories..."
ssh $HOST "cd $INSTALL_DIR && git clone https://github.com/rng-ops/CIRISNode.git --branch feature/eee-integration 2>/dev/null || (cd CIRISNode && git pull)"
ssh $HOST "cd $INSTALL_DIR && git clone https://github.com/rng-ops/ethicsengine_enterprise.git --branch feature/he300-api 2>/dev/null || (cd ethicsengine_enterprise && git pull)"
ssh $HOST "cd $INSTALL_DIR && git clone https://github.com/rng-ops/he300-integration.git 2>/dev/null || (cd he300-integration && git pull)"
echo "✅ Repositories cloned"

# Step 4: Generate .env file
echo ""
echo "[4/6] Generating configuration..."
ssh $HOST "cat > $INSTALL_DIR/.env << 'EOF'
ENVIRONMENT=staging
POSTGRES_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')
DB_PASSWORD=\${POSTGRES_PASSWORD}
REDIS_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')
JWT_SECRET=$(openssl rand -base64 32)
WEBHOOK_SECRET=$(openssl rand -hex 32)
CIRISNODE_PORT=8000
EEE_PORT=8080
DASHBOARD_PORT=3000
DEFAULT_MODEL=Qwen/Qwen2.5-7B-Instruct
MODEL_QUANTIZATION=Q4_K_M
MODEL_CACHE_DIR=$INSTALL_DIR/data/models
LOG_LEVEL=INFO
DATA_DIR=$INSTALL_DIR/data
LOG_DIR=$INSTALL_DIR/logs
EOF"
echo "✅ Configuration generated"

# Step 5: Copy docker-compose.yml
echo ""
echo "[5/6] Setting up Docker Compose..."
scp /Users/a/projects/ethics/staging/docker/docker-compose.he300.yml $HOST:$INSTALL_DIR/docker-compose.yml 2>/dev/null || {
    # Create compose file inline if scp fails
    ssh $HOST "cat > $INSTALL_DIR/docker-compose.yml << 'COMPOSE'
version: '3.8'

services:
  cirisnode:
    build:
      context: ./CIRISNode
      dockerfile: Dockerfile
    ports:
      - \"\${CIRISNODE_PORT:-8000}:8000\"
    environment:
      DATABASE_URL: postgresql://ciris:\${DB_PASSWORD}@db:5432/cirisnode
      REDIS_URL: redis://:\${REDIS_PASSWORD}@redis:6379/0
      JWT_SECRET: \${JWT_SECRET}
      EEE_BASE_URL: http://ethicsengine:8080
      LOG_LEVEL: \${LOG_LEVEL:-INFO}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - he300-net
    restart: unless-stopped

  ethicsengine:
    build:
      context: ./ethicsengine_enterprise
      dockerfile: Dockerfile
    ports:
      - \"\${EEE_PORT:-8080}:8080\"
    environment:
      REDIS_URL: redis://:\${REDIS_PASSWORD}@redis:6379/1
      MODEL_CACHE_DIR: /models
      DEFAULT_MODEL: \${DEFAULT_MODEL:-Qwen/Qwen2.5-7B-Instruct}
      LOG_LEVEL: \${LOG_LEVEL:-INFO}
    volumes:
      - \${DATA_DIR:-./data}/models:/models
    networks:
      - he300-net
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: \${DB_PASSWORD}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - he300-net
    healthcheck:
      test: [\"CMD-SHELL\", \"pg_isready -U postgres\"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    command: redis-server --requirepass \${REDIS_PASSWORD}
    volumes:
      - redis_data:/data
    networks:
      - he300-net
    healthcheck:
      test: [\"CMD\", \"redis-cli\", \"-a\", \"\${REDIS_PASSWORD}\", \"ping\"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

volumes:
  postgres_data:
  redis_data:

networks:
  he300-net:
    driver: bridge
COMPOSE"
}
echo "✅ Docker Compose configured"

# Step 6: Build and start
echo ""
echo "[6/6] Building and starting services..."
ssh $HOST "cd $INSTALL_DIR && docker compose build && docker compose up -d"

echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "Endpoints:"
echo "  CIRISNode:     http://163.192.58.165:8000"
echo "  EthicsEngine:  http://163.192.58.165:8080"
echo ""
echo "Check status:"
echo "  ssh $HOST 'cd $INSTALL_DIR && docker compose ps'"
echo ""
echo "View logs:"
echo "  ssh $HOST 'cd $INSTALL_DIR && docker compose logs -f'"
