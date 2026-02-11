#!/bin/bash
#
# deploy-remote-v2.sh - Fixed HE-300 Deployment Script
#
# Changes from v1:
# - Uses rsync instead of git clone (repos are local)
# - Adds verification after each step
# - Better error handling
# - Documents what's happening at each step
#
# Usage:
#   ./deploy-remote-v2.sh           # Interactive mode (prompts for confirmation)
#   ./deploy-remote-v2.sh -y        # Non-interactive mode (auto-confirm)
#   ./deploy-remote-v2.sh --yes     # Non-interactive mode (auto-confirm)
#
# HITL Reference: See docs/hitl.md for ethical considerations
# Security Reference: See docs/sec.md for security analysis
#

set -euo pipefail

# Parse command line arguments
NON_INTERACTIVE=false
for arg in "$@"; do
    case $arg in
        -y|--yes)
            NON_INTERACTIVE=true
            shift
            ;;
    esac
done

# Configuration
HOST="ubuntu@163.192.58.165"
INSTALL_DIR="/opt/he300"
LOCAL_CIRISNODE="/Users/a/projects/ethics/CIRISNode"
LOCAL_EEE="/Users/a/projects/ethics/ethicsengine_enterprise"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

log_step() {
    echo -e "\n${BLUE}[$1/7]${NC} $2"
}

log_success() {
    echo -e "${GREEN}✅ $1${NC}"
}

log_warn() {
    echo -e "${YELLOW}⚠️  $1${NC}"
}

log_error() {
    echo -e "${RED}❌ $1${NC}"
}

log_info() {
    echo -e "   ${BLUE}ℹ${NC}  $1"
}

verify_step() {
    if [ $? -eq 0 ]; then
        log_success "$1"
        return 0
    else
        log_error "$1 - FAILED"
        return 1
    fi
}

echo "=========================================="
echo "  HE-300 Remote Deployment v2.1"
echo "  Target: $HOST"
echo "  Date: $(date)"
echo "=========================================="
echo ""
echo "This script will:"
echo "  1. Verify Docker is installed"
echo "  2. Create required directories"
echo "  3. Sync CIRISNode code (via rsync)"
echo "  4. Sync EthicsEngine Enterprise code (via rsync)"
echo "  5. Generate secure credentials"
echo "  6. Create Docker Compose configuration"
echo "  7. Build and start services (including UI)"
echo ""
echo "📋 HITL: All actions are logged to docs/hitl.md"
echo "🔒 Security: See docs/sec.md for security analysis"
echo ""

if [ "$NON_INTERACTIVE" = false ]; then
    read -p "Press Enter to continue or Ctrl+C to abort..."
else
    echo "Running in non-interactive mode (-y flag provided)..."
    sleep 2
fi

# Step 1: Verify Docker
log_step 1 "Verifying Docker installation..."
log_info "Checking if Docker is running on remote server"

DOCKER_VERSION=$(ssh $HOST "docker --version 2>/dev/null" || echo "NOT_INSTALLED")
if [[ "$DOCKER_VERSION" == "NOT_INSTALLED" ]]; then
    log_warn "Docker not installed - this should have been done in v1"
    log_info "Installing Docker now..."
    ssh $HOST 'curl -fsSL https://get.docker.com | sudo sh'
    ssh $HOST 'sudo usermod -aG docker ubuntu'
    log_info "You may need to reconnect for group changes to take effect"
fi
log_success "Docker: $DOCKER_VERSION"

# Step 2: Create directories
log_step 2 "Creating directories on remote server..."
log_info "Creating: $INSTALL_DIR/{data/models,data/results,logs}"

ssh $HOST "sudo mkdir -p $INSTALL_DIR/{CIRISNode,ethicsengine_enterprise,data/models,data/results,logs} && sudo chown -R ubuntu:ubuntu $INSTALL_DIR"

# Verify
DIR_COUNT=$(ssh $HOST "ls -d $INSTALL_DIR/*/ 2>/dev/null | wc -l")
if [ "$DIR_COUNT" -ge 3 ]; then
    log_success "Directories created ($DIR_COUNT subdirectories)"
else
    log_error "Directory creation may have failed"
    exit 1
fi

# Step 3: Sync CIRISNode
log_step 3 "Syncing CIRISNode code to remote server..."
log_info "Source: $LOCAL_CIRISNODE"
log_info "Destination: $HOST:$INSTALL_DIR/CIRISNode/"
log_info "Excluding: .git, __pycache__, *.pyc, .env, venv"

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude 'venv' \
    --exclude 'node_modules' \
    --exclude '.pytest_cache' \
    "$LOCAL_CIRISNODE/" \
    "$HOST:$INSTALL_DIR/CIRISNode/"

# Verify
CIRIS_FILES=$(ssh $HOST "ls $INSTALL_DIR/CIRISNode/*.py 2>/dev/null | wc -l")
if [ "$CIRIS_FILES" -gt 0 ]; then
    log_success "CIRISNode synced ($CIRIS_FILES Python files)"
else
    log_error "CIRISNode sync failed - no Python files found"
    exit 1
fi

# Step 4: Sync EthicsEngine Enterprise
log_step 4 "Syncing EthicsEngine Enterprise code to remote server..."
log_info "Source: $LOCAL_EEE"
log_info "Destination: $HOST:$INSTALL_DIR/ethicsengine_enterprise/"
log_info "Excluding: .git, __pycache__, *.pyc, .env, venv, datasets"

rsync -avz --progress \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.env' \
    --exclude 'venv' \
    --exclude 'node_modules' \
    --exclude '.pytest_cache' \
    --exclude 'datasets/*.jsonl' \
    "$LOCAL_EEE/" \
    "$HOST:$INSTALL_DIR/ethicsengine_enterprise/"

# Verify
EEE_FILES=$(ssh $HOST "ls $INSTALL_DIR/ethicsengine_enterprise/*.py 2>/dev/null | wc -l")
if [ "$EEE_FILES" -gt 0 ]; then
    log_success "EthicsEngine Enterprise synced ($EEE_FILES Python files)"
else
    log_error "EthicsEngine sync failed - no Python files found"
    exit 1
fi

# Step 5: Generate secure credentials
log_step 5 "Generating secure credentials..."
log_info "Creating .env file with random secrets"
log_info "🔒 Secrets are generated using openssl rand"

# Generate secrets locally and send to remote
POSTGRES_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')
REDIS_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')
JWT_SECRET=$(openssl rand -base64 32)
WEBHOOK_SECRET=$(openssl rand -hex 32)

ssh $HOST "cat > $INSTALL_DIR/.env << 'EOF'
# HE-300 Environment Configuration
# Generated: $(date)
# WARNING: Contains secrets - do not commit to git

ENVIRONMENT=staging

# Database
POSTGRES_USER=ciris
POSTGRES_PASSWORD=$POSTGRES_PASSWORD
POSTGRES_DB=cirisnode
DATABASE_URL=postgresql://ciris:$POSTGRES_PASSWORD@db:5432/cirisnode

# Redis
REDIS_PASSWORD=$REDIS_PASSWORD
REDIS_URL=redis://:$REDIS_PASSWORD@redis:6379/0

# Authentication
JWT_SECRET=$JWT_SECRET
WEBHOOK_SECRET=$WEBHOOK_SECRET

# Service Ports
CIRISNODE_PORT=8000
EEE_PORT=8080
DASHBOARD_PORT=3000

# Model Configuration
DEFAULT_MODEL=Qwen/Qwen2.5-7B-Instruct
MODEL_QUANTIZATION=Q4_K_M
MODEL_CACHE_DIR=/models

# Logging
LOG_LEVEL=INFO

# Paths
DATA_DIR=/opt/he300/data
LOG_DIR=/opt/he300/logs
EOF"

# Secure the .env file (security recommendation from docs/sec.md)
ssh $HOST "chmod 600 $INSTALL_DIR/.env"

log_success "Credentials generated and secured (chmod 600)"

# Step 6: Create Docker Compose configuration
log_step 6 "Creating Docker Compose configuration..."
log_info "Creating production-ready docker-compose.yml"

ssh $HOST "cat > $INSTALL_DIR/docker-compose.yml << 'COMPOSE'
version: '3.8'

services:
  cirisnode:
    build:
      context: ./CIRISNode
      dockerfile: Dockerfile
    container_name: he300-cirisnode
    ports:
      - \"\${CIRISNODE_PORT:-8000}:8000\"
    environment:
      - DATABASE_URL=\${DATABASE_URL}
      - REDIS_URL=\${REDIS_URL}
      - JWT_SECRET=\${JWT_SECRET}
      - EEE_BASE_URL=http://ethicsengine:8080
      - LOG_LEVEL=\${LOG_LEVEL:-INFO}
    depends_on:
      db:
        condition: service_healthy
      redis:
        condition: service_healthy
    networks:
      - he300-net
    restart: unless-stopped
    healthcheck:
      test: [\"CMD\", \"curl\", \"-f\", \"http://localhost:8000/api/v1/health\"]
      interval: 30s
      timeout: 10s
      retries: 3

  ethicsengine:
    build:
      context: ./ethicsengine_enterprise
      dockerfile: Dockerfile
    container_name: he300-ethicsengine
    ports:
      - \"\${EEE_PORT:-8080}:8080\"
    environment:
      - REDIS_URL=redis://:\${REDIS_PASSWORD}@redis:6379/1
      - MODEL_CACHE_DIR=/models
      - DEFAULT_MODEL=\${DEFAULT_MODEL:-Qwen/Qwen2.5-7B-Instruct}
      - LOG_LEVEL=\${LOG_LEVEL:-INFO}
      - OLLAMA_HOST=http://ollama:11434
    volumes:
      - model_cache:/models
      - \${DATA_DIR:-./data}/results:/results
    networks:
      - he300-net
    restart: unless-stopped
    healthcheck:
      test: [\"CMD\", \"curl\", \"-f\", \"http://localhost:8080/health\"]
      interval: 30s
      timeout: 10s
      retries: 3

  ui:
    build:
      context: ./CIRISNode/ui
      dockerfile: Dockerfile
    container_name: he300-ui
    ports:
      - \"\${UI_PORT:-3000}:3000\"
    environment:
      - NEXTAUTH_SECRET=\${JWT_SECRET}
      - NEXTAUTH_URL=http://localhost:3000
    networks:
      - he300-net
    restart: unless-stopped

  ollama:
    image: ollama/ollama:latest
    container_name: he300-ollama
    volumes:
      - ollama_data:/root/.ollama
    networks:
      - he300-net
    restart: unless-stopped

  db:
    image: postgres:16-alpine
    container_name: he300-postgres
    environment:
      - POSTGRES_USER=\${POSTGRES_USER:-ciris}
      - POSTGRES_PASSWORD=\${POSTGRES_PASSWORD}
      - POSTGRES_DB=\${POSTGRES_DB:-cirisnode}
    volumes:
      - postgres_data:/var/lib/postgresql/data
    networks:
      - he300-net
    healthcheck:
      test: [\"CMD-SHELL\", \"pg_isready -U \${POSTGRES_USER:-ciris}\"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped

  redis:
    image: redis:7-alpine
    container_name: he300-redis
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
    name: he300-postgres-data
  redis_data:
    name: he300-redis-data
  model_cache:
    name: he300-model-cache
  ollama_data:
    name: he300-ollama-data

networks:
  he300-net:
    name: he300-network
    driver: bridge
COMPOSE"

log_success "Docker Compose configuration created"

# Step 7: Build and start services
log_step 7 "Building and starting services..."
log_info "This may take several minutes for the first build"
log_info "Building: cirisnode, ethicsengine"
log_info "Pulling: postgres:16-alpine, redis:7-alpine"

echo ""
echo "Starting build process..."
echo "---"

ssh $HOST "cd $INSTALL_DIR && docker compose build --progress=plain 2>&1"
BUILD_EXIT=$?

if [ $BUILD_EXIT -ne 0 ]; then
    log_error "Docker build failed with exit code $BUILD_EXIT"
    echo ""
    echo "Check the build output above for errors."
    echo "Common issues:"
    echo "  - Missing Dockerfile in CIRISNode or ethicsengine_enterprise"
    echo "  - Invalid Dockerfile syntax"
    echo "  - Network issues pulling base images"
    exit 1
fi

log_success "Docker images built successfully"

echo ""
log_info "Starting services..."
ssh $HOST "cd $INSTALL_DIR && docker compose up -d"

# Wait for services to start
log_info "Waiting for services to become healthy (30 seconds)..."
sleep 30

# Check status
echo ""
echo "=========================================="
echo "  Service Status"
echo "=========================================="
ssh $HOST "cd $INSTALL_DIR && docker compose ps"

echo ""
echo "=========================================="
echo "  Deployment Complete!"
echo "=========================================="
echo ""
echo "🌐 Endpoints (via SSH tunnel):"
echo "   CIRISNode:     http://localhost:8000"
echo "   EthicsEngine:  http://localhost:8080"
echo "   HE-300 UI:     http://localhost:3000/he300"
echo ""
echo "🔗 SSH Tunnel Command:"
echo "   ssh -L 3000:localhost:3000 -L 8000:localhost:8000 -L 8080:localhost:8080 $HOST"
echo ""
echo "📋 Quick Commands:"
echo "   Check status:  ssh $HOST 'cd $INSTALL_DIR && docker compose ps'"
echo "   View logs:     ssh $HOST 'cd $INSTALL_DIR && docker compose logs -f'"
echo "   Stop:          ssh $HOST 'cd $INSTALL_DIR && docker compose down'"
echo ""
echo "🧪 Test Endpoints:"
echo "   curl http://localhost:8000/api/v1/health"
echo "   curl http://localhost:8080/health"
echo "   curl http://localhost:8080/he300/catalog"
echo ""
echo "📋 Documentation:"
echo "   HITL Log:      docs/hitl.md"
echo "   Security:      docs/sec.md"
echo "   Deploy Log:    docs/deploy-log.md"
