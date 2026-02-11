#!/bin/bash
#===============================================================================
#
#  HE-300 Benchmark Suite Installer
#
#  Installs the complete HE-300 ethical benchmark stack via Docker Compose:
#  - CIRISNode with EthicsEngine Enterprise integration
#  - GPU-accelerated inference (NVIDIA or Apple Silicon)
#  - Results Dashboard
#
#  Supported Platforms:
#  - Ubuntu 22.04/24.04
#  - Lambda Stack 22.04/24.04
#  - macOS 13+ (Apple Silicon M1/M2/M3)
#  - Debian 12+
#
#  Usage: ./install.sh [OPTIONS]
#
#  Options:
#    --minimal       Install core components only (no dashboard)
#    --no-gpu        Disable GPU support
#    --dev           Development mode (build from source)
#    --help          Show this help message
#
#===============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
PURPLE='\033[0;35m'
CYAN='\033[0;36m'
NC='\033[0m' # No Color

# Configuration
INSTALL_DIR="${INSTALL_DIR:-$HOME/he300}"
DATA_DIR="${DATA_DIR:-$INSTALL_DIR/data}"
CONFIG_DIR="${CONFIG_DIR:-$INSTALL_DIR/config}"

# Options
MINIMAL=false
NO_GPU=false
DEV_MODE=false

#===============================================================================
# Utility Functions
#===============================================================================

log_info() { echo -e "${BLUE}[INFO]${NC} $1"; }
log_success() { echo -e "${GREEN}[OK]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

log_step() {
    echo -e "\n${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
    echo -e "${PURPLE}  $1${NC}"
    echo -e "${PURPLE}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"
}

detect_platform() {
    log_step "Detecting Platform"

    PLATFORM="unknown"
    ARCH=$(uname -m)

    case "$(uname -s)" in
        Linux*)
            PLATFORM="linux"
            if [ -f /etc/os-release ]; then
                . /etc/os-release
                DISTRO=$ID
                DISTRO_VERSION=$VERSION_ID
            fi
            ;;
        Darwin*)
            PLATFORM="macos"
            DISTRO="macos"
            DISTRO_VERSION=$(sw_vers -productVersion)
            ;;
    esac

    log_info "Platform: $PLATFORM"
    log_info "Architecture: $ARCH"
    log_info "Distribution: ${DISTRO:-unknown} ${DISTRO_VERSION:-}"

    # Detect GPU
    GPU_TYPE="none"
    if [ "$PLATFORM" = "macos" ]; then
        if [[ "$ARCH" == "arm64" ]]; then
            GPU_TYPE="apple-silicon"
            log_success "Apple Silicon detected (Metal acceleration)"
        fi
    elif [ "$PLATFORM" = "linux" ]; then
        if command -v nvidia-smi &> /dev/null; then
            GPU_INFO=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
            if [ -n "$GPU_INFO" ]; then
                GPU_TYPE="nvidia"
                log_success "NVIDIA GPU detected: $GPU_INFO"
            fi
        fi
    fi

    if [ "$GPU_TYPE" = "none" ] && [ "$NO_GPU" = false ]; then
        log_warn "No GPU detected. Running in CPU-only mode."
        NO_GPU=true
    fi
}

check_dependencies() {
    log_step "Checking Dependencies"

    local missing=()

    # Check Docker
    if ! command -v docker &> /dev/null; then
        missing+=("docker")
    else
        log_success "Docker: $(docker --version | cut -d' ' -f3 | tr -d ',')"
    fi

    # Check Docker Compose
    if ! docker compose version &> /dev/null; then
        missing+=("docker-compose")
    else
        log_success "Docker Compose: $(docker compose version --short)"
    fi

    # Check Git
    if ! command -v git &> /dev/null; then
        missing+=("git")
    else
        log_success "Git: $(git --version | cut -d' ' -f3)"
    fi

    # Check curl
    if ! command -v curl &> /dev/null; then
        missing+=("curl")
    else
        log_success "curl: available"
    fi

    if [ ${#missing[@]} -gt 0 ]; then
        log_error "Missing dependencies: ${missing[*]}"
        echo ""
        echo "Please install the missing dependencies:"

        if [ "$PLATFORM" = "macos" ]; then
            echo "  brew install ${missing[*]}"
            echo ""
            echo "For Docker, download from: https://docker.com/products/docker-desktop"
        elif [ "$PLATFORM" = "linux" ]; then
            echo "  sudo apt-get install ${missing[*]}"
            echo ""
            echo "For Docker: https://docs.docker.com/engine/install/"
        fi
        exit 1
    fi

    # Check Docker is running
    if ! docker info &> /dev/null; then
        log_error "Docker is not running. Please start Docker and try again."
        exit 1
    fi
    log_success "Docker daemon is running"
}

check_nvidia_toolkit() {
    if [ "$GPU_TYPE" = "nvidia" ]; then
        log_step "Checking NVIDIA Container Toolkit"

        if docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi &> /dev/null; then
            log_success "NVIDIA Container Toolkit is working"
        else
            log_warn "NVIDIA Container Toolkit not configured"
            echo ""
            echo "To enable GPU support, install NVIDIA Container Toolkit:"
            echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
            echo ""
            read -p "Continue without GPU support? (y/N): " -n 1 -r
            echo
            if [[ ! $REPLY =~ ^[Yy]$ ]]; then
                exit 1
            fi
            NO_GPU=true
        fi
    fi
}

create_directories() {
    log_step "Creating Directory Structure"

    mkdir -p "$INSTALL_DIR"/{config,logs}
    mkdir -p "$DATA_DIR"/{models,results,artifacts}

    log_success "Created: $INSTALL_DIR"
    log_success "Created: $DATA_DIR"
}

clone_repositories() {
    log_step "Cloning Repositories"

    cd "$INSTALL_DIR"

    # Clone or update CIRISNode
    if [ -d "CIRISNode" ]; then
        log_info "Updating CIRISNode..."
        cd CIRISNode && git pull --quiet && cd ..
    else
        log_info "Cloning CIRISNode..."
        git clone --quiet https://github.com/rng-ops/CIRISNode.git --branch feature/eee-integration
    fi
    log_success "CIRISNode ready"

    # Clone or update EthicsEngine Enterprise
    if [ -d "ethicsengine_enterprise" ]; then
        log_info "Updating EthicsEngine Enterprise..."
        cd ethicsengine_enterprise && git pull --quiet && cd ..
    else
        log_info "Cloning EthicsEngine Enterprise..."
        git clone --quiet https://github.com/rng-ops/ethicsengine_enterprise.git --branch feature/he300-api
    fi
    log_success "EthicsEngine Enterprise ready"

    # Clone staging repo for docker-compose files
    if [ -d "he300-integration" ]; then
        log_info "Updating he300-integration..."
        cd he300-integration && git pull --quiet && cd ..
    else
        log_info "Cloning he300-integration..."
        git clone --quiet https://github.com/rng-ops/he300-integration.git
    fi
    log_success "Integration configs ready"
}

generate_env_file() {
    log_step "Generating Configuration"

    ENV_FILE="$INSTALL_DIR/.env"

    if [ -f "$ENV_FILE" ]; then
        log_warn "Existing .env file found. Backing up..."
        cp "$ENV_FILE" "${ENV_FILE}.backup.$(date +%s)"
    fi

    # Generate secrets
    DB_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')
    REDIS_PASSWORD=$(openssl rand -base64 24 | tr -d '=+/')
    JWT_SECRET=$(openssl rand -base64 32)
    WEBHOOK_SECRET=$(openssl rand -hex 32)

    cat > "$ENV_FILE" << EOF
# HE-300 Configuration
# Generated: $(date -Iseconds)

# Environment
ENVIRONMENT=dev

# Database
POSTGRES_PASSWORD=$DB_PASSWORD
DB_PASSWORD=$DB_PASSWORD

# Redis
REDIS_PASSWORD=$REDIS_PASSWORD

# JWT Authentication
JWT_SECRET=$JWT_SECRET

# Webhook
WEBHOOK_SECRET=$WEBHOOK_SECRET

# Service Ports
CIRISNODE_PORT=8000
EEE_PORT=8080
DASHBOARD_PORT=3000

# Model Configuration
DEFAULT_MODEL=Qwen/Qwen2.5-7B-Instruct
MODEL_QUANTIZATION=Q4_K_M
MODEL_CACHE_DIR=$DATA_DIR/models

# GPU Configuration
ENABLE_GPU=$( [ "$NO_GPU" = true ] && echo "false" || echo "true" )
CUDA_VISIBLE_DEVICES=0

# Logging
LOG_LEVEL=INFO

# Paths
DATA_DIR=$DATA_DIR
LOG_DIR=$INSTALL_DIR/logs
EOF

    chmod 600 "$ENV_FILE"
    log_success "Configuration saved to $ENV_FILE"
}

create_docker_compose() {
    log_step "Creating Docker Compose Configuration"

    COMPOSE_FILE="$INSTALL_DIR/docker-compose.yml"

    # Determine GPU config
    GPU_CONFIG=""
    if [ "$NO_GPU" = false ]; then
        if [ "$GPU_TYPE" = "nvidia" ]; then
            GPU_CONFIG='
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]'
        fi
    fi

    cat > "$COMPOSE_FILE" << EOF
version: '3.8'

services:
  cirisnode:
    build:
      context: ./CIRISNode
      dockerfile: Dockerfile
    ports:
      - "\${CIRISNODE_PORT:-8000}:8000"
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
    volumes:
      - \${DATA_DIR}/results:/app/results
      - \${LOG_DIR:-./logs}/cirisnode:/app/logs
    networks:
      - he300-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3$GPU_CONFIG

  ethicsengine:
    build:
      context: ./ethicsengine_enterprise
      dockerfile: Dockerfile
    environment:
      DATABASE_URL: postgresql://eee:\${DB_PASSWORD}@db:5432/ethicsengine
      REDIS_URL: redis://:\${REDIS_PASSWORD}@redis:6379/1
      MODEL_CACHE_DIR: /models
      DEFAULT_MODEL: \${DEFAULT_MODEL:-Qwen/Qwen2.5-7B-Instruct}
      LOG_LEVEL: \${LOG_LEVEL:-INFO}
    volumes:
      - \${DATA_DIR}/models:/models
      - \${LOG_DIR:-./logs}/eee:/app/logs
    networks:
      - he300-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8080/health"]
      interval: 30s
      timeout: 10s
      retries: 3$GPU_CONFIG

  db:
    image: postgres:16-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: \${DB_PASSWORD}
      POSTGRES_MULTIPLE_DATABASES: cirisnode,ethicsengine
    volumes:
      - postgres_data:/var/lib/postgresql/data
      - ./he300-integration/docker/init-db.sql:/docker-entrypoint-initdb.d/init.sql
    networks:
      - he300-net
    healthcheck:
      test: ["CMD-SHELL", "pg_isready -U postgres"]
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
      test: ["CMD", "redis-cli", "-a", "\${REDIS_PASSWORD}", "ping"]
      interval: 10s
      timeout: 5s
      retries: 5
    restart: unless-stopped
EOF

    # Add dashboard if not minimal
    if [ "$MINIMAL" = false ]; then
        cat >> "$COMPOSE_FILE" << 'EOF'

  dashboard:
    build:
      context: ./he300-integration/dashboard
      dockerfile: Dockerfile
    ports:
      - "${DASHBOARD_PORT:-3000}:3000"
    environment:
      DATABASE_URL: postgresql://dashboard:${DB_PASSWORD}@db:5432/he300_dashboard
      WEBHOOK_SECRET: ${WEBHOOK_SECRET}
      NEXT_PUBLIC_APP_URL: http://localhost:${DASHBOARD_PORT:-3000}
    depends_on:
      db:
        condition: service_healthy
    networks:
      - he300-net
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "wget", "-q", "--spider", "http://localhost:3000/api/health"]
      interval: 30s
      timeout: 10s
      retries: 3
EOF
    fi

    # Add volumes and networks
    cat >> "$COMPOSE_FILE" << 'EOF'

volumes:
  postgres_data:
  redis_data:

networks:
  he300-net:
    driver: bridge
EOF

    log_success "Docker Compose configuration created"
}

create_helper_scripts() {
    log_step "Creating Helper Scripts"

    # Start script
    cat > "$INSTALL_DIR/start.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
docker compose up -d
echo "HE-300 services starting..."
echo ""
echo "Endpoints:"
echo "  CIRISNode:     http://localhost:${CIRISNODE_PORT:-8000}"
echo "  EthicsEngine:  http://localhost:${EEE_PORT:-8080}"
echo "  Dashboard:     http://localhost:${DASHBOARD_PORT:-3000}"
echo ""
echo "Run './status.sh' to check service status"
EOF
    chmod +x "$INSTALL_DIR/start.sh"

    # Stop script
    cat > "$INSTALL_DIR/stop.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
docker compose down
echo "HE-300 services stopped"
EOF
    chmod +x "$INSTALL_DIR/stop.sh"

    # Status script
    cat > "$INSTALL_DIR/status.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
echo "=== HE-300 Service Status ==="
docker compose ps
echo ""
echo "=== Health Checks ==="
curl -sf http://localhost:${CIRISNODE_PORT:-8000}/health && echo "CIRISNode: ✅" || echo "CIRISNode: ❌"
curl -sf http://localhost:${EEE_PORT:-8080}/health && echo "EthicsEngine: ✅" || echo "EthicsEngine: ❌"
curl -sf http://localhost:${DASHBOARD_PORT:-3000}/api/health && echo "Dashboard: ✅" || echo "Dashboard: ❌"
EOF
    chmod +x "$INSTALL_DIR/status.sh"

    # Logs script
    cat > "$INSTALL_DIR/logs.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"
SERVICE="${1:-cirisnode}"
docker compose logs -f "$SERVICE"
EOF
    chmod +x "$INSTALL_DIR/logs.sh"

    # Benchmark script
    cat > "$INSTALL_DIR/benchmark.sh" << 'EOF'
#!/bin/bash
cd "$(dirname "$0")"

MODEL="${1:-Qwen/Qwen2.5-7B-Instruct}"
SAMPLE_SIZE="${2:-300}"
CATEGORIES="${3:-commonsense,deontology,justice,utilitarianism,virtue}"

echo "Starting HE-300 benchmark..."
echo "  Model: $MODEL"
echo "  Sample Size: $SAMPLE_SIZE"
echo "  Categories: $CATEGORIES"
echo ""

curl -X POST "http://localhost:${CIRISNODE_PORT:-8000}/api/benchmarks/run" \
  -H "Content-Type: application/json" \
  -d "{
    \"model\": \"$MODEL\",
    \"sample_size\": $SAMPLE_SIZE,
    \"categories\": \"$CATEGORIES\"
  }"
EOF
    chmod +x "$INSTALL_DIR/benchmark.sh"

    log_success "Helper scripts created"
}

build_images() {
    log_step "Building Docker Images"

    cd "$INSTALL_DIR"

    log_info "This may take several minutes..."

    if docker compose build; then
        log_success "Docker images built successfully"
    else
        log_error "Failed to build Docker images"
        exit 1
    fi
}

print_summary() {
    echo -e "\n${GREEN}"
    cat << 'EOF'
  ╔═══════════════════════════════════════════════════════════════════╗
  ║                                                                   ║
  ║   ✅  HE-300 Benchmark Suite Installation Complete!               ║
  ║                                                                   ║
  ╚═══════════════════════════════════════════════════════════════════╝
EOF
    echo -e "${NC}"

    echo "Installation Directory: $INSTALL_DIR"
    echo ""
    echo "Quick Start:"
    echo "─────────────────────────────────────────────────────────────────"
    echo ""
    echo "  1. Start services:"
    echo "     cd $INSTALL_DIR && ./start.sh"
    echo ""
    echo "  2. Check status:"
    echo "     ./status.sh"
    echo ""
    echo "  3. Run benchmark:"
    echo "     ./benchmark.sh \"Qwen/Qwen2.5-7B-Instruct\" 300"
    echo ""
    echo "  4. View logs:"
    echo "     ./logs.sh cirisnode"
    echo "     ./logs.sh ethicsengine"
    echo ""
    echo "Endpoints (after starting):"
    echo "─────────────────────────────────────────────────────────────────"
    echo "  CIRISNode API:     http://localhost:8000"
    echo "  EthicsEngine API:  http://localhost:8080"

    if [ "$MINIMAL" = false ]; then
        echo "  Dashboard:         http://localhost:3000"
    fi

    echo ""
    echo "Configuration: $INSTALL_DIR/.env"
    echo ""
}

print_usage() {
    cat << EOF
HE-300 Benchmark Suite Installer

Usage: $0 [OPTIONS]

Options:
    --minimal       Install core components only (no dashboard)
    --no-gpu        Disable GPU support
    --dev           Development mode (verbose output)
    --help          Show this help message

Environment Variables:
    INSTALL_DIR     Installation directory (default: ~/he300)
    DATA_DIR        Data directory (default: \$INSTALL_DIR/data)

Examples:
    ./install.sh                    # Full installation
    ./install.sh --minimal          # Core components only
    ./install.sh --no-gpu           # CPU-only mode

EOF
}

parse_args() {
    while [[ $# -gt 0 ]]; do
        case $1 in
            --minimal)
                MINIMAL=true
                shift
                ;;
            --no-gpu)
                NO_GPU=true
                shift
                ;;
            --dev)
                DEV_MODE=true
                set -x
                shift
                ;;
            --help)
                print_usage
                exit 0
                ;;
            *)
                log_error "Unknown option: $1"
                print_usage
                exit 1
                ;;
        esac
    done
}

#===============================================================================
# Main
#===============================================================================

main() {
    parse_args "$@"

    echo -e "${CYAN}"
    cat << 'EOF'

  ██╗  ██╗███████╗      ██████╗  ██████╗  ██████╗
  ██║  ██║██╔════╝      ╚════██╗██╔═████╗██╔═████╗
  ███████║█████╗   █████╗ █████╔╝██║██╔██║██║██╔██║
  ██╔══██║██╔══╝   ╚════╝ ╚═══██╗████╔╝██║████╔╝██║
  ██║  ██║███████╗      ██████╔╝╚██████╔╝╚██████╔╝
  ╚═╝  ╚═╝╚══════╝      ╚═════╝  ╚═════╝  ╚═════╝

  Ethical Benchmark Suite - Docker Compose Installer

EOF
    echo -e "${NC}"

    detect_platform
    check_dependencies
    check_nvidia_toolkit
    create_directories
    clone_repositories
    generate_env_file
    create_docker_compose
    create_helper_scripts

    echo ""
    read -p "Build Docker images now? (Y/n): " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Nn]$ ]]; then
        build_images
    fi

    print_summary
}

main "$@"
