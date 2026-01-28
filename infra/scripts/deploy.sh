#!/usr/bin/env bash
# =============================================================================
# Deploy Script
# =============================================================================
#
# Deploys HE-300 stack to target environment.
#
# Usage:
#   ./scripts/deploy.sh [environment] [options]
#
# Environments:
#   staging     Deploy to staging (default)
#   production  Deploy to production (requires approval)
#
# Options:
#   --tag TAG       Image tag to deploy
#   --dry-run       Show what would be done
#   --rollback      Rollback to previous version
#   --help          Show this help
#
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Defaults
ENVIRONMENT="${1:-staging}"
TAG=""
DRY_RUN=false
ROLLBACK=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
    exit 1
}

# Parse arguments
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --tag)
            TAG="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --rollback)
            ROLLBACK=true
            shift
            ;;
        --help|-h)
            head -25 "$0" | grep -E "^#" | sed 's/^# *//'
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Get tag from VERSION if not specified
if [ -z "$TAG" ]; then
    TAG=$(cat "$ROOT_DIR/VERSION" 2>/dev/null | tr -d '[:space:]' || echo "latest")
fi

log_info "Deployment Configuration:"
echo "  Environment: $ENVIRONMENT"
echo "  Tag:         $TAG"
echo "  Dry Run:     $DRY_RUN"
echo "  Rollback:    $ROLLBACK"
echo ""

# Production requires confirmation
if [ "$ENVIRONMENT" = "production" ] && [ "$DRY_RUN" = false ]; then
    log_warn "You are about to deploy to PRODUCTION!"
    echo ""
    read -p "Type 'yes' to confirm: " CONFIRM
    if [ "$CONFIRM" != "yes" ]; then
        log_info "Deployment cancelled"
        exit 0
    fi
fi

# Select compose file
case $ENVIRONMENT in
    staging)
        COMPOSE_FILE="$ROOT_DIR/docker/docker-compose.he300.yml"
        ;;
    production)
        COMPOSE_FILE="$ROOT_DIR/docker/docker-compose.prod.yml"
        ;;
    *)
        log_error "Unknown environment: $ENVIRONMENT"
        ;;
esac

if [ "$ROLLBACK" = true ]; then
    log_info "Rolling back to previous deployment..."
    
    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would rollback deployment"
        exit 0
    fi
    
    # For Docker Compose, this just restarts with previous images
    docker compose -f "$COMPOSE_FILE" down
    docker compose -f "$COMPOSE_FILE" up -d
    
    log_success "Rollback complete"
    exit 0
fi

# Deploy
log_info "Deploying v$TAG to $ENVIRONMENT..."

if [ "$DRY_RUN" = true ]; then
    log_info "[DRY RUN] Would execute:"
    echo "  docker compose -f $COMPOSE_FILE pull"
    echo "  docker compose -f $COMPOSE_FILE up -d"
    exit 0
fi

# Pull latest images
export IMAGE_TAG="$TAG"
log_info "Pulling images..."
docker compose -f "$COMPOSE_FILE" pull

# Deploy
log_info "Starting services..."
docker compose -f "$COMPOSE_FILE" up -d

# Wait for health
log_info "Waiting for services to be healthy..."
sleep 10

# Health check
CIRISNODE_HEALTHY=false
EEE_HEALTHY=false

for i in {1..30}; do
    if curl -sf "http://localhost:8000/health" > /dev/null 2>&1; then
        CIRISNODE_HEALTHY=true
    fi
    if curl -sf "http://localhost:8080/health" > /dev/null 2>&1; then
        EEE_HEALTHY=true
    fi
    
    if [ "$CIRISNODE_HEALTHY" = true ] && [ "$EEE_HEALTHY" = true ]; then
        break
    fi
    
    echo -ne "\r  Checking health... ($i/30)"
    sleep 2
done

echo ""

if [ "$CIRISNODE_HEALTHY" = false ]; then
    log_warn "CIRISNode health check failed"
fi

if [ "$EEE_HEALTHY" = false ]; then
    log_warn "EEE health check failed"
fi

if [ "$CIRISNODE_HEALTHY" = true ] && [ "$EEE_HEALTHY" = true ]; then
    log_success "Deployment successful!"
    echo ""
    echo "  CIRISNode: http://localhost:8000"
    echo "  EEE:       http://localhost:8080"
else
    log_warn "Deployment completed with warnings. Check service logs."
    docker compose -f "$COMPOSE_FILE" logs --tail=50
fi
