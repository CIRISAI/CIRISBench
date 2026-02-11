#!/usr/bin/env bash
# =============================================================================
# Docker Image Builder Script
# =============================================================================
#
# Builds Docker images for CIRISNode and EthicsEngine Enterprise.
#
# Usage:
#   ./scripts/build-images.sh [target] [options]
#
# Targets:
#   all         Build all images (default)
#   cirisnode   Build CIRISNode only
#   eee         Build EthicsEngine only
#   push        Build and push all
#   ci          CI build (no push)
#
# Options:
#   --tag TAG           Image tag (default: VERSION or latest)
#   --registry REG      Docker registry (default: ghcr.io)
#   --org ORG           Docker org (default: rng-ops)
#   --no-cache          Build without cache
#   --platform PLAT     Target platform (default: linux/amd64)
#   --help              Show this help
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
TARGET="${1:-all}"
TAG=""
REGISTRY="${DOCKER_REGISTRY:-ghcr.io}"
ORG="${DOCKER_ORG:-rng-ops}"
NO_CACHE=""
PLATFORM="linux/amd64"
PUSH=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SUBMODULES_DIR="$ROOT_DIR/submodules"

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $*"
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
        --registry)
            REGISTRY="$2"
            shift 2
            ;;
        --org)
            ORG="$2"
            shift 2
            ;;
        --no-cache)
            NO_CACHE="--no-cache"
            shift
            ;;
        --platform)
            PLATFORM="$2"
            shift 2
            ;;
        --help|-h)
            head -30 "$0" | grep -E "^#" | sed 's/^# *//'
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

# Determine tag
if [ -z "$TAG" ]; then
    if [ -f "$ROOT_DIR/VERSION" ]; then
        TAG=$(cat "$ROOT_DIR/VERSION" | tr -d '[:space:]')
    else
        TAG="latest"
    fi
fi

# Build metadata
BUILD_DATE=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
GIT_HASH=$(git rev-parse --short HEAD 2>/dev/null || echo "unknown")

log_info "Build Configuration:"
echo "  Registry:  $REGISTRY"
echo "  Org:       $ORG"
echo "  Tag:       $TAG"
echo "  Platform:  $PLATFORM"
echo "  Git Hash:  $GIT_HASH"
echo ""

build_cirisnode() {
    local context="$SUBMODULES_DIR/cirisnode"
    local image="$REGISTRY/$ORG/cirisnode:$TAG"

    if [ ! -d "$context" ]; then
        log_error "CIRISNode not found at $context"
    fi

    log_info "Building CIRISNode: $image"

    docker build \
        $NO_CACHE \
        --platform "$PLATFORM" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$TAG" \
        --build-arg GIT_HASH="$GIT_HASH" \
        -t "$image" \
        -t "$REGISTRY/$ORG/cirisnode:latest" \
        "$context"

    log_success "Built: $image"

    if [ "$PUSH" = true ]; then
        log_info "Pushing $image..."
        docker push "$image"
        docker push "$REGISTRY/$ORG/cirisnode:latest"
        log_success "Pushed: $image"
    fi
}

build_eee() {
    local context="$SUBMODULES_DIR/ethicsengine"
    local image="$REGISTRY/$ORG/ethicsengine-enterprise:$TAG"

    if [ ! -d "$context" ]; then
        log_error "EthicsEngine not found at $context"
    fi

    log_info "Building EthicsEngine Enterprise: $image"

    docker build \
        $NO_CACHE \
        --platform "$PLATFORM" \
        --build-arg BUILD_DATE="$BUILD_DATE" \
        --build-arg VERSION="$TAG" \
        --build-arg GIT_HASH="$GIT_HASH" \
        -t "$image" \
        -t "$REGISTRY/$ORG/ethicsengine-enterprise:latest" \
        "$context"

    log_success "Built: $image"

    if [ "$PUSH" = true ]; then
        log_info "Pushing $image..."
        docker push "$image"
        docker push "$REGISTRY/$ORG/ethicsengine-enterprise:latest"
        log_success "Pushed: $image"
    fi
}

# Execute based on target
case $TARGET in
    all)
        build_cirisnode
        build_eee
        ;;
    cirisnode)
        build_cirisnode
        ;;
    eee)
        build_eee
        ;;
    push)
        PUSH=true
        build_cirisnode
        build_eee
        ;;
    ci)
        # CI build - no push, use Git SHA as tag
        TAG="$GIT_HASH"
        build_cirisnode
        build_eee
        ;;
    *)
        log_error "Unknown target: $TARGET"
        ;;
esac

log_success "Build complete!"
