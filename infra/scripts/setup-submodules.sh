#!/usr/bin/env bash
# =============================================================================
# Setup Submodules Script
# =============================================================================
#
# Initializes and configures git submodules for CIRISNode and EthicsEngine.
#
# Usage:
#   ./scripts/setup-submodules.sh [options]
#
# Options:
#   --cirisnode-branch BRANCH   CIRISNode branch (default: feature/eee-integration)
#   --eee-branch BRANCH         EEE branch (default: feature/he300-api)
#   --use-upstream              Use upstream repos instead of forks
#   --force                     Force reset submodules
#   --help                      Show this help
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
CIRISNODE_BRANCH="${CIRISNODE_BRANCH:-feature/eee-integration}"
EEE_BRANCH="${EEE_BRANCH:-feature/he300-api}"
USE_UPSTREAM=false
FORCE=false

# Fork URLs
CIRISNODE_FORK="https://github.com/rng-ops/CIRISNode.git"
EEE_FORK="https://github.com/rng-ops/ethicsengine_enterprise.git"

# Upstream URLs
CIRISNODE_UPSTREAM="https://github.com/CIRISAI/CIRISNode.git"
EEE_UPSTREAM="https://github.com/emooreatx/ethicsengine_enterprise.git"

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SUBMODULES_DIR="$ROOT_DIR/submodules"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

usage() {
    head -30 "$0" | grep -E "^#" | sed 's/^# *//'
    exit 0
}

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
while [[ $# -gt 0 ]]; do
    case $1 in
        --cirisnode-branch)
            CIRISNODE_BRANCH="$2"
            shift 2
            ;;
        --eee-branch)
            EEE_BRANCH="$2"
            shift 2
            ;;
        --use-upstream)
            USE_UPSTREAM=true
            shift
            ;;
        --force)
            FORCE=true
            shift
            ;;
        --help|-h)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            ;;
    esac
done

# Determine which URLs to use
if [ "$USE_UPSTREAM" = true ]; then
    CIRISNODE_URL="$CIRISNODE_UPSTREAM"
    EEE_URL="$EEE_UPSTREAM"
    log_info "Using upstream repositories"
else
    CIRISNODE_URL="$CIRISNODE_FORK"
    EEE_URL="$EEE_FORK"
    log_info "Using fork repositories"
fi

# -----------------------------------------------------------------------------
# Create submodules directory
# -----------------------------------------------------------------------------

log_info "Setting up submodules in $SUBMODULES_DIR"
mkdir -p "$SUBMODULES_DIR"

# -----------------------------------------------------------------------------
# Initialize .gitmodules if needed
# -----------------------------------------------------------------------------

GITMODULES_FILE="$SUBMODULES_DIR/.gitmodules"

cat > "$GITMODULES_FILE" << EOF
[submodule "cirisnode"]
    path = cirisnode
    url = $CIRISNODE_URL
    branch = $CIRISNODE_BRANCH

[submodule "ethicsengine"]
    path = ethicsengine
    url = $EEE_URL
    branch = $EEE_BRANCH
EOF

log_success "Created .gitmodules"

# -----------------------------------------------------------------------------
# Clone/Update CIRISNode
# -----------------------------------------------------------------------------

CIRISNODE_DIR="$SUBMODULES_DIR/cirisnode"

if [ -d "$CIRISNODE_DIR/.git" ]; then
    if [ "$FORCE" = true ]; then
        log_warn "Force resetting CIRISNode"
        rm -rf "$CIRISNODE_DIR"
    else
        log_info "Updating existing CIRISNode checkout"
        cd "$CIRISNODE_DIR"
        git fetch origin
        git checkout "$CIRISNODE_BRANCH" 2>/dev/null || git checkout -b "$CIRISNODE_BRANCH" "origin/$CIRISNODE_BRANCH"
        git pull origin "$CIRISNODE_BRANCH" --rebase || true
        cd "$ROOT_DIR"
        log_success "Updated CIRISNode to $CIRISNODE_BRANCH"
    fi
fi

if [ ! -d "$CIRISNODE_DIR/.git" ]; then
    log_info "Cloning CIRISNode from $CIRISNODE_URL"
    git clone --branch "$CIRISNODE_BRANCH" "$CIRISNODE_URL" "$CIRISNODE_DIR" || \
        git clone "$CIRISNODE_URL" "$CIRISNODE_DIR"

    cd "$CIRISNODE_DIR"
    git checkout "$CIRISNODE_BRANCH" 2>/dev/null || git checkout -b "$CIRISNODE_BRANCH"

    # Add upstream remote if using fork
    if [ "$USE_UPSTREAM" = false ]; then
        git remote add upstream "$CIRISNODE_UPSTREAM" 2>/dev/null || true
        git fetch upstream
    fi

    cd "$ROOT_DIR"
    log_success "Cloned CIRISNode"
fi

# -----------------------------------------------------------------------------
# Clone/Update EthicsEngine
# -----------------------------------------------------------------------------

EEE_DIR="$SUBMODULES_DIR/ethicsengine"

if [ -d "$EEE_DIR/.git" ]; then
    if [ "$FORCE" = true ]; then
        log_warn "Force resetting EthicsEngine"
        rm -rf "$EEE_DIR"
    else
        log_info "Updating existing EthicsEngine checkout"
        cd "$EEE_DIR"
        git fetch origin
        git checkout "$EEE_BRANCH" 2>/dev/null || git checkout -b "$EEE_BRANCH" "origin/$EEE_BRANCH"
        git pull origin "$EEE_BRANCH" --rebase || true
        cd "$ROOT_DIR"
        log_success "Updated EthicsEngine to $EEE_BRANCH"
    fi
fi

if [ ! -d "$EEE_DIR/.git" ]; then
    log_info "Cloning EthicsEngine from $EEE_URL"
    git clone --branch "$EEE_BRANCH" "$EEE_URL" "$EEE_DIR" || \
        git clone "$EEE_URL" "$EEE_DIR"

    cd "$EEE_DIR"
    git checkout "$EEE_BRANCH" 2>/dev/null || git checkout -b "$EEE_BRANCH"

    # Add upstream remote if using fork
    if [ "$USE_UPSTREAM" = false ]; then
        git remote add upstream "$EEE_UPSTREAM" 2>/dev/null || true
        git fetch upstream
    fi

    cd "$ROOT_DIR"
    log_success "Cloned EthicsEngine"
fi

# -----------------------------------------------------------------------------
# Create symlinks for convenience
# -----------------------------------------------------------------------------

log_info "Creating convenience symlinks"

# Symlink to Docker dirs
ln -sf "$CIRISNODE_DIR" "$ROOT_DIR/docker/cirisnode-src" 2>/dev/null || true
ln -sf "$EEE_DIR" "$ROOT_DIR/docker/eee-src" 2>/dev/null || true

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------

echo ""
log_success "Submodule setup complete!"
echo ""
echo "  CIRISNode:      $CIRISNODE_DIR"
echo "    Branch:       $CIRISNODE_BRANCH"
echo "    Remote:       $CIRISNODE_URL"
echo ""
echo "  EthicsEngine:   $EEE_DIR"
echo "    Branch:       $EEE_BRANCH"
echo "    Remote:       $EEE_URL"
echo ""
echo "Next steps:"
echo "  1. cp .env.example .env"
echo "  2. Edit .env with your settings"
echo "  3. make dev-up"
