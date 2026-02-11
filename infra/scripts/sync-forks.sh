#!/usr/bin/env bash
# =============================================================================
# Fork Synchronization Script
# =============================================================================
#
# Synchronizes forks with their upstream repositories.
#
# Usage:
#   ./scripts/sync-forks.sh [options]
#
# Options:
#   --cirisnode         Sync CIRISNode only
#   --eee               Sync EthicsEngine only
#   --strategy STRAT    Sync strategy: rebase (default), merge, reset
#   --dry-run           Show what would be done
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
SYNC_CIRISNODE=true
SYNC_EEE=true
STRATEGY="rebase"
DRY_RUN=false

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

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $*"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cirisnode)
            SYNC_CIRISNODE=true
            SYNC_EEE=false
            shift
            ;;
        --eee)
            SYNC_CIRISNODE=false
            SYNC_EEE=true
            shift
            ;;
        --strategy)
            STRATEGY="$2"
            shift 2
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --help|-h)
            head -20 "$0" | grep -E "^#" | sed 's/^# *//'
            exit 0
            ;;
        *)
            shift
            ;;
    esac
done

sync_repo() {
    local name="$1"
    local dir="$2"
    local upstream="$3"
    local branch="$4"

    log_info "Syncing $name..."

    if [ ! -d "$dir/.git" ]; then
        log_warn "$name not found at $dir"
        return 1
    fi

    cd "$dir"

    # Ensure upstream remote exists
    if ! git remote get-url upstream >/dev/null 2>&1; then
        log_info "Adding upstream remote: $upstream"
        git remote add upstream "$upstream"
    fi

    # Fetch upstream
    log_info "Fetching upstream..."
    git fetch upstream

    # Get current branch
    local current_branch=$(git branch --show-current)
    log_info "Current branch: $current_branch"

    # Check how far behind we are
    local behind=$(git rev-list --count HEAD..upstream/main 2>/dev/null || echo "0")
    local ahead=$(git rev-list --count upstream/main..HEAD 2>/dev/null || echo "0")

    log_info "Status: $behind commits behind, $ahead commits ahead of upstream/main"

    if [ "$behind" = "0" ]; then
        log_success "$name is up to date"
        cd "$ROOT_DIR"
        return 0
    fi

    if [ "$DRY_RUN" = true ]; then
        log_info "[DRY RUN] Would sync $behind commits from upstream"
        cd "$ROOT_DIR"
        return 0
    fi

    # Perform sync based on strategy
    case $STRATEGY in
        rebase)
            log_info "Rebasing on upstream/main..."
            if ! git rebase upstream/main; then
                log_error "Rebase failed. Resolve conflicts and run: git rebase --continue"
                git rebase --abort
                cd "$ROOT_DIR"
                return 1
            fi
            ;;
        merge)
            log_info "Merging upstream/main..."
            if ! git merge upstream/main -m "Merge upstream changes"; then
                log_error "Merge failed. Resolve conflicts and commit."
                cd "$ROOT_DIR"
                return 1
            fi
            ;;
        reset)
            log_warn "Resetting to upstream/main (DESTRUCTIVE)..."
            git reset --hard upstream/main
            ;;
        *)
            log_error "Unknown strategy: $STRATEGY"
            ;;
    esac

    log_success "$name synced successfully"

    # Push to fork
    log_info "Pushing to origin..."
    git push origin "$current_branch" --force-with-lease || log_warn "Push failed (may need manual push)"

    cd "$ROOT_DIR"
    return 0
}

# Sync CIRISNode
if [ "$SYNC_CIRISNODE" = true ]; then
    sync_repo \
        "CIRISNode" \
        "$SUBMODULES_DIR/cirisnode" \
        "https://github.com/CIRISAI/CIRISNode.git" \
        "main" || true
fi

# Sync EthicsEngine
if [ "$SYNC_EEE" = true ]; then
    sync_repo \
        "EthicsEngine" \
        "$SUBMODULES_DIR/ethicsengine" \
        "https://github.com/emooreatx/ethicsengine_enterprise.git" \
        "main" || true
fi

log_success "Fork sync complete!"
