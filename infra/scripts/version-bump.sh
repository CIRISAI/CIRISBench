#!/usr/bin/env bash
# =============================================================================
# Version Bump Script
# =============================================================================
#
# Manages version bumping and release tagging.
#
# Usage:
#   ./scripts/version-bump.sh [patch|minor|major|release]
#
# =============================================================================

set -euo pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
BLUE='\033[0;34m'
NC='\033[0m'

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
VERSION_FILE="$ROOT_DIR/VERSION"

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

# Get current version
if [ ! -f "$VERSION_FILE" ]; then
    echo "0.1.0" > "$VERSION_FILE"
fi

CURRENT_VERSION=$(cat "$VERSION_FILE" | tr -d '[:space:]')
log_info "Current version: $CURRENT_VERSION"

# Parse version
IFS='.' read -r MAJOR MINOR PATCH <<< "$CURRENT_VERSION"
MAJOR=${MAJOR:-0}
MINOR=${MINOR:-1}
PATCH=${PATCH:-0}

# Determine bump type
BUMP_TYPE="${1:-patch}"

case $BUMP_TYPE in
    patch)
        PATCH=$((PATCH + 1))
        ;;
    minor)
        MINOR=$((MINOR + 1))
        PATCH=0
        ;;
    major)
        MAJOR=$((MAJOR + 1))
        MINOR=0
        PATCH=0
        ;;
    release)
        # Don't bump, just create release tag
        log_info "Creating release for v$CURRENT_VERSION"

        # Verify we're on a clean state
        if ! git diff --quiet HEAD 2>/dev/null; then
            log_error "Working directory not clean. Commit or stash changes first."
        fi

        # Create tag
        git tag -a "v$CURRENT_VERSION" -m "Release v$CURRENT_VERSION"
        log_success "Created tag v$CURRENT_VERSION"

        echo ""
        echo "To push the release:"
        echo "  git push origin v$CURRENT_VERSION"
        exit 0
        ;;
    *)
        log_error "Unknown bump type: $BUMP_TYPE. Use: patch, minor, major, release"
        ;;
esac

NEW_VERSION="$MAJOR.$MINOR.$PATCH"
log_info "New version: $NEW_VERSION"

# Update VERSION file
echo "$NEW_VERSION" > "$VERSION_FILE"
log_success "Updated VERSION file"

# Update submodule package versions if they exist
if [ -f "$ROOT_DIR/submodules/cirisnode/cirisnode/__init__.py" ]; then
    sed -i.bak "s/__version__ = .*/__version__ = \"$NEW_VERSION\"/" \
        "$ROOT_DIR/submodules/cirisnode/cirisnode/__init__.py" 2>/dev/null || true
    rm -f "$ROOT_DIR/submodules/cirisnode/cirisnode/__init__.py.bak"
fi

# Commit version bump
git add "$VERSION_FILE"
git commit -m "chore: Bump version to $NEW_VERSION" || true

log_success "Version bumped to $NEW_VERSION"
echo ""
echo "Next steps:"
echo "  1. Push changes: git push"
echo "  2. Create release: ./scripts/version-bump.sh release"
