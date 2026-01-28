#!/usr/bin/env bash
# =============================================================================
# Unified Test Runner Script
# =============================================================================
#
# Runs tests across CIRISNode and EthicsEngine Enterprise.
#
# Usage:
#   ./scripts/run-tests.sh [type] [options]
#
# Types:
#   unit          Run unit tests (default)
#   integration   Run integration tests
#   e2e           Run end-to-end tests
#   regression    Run full regression suite
#   coverage      Run with coverage report
#   lint          Run linters only
#   format        Run formatters
#   ci            CI-optimized test run
#
# Options:
#   --cirisnode-only    Only run CIRISNode tests
#   --eee-only          Only run EEE tests
#   --verbose           Verbose output
#   --failfast          Stop on first failure
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
TEST_TYPE="${1:-unit}"
CIRISNODE_ONLY=false
EEE_ONLY=false
VERBOSE=false
FAILFAST=false

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(dirname "$SCRIPT_DIR")"
SUBMODULES_DIR="$ROOT_DIR/submodules"

# -----------------------------------------------------------------------------
# Functions
# -----------------------------------------------------------------------------

log_info() {
    echo -e "${BLUE}[INFO]${NC} $*"
}

log_success() {
    echo -e "${GREEN}[PASS]${NC} $*"
}

log_warn() {
    echo -e "${YELLOW}[WARN]${NC} $*"
}

log_error() {
    echo -e "${RED}[FAIL]${NC} $*"
}

# Parse arguments
shift || true
while [[ $# -gt 0 ]]; do
    case $1 in
        --cirisnode-only)
            CIRISNODE_ONLY=true
            shift
            ;;
        --eee-only)
            EEE_ONLY=true
            shift
            ;;
        --verbose|-v)
            VERBOSE=true
            shift
            ;;
        --failfast|-x)
            FAILFAST=true
            shift
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

# Build pytest args
PYTEST_ARGS="-v"
if [ "$FAILFAST" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS -x"
fi
if [ "$VERBOSE" = true ]; then
    PYTEST_ARGS="$PYTEST_ARGS --tb=long"
else
    PYTEST_ARGS="$PYTEST_ARGS --tb=short"
fi

# -----------------------------------------------------------------------------
# Test Functions
# -----------------------------------------------------------------------------

run_cirisnode_tests() {
    local test_dir="$SUBMODULES_DIR/cirisnode"
    
    if [ ! -d "$test_dir" ]; then
        log_warn "CIRISNode not found at $test_dir"
        return 1
    fi
    
    log_info "Running CIRISNode tests..."
    cd "$test_dir"
    
    # Install dependencies if needed
    pip install -q -r requirements.txt 2>/dev/null || true
    pip install -q pytest pytest-asyncio pytest-cov 2>/dev/null || true
    
    # Set environment
    export EEE_ENABLED=false
    export JWT_SECRET=test-secret
    
    pytest tests/ $PYTEST_ARGS "$@"
    local result=$?
    
    cd "$ROOT_DIR"
    return $result
}

run_eee_tests() {
    local test_dir="$SUBMODULES_DIR/ethicsengine"
    
    if [ ! -d "$test_dir" ]; then
        log_warn "EthicsEngine not found at $test_dir"
        return 1
    fi
    
    log_info "Running EthicsEngine tests..."
    cd "$test_dir"
    
    # Install dependencies if needed
    pip install -q -r requirements.txt 2>/dev/null || true
    pip install -q pytest pytest-asyncio pytest-cov 2>/dev/null || true
    
    # Set environment
    export FF_MOCK_LLM=true
    export HE300_ENABLED=true
    
    pytest tests/ $PYTEST_ARGS "$@"
    local result=$?
    
    cd "$ROOT_DIR"
    return $result
}

run_integration_tests() {
    log_info "Running integration tests..."
    
    cd "$ROOT_DIR"
    pip install -q pytest pytest-asyncio httpx 2>/dev/null || true
    
    pytest tests/test_integration.py $PYTEST_ARGS "$@"
}

run_e2e_tests() {
    log_info "Running E2E tests..."
    
    cd "$ROOT_DIR"
    pip install -q pytest pytest-asyncio httpx 2>/dev/null || true
    
    pytest tests/test_e2e_he300.py $PYTEST_ARGS "$@"
}

run_lint() {
    log_info "Running linters..."
    
    pip install -q ruff black isort mypy 2>/dev/null || true
    
    local failed=false
    
    if [ -d "$SUBMODULES_DIR/cirisnode" ]; then
        log_info "Linting CIRISNode..."
        ruff check "$SUBMODULES_DIR/cirisnode/cirisnode" || failed=true
    fi
    
    if [ -d "$SUBMODULES_DIR/ethicsengine" ]; then
        log_info "Linting EthicsEngine..."
        ruff check "$SUBMODULES_DIR/ethicsengine/api" "$SUBMODULES_DIR/ethicsengine/core" || failed=true
    fi
    
    if [ "$failed" = true ]; then
        return 1
    fi
}

run_format() {
    log_info "Running formatters..."
    
    pip install -q black isort 2>/dev/null || true
    
    if [ -d "$SUBMODULES_DIR/cirisnode" ]; then
        log_info "Formatting CIRISNode..."
        black "$SUBMODULES_DIR/cirisnode/cirisnode" "$SUBMODULES_DIR/cirisnode/tests"
        isort "$SUBMODULES_DIR/cirisnode/cirisnode" "$SUBMODULES_DIR/cirisnode/tests"
    fi
    
    if [ -d "$SUBMODULES_DIR/ethicsengine" ]; then
        log_info "Formatting EthicsEngine..."
        black "$SUBMODULES_DIR/ethicsengine/api" "$SUBMODULES_DIR/ethicsengine/core" "$SUBMODULES_DIR/ethicsengine/tests"
        isort "$SUBMODULES_DIR/ethicsengine/api" "$SUBMODULES_DIR/ethicsengine/core" "$SUBMODULES_DIR/ethicsengine/tests"
    fi
}

# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

log_info "Test type: $TEST_TYPE"

case $TEST_TYPE in
    unit)
        exit_code=0
        if [ "$EEE_ONLY" != true ]; then
            run_cirisnode_tests || exit_code=1
        fi
        if [ "$CIRISNODE_ONLY" != true ]; then
            run_eee_tests || exit_code=1
        fi
        exit $exit_code
        ;;
    integration)
        run_integration_tests
        ;;
    e2e)
        run_e2e_tests
        ;;
    regression)
        exit_code=0
        run_cirisnode_tests || exit_code=1
        run_eee_tests || exit_code=1
        run_integration_tests || exit_code=1
        run_e2e_tests || exit_code=1
        exit $exit_code
        ;;
    coverage)
        exit_code=0
        if [ "$EEE_ONLY" != true ]; then
            run_cirisnode_tests --cov=cirisnode --cov-report=xml --cov-report=html || exit_code=1
        fi
        if [ "$CIRISNODE_ONLY" != true ]; then
            run_eee_tests --cov=api --cov=core --cov-report=xml --cov-report=html || exit_code=1
        fi
        exit $exit_code
        ;;
    lint)
        run_lint
        ;;
    format)
        run_format
        ;;
    ci)
        # CI-optimized: fast, fail on first error
        FAILFAST=true
        PYTEST_ARGS="$PYTEST_ARGS -x --tb=short"
        exit_code=0
        run_cirisnode_tests || exit_code=1
        run_eee_tests || exit_code=1
        exit $exit_code
        ;;
    *)
        log_error "Unknown test type: $TEST_TYPE"
        exit 1
        ;;
esac

log_success "Tests completed!"
