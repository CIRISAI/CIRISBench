# =============================================================================
# HE-300 Benchmark Integration - Makefile
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := help

# Load environment
-include .env
export

# Version
VERSION := $(shell cat VERSION 2>/dev/null || echo "0.1.0")
GIT_HASH := $(shell git rev-parse --short HEAD 2>/dev/null || echo "unknown")
BUILD_DATE := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

# Docker
DOCKER_REGISTRY ?= ghcr.io
DOCKER_ORG ?= rng-ops
IMAGE_TAG ?= $(VERSION)

# Directories
ROOT_DIR := $(shell dirname $(realpath $(firstword $(MAKEFILE_LIST))))
DOCKER_DIR := $(ROOT_DIR)/docker
SCRIPTS_DIR := $(ROOT_DIR)/scripts
CONFIG_DIR := $(ROOT_DIR)/config
TESTS_DIR := $(ROOT_DIR)/tests

# Colors
GREEN := \033[0;32m
YELLOW := \033[0;33m
RED := \033[0;31m
NC := \033[0m

# =============================================================================
# Help
# =============================================================================

.PHONY: help
help: ## Show this help
	@echo "HE-300 Benchmark Integration - Makefile"
	@echo ""
	@echo "Usage: make [target]"
	@echo ""
	@awk 'BEGIN {FS = ":.*##"; printf "Targets:\n"} /^[a-zA-Z_-]+:.*?##/ { printf "  $(GREEN)%-20s$(NC) %s\n", $$1, $$2 }' $(MAKEFILE_LIST)

# =============================================================================
# Setup
# =============================================================================

.PHONY: setup
setup: ## Initial setup - submodules, dependencies, environment
	@echo "$(GREEN)Setting up HE-300 integration environment...$(NC)"
	@$(SCRIPTS_DIR)/setup-submodules.sh
	@cp -n .env.example .env 2>/dev/null || true
	@echo "$(GREEN)Setup complete. Edit .env as needed.$(NC)"

.PHONY: submodule-update
submodule-update: ## Update submodules to latest from configured branches
	@echo "$(GREEN)Updating submodules...$(NC)"
	@git submodule update --remote --merge

.PHONY: submodule-sync
submodule-sync: ## Sync submodules with upstream
	@echo "$(GREEN)Syncing submodules with upstream...$(NC)"
	@$(SCRIPTS_DIR)/sync-forks.sh

# =============================================================================
# Development
# =============================================================================

.PHONY: dev-up
dev-up: ## Start development environment
	@echo "$(GREEN)Starting development stack...$(NC)"
	@docker compose -f $(DOCKER_DIR)/docker-compose.dev.yml up -d
	@echo "$(GREEN)Development environment ready$(NC)"
	@echo "  CIRISNode API: http://localhost:8000"
	@echo "  EEE API: http://localhost:8080"
	@echo "  UI: http://localhost:3004"

.PHONY: dev-down
dev-down: ## Stop development environment
	@echo "$(YELLOW)Stopping development stack...$(NC)"
	@docker compose -f $(DOCKER_DIR)/docker-compose.dev.yml down

.PHONY: dev-logs
dev-logs: ## View development logs
	@docker compose -f $(DOCKER_DIR)/docker-compose.dev.yml logs -f

.PHONY: dev-restart
dev-restart: dev-down dev-up ## Restart development environment

# =============================================================================
# Testing
# =============================================================================

.PHONY: test
test: test-unit test-integration ## Run all tests
	@echo "$(GREEN)All tests completed$(NC)"

.PHONY: test-unit
test-unit: ## Run unit tests for both projects
	@echo "$(GREEN)Running unit tests...$(NC)"
	@$(SCRIPTS_DIR)/run-tests.sh unit

.PHONY: test-integration
test-integration: ## Run integration tests
	@echo "$(GREEN)Running integration tests...$(NC)"
	@$(SCRIPTS_DIR)/run-tests.sh integration

.PHONY: test-e2e
test-e2e: ## Run end-to-end tests
	@echo "$(GREEN)Running E2E tests...$(NC)"
	@docker compose -f $(DOCKER_DIR)/docker-compose.test.yml up -d
	@$(SCRIPTS_DIR)/run-tests.sh e2e
	@docker compose -f $(DOCKER_DIR)/docker-compose.test.yml down

.PHONY: test-regression
test-regression: ## Run regression test suite
	@echo "$(GREEN)Running regression tests...$(NC)"
	@$(SCRIPTS_DIR)/run-tests.sh regression

.PHONY: test-coverage
test-coverage: ## Run tests with coverage report
	@echo "$(GREEN)Running tests with coverage...$(NC)"
	@$(SCRIPTS_DIR)/run-tests.sh coverage

# =============================================================================
# Benchmarks
# =============================================================================

.PHONY: benchmark
benchmark: ## Run full HE-300 benchmark (300 scenarios)
	@echo "$(GREEN)Running HE-300 benchmark...$(NC)"
	@$(SCRIPTS_DIR)/run-benchmark.sh --full

.PHONY: benchmark-quick
benchmark-quick: ## Run quick HE-300 benchmark (30 scenarios)
	@echo "$(GREEN)Running quick HE-300 benchmark...$(NC)"
	@$(SCRIPTS_DIR)/run-benchmark.sh --quick

.PHONY: benchmark-mock
benchmark-mock: ## Run HE-300 benchmark with mock LLM
	@echo "$(GREEN)Running mock HE-300 benchmark...$(NC)"
	@$(SCRIPTS_DIR)/run-benchmark.sh --mock

# =============================================================================
# Docker
# =============================================================================

.PHONY: build
build: ## Build all Docker images
	@echo "$(GREEN)Building Docker images...$(NC)"
	@$(SCRIPTS_DIR)/build-images.sh all

.PHONY: build-cirisnode
build-cirisnode: ## Build CIRISNode image
	@$(SCRIPTS_DIR)/build-images.sh cirisnode

.PHONY: build-eee
build-eee: ## Build EthicsEngine Enterprise image
	@$(SCRIPTS_DIR)/build-images.sh eee

.PHONY: push
push: ## Push images to registry
	@echo "$(GREEN)Pushing images to $(DOCKER_REGISTRY)/$(DOCKER_ORG)...$(NC)"
	@$(SCRIPTS_DIR)/build-images.sh push

.PHONY: pull
pull: ## Pull latest images from registry
	@docker compose -f $(DOCKER_DIR)/docker-compose.he300.yml pull

# =============================================================================
# Versioning & Releases
# =============================================================================

.PHONY: version
version: ## Show current version
	@echo "Version: $(VERSION)"
	@echo "Git Hash: $(GIT_HASH)"
	@echo "Build Date: $(BUILD_DATE)"

.PHONY: version-patch
version-patch: ## Bump patch version (x.y.Z)
	@$(SCRIPTS_DIR)/version-bump.sh patch

.PHONY: version-minor
version-minor: ## Bump minor version (x.Y.z)
	@$(SCRIPTS_DIR)/version-bump.sh minor

.PHONY: version-major
version-major: ## Bump major version (X.y.z)
	@$(SCRIPTS_DIR)/version-bump.sh major

.PHONY: release
release: ## Create a new release
	@echo "$(GREEN)Creating release v$(VERSION)...$(NC)"
	@$(SCRIPTS_DIR)/version-bump.sh release

.PHONY: changelog
changelog: ## Generate changelog
	@echo "$(GREEN)Generating changelog...$(NC)"
	@git log --oneline --no-decorate $(shell git describe --tags --abbrev=0 2>/dev/null || echo "HEAD~10")..HEAD

# =============================================================================
# Production
# =============================================================================

.PHONY: prod-up
prod-up: ## Start production environment
	@echo "$(GREEN)Starting production stack...$(NC)"
	@docker compose -f $(DOCKER_DIR)/docker-compose.prod.yml up -d

.PHONY: prod-down
prod-down: ## Stop production environment
	@docker compose -f $(DOCKER_DIR)/docker-compose.prod.yml down

.PHONY: deploy
deploy: ## Deploy to production
	@echo "$(GREEN)Deploying v$(VERSION)...$(NC)"
	@$(SCRIPTS_DIR)/deploy.sh

# =============================================================================
# HE-300 Specific
# =============================================================================

.PHONY: he300-up
he300-up: ## Start HE-300 benchmark stack
	@echo "$(GREEN)Starting HE-300 benchmark stack...$(NC)"
	@docker compose -f $(DOCKER_DIR)/docker-compose.he300.yml up -d

.PHONY: he300-down
he300-down: ## Stop HE-300 benchmark stack
	@docker compose -f $(DOCKER_DIR)/docker-compose.he300.yml down

.PHONY: he300-logs
he300-logs: ## View HE-300 stack logs
	@docker compose -f $(DOCKER_DIR)/docker-compose.he300.yml logs -f

# =============================================================================
# Cleanup
# =============================================================================

.PHONY: clean
clean: ## Clean build artifacts
	@echo "$(YELLOW)Cleaning build artifacts...$(NC)"
	@find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	@find . -type d -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true
	@find . -type f -name "*.pyc" -delete 2>/dev/null || true

.PHONY: clean-docker
clean-docker: ## Clean Docker resources
	@echo "$(YELLOW)Cleaning Docker resources...$(NC)"
	@docker compose -f $(DOCKER_DIR)/docker-compose.dev.yml down -v --rmi local 2>/dev/null || true
	@docker compose -f $(DOCKER_DIR)/docker-compose.test.yml down -v --rmi local 2>/dev/null || true
	@docker system prune -f

.PHONY: clean-all
clean-all: clean clean-docker ## Clean everything

# =============================================================================
# Linting & Formatting
# =============================================================================

.PHONY: lint
lint: ## Run linters
	@echo "$(GREEN)Running linters...$(NC)"
	@$(SCRIPTS_DIR)/run-tests.sh lint

.PHONY: format
format: ## Format code
	@echo "$(GREEN)Formatting code...$(NC)"
	@$(SCRIPTS_DIR)/run-tests.sh format

# =============================================================================
# CI Helpers
# =============================================================================

.PHONY: ci-setup
ci-setup: ## CI setup (no submodule init)
	@echo "$(GREEN)CI Setup...$(NC)"
	@pip install -r requirements-ci.txt 2>/dev/null || true

.PHONY: ci-test
ci-test: ## CI test run
	@$(SCRIPTS_DIR)/run-tests.sh ci

.PHONY: ci-build
ci-build: ## CI build
	@$(SCRIPTS_DIR)/build-images.sh ci
