#!/usr/bin/env bash
#
# setup-vault.sh - Initialize Vault with HE-300 secrets and configuration
#
# Usage:
#   ./setup-vault.sh [options]
#
# Options:
#   --vault-addr    Vault server address (default: $VAULT_ADDR or http://127.0.0.1:8200)
#   --init          Initialize a new Vault instance
#   --unseal        Unseal Vault using provided keys
#   --configure     Configure auth methods and policies
#   --secrets       Generate and store initial secrets
#   --all           Run all initialization steps
#   --dry-run       Show what would be done without making changes
#
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VAULT_ADDR="${VAULT_ADDR:-http://127.0.0.1:8200}"
DRY_RUN=false
DO_INIT=false
DO_UNSEAL=false
DO_CONFIGURE=false
DO_SECRETS=false

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

log_info() { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[OK]${NC} $*"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

usage() {
    sed -n '3,14p' "$0" | sed 's/^#//'
    exit 1
}

check_vault() {
    if ! command -v vault &> /dev/null; then
        log_error "Vault CLI not found. Install from: https://www.vaultproject.io/downloads"
        exit 1
    fi

    if ! vault status &> /dev/null; then
        log_warn "Cannot connect to Vault at $VAULT_ADDR"
        return 1
    fi

    return 0
}

generate_secret() {
    local length="${1:-32}"
    openssl rand -hex "$length"
}

generate_base64_secret() {
    local length="${1:-32}"
    openssl rand -base64 "$length" | tr -d '\n'
}

init_vault() {
    log_info "Initializing Vault..."

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would initialize Vault with 5 key shares, 3 threshold"
        return
    fi

    local init_output
    init_output=$(vault operator init -key-shares=5 -key-threshold=3 -format=json)

    # Save init output securely
    local init_file="${SCRIPT_DIR}/../.vault-init.json"
    echo "$init_output" > "$init_file"
    chmod 600 "$init_file"

    log_success "Vault initialized. Keys saved to $init_file"
    log_warn "IMPORTANT: Back up this file securely and delete it from disk!"

    # Extract root token
    VAULT_TOKEN=$(echo "$init_output" | jq -r '.root_token')
    export VAULT_TOKEN
}

unseal_vault() {
    log_info "Unsealing Vault..."

    local init_file="${SCRIPT_DIR}/../.vault-init.json"
    if [[ ! -f "$init_file" ]]; then
        log_error "Init file not found: $init_file"
        log_info "Provide unseal keys manually or run with --init first"
        exit 1
    fi

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would unseal Vault using saved keys"
        return
    fi

    # Extract unseal keys
    local keys
    keys=$(jq -r '.unseal_keys_b64[]' "$init_file" | head -3)

    for key in $keys; do
        vault operator unseal "$key" > /dev/null
    done

    log_success "Vault unsealed"
}

configure_policies() {
    log_info "Configuring Vault policies..."

    local policy_dir="${SCRIPT_DIR}/../config/vault/policies"

    for policy_file in "$policy_dir"/*.hcl; do
        local policy_name
        policy_name=$(basename "$policy_file" .hcl)

        if $DRY_RUN; then
            log_info "[DRY-RUN] Would create policy: $policy_name"
            continue
        fi

        vault policy write "$policy_name" "$policy_file"
        log_success "Created policy: $policy_name"
    done
}

configure_auth() {
    log_info "Configuring auth methods..."

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would configure AppRole and JWT auth methods"
        return
    fi

    # Enable AppRole if not already enabled
    if ! vault auth list | grep -q "approle/"; then
        vault auth enable approle
        log_success "Enabled AppRole auth"
    fi

    # Enable JWT for GitHub OIDC if not already enabled
    if ! vault auth list | grep -q "jwt-github/"; then
        vault auth enable -path=jwt-github jwt
        log_success "Enabled JWT auth for GitHub"
    fi

    # Configure GitHub OIDC
    vault write auth/jwt-github/config \
        oidc_discovery_url="https://token.actions.githubusercontent.com" \
        bound_issuer="https://token.actions.githubusercontent.com"

    # Configure roles
    vault write auth/jwt-github/role/he300-deploy \
        role_type="jwt" \
        user_claim="actor" \
        bound_audiences="sigstore" \
        bound_claims='{"repository":"rng-ops/he300-integration"}' \
        policies="he300-deploy" \
        ttl="1h" \
        max_ttl="4h"

    vault write auth/jwt-github/role/terraform-deploy \
        role_type="jwt" \
        user_claim="actor" \
        bound_audiences="sigstore" \
        bound_claims='{"repository":"rng-ops/he300-integration","ref":"refs/heads/main"}' \
        policies="terraform-deploy" \
        ttl="1h" \
        max_ttl="2h"

    log_success "Configured GitHub OIDC roles"

    # Configure AppRoles
    vault write auth/approle/role/he300-installer \
        policies="he300-deploy" \
        token_ttl="30m" \
        token_max_ttl="1h" \
        secret_id_ttl="24h" \
        secret_id_num_uses=10

    vault write auth/approle/role/he300-gpu-host \
        policies="he300-deploy" \
        token_ttl="1h" \
        token_max_ttl="24h" \
        token_num_uses=0 \
        secret_id_ttl=0 \
        secret_id_num_uses=0

    vault write auth/approle/role/he300-dashboard \
        policies="dashboard" \
        token_ttl="1h" \
        token_max_ttl="24h" \
        token_num_uses=0 \
        secret_id_ttl=0 \
        secret_id_num_uses=0

    log_success "Configured AppRole roles"
}

enable_secrets_engine() {
    log_info "Enabling secrets engines..."

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would enable KV v2 secrets engine at secret/"
        return
    fi

    # Enable KV v2 if not already enabled
    if ! vault secrets list | grep -q "secret/"; then
        vault secrets enable -path=secret -version=2 kv
        log_success "Enabled KV v2 secrets engine"
    fi
}

generate_and_store_secrets() {
    log_info "Generating and storing secrets..."

    # API Keys
    local cirisnode_key=$(generate_secret 32)
    local ethicsengine_key=$(generate_secret 32)
    local dashboard_key=$(generate_secret 32)

    # Database passwords
    local postgres_password=$(generate_secret 24)
    local redis_password=$(generate_secret 24)
    local dashboard_db_password=$(generate_secret 24)

    # JWT secret
    local jwt_secret=$(generate_base64_secret 48)

    # Webhook secrets
    local github_webhook_secret=$(generate_secret 32)
    local dashboard_webhook_secret=$(generate_secret 32)

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would store secrets at secret/he300/*"
        log_info "  - API keys: cirisnode, ethicsengine, dashboard"
        log_info "  - Database: postgres, redis, dashboard"
        log_info "  - JWT signing key"
        log_info "  - Webhook secrets"
        return
    fi

    # Store API keys
    vault kv put secret/he300/api-keys/cirisnode \
        api_key="$cirisnode_key" \
        description="CIRISNode API authentication key"

    vault kv put secret/he300/api-keys/ethicsengine \
        api_key="$ethicsengine_key" \
        description="EthicsEngine Enterprise API key"

    vault kv put secret/he300/api-keys/dashboard \
        api_key="$dashboard_key" \
        description="Dashboard API key"

    log_success "Stored API keys"

    # Store database credentials
    vault kv put secret/he300/database/postgres \
        username="he300" \
        password="$postgres_password" \
        host="postgres" \
        port="5432" \
        database="he300"

    vault kv put secret/he300/database/redis \
        password="$redis_password" \
        host="redis" \
        port="6379"

    vault kv put secret/he300/database/dashboard \
        username="dashboard" \
        password="$dashboard_db_password" \
        host="postgres" \
        port="5432" \
        database="he300_dashboard"

    log_success "Stored database credentials"

    # Store JWT secret
    vault kv put secret/he300/jwt/signing-key \
        secret="$jwt_secret" \
        algorithm="HS256" \
        issuer="he300-benchmark"

    log_success "Stored JWT signing key"

    # Store webhook secrets
    vault kv put secret/he300/webhook/github \
        secret="$github_webhook_secret"

    vault kv put secret/he300/webhook/dashboard \
        secret="$dashboard_webhook_secret"

    log_success "Stored webhook secrets"

    # Output summary
    echo ""
    log_info "=== Secret Generation Complete ==="
    log_info "Secrets stored at: secret/he300/*"
    log_info ""
    log_info "To retrieve a secret:"
    log_info "  vault kv get secret/he300/api-keys/cirisnode"
    log_info ""
    log_info "To get AppRole credentials:"
    log_info "  vault read auth/approle/role/he300-installer/role-id"
    log_info "  vault write -f auth/approle/role/he300-installer/secret-id"
}

setup_wireguard_keys() {
    log_info "Generating WireGuard keys..."

    if ! command -v wg &> /dev/null; then
        log_warn "WireGuard tools not found. Skipping key generation."
        log_info "Install wireguard-tools and run: ./setup-vault.sh --wireguard"
        return
    fi

    # Generate GPU host keys
    local gpu_private=$(wg genkey)
    local gpu_public=$(echo "$gpu_private" | wg pubkey)

    # Generate test runner keys
    local runner_private=$(wg genkey)
    local runner_public=$(echo "$runner_private" | wg pubkey)

    if $DRY_RUN; then
        log_info "[DRY-RUN] Would store WireGuard keys at secret/he300/wireguard/*"
        return
    fi

    vault kv put secret/he300/wireguard/gpu-host \
        private_key="$gpu_private" \
        public_key="$gpu_public" \
        address="10.0.0.2/24" \
        listen_port="51820"

    vault kv put secret/he300/wireguard/test-runner \
        private_key="$runner_private" \
        public_key="$runner_public" \
        address="10.0.0.1/24"

    log_success "Stored WireGuard keys"

    echo ""
    log_info "WireGuard Public Keys:"
    log_info "  GPU Host:    $gpu_public"
    log_info "  Test Runner: $runner_public"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vault-addr)
            VAULT_ADDR="$2"
            shift 2
            ;;
        --init)
            DO_INIT=true
            shift
            ;;
        --unseal)
            DO_UNSEAL=true
            shift
            ;;
        --configure)
            DO_CONFIGURE=true
            shift
            ;;
        --secrets)
            DO_SECRETS=true
            shift
            ;;
        --all)
            DO_INIT=true
            DO_UNSEAL=true
            DO_CONFIGURE=true
            DO_SECRETS=true
            shift
            ;;
        --dry-run)
            DRY_RUN=true
            shift
            ;;
        --wireguard)
            setup_wireguard_keys
            exit 0
            ;;
        -h|--help)
            usage
            ;;
        *)
            log_error "Unknown option: $1"
            usage
            ;;
    esac
done

# If no options specified, show usage
if ! $DO_INIT && ! $DO_UNSEAL && ! $DO_CONFIGURE && ! $DO_SECRETS; then
    usage
fi

# Export Vault address
export VAULT_ADDR

log_info "Using Vault at: $VAULT_ADDR"

# Run requested operations
if $DO_INIT; then
    init_vault
fi

if $DO_UNSEAL; then
    unseal_vault
fi

if $DO_CONFIGURE; then
    check_vault || exit 1
    enable_secrets_engine
    configure_policies
    configure_auth
fi

if $DO_SECRETS; then
    check_vault || exit 1
    generate_and_store_secrets
    setup_wireguard_keys
fi

log_success "Vault setup complete!"
