# HE-300 Distributed Runner Architecture

## Overview

This document describes the architecture for deploying and running HE-300 benchmarks across distributed infrastructure, where:

- **GPU Host** (Lambda Labs A10, local Mac, etc.) runs the inference server and CIRISNode sidecar
- **Test Runner** (GitHub Actions, local dev, CI server) orchestrates benchmarks and collects results

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              TEST RUNNER                                     │
│  (GitHub Actions / Local Dev / Jenkins / GitLab CI)                         │
│                                                                             │
│  ┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐         │
│  │  Benchmark      │    │  Results        │    │  Report         │         │
│  │  Orchestrator   │───▶│  Collector      │───▶│  Generator      │         │
│  └────────┬────────┘    └─────────────────┘    └─────────────────┘         │
│           │                                                                 │
└───────────┼─────────────────────────────────────────────────────────────────┘
            │
            │ HTTPS / WireGuard VPN
            │ Port 8000 (CIRISNode)
            │ Port 8080 (EthicsEngine)
            ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                              GPU HOST                                        │
│  Lambda A10 / Mac M-series / Custom GPU Server                              │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                     Docker Network: he300-net                        │   │
│  │                                                                      │   │
│  │  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐   ┌─────────┐ │   │
│  │  │  CIRISNode  │──▶│  Ethics     │──▶│   Ollama    │──▶│  GPU    │ │   │
│  │  │  :8000      │   │  Engine     │   │   :11434    │   │  A10    │ │   │
│  │  │  (Sidecar)  │   │  :8080      │   │             │   │  24GB   │ │   │
│  │  └─────────────┘   └─────────────┘   └─────────────┘   └─────────┘ │   │
│  │        │                 │                                          │   │
│  │        ▼                 ▼                                          │   │
│  │  ┌─────────────┐   ┌─────────────┐                                  │   │
│  │  │   Redis     │   │  PostgreSQL │                                  │   │
│  │  │   :6379     │   │   :5432     │                                  │   │
│  │  └─────────────┘   └─────────────┘                                  │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────┐                                                        │
│  │   WireGuard     │◀── Tunnel to Test Runner (optional)                   │
│  │   wg0: 10.0.0.x │                                                        │
│  └─────────────────┘                                                        │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Deployment Modes

### Mode 1: Local Development

```
┌─────────────────────────────────────────┐
│          Developer Machine              │
│                                         │
│  pytest ──▶ CIRISNode ──▶ EEE ──▶ LLM  │
│  (local)    (docker)     (docker) (local)│
│                                         │
│  All on localhost, no VPN needed        │
└─────────────────────────────────────────┘
```

**Use case**: Developer testing on local machine with GPU (Mac M-series, Linux + NVIDIA)

**Connection**: `localhost:8000` → `localhost:8080` → `localhost:11434`

---

### Mode 2: Remote GPU (Direct HTTPS)

```
┌────────────────┐         HTTPS          ┌────────────────┐
│  Test Runner   │ ──────────────────────▶│   GPU Host     │
│  (GitHub/CI)   │     (Public IP)        │  (Lambda A10)  │
│                │                         │                │
│  Port 443 ─────┼─────────────────────── │──▶ :8000       │
└────────────────┘                         └────────────────┘
```

**Use case**: CI runner connecting to cloud GPU with public IP

**Connection**: Test runner → `https://gpu-host.example.com:8000`

**Security**: 
- HTTPS with Let's Encrypt certificate
- API key authentication
- IP allowlist (GitHub Actions IP ranges)

---

### Mode 3: Remote GPU (WireGuard VPN)

```
┌────────────────┐                         ┌────────────────┐
│  Test Runner   │                         │   GPU Host     │
│  wg0: 10.0.0.1 │ ══════════════════════ │  wg0: 10.0.0.2 │
│                │     WireGuard VPN       │                │
│  10.0.0.2:8000 │◀────────────────────── │  :8000         │
└────────────────┘                         └────────────────┘
```

**Use case**: Private GPU host behind NAT, GitHub Actions with self-hosted runner

**Connection**: Test runner → `10.0.0.2:8000` (via WireGuard tunnel)

**Security**:
- WireGuard encryption
- No public IP exposure
- Mutual authentication via public keys

---

## Installer Package

### Package Contents

```
he300-installer/
├── install.sh                    # Main installer script
├── uninstall.sh                  # Removal script
├── config/
│   ├── install.yaml              # Installation configuration
│   ├── models.yaml               # Model weight configurations
│   └── wireguard/
│       ├── wg0.conf.template     # WireGuard config template
│       └── generate-keys.sh      # Key generation script
├── docker/
│   ├── docker-compose.gpu.yml    # GPU host compose file
│   ├── docker-compose.cpu.yml    # CPU-only fallback
│   └── .env.template             # Environment template
├── systemd/
│   ├── he300-stack.service       # Systemd service unit
│   └── he300-health.timer        # Health check timer
├── scripts/
│   ├── setup-nvidia.sh           # NVIDIA driver setup
│   ├── setup-ollama.sh           # Ollama installation
│   ├── pull-models.sh            # Model weight download
│   ├── health-check.sh           # Stack health verification
│   └── benchmark-test.sh         # Quick validation benchmark
└── docs/
    └── INSTALL.md                # Installation guide
```

### Installation Configuration (`install.yaml`)

```yaml
# Target host configuration
target:
  host: "gpu-server.example.com"
  user: "ubuntu"
  ssh_key: "~/.ssh/id_ed25519"
  port: 22

# OS detection (auto or manual)
os:
  detect: auto
  # manual: "ubuntu-22.04" | "ubuntu-24.04" | "lambda-22.04" | "lambda-24.04"

# GPU configuration
gpu:
  type: auto  # auto-detect, or: a10, a100, rtx4090, m1, m2, m3
  memory: auto
  nvidia_driver: auto  # or specific version: "535.104.05"

# Model configuration
models:
  default: "llama-3.2-3B-instruct"
  pull_on_install:
    - "llama-3.2-3B-instruct"
  quantization: "Q4_K_M"  # Q4_K_M, Q5_K_M, Q8_0, fp16

# Network configuration
network:
  mode: "direct"  # direct, wireguard, tailscale
  
  direct:
    https: true
    certbot: true
    domain: "gpu-server.example.com"
  
  wireguard:
    enabled: false
    interface: "wg0"
    listen_port: 51820
    address: "10.0.0.2/24"
    peer_public_key: ""  # Test runner's public key
    peer_endpoint: ""    # Test runner's public IP:port
    peer_allowed_ips: "10.0.0.1/32"
  
  tailscale:
    enabled: false
    auth_key: ""  # Tailscale auth key

# Service configuration
services:
  cirisnode:
    port: 8000
    replicas: 1
  
  ethicsengine:
    port: 8080
    replicas: 1
  
  ollama:
    port: 11434
    gpu_layers: -1  # All layers on GPU
  
  redis:
    port: 6379
    persistence: true
  
  postgres:
    port: 5432
    persistence: true

# Security
security:
  api_key: ""  # Generated if empty
  jwt_secret: ""  # Generated if empty
  allowed_ips: []  # Empty = allow all, or list of CIDRs
```

### Model Weight Configuration (`models.yaml`)

```yaml
# Model registry
models:
  # Llama 3.2 variants
  llama-3.2-3B-instruct:
    ollama_name: "llama3.2:3b-instruct-q4_K_M"
    context_length: 8192
    recommended_gpu_memory: 4  # GB
    quantizations:
      Q4_K_M: "llama3.2:3b-instruct-q4_K_M"
      Q5_K_M: "llama3.2:3b-instruct-q5_K_M"
      Q8_0: "llama3.2:3b-instruct-q8_0"
      fp16: "llama3.2:3b-instruct-fp16"
  
  llama-3.2-8B-instruct:
    ollama_name: "llama3.2:8b-instruct-q4_K_M"
    context_length: 8192
    recommended_gpu_memory: 8
    quantizations:
      Q4_K_M: "llama3.2:8b-instruct-q4_K_M"
      Q5_K_M: "llama3.2:8b-instruct-q5_K_M"
      Q8_0: "llama3.2:8b-instruct-q8_0"
      fp16: "llama3.2:8b-instruct-fp16"
  
  # Mistral variants
  mistral-7B-instruct:
    ollama_name: "mistral:7b-instruct-q4_K_M"
    context_length: 8192
    recommended_gpu_memory: 8
    quantizations:
      Q4_K_M: "mistral:7b-instruct-q4_K_M"
      Q5_K_M: "mistral:7b-instruct-q5_K_M"
      Q8_0: "mistral:7b-instruct-q8_0"

# A10 24GB optimized configurations
gpu_profiles:
  a10-24gb:
    max_model_size: "13B"
    recommended_models:
      - "llama-3.2-8B-instruct"
      - "mistral-7B-instruct"
    max_batch_size: 50
    concurrent_requests: 4
    
  a100-40gb:
    max_model_size: "70B"
    recommended_models:
      - "llama-3.2-70B-instruct"
    max_batch_size: 100
    concurrent_requests: 8

  mac-m1-16gb:
    max_model_size: "7B"
    recommended_models:
      - "llama-3.2-3B-instruct"
    max_batch_size: 20
    concurrent_requests: 2
```

---

## Installation Flow

### Phase 1: Remote Setup

```bash
# From local machine / CI runner
./install.sh --config install.yaml

# What happens:
# 1. SSH to target host
# 2. Detect OS and GPU
# 3. Install Docker if needed
# 4. Install NVIDIA Container Toolkit (if NVIDIA GPU)
# 5. Pull Docker images
# 6. Configure services
# 7. Set up networking (direct/WireGuard/Tailscale)
# 8. Start stack
# 9. Pull model weights
# 10. Run health check
# 11. Run quick benchmark validation
```

### Phase 2: OS-Specific Setup

```
┌─────────────────────────────────────────────────────────────────┐
│                    OS Detection & Setup                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Lambda Stack 22.04 / 24.04                                     │
│  ├── Docker: pre-installed                                      │
│  ├── NVIDIA: pre-installed (Lambda drivers)                    │
│  ├── CUDA: pre-installed                                        │
│  └── Action: Install NVIDIA Container Toolkit only              │
│                                                                  │
│  Ubuntu 22.04 / 24.04                                           │
│  ├── Docker: apt install docker.io                              │
│  ├── NVIDIA: Install from nvidia-driver-535                    │
│  ├── CUDA: Install nvidia-container-toolkit                    │
│  └── Action: Full driver + toolkit installation                 │
│                                                                  │
│  GPU Base 24.04                                                  │
│  ├── Docker: pre-installed                                      │
│  ├── NVIDIA: pre-installed                                      │
│  └── Action: Minimal setup, verify only                         │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 3: Network Configuration

#### Direct HTTPS (Public IP)

```bash
# Auto-configure with certbot
./install.sh --network direct --domain gpu.example.com

# Firewall rules
ufw allow 8000/tcp  # CIRISNode
ufw allow 8080/tcp  # EthicsEngine
ufw allow 443/tcp   # HTTPS proxy
```

#### WireGuard VPN

```bash
# On GPU host (generates keys)
./install.sh --network wireguard --wg-address 10.0.0.2/24

# Output: GPU host public key
# WG_PUBLIC_KEY=xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx

# On test runner (GitHub self-hosted runner)
./scripts/setup-wireguard-client.sh \
  --address 10.0.0.1/24 \
  --peer-key "xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx" \
  --peer-endpoint "gpu-host-ip:51820"
```

#### Tailscale (Simplest)

```bash
# On GPU host
./install.sh --network tailscale --ts-authkey "tskey-xxx"

# On test runner
tailscale up --authkey "tskey-xxx"
# Access via: gpu-host.ts.net:8000
```

---

## Test Runner Configuration

### GitHub Actions (Remote GPU)

```yaml
# .github/workflows/he300-benchmark.yml
name: HE-300 Benchmark

on:
  workflow_dispatch:
    inputs:
      gpu_host:
        description: 'GPU host address'
        default: '10.0.0.2'  # WireGuard address
      model:
        description: 'Model to test'
        default: 'llama-3.2-3B-instruct'

jobs:
  benchmark:
    runs-on: ubuntu-latest  # or self-hosted with WireGuard
    
    steps:
      - uses: actions/checkout@v4
      
      # Option A: Direct HTTPS connection
      - name: Run benchmark (HTTPS)
        if: ${{ !inputs.use_wireguard }}
        env:
          CIRISNODE_URL: https://${{ inputs.gpu_host }}:8000
          API_KEY: ${{ secrets.CIRISNODE_API_KEY }}
        run: ./scripts/run_he300_benchmark.sh
      
      # Option B: WireGuard tunnel
      - name: Setup WireGuard
        if: ${{ inputs.use_wireguard }}
        run: |
          sudo apt-get install -y wireguard
          echo "${{ secrets.WG_PRIVATE_KEY }}" | sudo tee /etc/wireguard/wg0.conf
          sudo wg-quick up wg0
      
      - name: Run benchmark (WireGuard)
        if: ${{ inputs.use_wireguard }}
        env:
          CIRISNODE_URL: http://10.0.0.2:8000
        run: ./scripts/run_he300_benchmark.sh
```

### GitHub Actions (Self-Hosted GPU Runner)

```yaml
# GPU host IS the runner - no network complexity
jobs:
  benchmark:
    runs-on: self-hosted  # Label for GPU machine
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Start stack
        run: docker compose -f docker/docker-compose.gpu.yml up -d
      
      - name: Run benchmark
        env:
          CIRISNODE_URL: http://localhost:8000
        run: ./scripts/run_he300_benchmark.sh
```

### Local Development

```bash
# Start local stack
make dev-up

# Run benchmark against local stack
export CIRISNODE_URL=http://localhost:8000
./scripts/run_he300_benchmark.sh --model ollama/llama3.2 --sample-size 50
```

---

## Network Security

### API Authentication

```
┌─────────────────────────────────────────────────────────────────┐
│                    Authentication Flow                           │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Test Runner                        GPU Host                     │
│       │                                  │                       │
│       │─── POST /auth/login ────────────▶│                       │
│       │    {api_key: "xxx"}              │                       │
│       │                                  │                       │
│       │◀── {token: "jwt.xxx"} ──────────│                       │
│       │                                  │                       │
│       │─── GET /he300/batch ────────────▶│                       │
│       │    Authorization: Bearer jwt.xxx │                       │
│       │                                  │                       │
│       │◀── {results: [...]} ────────────│                       │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### IP Allowlisting

```yaml
# install.yaml
security:
  allowed_ips:
    # GitHub Actions IP ranges
    - "140.82.112.0/20"
    - "143.55.64.0/20"
    # Your office
    - "203.0.113.50/32"
    # WireGuard subnet
    - "10.0.0.0/24"
```

### WireGuard Configuration

```ini
# GPU Host: /etc/wireguard/wg0.conf
[Interface]
PrivateKey = <gpu-host-private-key>
Address = 10.0.0.2/24
ListenPort = 51820

[Peer]
# Test Runner (GitHub self-hosted or CI server)
PublicKey = <test-runner-public-key>
AllowedIPs = 10.0.0.1/32
# If test runner has static IP
Endpoint = runner.example.com:51820
PersistentKeepalive = 25
```

```ini
# Test Runner: /etc/wireguard/wg0.conf
[Interface]
PrivateKey = <test-runner-private-key>
Address = 10.0.0.1/24

[Peer]
# GPU Host
PublicKey = <gpu-host-public-key>
AllowedIPs = 10.0.0.2/32
Endpoint = gpu-host.example.com:51820
PersistentKeepalive = 25
```

---

## Deployment Commands

### Full Installation

```bash
# Interactive mode
./install.sh

# Non-interactive with config file
./install.sh --config install.yaml

# Dry run (show what would happen)
./install.sh --config install.yaml --dry-run
```

### Model Management

```bash
# Pull additional model
./install.sh --pull-model llama-3.2-8B-instruct --quantization Q5_K_M

# List available models
./install.sh --list-models

# Switch default model
./install.sh --set-model mistral-7B-instruct
```

### Stack Management

```bash
# Check status
./install.sh --status

# Restart stack
./install.sh --restart

# View logs
./install.sh --logs

# Update to latest version
./install.sh --update

# Uninstall
./uninstall.sh
```

---

## Lambda Labs Specific Notes

### Quick Start on Lambda A10

```bash
# 1. SSH to Lambda instance
ssh ubuntu@<lambda-ip>

# 2. Clone installer
git clone https://github.com/rng-ops/he300-integration.git
cd he300-integration/installer

# 3. Run installer (Lambda has Docker + NVIDIA pre-installed)
./install.sh --quick-start

# Lambda-specific optimizations applied:
# - Uses Lambda's NVIDIA drivers (no reinstall)
# - Optimizes for A10 24GB VRAM
# - Configures for high-throughput inference
```

### Lambda Stack Detection

```bash
# Installer auto-detects Lambda Stack via:
if [ -f /etc/lambda-stack ]; then
    OS_TYPE="lambda"
    # Skip driver installation
    # Use Lambda's optimized CUDA
fi
```

### A10 24GB Optimization

```yaml
# Automatic configuration for A10:
ollama:
  gpu_layers: -1           # All layers on GPU
  num_parallel: 4          # 4 concurrent requests
  num_ctx: 8192            # Full context window
  
services:
  max_batch_size: 50       # Optimal for A10 memory
  queue_size: 200          # Deep queue for throughput
```

---

## Monitoring & Health Checks

### Health Check Endpoints

```bash
# CIRISNode health
curl http://localhost:8000/health

# EthicsEngine health  
curl http://localhost:8080/health

# Ollama health
curl http://localhost:11434/api/tags

# Full stack health
./scripts/health-check.sh
```

### Prometheus Metrics (Optional)

```yaml
# docker-compose.gpu.yml includes:
services:
  prometheus:
    image: prom/prometheus
    ports:
      - "9090:9090"
  
  grafana:
    image: grafana/grafana
    ports:
      - "3000:3000"
```

### Alerting

```bash
# Health check runs every 5 minutes via systemd timer
# Sends alert on failure:
./scripts/health-check.sh --alert-webhook https://hooks.slack.com/xxx
```

---

## Troubleshooting

### Common Issues

| Issue | Cause | Solution |
|-------|-------|----------|
| GPU not detected | NVIDIA drivers not loaded | `sudo nvidia-smi` to verify, reinstall drivers if needed |
| OOM during inference | Model too large | Use smaller quantization (Q4_K_M) or smaller model |
| Connection refused | Firewall blocking | Check `ufw status`, allow ports 8000, 8080 |
| WireGuard no handshake | Wrong keys or endpoint | Verify public keys match, check endpoint reachability |
| Slow inference | Wrong GPU layers config | Ensure `gpu_layers: -1` in Ollama config |

### Debug Commands

```bash
# Check GPU utilization
nvidia-smi -l 1

# Check Docker containers
docker ps -a
docker logs cirisnode
docker logs eee
docker logs ollama

# Check WireGuard status
sudo wg show

# Test internal connectivity
docker exec cirisnode curl http://eee:8080/health
docker exec eee curl http://ollama:11434/api/tags
```

---

## Summary

| Mode | Complexity | Security | Best For |
|------|------------|----------|----------|
| Local Dev | Low | N/A | Development, testing |
| Direct HTTPS | Medium | API key + TLS | Cloud GPU with public IP |
| WireGuard | Medium-High | Full encryption | Private GPU, NAT traversal |
| Tailscale | Low | Full encryption | Quick setup, team access |
| Self-hosted Runner | Low | Physical security | GPU machine is CI runner |

### Recommended Setup for Lambda A10

1. **Development**: Local mode with SSH tunnel
2. **CI/CD**: WireGuard VPN with GitHub self-hosted runner
3. **Production**: Direct HTTPS with IP allowlist

---

## HashiCorp Vault Integration

### Secrets Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           VAULT SERVER                                       │
│                    (Self-hosted or HCP Vault)                               │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Secret Paths                                 │   │
│  │                                                                      │   │
│  │  secret/he300/                                                       │   │
│  │  ├── api-keys/                                                       │   │
│  │  │   ├── cirisnode          # CIRISNode API key                     │   │
│  │  │   ├── ethicsengine       # EthicsEngine API key                  │   │
│  │  │   └── github-actions     # GitHub Actions token                  │   │
│  │  │                                                                   │   │
│  │  ├── database/                                                       │   │
│  │  │   ├── postgres           # PostgreSQL credentials                │   │
│  │  │   └── redis              # Redis password                        │   │
│  │  │                                                                   │   │
│  │  ├── wireguard/                                                      │   │
│  │  │   ├── gpu-host           # GPU host private key                  │   │
│  │  │   └── test-runner        # Test runner private key               │   │
│  │  │                                                                   │   │
│  │  ├── tls/                                                            │   │
│  │  │   ├── cert               # TLS certificate                       │   │
│  │  │   └── key                # TLS private key                       │   │
│  │  │                                                                   │   │
│  │  └── jwt/                                                            │   │
│  │      └── signing-key        # JWT signing secret                    │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                         Auth Methods                                 │   │
│  │                                                                      │   │
│  │  ├── AppRole    → CI/CD pipelines, installer scripts               │   │
│  │  ├── JWT/OIDC   → GitHub Actions (OIDC federation)                 │   │
│  │  ├── Token      → Manual operations, debugging                     │   │
│  │  └── Kubernetes → K8s deployments (future)                         │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Vault Configuration

```hcl
# terraform/vault/policies/he300-deploy.hcl
path "secret/data/he300/*" {
  capabilities = ["read", "list"]
}

path "secret/data/he300/api-keys/*" {
  capabilities = ["read"]
}

path "secret/data/he300/wireguard/*" {
  capabilities = ["read"]
}

# Dynamic database credentials
path "database/creds/he300-postgres" {
  capabilities = ["read"]
}
```

```hcl
# terraform/vault/auth/github-oidc.hcl
resource "vault_jwt_auth_backend" "github" {
  path               = "jwt-github"
  oidc_discovery_url = "https://token.actions.githubusercontent.com"
  bound_issuer       = "https://token.actions.githubusercontent.com"
}

resource "vault_jwt_auth_backend_role" "he300_deploy" {
  backend        = vault_jwt_auth_backend.github.path
  role_name      = "he300-deploy"
  token_policies = ["he300-deploy"]

  bound_claims = {
    repository = "rng-ops/he300-integration"
  }
  
  user_claim = "actor"
  role_type  = "jwt"
}
```

### GitHub Actions + Vault OIDC

```yaml
# .github/workflows/he300-benchmark.yml
jobs:
  benchmark:
    permissions:
      id-token: write  # Required for OIDC
      contents: read
    
    steps:
      - name: Import secrets from Vault
        uses: hashicorp/vault-action@v2
        with:
          url: ${{ secrets.VAULT_ADDR }}
          method: jwt
          role: he300-deploy
          jwtGithubAudience: sigstore
          secrets: |
            secret/data/he300/api-keys/cirisnode api_key | CIRISNODE_API_KEY ;
            secret/data/he300/wireguard/test-runner private_key | WG_PRIVATE_KEY ;
            secret/data/he300/jwt/signing-key secret | JWT_SECRET
      
      - name: Run benchmark
        env:
          CIRISNODE_API_KEY: ${{ env.CIRISNODE_API_KEY }}
        run: ./scripts/run_he300_benchmark.sh
```

### Installer Vault Integration

```bash
# install.sh vault integration
vault_fetch_secrets() {
    local vault_addr="$1"
    local vault_token="$2"
    
    # Fetch all required secrets
    export CIRISNODE_API_KEY=$(vault kv get -field=api_key secret/he300/api-keys/cirisnode)
    export POSTGRES_PASSWORD=$(vault kv get -field=password secret/he300/database/postgres)
    export REDIS_PASSWORD=$(vault kv get -field=password secret/he300/database/redis)
    export JWT_SECRET=$(vault kv get -field=secret secret/he300/jwt/signing-key)
    
    # WireGuard keys
    export WG_PRIVATE_KEY=$(vault kv get -field=private_key secret/he300/wireguard/gpu-host)
}

# AppRole authentication for automated deployments
vault_login_approle() {
    local role_id="$1"
    local secret_id="$2"
    
    VAULT_TOKEN=$(vault write -field=token auth/approle/login \
        role_id="$role_id" \
        secret_id="$secret_id")
    export VAULT_TOKEN
}
```

### Secret Rotation

```yaml
# config/vault/secret-rotation.yaml
rotation:
  api_keys:
    interval: 30d
    notify:
      - slack: "#he300-ops"
      - email: "ops@example.com"
  
  database:
    postgres:
      interval: 7d
      method: dynamic  # Use Vault dynamic credentials
    redis:
      interval: 30d
      method: static
  
  wireguard:
    interval: 90d
    method: manual  # Requires coordinated rotation
  
  jwt:
    signing_key:
      interval: 90d
      grace_period: 24h  # Both keys valid during transition
```

---

## Terraform Infrastructure

### Directory Structure

```
terraform/
├── modules/
│   ├── gpu-instance/           # GPU instance provisioning
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   ├── outputs.tf
│   │   └── userdata.sh.tpl
│   │
│   ├── docker-registry/        # Private registry for images
│   │   ├── main.tf
│   │   └── variables.tf
│   │
│   ├── wireguard/              # WireGuard VPN setup
│   │   ├── main.tf
│   │   ├── variables.tf
│   │   └── wg0.conf.tpl
│   │
│   ├── vault/                  # Vault cluster setup
│   │   ├── main.tf
│   │   ├── policies/
│   │   └── auth/
│   │
│   └── results-dashboard/      # Web dashboard infrastructure
│       ├── main.tf
│       ├── variables.tf
│       └── outputs.tf
│
├── environments/
│   ├── dev/
│   │   ├── main.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   │
│   ├── staging/
│   │   ├── main.tf
│   │   ├── terraform.tfvars
│   │   └── backend.tf
│   │
│   └── prod/
│       ├── main.tf
│       ├── terraform.tfvars
│       └── backend.tf
│
├── packer/                     # Image building
│   ├── he300-gpu.pkr.hcl       # GPU host image
│   ├── he300-runner.pkr.hcl    # Test runner image
│   └── scripts/
│       ├── install-docker.sh
│       ├── install-nvidia.sh
│       └── install-he300.sh
│
└── scripts/
    ├── apply.sh
    ├── plan.sh
    └── destroy.sh
```

### GPU Instance Module

```hcl
# terraform/modules/gpu-instance/main.tf

variable "instance_type" {
  description = "GPU instance type"
  type        = string
  default     = "gpu_1x_a10"  # Lambda Labs instance type
}

variable "region" {
  description = "Deployment region"
  type        = string
  default     = "us-west-1"
}

variable "os_image" {
  description = "Operating system image"
  type        = string
  default     = "ubuntu-22.04-cuda-12.1"
}

variable "vault_addr" {
  description = "Vault server address"
  type        = string
}

# Lambda Labs provider (or use generic cloud provider)
resource "lambdalabs_instance" "gpu_host" {
  name          = "he300-gpu-${var.environment}"
  instance_type = var.instance_type
  region        = var.region
  
  ssh_key_names = [var.ssh_key_name]
  
  # Custom image with pre-installed stack
  image_id = var.custom_image_id != "" ? var.custom_image_id : null
  
  tags = {
    Environment = var.environment
    Purpose     = "he300-benchmark"
    ManagedBy   = "terraform"
  }
}

# Alternative: AWS GPU instance
resource "aws_instance" "gpu_host" {
  count = var.cloud_provider == "aws" ? 1 : 0
  
  ami           = data.aws_ami.gpu_base.id
  instance_type = "g5.xlarge"  # NVIDIA A10G
  
  vpc_security_group_ids = [aws_security_group.he300.id]
  subnet_id              = var.subnet_id
  
  user_data = templatefile("${path.module}/userdata.sh.tpl", {
    vault_addr     = var.vault_addr
    vault_role_id  = var.vault_role_id
    environment    = var.environment
  })
  
  root_block_device {
    volume_size = 200
    volume_type = "gp3"
  }
  
  tags = {
    Name        = "he300-gpu-${var.environment}"
    Environment = var.environment
  }
}

# Security group
resource "aws_security_group" "he300" {
  name        = "he300-${var.environment}"
  description = "HE-300 benchmark stack"
  vpc_id      = var.vpc_id
  
  # SSH
  ingress {
    from_port   = 22
    to_port     = 22
    protocol    = "tcp"
    cidr_blocks = var.admin_cidrs
  }
  
  # CIRISNode API
  ingress {
    from_port   = 8000
    to_port     = 8000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }
  
  # EthicsEngine API
  ingress {
    from_port   = 8080
    to_port     = 8080
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }
  
  # WireGuard
  ingress {
    from_port   = 51820
    to_port     = 51820
    protocol    = "udp"
    cidr_blocks = ["0.0.0.0/0"]
  }
  
  # Results dashboard
  ingress {
    from_port   = 3000
    to_port     = 3000
    protocol    = "tcp"
    cidr_blocks = var.allowed_cidrs
  }
  
  egress {
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }
}

output "instance_ip" {
  value = coalesce(
    try(lambdalabs_instance.gpu_host.ip, null),
    try(aws_instance.gpu_host[0].public_ip, null)
  )
}

output "instance_id" {
  value = coalesce(
    try(lambdalabs_instance.gpu_host.id, null),
    try(aws_instance.gpu_host[0].id, null)
  )
}
```

### Packer Image Build

```hcl
# terraform/packer/he300-gpu.pkr.hcl

packer {
  required_plugins {
    amazon = {
      version = ">= 1.2.0"
      source  = "github.com/hashicorp/amazon"
    }
  }
}

variable "vault_addr" {
  type    = string
  default = env("VAULT_ADDR")
}

variable "base_ami" {
  type    = string
  default = "ami-0c55b159cbfafe1f0"  # Ubuntu 22.04 base
}

source "amazon-ebs" "he300_gpu" {
  ami_name      = "he300-gpu-{{timestamp}}"
  instance_type = "g5.xlarge"
  region        = "us-west-2"
  source_ami    = var.base_ami
  ssh_username  = "ubuntu"
  
  ami_description = "HE-300 Benchmark GPU Image with CIRISNode + EthicsEngine"
  
  tags = {
    Name        = "he300-gpu"
    BuildTime   = "{{timestamp}}"
    Environment = "base"
  }
  
  launch_block_device_mappings {
    device_name           = "/dev/sda1"
    volume_size           = 200
    volume_type           = "gp3"
    delete_on_termination = true
  }
}

build {
  sources = ["source.amazon-ebs.he300_gpu"]
  
  # Install Docker
  provisioner "shell" {
    script = "${path.root}/scripts/install-docker.sh"
  }
  
  # Install NVIDIA drivers + container toolkit
  provisioner "shell" {
    script = "${path.root}/scripts/install-nvidia.sh"
  }
  
  # Install Vault CLI
  provisioner "shell" {
    inline = [
      "curl -fsSL https://apt.releases.hashicorp.com/gpg | sudo apt-key add -",
      "sudo apt-add-repository \"deb [arch=amd64] https://apt.releases.hashicorp.com $(lsb_release -cs) main\"",
      "sudo apt-get update && sudo apt-get install -y vault"
    ]
  }
  
  # Pre-pull Docker images
  provisioner "shell" {
    inline = [
      "sudo docker pull ghcr.io/rng-ops/cirisnode:latest",
      "sudo docker pull ghcr.io/rng-ops/ethicsengine:latest",
      "sudo docker pull ollama/ollama:latest",
      "sudo docker pull postgres:15-alpine",
      "sudo docker pull redis:7-alpine"
    ]
  }
  
  # Copy installer scripts
  provisioner "file" {
    source      = "../../installer/"
    destination = "/opt/he300/"
  }
  
  # Pre-configure systemd services
  provisioner "shell" {
    script = "${path.root}/scripts/install-he300.sh"
  }
}
```

### Terraform Workflow

```yaml
# .github/workflows/terraform.yml
name: Terraform Infrastructure

on:
  push:
    paths:
      - 'terraform/**'
  pull_request:
    paths:
      - 'terraform/**'
  workflow_dispatch:
    inputs:
      action:
        description: 'Terraform action'
        required: true
        default: 'plan'
        type: choice
        options:
          - plan
          - apply
          - destroy
      environment:
        description: 'Target environment'
        required: true
        default: 'dev'
        type: choice
        options:
          - dev
          - staging
          - prod

permissions:
  id-token: write
  contents: read
  pull-requests: write

jobs:
  terraform:
    runs-on: ubuntu-latest
    environment: ${{ inputs.environment || 'dev' }}
    
    defaults:
      run:
        working-directory: terraform/environments/${{ inputs.environment || 'dev' }}
    
    steps:
      - uses: actions/checkout@v4
      
      - name: Configure AWS credentials
        uses: aws-actions/configure-aws-credentials@v4
        with:
          role-to-assume: ${{ secrets.AWS_ROLE_ARN }}
          aws-region: us-west-2
      
      - name: Import Vault secrets
        uses: hashicorp/vault-action@v2
        with:
          url: ${{ secrets.VAULT_ADDR }}
          method: jwt
          role: terraform-deploy
          secrets: |
            secret/data/he300/terraform/aws access_key | AWS_ACCESS_KEY_ID ;
            secret/data/he300/terraform/aws secret_key | AWS_SECRET_ACCESS_KEY
      
      - uses: hashicorp/setup-terraform@v3
        with:
          terraform_version: 1.6.0
      
      - name: Terraform Init
        run: terraform init
      
      - name: Terraform Plan
        if: inputs.action == 'plan' || github.event_name == 'pull_request'
        run: terraform plan -out=tfplan
      
      - name: Terraform Apply
        if: inputs.action == 'apply'
        run: terraform apply -auto-approve tfplan
      
      - name: Terraform Destroy
        if: inputs.action == 'destroy'
        run: terraform destroy -auto-approve
```

---

## Results Dashboard Web Interface

### Dashboard Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        RESULTS DASHBOARD                                     │
│                     (Next.js + React + Tailwind)                            │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │                          /dashboard                                  │   │
│  │  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐               │   │
│  │  │  Benchmark   │  │   Model      │  │  Historical  │               │   │
│  │  │  Overview    │  │  Comparison  │  │   Trends     │               │   │
│  │  └──────────────┘  └──────────────┘  └──────────────┘               │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                    Latest Results                              │  │   │
│  │  │  ┌─────────────────────────────────────────────────────────┐  │  │   │
│  │  │  │  Run #42 | llama-3.2-8B | 2026-01-03 14:32             │  │  │   │
│  │  │  │  ├── Commonsense:  92.4% (46/50)                       │  │  │   │
│  │  │  │  ├── Deontology:   88.0% (44/50)                       │  │  │   │
│  │  │  │  ├── Justice:      90.0% (45/50)                       │  │  │   │
│  │  │  │  ├── Virtue:       86.0% (43/50)                       │  │  │   │
│  │  │  │  └── Mixed:        85.0% (51/60)                       │  │  │   │
│  │  │  │  ─────────────────────────────────────────────────────  │  │  │   │
│  │  │  │  Overall: 88.1% (229/260) | Duration: 12m 34s          │  │  │   │
│  │  │  └─────────────────────────────────────────────────────────┘  │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  │  ┌───────────────────────────────────────────────────────────────┐  │   │
│  │  │                   Category Breakdown                          │  │   │
│  │  │   [====== Radar Chart ======]  [===== Bar Chart =====]       │  │   │
│  │  └───────────────────────────────────────────────────────────────┘  │   │
│  │                                                                      │   │
│  └──────────────────────────────────────────────────────────────────────┘   │
│                                                                             │
│  API: /api/results/*                                                        │
│  Data: PostgreSQL + S3 (artifacts)                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Dashboard Directory Structure

```
dashboard/
├── package.json
├── next.config.js
├── tailwind.config.js
├── tsconfig.json
├── Dockerfile
├── docker-compose.yml
│
├── prisma/
│   └── schema.prisma           # Database schema
│
├── src/
│   ├── app/
│   │   ├── layout.tsx
│   │   ├── page.tsx            # Landing page
│   │   ├── dashboard/
│   │   │   ├── page.tsx        # Main dashboard
│   │   │   ├── runs/
│   │   │   │   ├── page.tsx    # Run history
│   │   │   │   └── [id]/
│   │   │   │       └── page.tsx # Run details
│   │   │   ├── compare/
│   │   │   │   └── page.tsx    # Model comparison
│   │   │   └── trends/
│   │   │       └── page.tsx    # Historical trends
│   │   │
│   │   └── api/
│   │       ├── results/
│   │       │   ├── route.ts    # GET all results
│   │       │   └── [id]/
│   │       │       └── route.ts # GET single result
│   │       ├── runs/
│   │       │   ├── route.ts    # List runs
│   │       │   └── [id]/
│   │       │       ├── route.ts
│   │       │       └── artifacts/
│   │       │           └── route.ts
│   │       ├── models/
│   │       │   └── route.ts    # Model performance
│   │       └── webhook/
│   │           └── route.ts    # CI/CD webhook
│   │
│   ├── components/
│   │   ├── ui/                 # Shadcn/ui components
│   │   ├── charts/
│   │   │   ├── RadarChart.tsx
│   │   │   ├── BarChart.tsx
│   │   │   ├── LineChart.tsx
│   │   │   └── Heatmap.tsx
│   │   ├── dashboard/
│   │   │   ├── RunCard.tsx
│   │   │   ├── CategoryBreakdown.tsx
│   │   │   ├── ModelComparison.tsx
│   │   │   └── TrendGraph.tsx
│   │   └── layout/
│   │       ├── Header.tsx
│   │       ├── Sidebar.tsx
│   │       └── Footer.tsx
│   │
│   ├── lib/
│   │   ├── prisma.ts           # Prisma client
│   │   ├── s3.ts               # S3 artifact storage
│   │   └── utils.ts
│   │
│   └── types/
│       └── results.ts
│
└── public/
    └── ...
```

### Database Schema

```prisma
// dashboard/prisma/schema.prisma

generator client {
  provider = "prisma-client-js"
}

datasource db {
  provider = "postgresql"
  url      = env("DATABASE_URL")
}

model BenchmarkRun {
  id            String   @id @default(cuid())
  createdAt     DateTime @default(now())
  completedAt   DateTime?
  
  // Configuration
  model         String   // llama-3.2-8B-instruct
  quantization  String   // Q4_K_M
  sampleSize    Int      // 260
  seed          Int      // 42
  
  // Environment
  gpuType       String?  // a10, a100, m1
  gpuMemory     Int?     // GB
  runnerType    String   // github-actions, local, jenkins
  environment   String   // dev, staging, prod
  
  // Status
  status        RunStatus @default(PENDING)
  errorMessage  String?
  
  // Results
  results       CategoryResult[]
  artifacts     Artifact[]
  
  // Metrics
  duration      Int?     // seconds
  tokensPerSec  Float?
  
  // Git info
  commitSha     String?
  branch        String?
  prNumber      Int?
  
  @@index([createdAt])
  @@index([model])
  @@index([status])
}

enum RunStatus {
  PENDING
  RUNNING
  COMPLETED
  FAILED
  CANCELLED
}

model CategoryResult {
  id            String   @id @default(cuid())
  runId         String
  run           BenchmarkRun @relation(fields: [runId], references: [id], onDelete: Cascade)
  
  category      String   // commonsense, deontology, justice, virtue, mixed
  total         Int      // 50 or 60
  correct       Int
  accuracy      Float    // 0.0 - 1.0
  
  // Detailed metrics
  avgLatency    Float?   // ms per scenario
  avgTokens     Int?     // tokens per response
  
  // Per-scenario results stored as JSON
  scenarios     Json?
  
  @@unique([runId, category])
}

model Artifact {
  id            String   @id @default(cuid())
  runId         String
  run           BenchmarkRun @relation(fields: [runId], references: [id], onDelete: Cascade)
  
  type          ArtifactType
  filename      String
  s3Key         String
  size          Int      // bytes
  contentType   String
  
  createdAt     DateTime @default(now())
}

enum ArtifactType {
  FULL_RESULTS    // Complete JSON results
  SUMMARY         // Summary report
  LOGS            // Execution logs
  TRACE           // OpenTelemetry trace
}

model Model {
  id            String   @id @default(cuid())
  name          String   @unique  // llama-3.2-8B-instruct
  displayName   String   // Llama 3.2 8B Instruct
  provider      String   // ollama, openai, anthropic
  
  // Aggregate stats (updated after each run)
  totalRuns     Int      @default(0)
  avgAccuracy   Float?
  bestAccuracy  Float?
  lastRunAt     DateTime?
  
  @@index([name])
}
```

### Dashboard Components

```tsx
// dashboard/src/components/dashboard/RunCard.tsx
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { formatDistanceToNow } from "date-fns"
import type { BenchmarkRun } from "@prisma/client"

interface RunCardProps {
  run: BenchmarkRun & {
    results: { category: string; accuracy: number; correct: number; total: number }[]
  }
}

export function RunCard({ run }: RunCardProps) {
  const overallAccuracy = run.results.reduce(
    (acc, r) => acc + r.correct, 0
  ) / run.results.reduce((acc, r) => acc + r.total, 0)
  
  const statusColors = {
    COMPLETED: "bg-green-500",
    RUNNING: "bg-blue-500",
    FAILED: "bg-red-500",
    PENDING: "bg-yellow-500",
    CANCELLED: "bg-gray-500",
  }
  
  return (
    <Card className="hover:shadow-lg transition-shadow">
      <CardHeader className="flex flex-row items-center justify-between">
        <div>
          <CardTitle className="text-lg">{run.model}</CardTitle>
          <p className="text-sm text-muted-foreground">
            {formatDistanceToNow(run.createdAt, { addSuffix: true })}
          </p>
        </div>
        <Badge className={statusColors[run.status]}>
          {run.status}
        </Badge>
      </CardHeader>
      <CardContent>
        <div className="grid grid-cols-5 gap-2 mb-4">
          {run.results.map((result) => (
            <div key={result.category} className="text-center">
              <div className="text-xs text-muted-foreground capitalize">
                {result.category}
              </div>
              <div className="text-lg font-semibold">
                {(result.accuracy * 100).toFixed(1)}%
              </div>
              <div className="text-xs">
                {result.correct}/{result.total}
              </div>
            </div>
          ))}
        </div>
        
        <div className="flex justify-between items-center border-t pt-4">
          <div>
            <span className="text-2xl font-bold">
              {(overallAccuracy * 100).toFixed(1)}%
            </span>
            <span className="text-muted-foreground ml-2">overall</span>
          </div>
          {run.duration && (
            <div className="text-sm text-muted-foreground">
              {Math.floor(run.duration / 60)}m {run.duration % 60}s
            </div>
          )}
        </div>
      </CardContent>
    </Card>
  )
}
```

```tsx
// dashboard/src/components/charts/RadarChart.tsx
"use client"

import {
  Radar,
  RadarChart as RechartsRadar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  ResponsiveContainer,
  Legend,
} from "recharts"

interface RadarChartProps {
  data: {
    category: string
    accuracy: number
    baseline?: number
  }[]
  showBaseline?: boolean
}

export function RadarChart({ data, showBaseline = true }: RadarChartProps) {
  const chartData = data.map((d) => ({
    category: d.category.charAt(0).toUpperCase() + d.category.slice(1),
    accuracy: d.accuracy * 100,
    baseline: d.baseline ? d.baseline * 100 : 80,
  }))
  
  return (
    <ResponsiveContainer width="100%" height={400}>
      <RechartsRadar cx="50%" cy="50%" outerRadius="80%" data={chartData}>
        <PolarGrid />
        <PolarAngleAxis dataKey="category" />
        <PolarRadiusAxis angle={90} domain={[0, 100]} />
        <Radar
          name="Accuracy"
          dataKey="accuracy"
          stroke="#8884d8"
          fill="#8884d8"
          fillOpacity={0.6}
        />
        {showBaseline && (
          <Radar
            name="Baseline"
            dataKey="baseline"
            stroke="#82ca9d"
            fill="#82ca9d"
            fillOpacity={0.3}
          />
        )}
        <Legend />
      </RechartsRadar>
    </ResponsiveContainer>
  )
}
```

### API Routes

```ts
// dashboard/src/app/api/webhook/route.ts
import { NextRequest, NextResponse } from "next/server"
import { prisma } from "@/lib/prisma"
import { z } from "zod"

const webhookSchema = z.object({
  run_id: z.string(),
  status: z.enum(["PENDING", "RUNNING", "COMPLETED", "FAILED", "CANCELLED"]),
  model: z.string(),
  quantization: z.string().optional(),
  sample_size: z.number(),
  results: z.array(z.object({
    category: z.string(),
    total: z.number(),
    correct: z.number(),
    accuracy: z.number(),
    scenarios: z.any().optional(),
  })).optional(),
  duration: z.number().optional(),
  error_message: z.string().optional(),
  commit_sha: z.string().optional(),
  branch: z.string().optional(),
  gpu_type: z.string().optional(),
})

export async function POST(request: NextRequest) {
  // Verify webhook signature
  const signature = request.headers.get("x-webhook-signature")
  if (!verifySignature(signature, await request.text())) {
    return NextResponse.json({ error: "Invalid signature" }, { status: 401 })
  }
  
  const body = await request.json()
  const data = webhookSchema.parse(body)
  
  // Upsert run
  const run = await prisma.benchmarkRun.upsert({
    where: { id: data.run_id },
    update: {
      status: data.status,
      duration: data.duration,
      errorMessage: data.error_message,
      completedAt: data.status === "COMPLETED" ? new Date() : undefined,
    },
    create: {
      id: data.run_id,
      model: data.model,
      quantization: data.quantization || "Q4_K_M",
      sampleSize: data.sample_size,
      seed: 42,
      status: data.status,
      gpuType: data.gpu_type,
      commitSha: data.commit_sha,
      branch: data.branch,
    },
  })
  
  // Insert results if completed
  if (data.status === "COMPLETED" && data.results) {
    await prisma.categoryResult.createMany({
      data: data.results.map((r) => ({
        runId: run.id,
        category: r.category,
        total: r.total,
        correct: r.correct,
        accuracy: r.accuracy,
        scenarios: r.scenarios,
      })),
      skipDuplicates: true,
    })
  }
  
  return NextResponse.json({ success: true, run_id: run.id })
}

function verifySignature(signature: string | null, body: string): boolean {
  if (!signature) return false
  const expected = crypto
    .createHmac("sha256", process.env.WEBHOOK_SECRET!)
    .update(body)
    .digest("hex")
  return crypto.timingSafeEqual(
    Buffer.from(signature),
    Buffer.from(`sha256=${expected}`)
  )
}
```

```ts
// dashboard/src/app/api/results/route.ts
import { NextRequest, NextResponse } from "next/server"
import { prisma } from "@/lib/prisma"

export async function GET(request: NextRequest) {
  const { searchParams } = new URL(request.url)
  
  const model = searchParams.get("model")
  const status = searchParams.get("status")
  const limit = parseInt(searchParams.get("limit") || "20")
  const offset = parseInt(searchParams.get("offset") || "0")
  
  const where = {
    ...(model && { model }),
    ...(status && { status: status as any }),
  }
  
  const [runs, total] = await Promise.all([
    prisma.benchmarkRun.findMany({
      where,
      include: {
        results: true,
      },
      orderBy: { createdAt: "desc" },
      take: limit,
      skip: offset,
    }),
    prisma.benchmarkRun.count({ where }),
  ])
  
  return NextResponse.json({
    runs,
    pagination: {
      total,
      limit,
      offset,
      hasMore: offset + runs.length < total,
    },
  })
}
```

### Dashboard Docker Setup

```yaml
# dashboard/docker-compose.yml
version: '3.8'

services:
  dashboard:
    build: .
    ports:
      - "3000:3000"
    environment:
      - DATABASE_URL=postgresql://postgres:postgres@db:5432/he300_dashboard
      - NEXTAUTH_SECRET=${NEXTAUTH_SECRET}
      - WEBHOOK_SECRET=${WEBHOOK_SECRET}
      - S3_BUCKET=${S3_BUCKET}
      - AWS_REGION=${AWS_REGION}
    depends_on:
      - db
    networks:
      - he300-net

  db:
    image: postgres:15-alpine
    environment:
      POSTGRES_USER: postgres
      POSTGRES_PASSWORD: postgres
      POSTGRES_DB: he300_dashboard
    volumes:
      - dashboard_db:/var/lib/postgresql/data
    networks:
      - he300-net

networks:
  he300-net:
    external: true

volumes:
  dashboard_db:
```

```dockerfile
# dashboard/Dockerfile
FROM node:20-alpine AS base

FROM base AS deps
WORKDIR /app
COPY package*.json ./
RUN npm ci

FROM base AS builder
WORKDIR /app
COPY --from=deps /app/node_modules ./node_modules
COPY . .
RUN npx prisma generate
RUN npm run build

FROM base AS runner
WORKDIR /app
ENV NODE_ENV=production

RUN addgroup --system --gid 1001 nodejs
RUN adduser --system --uid 1001 nextjs

COPY --from=builder /app/public ./public
COPY --from=builder /app/.next/standalone ./
COPY --from=builder /app/.next/static ./.next/static
COPY --from=builder /app/prisma ./prisma

USER nextjs

EXPOSE 3000
ENV PORT 3000

CMD ["node", "server.js"]
```

### CI/CD Webhook Integration

```bash
# scripts/run_he300_benchmark.sh - add webhook calls

send_webhook() {
    local status="$1"
    local results_file="$2"
    
    curl -X POST "${DASHBOARD_WEBHOOK_URL}/api/webhook" \
        -H "Content-Type: application/json" \
        -H "x-webhook-signature: sha256=$(echo -n "$payload" | openssl dgst -sha256 -hmac "$WEBHOOK_SECRET" | cut -d' ' -f2)" \
        -d @- <<EOF
{
    "run_id": "${RUN_ID}",
    "status": "${status}",
    "model": "${MODEL}",
    "quantization": "${QUANTIZATION}",
    "sample_size": ${SAMPLE_SIZE},
    "results": $(cat "$results_file" 2>/dev/null || echo "null"),
    "duration": ${DURATION:-0},
    "commit_sha": "${GITHUB_SHA:-}",
    "branch": "${GITHUB_REF_NAME:-}",
    "gpu_type": "${GPU_TYPE:-}"
}
EOF
}

# At start of benchmark
send_webhook "RUNNING" ""

# On completion
send_webhook "COMPLETED" "$RESULTS_FILE"

# On failure
send_webhook "FAILED" ""
```

---

## Complete Installer Package Structure

```
installer/
├── install.sh                      # Main installer
├── uninstall.sh                    # Removal script
├── VERSION
│
├── config/
│   ├── install.yaml                # Installation config
│   ├── models.yaml                 # Model weights
│   ├── vault/
│   │   ├── policies/               # Vault policies
│   │   └── secrets.yaml.template   # Secret structure template
│   └── wireguard/
│       ├── wg0.conf.template
│       └── generate-keys.sh
│
├── docker/
│   ├── docker-compose.gpu.yml
│   ├── docker-compose.cpu.yml
│   ├── docker-compose.dashboard.yml
│   └── .env.template
│
├── terraform/
│   ├── modules/                    # Reusable modules
│   ├── environments/               # Per-environment configs
│   └── packer/                     # Image building
│
├── dashboard/                      # Results web interface
│   ├── src/
│   ├── prisma/
│   └── Dockerfile
│
├── systemd/
│   ├── he300-stack.service
│   └── he300-health.timer
│
├── scripts/
│   ├── setup-nvidia.sh
│   ├── setup-ollama.sh
│   ├── setup-vault.sh
│   ├── setup-wireguard.sh
│   ├── pull-models.sh
│   ├── health-check.sh
│   └── benchmark-test.sh
│
└── docs/
    ├── INSTALL.md
    ├── VAULT.md
    ├── TERRAFORM.md
    └── DASHBOARD.md
```

---

### Next Steps

1. **Build installer package** → `scripts/build-installer.sh`
2. **Create systemd units** → `systemd/he300-stack.service`
3. **WireGuard automation** → `config/wireguard/`
4. **Model management CLI** → `scripts/model-manager.sh`
5. **Monitoring stack** → `docker/docker-compose.monitoring.yml`
6. **Vault setup scripts** → `scripts/setup-vault.sh`
7. **Terraform modules** → `terraform/modules/`
8. **Dashboard implementation** → `dashboard/`
9. **Packer image builds** → `terraform/packer/`
10. **GitHub Actions for Terraform** → `.github/workflows/terraform.yml`
