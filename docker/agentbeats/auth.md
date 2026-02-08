# CIRISBench Authentication

CIRISBench operates in two deployment modes with different auth models,
controlled by the `AGENTBEATS_MODE` environment variable.

## Deployment Modes

| Mode | `AGENTBEATS_MODE` | Auth at CIRISNode | Auth at Engine | Quota |
|------|-------------------|-------------------|----------------|-------|
| **Standalone** | `true` | Bypassed | API key via `ENGINE_API_KEYS` | None |
| **Managed** | unset / `""` | JWT + API key | Service JWT from CIRISNode | Tiered |

### Standalone Mode (AgentBeats / Hackathon)

Used when running the unified Docker image (`cirisbench:agentbeats`).
The Dockerfile sets `AGENTBEATS_MODE=true`. The entrypoint automatically
wires `AGENTBEATS_API_KEY` to `ENGINE_API_KEYS` so the Engine validates
incoming API keys from the AgentBeats platform.

**Request flow:**
```
AgentBeats Platform
  │  Authorization: Bearer ${AGENTBEATS_API_KEY}
  ▼
CIRISNode (:8000)           ← auth bypassed (AGENTBEATS_MODE=true)
  │  forwards Authorization header
  ▼
EthicsEngine (:8080)        ← validates Bearer token against ENGINE_API_KEYS
  │
  ▼
Purple Agent (under test)
```

### Managed Mode (ethicsengine.org)

Used for the hosted platform. CIRISNode enforces JWT auth and tiered
usage quotas before proxying to the Engine.

**Request flow:**
```
Frontend (ethicsengine.org)
  │  Authorization: Bearer ${JWT}
  ▼
CIRISNode (:443)            ← JWT validated via validate_a2a_auth
  │                         ← quota checked (Community/Pro/Enterprise)
  │  Authorization: Bearer ${SERVICE_JWT}
  ▼
EthicsEngine (:8080)        ← validates service JWT
  │
  ▼
Purple Agent (under test)
```

**Subscription tiers:**

| Tier | Price | Limit | Window |
|------|-------|-------|--------|
| Community | Free | 1 eval | Per week |
| Pro | $399/mo | 100 evals | Per month |
| Enterprise | Custom | Unlimited | - |

Tier data is stored in the `tenant_tiers` PostgreSQL table, written by
Stripe webhook handlers when subscriptions change.

## Environment Variables

### Provided by AgentBeats Platform (Standalone)

| Variable | Description | Provided By |
|----------|-------------|-------------|
| `AGENTBEATS_API_KEY` | API key for incoming requests from AgentBeats | AgentBeats Platform |
| `AGENTBEATS_WEBHOOK_SECRET` | Secret for signing result callbacks | AgentBeats Platform |
| `AGENTBEATS_CALLBACK_URL` | URL to send benchmark results | AgentBeats Platform |
| `AGENTBEATS_RUN_ID` | Unique identifier for this benchmark run | AgentBeats Platform |

### Internal (Set by Dockerfile / Entrypoint)

| Variable | Default | Description |
|----------|---------|-------------|
| `AGENTBEATS_MODE` | `""` (managed) | `true` for standalone, `""` for managed |
| `ENGINE_API_KEYS` | (wired from `AGENTBEATS_API_KEY`) | Comma-separated valid API keys for Engine |
| `AUTH_ENABLED` | `true` | Set `false` to disable all Engine auth (dev only) |
| `JWT_SECRET` | - | Shared secret for HS256 JWT (must match CIRISNode + Engine) |

## Authentication Flows

### 1. Incoming Requests (AgentBeats -> CIRISBench)

AgentBeats sends requests with an API key:

```http
POST /api/v1/agentbeats/run HTTP/1.1
Host: cirisbench:8000
Authorization: Bearer ${AGENTBEATS_API_KEY}
X-AgentBeats-Run-ID: ${AGENTBEATS_RUN_ID}
Content-Type: application/json

{
  "agent_url": "https://agent-to-test.example.com/a2a",
  "sample_size": 300,
  "concurrency": 50,
  "protocol": "a2a"
}
```

In standalone mode, CIRISNode passes this through to the Engine.
The Engine validates the Bearer token against `ENGINE_API_KEYS`.

### 2. Outgoing Callbacks (CIRISBench -> AgentBeats)

CIRISBench sends results back with HMAC signature:

```http
POST ${AGENTBEATS_CALLBACK_URL} HTTP/1.1
Content-Type: application/json
X-AgentBeats-Signature: sha256=${HMAC_SIGNATURE}
X-AgentBeats-Run-ID: ${AGENTBEATS_RUN_ID}

{
  "status": "completed",
  "results": {
    "accuracy": 0.847,
    "categories": {...}
  }
}
```

Signature is computed as:
```
HMAC-SHA256(AGENTBEATS_WEBHOOK_SECRET, request_body)
```

### 3. Purple Agent Calls (CIRISBench -> Agent Being Tested)

CIRISBench calls the purple agent using standard A2A:

```http
POST ${purple_agent_url} HTTP/1.1
Content-Type: application/json
X-CIRISBench-Scenario-ID: commonsense_001

{
  "jsonrpc": "2.0",
  "method": "evaluate",
  "params": {
    "scenario": "You find a wallet with $500...",
    "options": ["Keep it", "Turn it in", "Take some"]
  },
  "id": 1
}
```

## No Auth Needed For

- `GET /health` - Health check (public)
- `GET /he300/catalog` - List scenarios (public)
- `GET /info` - Service info (public)
- `GET /api/v1/agentbeats/status` - Runner status (public)

## Local Development

For local testing without AgentBeats:

```bash
# Generate test credentials
export AGENTBEATS_API_KEY=$(openssl rand -hex 32)

# Run container (standalone mode, auth via API key)
docker run -p 8000:8000 -p 8080:8080 \
  -e AGENTBEATS_API_KEY=$AGENTBEATS_API_KEY \
  ghcr.io/cirisai/cirisbench:agentbeats

# Test authenticated endpoint via CIRISNode proxy
curl -X POST http://localhost:8000/api/v1/agentbeats/run \
  -H "Authorization: Bearer $AGENTBEATS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"agent_url":"http://host.docker.internal:9000/a2a","sample_size":50}'

# Or hit Engine directly (also accepts the API key)
curl -X POST http://localhost:8080/he300/agentbeats/run \
  -H "Authorization: Bearer $AGENTBEATS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"agent_url":"http://host.docker.internal:9000/a2a","sample_size":50}'
```

## Engine Auth Details

The Engine's `require_auth` dependency accepts credentials in this order:

1. `Authorization: Bearer <token>` — tries JWT decode first, then checks against `ENGINE_API_KEYS`
2. `X-API-Key: <key>` — checks against `ENGINE_API_KEYS`
3. If `AUTH_ENABLED=false` — returns `"anonymous"` (dev mode only)

This means `AGENTBEATS_API_KEY` works as a Bearer token without needing
to be a valid JWT, because the Engine falls back to API key validation.
