# CIRISBench Authentication for AgentBeats

## Environment Variables (Provided by AgentBeats)

When AgentBeats runs the CIRISBench container, it injects these credentials:

| Variable | Description | Provided By |
|----------|-------------|-------------|
| `AGENTBEATS_API_KEY` | API key for incoming requests from AgentBeats | AgentBeats Platform |
| `AGENTBEATS_WEBHOOK_SECRET` | Secret for signing result callbacks | AgentBeats Platform |
| `AGENTBEATS_CALLBACK_URL` | URL to send benchmark results | AgentBeats Platform |
| `AGENTBEATS_RUN_ID` | Unique identifier for this benchmark run | AgentBeats Platform |

## Authentication Flow

### 1. Incoming Requests (AgentBeats → CIRISBench)

AgentBeats sends requests with an API key:

```http
POST /a2a HTTP/1.1
Host: cirisbench:8000
Authorization: Bearer ${AGENTBEATS_API_KEY}
X-AgentBeats-Run-ID: ${AGENTBEATS_RUN_ID}
Content-Type: application/json

{
  "jsonrpc": "2.0",
  "method": "benchmark.run",
  "params": {
    "purple_agent_url": "https://agent-to-test.example.com/a2a",
    "sample_size": 300
  },
  "id": 1
}
```

CIRISBench validates:
- `Authorization` header matches `AGENTBEATS_API_KEY`
- Request is well-formed

### 2. Outgoing Callbacks (CIRISBench → AgentBeats)

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

### 3. Purple Agent Calls (CIRISBench → Agent Being Tested)

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

## No API Key Needed For

- `GET /health` - Health check (public)
- `GET /he300/catalog` - List scenarios (public)
- `GET /info` - Service info (public)

## Local Development

For local testing without AgentBeats:

```bash
# Generate test credentials
export AGENTBEATS_API_KEY=$(openssl rand -hex 32)
export AGENTBEATS_WEBHOOK_SECRET=$(openssl rand -hex 32)

# Run container
docker run -p 8000:8000 -p 8080:8080 \
  -e AGENTBEATS_API_KEY=$AGENTBEATS_API_KEY \
  -e AGENTBEATS_WEBHOOK_SECRET=$AGENTBEATS_WEBHOOK_SECRET \
  ghcr.io/cirisai/cirisbench:agentbeats

# Test authenticated endpoint
curl -X POST http://localhost:8000/a2a \
  -H "Authorization: Bearer $AGENTBEATS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"jsonrpc":"2.0","method":"benchmark.run","params":{},"id":1}'
```
