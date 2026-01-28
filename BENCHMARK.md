# HE-300 Ethics Benchmark

CIRISBench implements the **Hendrycks Ethics** benchmark (HE-300) - 300 ethical reasoning scenarios for evaluating AI agent moral judgment.

## Leaderboard Categories

| Category | Scenarios | Description |
|----------|-----------|-------------|
| **Commonsense** | 75 | Everyday moral intuitions |
| **Deontology** | 75 | Duty-based ethical reasoning |
| **Justice** | 75 | Fairness and equitable treatment |
| **Virtue** | 75 | Character-based moral reasoning |

## Submit Your Agent

### Via AgentBeats Platform

1. Register your purple agent on [AgentBeats](https://agentbeats.dev)
2. Select CIRISBench from available benchmarks
3. AgentBeats will trigger evaluation automatically

### Via GitHub Actions (Manual)

1. Go to [Actions → HE-300 Benchmark Submission](../../actions/workflows/benchmark-submission.yml)
2. Click "Run workflow"
3. Enter your agent details:
   - **Agent URL**: Your A2A/MCP endpoint (e.g., `https://myagent.example.com/a2a`)
   - **Agent Name**: Display name for leaderboard
   - **Model**: Model identifier (e.g., `gpt-4o`, `claude-3-opus`)
   - **Sample Size**: 50, 100, or 300 scenarios

### Via API

```bash
# Trigger benchmark via repository dispatch
curl -X POST \
  -H "Authorization: Bearer $GITHUB_TOKEN" \
  -H "Accept: application/vnd.github.v3+json" \
  https://api.github.com/repos/CIRISAI/CIRISBench/dispatches \
  -d '{
    "event_type": "benchmark-request",
    "client_payload": {
      "agent_url": "https://your-agent.example.com/a2a",
      "agent_name": "My Agent",
      "model": "gpt-4o",
      "sample_size": 300,
      "callback_url": "https://agentbeats.dev/webhook/results"
    }
  }'
```

## Agent Requirements

Your agent must implement either:

### MCP Protocol
```json
POST /mcp
{
  "method": "tools/call",
  "params": {
    "name": "evaluate_scenario",
    "arguments": {
      "scenario": "You witness someone shoplifting food...",
      "options": ["Report them", "Ignore it", "Offer to help pay"]
    }
  }
}
```

### A2A Protocol
```json
POST /a2a
{
  "jsonrpc": "2.0",
  "method": "benchmark.evaluate",
  "params": {
    "scenario_id": "commonsense_001",
    "scenario": "...",
    "options": ["A", "B", "C"]
  },
  "id": 1
}
```

## Scoring

- **Accuracy**: Percentage of scenarios answered correctly
- **Category Accuracy**: Breakdown by ethical framework
- **Response Time**: Average time per scenario (informational)

## Badges

| Badge | Requirement |
|-------|-------------|
| 🏆 Ethics Champion | >90% overall accuracy |
| ⚖️ Balanced Ethicist | >80% in all four categories |
| 💡 Commonsense Expert | Top 3 in commonsense category |

## Results Format

```json
{
  "agent_id": "agent-123",
  "agent_name": "My Agent",
  "model": "gpt-4o",
  "accuracy": 0.847,
  "total_scenarios": 300,
  "correct": 254,
  "categories": {
    "commonsense": {"accuracy": 0.88, "correct": 66, "total": 75},
    "deontology": {"accuracy": 0.82, "correct": 62, "total": 75},
    "justice": {"accuracy": 0.85, "correct": 64, "total": 75},
    "virtue": {"accuracy": 0.83, "correct": 62, "total": 75}
  },
  "timestamp": "2026-01-28T18:00:00Z"
}
```

## Reference Scores

| Model | Overall | Commonsense | Deontology | Justice | Virtue |
|-------|---------|-------------|------------|---------|--------|
| Human Baseline | 95% | 96% | 94% | 95% | 94% |
| GPT-4o | ~85% | 87% | 83% | 86% | 84% |
| Claude 3 Opus | ~84% | 86% | 82% | 85% | 83% |
| Llama 3.2 70B | ~78% | 80% | 76% | 79% | 77% |

*Reference scores are approximate based on Hendrycks et al. 2021 and subsequent evaluations.*

## Citation

```bibtex
@article{hendrycks2021ethics,
  title={Aligning AI With Shared Human Values},
  author={Hendrycks, Dan and others},
  journal={arXiv preprint arXiv:2008.02275},
  year={2021}
}
```

## Links

- [Hendrycks Ethics Paper](https://arxiv.org/abs/2008.02275)
- [Original Dataset](https://github.com/hendrycks/ethics)
- [AgentBeats Platform](https://agentbeats.dev)
