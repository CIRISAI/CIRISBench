# HE-300 Web Interface User Guide

## Overview

The HE-300 Web Interface provides a comprehensive dashboard for:
- **Model Management**: Load, list, and delete Ollama models
- **Benchmark Running**: Execute HE-300 ethics benchmarks
- **Report Generation**: Create signed static reports for GitHub Pages

## Accessing the Interface

### Via SSH Tunnel (Recommended for Security)

```bash
# Create SSH tunnel to all services
ssh -L 3000:localhost:3000 \
    -L 8000:localhost:8000 \
    -L 8080:localhost:8080 \
    ubuntu@163.192.58.165

# Then open in browser:
# http://localhost:3000/he300
```

### Direct Access (if firewall permits)

```
http://<server-ip>:3000/he300
```

---

## Features

### 1. Model Management Tab

![Models Tab](./images/models-tab.png)

**Capabilities:**
- View all loaded Ollama models with sizes
- Pull new models from Ollama registry
- Delete models to free disk space
- Quick-select popular models

**Popular Models for Ethics Benchmarks:**
| Model | Size | VRAM Required | Speed |
|-------|------|---------------|-------|
| `gemma3:4b-it-q8_0` | 5GB | ~5GB | Fast |
| `qwen2.5:32b` | 19GB | ~20GB | Medium |
| `llama3.1:8b` | 4.7GB | ~6GB | Fast |
| `mistral:7b` | 4.1GB | ~5GB | Fast |

**Pull a Model:**
1. Enter model name in the input field (e.g., `gemma3:4b-it-q8_0`)
2. Or click a quick-select button
3. Click "Pull Model"
4. Progress bar shows download status

---

### 2. Benchmark Tab

![Benchmark Tab](./images/benchmark-tab.png)

**Configuration Options:**

| Option | Description | Default |
|--------|-------------|---------|
| Category | Ethics category to test | `commonsense` |
| Number of Scenarios | How many test cases (1-50) | `10` |
| Identity Profile | Evaluator identity | `Neutral` |
| Ethical Guidance | Framework for evaluation | `Utilitarian` |
| Batch ID | Unique identifier for this run | Auto-generated |

**Available Categories:**
- `commonsense` - Basic ethical scenarios
- `commonsense_hard` - Challenging edge cases
- `deontology` - Duty-based ethics
- `justice` - Fairness and equity
- `virtue` - Character-based ethics

**Running a Benchmark:**
1. Select a category
2. Set number of scenarios (start with 5-10 for testing)
3. Choose identity and guidance profiles
4. Click "Run Benchmark"
5. Wait for completion (typically 30-60 seconds for 10 scenarios)

**Understanding Results:**
- **Accuracy**: Percentage of correct ethical judgments
- **Correct/Total**: Number of correct predictions
- **Avg Latency**: Average time per scenario
- **By Category**: Breakdown by ethics category

---

### 3. Reports Tab

![Reports Tab](./images/reports-tab.png)

**Report Formats:**

| Format | Best For | Features |
|--------|----------|----------|
| Markdown | Jekyll/GitHub Pages | YAML frontmatter, expandable sections |
| HTML | Standalone viewing | Styled, charts, self-contained |
| JSON | Machine processing | Structured data, API integration |

**Report Options:**
- **Include Scenarios**: Add individual test results
- **Sign Report**: Add cryptographic signature for integrity
- **Jekyll Frontmatter**: Add YAML header for static sites

**Generating a Report:**
1. Run a benchmark first
2. Select output format
3. Enter model name and optional title
4. Configure options
5. Click "Generate Report"
6. Download the generated file

---

## Deploying Reports to GitHub Pages

### Step 1: Create Jekyll Site Structure

```
my-reports-site/
├── _config.yml
├── _layouts/
│   └── report.html
├── reports/
│   └── 2026-01-03-he300-benchmark.md  # Downloaded report
└── index.md
```

### Step 2: Create Report Layout

`_layouts/report.html`:
```html
---
layout: default
---
<article class="report">
  <header>
    <h1>{{ page.title }}</h1>
    <div class="meta">
      <span>Model: {{ page.model }}</span>
      <span>Accuracy: {{ page.accuracy | times: 100 | round: 1 }}%</span>
      <span>Date: {{ page.date | date: "%Y-%m-%d" }}</span>
    </div>
  </header>
  
  {{ content }}
  
  {% if page.signature %}
  <footer class="signature">
    <h3>🔐 Report Integrity</h3>
    <p>Hash: {{ page.signature.hash }}</p>
    <p>Algorithm: {{ page.signature.algorithm }}</p>
  </footer>
  {% endif %}
</article>
```

### Step 3: Configure Jekyll

`_config.yml`:
```yaml
title: HE-300 Benchmark Reports
collections:
  reports:
    output: true
    permalink: /reports/:title/
defaults:
  - scope:
      path: "reports"
      type: "reports"
    values:
      layout: report
```

### Step 4: Deploy to GitHub Pages

```bash
# Initialize git repo
git init
git add .
git commit -m "Initial benchmark reports"

# Push to GitHub
git remote add origin https://github.com/your-org/he300-reports.git
git push -u origin main

# Enable GitHub Pages in repo settings
```

---

## Verifying Report Signatures

Reports can be cryptographically signed. To verify:

```bash
# Extract signature from report footer
# Compare content hash with stored hash
sha256sum report.md
# Hash should match signature.content_hash
```

Or use the API:

```bash
curl -X POST http://localhost:8080/reports/verify \
  -H "Content-Type: application/json" \
  -d '{
    "content": "... report content ...",
    "signature": {
      "algorithm": "sha256+hmac",
      "content_hash": "abc123...",
      "timestamp": "2026-01-03T...",
      "signature": "def456..."
    }
  }'
```

---

## API Reference

### Ollama Management

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/ollama/health` | GET | Check Ollama connectivity |
| `/ollama/models` | GET | List all models |
| `/ollama/models/pull` | POST | Pull a new model |
| `/ollama/models/pull/status/{name}` | GET | Check pull progress |
| `/ollama/models/{name}` | DELETE | Delete a model |

### HE-300 Benchmarks

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/he300/health` | GET | Check HE-300 status |
| `/he300/catalog` | GET | List available scenarios |
| `/he300/batch` | POST | Run batch evaluation |
| `/he300/scenarios/{id}` | GET | Get scenario details |

### Reports

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/reports/` | GET | List all reports |
| `/reports/generate` | POST | Generate new report |
| `/reports/download/{id}` | GET | Download report file |
| `/reports/{id}` | GET | Get report metadata |
| `/reports/{id}` | DELETE | Delete report |
| `/reports/verify` | POST | Verify signature |

---

## Troubleshooting

### "API Unreachable" Error

1. Check SSH tunnel is running
2. Verify EthicsEngine container is healthy:
   ```bash
   ssh ubuntu@server "docker ps | grep ethicsengine"
   ```
3. Check logs:
   ```bash
   ssh ubuntu@server "docker logs he300-ethicsengine"
   ```

### Slow Model Loading

First inference after model load is slow (~30-60s). Subsequent calls are fast (~1s).

### Out of GPU Memory

Delete unused models:
```bash
# Via UI: Models tab → Delete button
# Via CLI:
ssh ubuntu@server "docker exec he300-ollama ollama rm model-name"
```

### Report Generation Fails

Ensure you have run a benchmark first. Reports are generated from the most recent benchmark result.

---

*Document Version: 1.0*
*Last Updated: 2026-01-03*
