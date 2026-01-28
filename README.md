# CIRISBench

Benchmarking and evaluation infrastructure for CIRIS AI agents.

## Overview

CIRISBench provides the evaluation layer for [CIRISNode](https://github.com/CIRISAI/CIRISNode), enabling systematic assessment of AI agent behavior through ethical benchmarks and evaluation frameworks.

## Repository Structure

```
CIRISBench/
├── engine/     # EthicsEngine Enterprise - benchmark evaluation engine
└── infra/      # CI/CD, Docker, Terraform infrastructure
```

### engine/ - EthicsEngine Enterprise

The core evaluation engine implementing:
- **HE-300 Benchmark**: 300-scenario ethical reasoning evaluation based on Hendrycks Ethics
- **Batch Evaluation API**: REST endpoints for benchmark execution
- **CIRIS Trace Validation**: Ed25519 signed trace verification
- **Report Generation**: HTML reports with GitHub Pages deployment

### infra/ - Integration Infrastructure

CI/CD and deployment infrastructure:
- **Docker Compose**: Development, staging, and production stacks
- **GitHub Actions**: Automated benchmark workflows
- **Terraform**: Infrastructure as code for cloud deployments
- **Dashboard**: Next.js monitoring UI

## Ecosystem

```
CIRISNode (agent framework)
    ↓ evaluated by
CIRISBench (this repo)
    ↓ runs on
CIRISBridge (infrastructure services)
```

## Quick Start

See [infra/QUICKSTART.md](infra/QUICKSTART.md) for setup instructions.

## License

See individual component licenses in `engine/` and `infra/` directories.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

For benchmark contributions, see [engine/docs/HE300_BENCHMARK.md](engine/docs/HE300_BENCHMARK.md).
