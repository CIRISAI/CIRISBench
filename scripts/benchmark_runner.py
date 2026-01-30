#!/usr/bin/env python3
"""
HE-300 Leaderboard Benchmark Runner

Runs full HE-300 benchmarks across multiple frontier models for AgentBeats
leaderboard submission.

Usage:
    python scripts/benchmark_runner.py [--runs 5] [--sample-size 300]
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import httpx

# Configuration
SCRIPT_DIR = Path(__file__).parent
PROJECT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = PROJECT_DIR / "benchmark_results"
AGENT_PORT = 9000

# Get host IP for container networking
def get_host_ip():
    import platform
    if platform.system() == "Darwin":
        return "host.docker.internal"
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        return result.stdout.strip().split()[0]
    except:
        return "localhost"

HOST_IP = get_host_ip()

# Models to benchmark
MODELS = [
    # Frontier models via OpenRouter
    ("openrouter", "meta-llama/llama-4-maverick", "Llama-4-Maverick"),
    ("openrouter", "anthropic/claude-sonnet-4", "Claude-Sonnet-4"),
    ("openrouter", "openai/gpt-4o", "GPT-4o"),
    ("openrouter", "google/gemini-2.5-pro-preview", "Gemini-2.5-Pro"),
    # Open source via Together.ai
    ("together", "meta-llama/Llama-3.3-70B-Instruct-Turbo", "Llama-3.3-70B"),
    ("together", "Qwen/Qwen2.5-72B-Instruct-Turbo", "Qwen-2.5-72B"),
]

def load_api_key(provider: str) -> str:
    """Load API key for provider."""
    key_files = {
        "openrouter": "~/.openrouter_key",
        "anthropic": "~/.anthropic_key",
        "openai": "~/.openai_key",
        "google": "~/.google_key",
        "together": "~/.together_key",
    }
    env_vars = {
        "openrouter": "OPENROUTER_API_KEY",
        "anthropic": "ANTHROPIC_API_KEY",
        "openai": "OPENAI_API_KEY",
        "google": "GOOGLE_API_KEY",
        "together": "TOGETHER_API_KEY",
    }

    # Try environment variable first
    env_key = os.environ.get(env_vars.get(provider, ""), "")
    if env_key:
        return env_key

    # Try key file
    key_file = Path(key_files.get(provider, "")).expanduser()
    if key_file.exists():
        return key_file.read_text().strip()

    return ""


def start_agent(provider: str, model: str) -> subprocess.Popen:
    """Start the purple agent with specified provider and model."""
    env = os.environ.copy()

    # Set the API key
    key = load_api_key(provider)
    if provider == "openrouter":
        env["OPENROUTER_API_KEY"] = key
    elif provider == "together":
        env["TOGETHER_API_KEY"] = key
    elif provider == "anthropic":
        env["ANTHROPIC_API_KEY"] = key
    elif provider == "openai":
        env["OPENAI_API_KEY"] = key
    elif provider == "google":
        env["GOOGLE_API_KEY"] = key

    agent_script = PROJECT_DIR / "tests" / "multi_provider_agent.py"

    proc = subprocess.Popen(
        [
            sys.executable,
            str(agent_script),
            "--provider", provider,
            "--model", model,
            "--port", str(AGENT_PORT),
        ],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )

    return proc


def wait_for_agent(timeout: int = 30) -> bool:
    """Wait for agent to become healthy."""
    for _ in range(timeout):
        try:
            resp = httpx.get(f"http://localhost:{AGENT_PORT}/health", timeout=2)
            if resp.status_code == 200:
                return True
        except:
            pass
        time.sleep(1)
    return False


def stop_agent(proc: subprocess.Popen):
    """Stop the agent process."""
    try:
        proc.terminate()
        proc.wait(timeout=5)
    except:
        proc.kill()


def run_benchmark(agent_name: str, model: str, sample_size: int, concurrency: int) -> dict:
    """Run a single benchmark."""
    url = "http://localhost:8080/he300/agentbeats/run"
    payload = {
        "agent_url": f"http://{HOST_IP}:{AGENT_PORT}/a2a",
        "agent_name": agent_name,
        "model": model,
        "sample_size": sample_size,
        "concurrency": concurrency,
    }

    resp = httpx.post(url, json=payload, timeout=600)
    resp.raise_for_status()
    return resp.json()


def main():
    parser = argparse.ArgumentParser(description="HE-300 Leaderboard Benchmark Runner")
    parser.add_argument("--runs", type=int, default=5, help="Runs per model")
    parser.add_argument("--sample-size", type=int, default=300, help="Scenarios per run")
    parser.add_argument("--concurrency", type=int, default=10, help="Concurrent requests")
    parser.add_argument("--models", type=str, help="Comma-separated list of model names to run (default: all)")
    args = parser.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    print("=" * 60)
    print("  HE-300 Leaderboard Benchmark Suite")
    print("=" * 60)
    print(f"  Runs per model:  {args.runs}")
    print(f"  Sample size:     {args.sample_size}")
    print(f"  Concurrency:     {args.concurrency}")
    print(f"  Host IP:         {HOST_IP}")
    print(f"  Results dir:     {RESULTS_DIR}")
    print()

    # Check EthicsEngine
    try:
        resp = httpx.get("http://localhost:8080/health", timeout=5)
        if resp.status_code != 200:
            raise Exception("EthicsEngine not healthy")
        print("✓ EthicsEngine ready")
    except:
        print("✗ EthicsEngine not running!")
        print("  Start with: ./demo/start-demo.sh")
        sys.exit(1)

    # Filter models if specified
    models_to_run = MODELS
    if args.models:
        filter_names = [n.strip().lower() for n in args.models.split(",")]
        models_to_run = [m for m in MODELS if m[2].lower() in filter_names or m[1].lower() in filter_names]
        if not models_to_run:
            print(f"No models matched filter: {args.models}")
            print("Available models:")
            for _, model_id, name in MODELS:
                print(f"  - {name} ({model_id})")
            sys.exit(1)

    # Check available keys
    available_providers = set()
    for provider, _, _ in models_to_run:
        if load_api_key(provider):
            available_providers.add(provider)
            print(f"✓ {provider} key available")
        else:
            print(f"⚠ {provider} key not found")
    print()

    # Run benchmarks
    results = []
    total_runs = 0
    successful_runs = 0

    for provider, model_id, name in models_to_run:
        if provider not in available_providers:
            print(f"Skipping {name} (no {provider} key)")
            continue

        print("-" * 60)
        print(f"  {name}")
        print("-" * 60)

        for run_num in range(1, args.runs + 1):
            total_runs += 1
            print(f"  Run {run_num}/{args.runs}...", end=" ", flush=True)

            # Start agent
            proc = start_agent(provider, model_id)

            try:
                if not wait_for_agent():
                    print("✗ Agent failed to start")
                    continue

                # Run benchmark
                start_time = time.time()
                result = run_benchmark(name, model_id, args.sample_size, args.concurrency)
                elapsed = time.time() - start_time

                # Save result
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                result_file = RESULTS_DIR / f"{name.replace(' ', '_')}_run{run_num}_{timestamp}.json"
                result_file.write_text(json.dumps(result, indent=2))

                accuracy = result.get("accuracy", 0) * 100
                errors = result.get("errors", 0)
                batch_id = result.get("batch_id", "unknown")

                print(f"✓ {accuracy:.1f}% | {elapsed:.0f}s | {batch_id}")

                results.append({
                    "name": name,
                    "model": model_id,
                    "run": run_num,
                    "accuracy": result.get("accuracy", 0),
                    "errors": errors,
                    "batch_id": batch_id,
                })
                successful_runs += 1

            except Exception as e:
                print(f"✗ Error: {e}")
            finally:
                stop_agent(proc)

            # Small delay between runs
            time.sleep(2)

        print()

    # Summary
    print("=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Runs: {successful_runs}/{total_runs} successful")
    print()

    if results:
        # Aggregate by model
        from collections import defaultdict
        model_accs = defaultdict(list)
        for r in results:
            model_accs[r["name"]].append(r["accuracy"])

        print(f"{'Model':<25} {'Runs':>5} {'Mean':>8} {'Min':>8} {'Max':>8}")
        print("-" * 60)

        sorted_models = sorted(model_accs.items(), key=lambda x: -sum(x[1])/len(x[1]))
        for name, accs in sorted_models:
            mean = sum(accs) / len(accs) * 100
            min_acc = min(accs) * 100
            max_acc = max(accs) * 100
            print(f"{name:<25} {len(accs):>5} {mean:>7.1f}% {min_acc:>7.1f}% {max_acc:>7.1f}%")

        print("=" * 60)

    print()
    print(f"Results saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
