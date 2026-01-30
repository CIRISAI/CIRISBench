#!/usr/bin/env python3
"""
Submit HE-300 benchmark results to AgentBeats leaderboard.

Pushes results to the CIRISBench-leaderboard GitHub repo which triggers
the AgentBeats webhook to update the public leaderboard.

Usage:
    python scripts/submit_to_agentbeats.py benchmark_results/*.json
    python scripts/submit_to_agentbeats.py --all-results
"""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

# Configuration
AGENTBEATS_AGENT_ID = "019c0710-78cf-75c3-bb2d-631789a667ef"
LEADERBOARD_REPO = Path.home() / "CIRISBench-leaderboard"
RESULTS_DIR = LEADERBOARD_REPO / "results"


def transform_result(result: dict, run_id: str = None) -> dict:
    """Transform benchmark result to AgentBeats leaderboard format."""
    categories = result.get("categories", {})

    # Generate a unique run ID
    if not run_id:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        agent_name = result.get("agent_name", "unknown").replace(" ", "-").lower()
        run_id = f"{agent_name}-{timestamp}"

    # Build submission in the expected format
    submission = {
        "id": AGENTBEATS_AGENT_ID,  # AgentBeats agent UUID
        "agent_id": run_id,  # Unique identifier for this run
        "agent_name": result.get("agent_name", "Unknown"),
        "model": result.get("model", "unknown"),
        "accuracy": result.get("accuracy", 0),
        "total_scenarios": result.get("total_scenarios", 0),
        "correct": result.get("correct", 0),
        "errors": result.get("errors", 0),
        # Flat category accuracies for DuckDB queries
        "commonsense_accuracy": categories.get("commonsense", {}).get("accuracy"),
        "commonsense_hard_accuracy": categories.get("commonsense_hard", {}).get("accuracy"),
        "deontology_accuracy": categories.get("deontology", {}).get("accuracy"),
        "justice_accuracy": categories.get("justice", {}).get("accuracy"),
        "virtue_accuracy": categories.get("virtue", {}).get("accuracy"),
        "categories": categories,
        "avg_latency_ms": result.get("avg_latency_ms", 0),
        "processing_time_ms": result.get("processing_time_ms", 0),
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        "run_id": run_id,
        "batch_id": result.get("batch_id", ""),
    }

    return submission


def write_result_to_repo(result: dict, dry_run: bool = False) -> str:
    """Write a result file to the leaderboard repo."""
    submission = transform_result(result)

    # Generate filename
    agent_name = submission["agent_name"].replace(" ", "-")
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")
    filename = f"{agent_name}-{timestamp}.json"
    filepath = RESULTS_DIR / filename

    print(f"\nWriting: {filename}")
    print(f"  Agent: {submission['agent_name']} ({submission['model']})")
    print(f"  Accuracy: {submission['accuracy']*100:.1f}%")

    if dry_run:
        print(f"  [DRY RUN] Would write to: {filepath}")
        return None

    # Write file
    RESULTS_DIR.mkdir(exist_ok=True)
    filepath.write_text(json.dumps(submission, indent=2))
    print(f"  ✓ Written to: {filepath}")

    return str(filepath)


def git_push_results(files: list, dry_run: bool = False):
    """Commit and push results to GitHub."""
    if not files:
        print("No files to push")
        return

    os.chdir(LEADERBOARD_REPO)

    # Git add
    print("\nCommitting to GitHub...")
    for f in files:
        subprocess.run(["git", "add", f], check=True)

    # Check for changes
    result = subprocess.run(["git", "status", "--porcelain"], capture_output=True, text=True)
    if not result.stdout.strip():
        print("No changes to commit")
        return

    # Commit
    num_results = len(files)
    commit_msg = f"Add {num_results} HE-300 benchmark result(s)\n\n🤖 Generated with CIRISBench"

    if dry_run:
        print(f"  [DRY RUN] Would commit: {commit_msg}")
        return

    subprocess.run(["git", "commit", "-m", commit_msg], check=True)

    # Push
    print("Pushing to GitHub...")
    result = subprocess.run(["git", "push"], capture_output=True, text=True)
    if result.returncode == 0:
        print("✓ Pushed to GitHub - AgentBeats webhook will update leaderboard")
    else:
        print(f"✗ Push failed: {result.stderr}")


def main():
    parser = argparse.ArgumentParser(description="Submit results to AgentBeats leaderboard")
    parser.add_argument("files", nargs="*", help="JSON result files to submit")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be submitted")
    parser.add_argument("--all-results", action="store_true", help="Submit all files in benchmark_results/")
    parser.add_argument("--no-push", action="store_true", help="Write files but don't push to GitHub")
    args = parser.parse_args()

    # Check leaderboard repo exists
    if not LEADERBOARD_REPO.exists():
        print(f"Error: Leaderboard repo not found at {LEADERBOARD_REPO}")
        print("Clone it with: git clone https://github.com/CIRISAI/CIRISBench-leaderboard.git ~/CIRISBench-leaderboard")
        sys.exit(1)

    # Collect results
    result_files = []

    if args.all_results:
        benchmark_dir = Path("benchmark_results")
        if benchmark_dir.exists():
            result_files.extend(benchmark_dir.glob("*.json"))

    for f in args.files:
        result_files.append(Path(f))

    if not result_files:
        print("No results to submit!")
        print("Usage:")
        print("  python scripts/submit_to_agentbeats.py benchmark_results/*.json")
        print("  python scripts/submit_to_agentbeats.py --all-results")
        sys.exit(1)

    # Parse and write results
    written_files = []
    for rf in result_files:
        try:
            result = json.loads(rf.read_text())
            written = write_result_to_repo(result, dry_run=args.dry_run)
            if written:
                written_files.append(written)
        except Exception as e:
            print(f"Warning: Could not process {rf}: {e}")

    print(f"\n{'='*50}")
    print(f"Processed: {len(written_files)} results")

    # Push to GitHub
    if not args.no_push and not args.dry_run and written_files:
        git_push_results(written_files, dry_run=args.dry_run)
        print(f"\nView leaderboard at: https://agentbeats.dev/")


if __name__ == "__main__":
    main()
