"""CIRISBench CLI entry point."""

import argparse
import sys
from pathlib import Path


def main() -> int:
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        prog="cirisbench",
        description="CIRISBench - Benchmarking infrastructure for CIRIS AI agents",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version and exit",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # benchmark command
    bench_parser = subparsers.add_parser("benchmark", help="Run HE-300 benchmark")
    bench_parser.add_argument(
        "--model",
        default="ollama/llama3.2",
        help="Model to evaluate (default: ollama/llama3.2)",
    )
    bench_parser.add_argument(
        "--sample-size",
        type=int,
        default=50,
        help="Number of scenarios to evaluate (default: 50)",
    )
    bench_parser.add_argument(
        "--mock",
        action="store_true",
        help="Use mock LLM for testing",
    )
    bench_parser.add_argument(
        "--output",
        type=Path,
        help="Output file for results (JSON)",
    )

    # server command
    server_parser = subparsers.add_parser("server", help="Start EthicsEngine server")
    server_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    server_parser.add_argument(
        "--port",
        type=int,
        default=8080,
        help="Port to bind to (default: 8080)",
    )
    server_parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development",
    )

    # info command
    subparsers.add_parser("info", help="Show system information")

    args = parser.parse_args()

    if args.version:
        from cirisbench import __version__
        print(f"cirisbench {__version__}")
        return 0

    if args.command == "benchmark":
        return run_benchmark(args)
    elif args.command == "server":
        return run_server(args)
    elif args.command == "info":
        return show_info()
    else:
        parser.print_help()
        return 0


def run_benchmark(args: argparse.Namespace) -> int:
    """Run HE-300 benchmark."""
    print(f"Running HE-300 benchmark...")
    print(f"  Model: {args.model}")
    print(f"  Sample size: {args.sample_size}")
    print(f"  Mock mode: {args.mock}")

    # Import engine components
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent / "engine"))
        from run_pipeline import main as run_engine_benchmark
        # TODO: Integrate with engine benchmark runner
        print("\nBenchmark runner not yet integrated. Use:")
        print("  cd engine && python run_pipeline.py")
    except ImportError as e:
        print(f"Error importing engine: {e}")
        return 1

    return 0


def run_server(args: argparse.Namespace) -> int:
    """Start EthicsEngine server."""
    import subprocess

    cmd = [
        sys.executable, "-m", "uvicorn",
        "api.main:app",
        "--host", args.host,
        "--port", str(args.port),
    ]
    if args.reload:
        cmd.append("--reload")

    engine_dir = Path(__file__).parent.parent / "engine"
    print(f"Starting EthicsEngine server at http://{args.host}:{args.port}")

    try:
        result = subprocess.run(cmd, cwd=engine_dir)
        return result.returncode
    except KeyboardInterrupt:
        print("\nServer stopped.")
        return 0


def show_info() -> int:
    """Show system information."""
    from cirisbench import __version__
    from cirisbench.config import settings

    print("CIRISBench System Information")
    print("=" * 40)
    print(f"Version: {__version__}")
    print(f"Root directory: {settings.root_dir}")
    print(f"Engine directory: {settings.engine_dir}")
    print(f"Infra directory: {settings.infra_dir}")
    print()
    print("Configuration:")
    print(f"  CIRISNode URL: {settings.cirisnode_url}")
    print(f"  LLM Model: {settings.llm_model}")
    print(f"  Ollama URL: {settings.ollama_base_url}")
    print(f"  HE-300 Sample Size: {settings.he300_sample_size}")
    print(f"  Mock LLM: {settings.ff_mock_llm}")
    print()
    print("Paths:")
    print(f"  Engine exists: {settings.engine_dir.exists()}")
    print(f"  Infra exists: {settings.infra_dir.exists()}")
    print(f"  Data exists: {settings.data_dir.exists()}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
