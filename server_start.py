#!/usr/bin/env python
"""
Ethics Engine API Server Startup Script
This script helps start and verify the API server is running correctly.
"""

import os
import sys
import time
import subprocess
import requests
import argparse
from typing import Optional, Tuple

def check_server_running(url: str, timeout: int = 5) -> bool:
    """Check if the server is already running at the given URL."""
    try:
        response = requests.get(f"{url}/health", timeout=timeout)
        return response.status_code == 200
    except:
        return False

def start_server(host: str = "127.0.0.1", 
                port: int = 8080, 
                reload: bool = True,
                log_level: str = "info",
                env_file: Optional[str] = None) -> Tuple[subprocess.Popen, str]:
    """Start the FastAPI server using uvicorn."""
    
    # Construct the base URL
    base_url = f"http://{host}:{port}"
    
    # Check if the server is already running
    if check_server_running(base_url):
        print(f"Server is already running at {base_url}")
        return None, base_url
    
    # Prepare the environment variables
    env = os.environ.copy()
    if env_file and os.path.exists(env_file):
        print(f"Loading environment variables from {env_file}")
        with open(env_file, 'r') as f:
            for line in f:
                if line.strip() and not line.startswith('#'):
                    key, value = line.strip().split('=', 1)
                    env[key] = value
    
    # Construct the command to start the server
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "api.main:app",  # Adjust this to your main app module path
        f"--host={host}", 
        f"--port={port}", 
        f"--log-level={log_level}"
    ]
    
    if reload:
        cmd.append("--reload")
    
    print(f"Starting server with command: {' '.join(cmd)}")
    
    # Start the server as a subprocess
    process = subprocess.Popen(
        cmd, 
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    
    # Wait for the server to start
    max_attempts = 10
    for attempt in range(max_attempts):
        print(f"Waiting for server to start (attempt {attempt+1}/{max_attempts})...")
        time.sleep(2)  # Give it time to start
        
        if check_server_running(base_url):
            print(f"Server started successfully at {base_url}")
            return process, base_url
    
    # If we get here, the server didn't start properly
    print("Failed to start server. Check the logs below:")
    stdout, stderr = process.communicate(timeout=5)
    print("STDOUT:", stdout)
    print("STDERR:", stderr)
    process.terminate()
    return None, base_url

def main():
    parser = argparse.ArgumentParser(description="Start and manage the Ethics Engine API server")
    parser.add_argument("--host", default="127.0.0.1", help="Host to bind the server to")
    parser.add_argument("--port", type=int, default=8080, help="Port to bind the server to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--log-level", default="info", choices=["debug", "info", "warning", "error", "critical"], 
                        help="Logging level")
    parser.add_argument("--env-file", default=".env", help="Path to .env file")
    
    args = parser.parse_args()
    
    process, base_url = start_server(
        host=args.host,
        port=args.port,
        reload=not args.no_reload,
        log_level=args.log_level,
        env_file=args.env_file
    )
    
    if process:
        print(f"Server is running at {base_url}")
        print("Press Ctrl+C to stop the server")
        try:
            # Stream the logs from the server
            for line in iter(process.stdout.readline, ""):
                print(line, end="")
        except KeyboardInterrupt:
            print("Stopping server...")
            process.terminate()
            process.wait(timeout=5)
            print("Server stopped")

if __name__ == "__main__":
    main()