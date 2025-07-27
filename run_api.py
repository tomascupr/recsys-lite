#!/usr/bin/env python3
"""
Script to run RecSys-Lite API server.

This script properly runs the FastAPI application using Uvicorn.
It resolves issues with the CLI approach that uses 'recsys-lite serve'
by directly launching the server with the right module path.

Usage:
    python run_api.py --model-dir model_artifacts/als --port 8000
"""

import argparse
from pathlib import Path

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run RecSys-Lite API server")
    parser.add_argument("--model-dir", type=str, default="model_artifacts/als", help="Model directory")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Host to listen on")
    parser.add_argument("--port", type=int, default=8000, help="Port to listen on")
    parser.add_argument("--log-level", type=str, default="info", help="Log level")

    args = parser.parse_args()

    model_path = Path(args.model_dir)
    if not model_path.exists():
        print(f"Error: Model directory '{args.model_dir}' does not exist")
        return 1

    print(f"Starting API server at http://{args.host}:{args.port}")
    uvicorn.run("recsys_lite.api.main:app", host=args.host, port=args.port, log_level=args.log_level, reload=False)
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
