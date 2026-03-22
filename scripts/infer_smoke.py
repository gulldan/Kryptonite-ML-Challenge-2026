"""Validate the inference runtime and thin API startup path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.serve import create_http_server
from kryptonite.serve.runtime import build_serve_runtime_report, render_serve_runtime_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deployment/infer.toml"),
        help="Path to the active serving config.",
    )
    parser.add_argument(
        "--env-file",
        type=Path,
        default=Path(".env"),
        help="Optional dotenv file with secrets.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in dotted.key=value form. Can be passed multiple times.",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    parser.add_argument(
        "--skip-startup-check",
        action="store_true",
        help="Only validate the runtime, do not bind an HTTP server socket.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(
        config_path=args.config,
        overrides=args.override,
        env_file=args.env_file,
    )
    report = build_serve_runtime_report(config=config)

    if args.output == "json":
        print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_serve_runtime_report(report))

    if not report.passed:
        raise SystemExit(1)

    if args.skip_startup_check:
        raise SystemExit(0)

    server = create_http_server(host="127.0.0.1", port=0, config=config)
    try:
        host, port = server.server_address
        print(f"API startup smoke: PASS ({host}:{port})")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
