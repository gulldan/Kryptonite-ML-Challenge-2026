"""Run the FastAPI adapter for the current inference backend."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.serve import run_http_server


def _env_flag(name: str) -> bool:
    return os.getenv(name, "").lower() in {"1", "true", "yes", "on"}


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
    parser.add_argument("--host", default="0.0.0.0", help="Bind address.")
    parser.add_argument("--port", type=int, default=8080, help="Bind port.")
    parser.add_argument(
        "--require-artifacts",
        action="store_true",
        default=_env_flag("KRYP_REQUIRE_DEPLOYMENT_ARTIFACTS"),
        help="Fail startup unless manifests/model bundle/demo subset are present.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(
        config_path=args.config,
        overrides=args.override,
        env_file=args.env_file,
    )
    run_http_server(
        host=args.host,
        port=args.port,
        config=config,
        require_artifacts=args.require_artifacts or config.deployment.require_artifacts,
    )


if __name__ == "__main__":
    main()
