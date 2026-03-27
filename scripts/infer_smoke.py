"""Validate the inference runtime and thin API startup path."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.deployment import render_artifact_report
from kryptonite.serve import Inferencer, create_http_server
from kryptonite.serve.deployment import build_infer_artifact_report
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
        "--require-artifacts",
        action="store_true",
        help="Fail unless manifests/model bundle/demo subset are present for target runs.",
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
    strict_artifacts = args.require_artifacts or config.deployment.require_artifacts
    report = build_serve_runtime_report(config=config)
    artifact_report = build_infer_artifact_report(config=config, strict=strict_artifacts)
    inferencer = None
    if report.passed and artifact_report.passed:
        inferencer = Inferencer.from_config(
            config=config,
            require_artifacts=strict_artifacts,
        )

    if args.output == "json":
        payload = {
            "runtime": report.to_dict(),
            "artifacts": artifact_report.to_dict(),
        }
        if inferencer is not None:
            payload["inferencer"] = inferencer.health_payload()["inferencer"]
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        sections = [
            render_serve_runtime_report(report),
            render_artifact_report(artifact_report),
        ]
        if inferencer is not None:
            inferencer_payload = inferencer.health_payload()["inferencer"]
            sections.append(
                "Inferencer smoke: PASS "
                f"(implementation={inferencer_payload['implementation']}, "
                f"embedding_dim={inferencer_payload['embedding_dim']}, "
                f"stage={inferencer_payload['default_stage']})"
            )
        print("\n\n".join(sections))

    if not report.passed or not artifact_report.passed:
        raise SystemExit(1)

    if args.skip_startup_check:
        raise SystemExit(0)

    server = create_http_server(
        host="127.0.0.1",
        port=0,
        config=config,
        require_artifacts=strict_artifacts,
    )
    try:
        host, port = server.server_address
        print(f"API startup smoke: PASS ({host}:{port})")
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
