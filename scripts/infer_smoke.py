"""Smoke-check the inference/runtime surface expected by the project."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.serve.deployment import build_infer_artifact_report
from kryptonite.serve.inferencer import Inferencer
from kryptonite.serve.runtime import (
    build_serve_runtime_report,
    build_service_metadata,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deployment/infer.toml"),
        help="Path to the serving TOML config.",
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
        "--require-artifacts",
        action="store_true",
        help="Fail if deployment artifacts are missing.",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def _render_health(payload: dict[str, object]) -> str:
    artifacts = payload.get("artifacts", {})
    inferencer = payload.get("inferencer", {})
    runtime = payload.get("runtime", {})
    model_bundle = payload.get("model_bundle", {})
    selection = payload.get("selection", {})

    lines = [
        f"Inference smoke: {'PASS' if payload.get('status') == 'ok' else 'DEGRADED'}",
        f"Requested backend: {payload.get('requested_backend')}",
        f"Selected backend: {payload.get('selected_backend')}",
        f"Selected provider: {payload.get('selected_provider') or '-'}",
        f"Selection reason: {selection.get('reason')}",
        f"Runtime device: {runtime.get('device')}",
        f"Inferencer implementation: {inferencer.get('implementation')}",
        f"Model bundle loaded: {'yes' if model_bundle.get('loaded') else 'no'}",
        (
            "Artifacts: "
            f"{'strict' if artifacts.get('strict') else 'advisory'} / "
            f"{'pass' if artifacts.get('passed') else 'fail'}"
        ),
    ]
    advisory_warning = payload.get("advisory_warning")
    if advisory_warning:
        lines.append(f"Advisory warning: {advisory_warning}")
    return "\n".join(lines)


def _build_advisory_payload(*, config, warning: str) -> dict[str, object]:
    artifact_report = build_infer_artifact_report(config=config, strict=False)
    runtime_report = build_serve_runtime_report(config=config, model_metadata=None)
    payload = build_service_metadata(
        config=config,
        report=runtime_report,
        artifact_report=artifact_report,
        enrollment_cache={
            "loaded": False,
            "reason": "advisory smoke skipped model bundle initialization",
        },
    )
    payload["inferencer"] = {"implementation": "uninitialized"}
    payload["model_bundle"] = {"loaded": False, "warning": warning}
    payload["advisory_warning"] = warning
    return payload


def main() -> None:
    args = parse_args()
    config = load_project_config(
        config_path=args.config,
        overrides=args.override,
        env_file=args.env_file,
    )
    try:
        inferencer = Inferencer.from_config(
            config=config,
            require_artifacts=args.require_artifacts,
        )
        payload = inferencer.health_payload()
    except ValueError as exc:
        if args.require_artifacts:
            raise
        payload = _build_advisory_payload(config=config, warning=str(exc))
    if args.output == "json":
        print(json.dumps(payload, indent=2, sort_keys=True))
    else:
        print(_render_health(payload))


if __name__ == "__main__":
    main()
