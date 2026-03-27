"""Probe a running Triton server with the generated encoder-boundary request."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.serve.triton_smoke import render_triton_smoke_result, run_triton_infer_smoke


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000",
        help="Triton HTTP endpoint root.",
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path("artifacts/triton-model-repository"),
        help="Repository root created by scripts/build_triton_model_repository.py.",
    )
    parser.add_argument(
        "--model-name",
        default="kryptonite_encoder",
        help="Model name to probe under the Triton repository.",
    )
    parser.add_argument(
        "--request-file",
        type=Path,
        default=None,
        help="Optional explicit request JSON path. Defaults to the generated smoke request.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=float,
        default=10.0,
        help="Timeout for health and infer HTTP calls.",
    )
    parser.add_argument(
        "--output",
        choices=("text", "json"),
        default="text",
        help="Output format.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    request_file = args.request_file
    if request_file is None:
        request_file = args.repository_root / "smoke" / f"{args.model_name}_infer_request.json"
    result = run_triton_infer_smoke(
        server_url=args.server_url,
        model_name=args.model_name,
        request_path=request_file,
        timeout_seconds=args.timeout_seconds,
    )
    if args.output == "json":
        print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    else:
        print(render_triton_smoke_result(result))


if __name__ == "__main__":
    main()
