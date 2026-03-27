"""Build a Triton model repository from the current encoder-boundary bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.serve.triton_repository import (
    TritonDynamicBatchingConfig,
    TritonRepositoryRequest,
    build_triton_model_repository,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deployment/infer.toml"),
        help="Path to the deployment config that defines the source model bundle.",
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
        "--backend-mode",
        choices=("onnx", "tensorrt"),
        default="onnx",
        help="Which Triton backend layout to generate.",
    )
    parser.add_argument(
        "--engine-path",
        type=str,
        default=None,
        help="Optional TensorRT engine path used when --backend-mode=tensorrt.",
    )
    parser.add_argument(
        "--output-root",
        type=str,
        default="artifacts/triton-model-repository",
        help="Where to write the Triton repository.",
    )
    parser.add_argument(
        "--model-name",
        default="kryptonite_encoder",
        help="Triton model name to create under the repository root.",
    )
    parser.add_argument(
        "--version",
        type=int,
        default=1,
        help="Version directory to create under the Triton model name.",
    )
    parser.add_argument(
        "--max-batch-size",
        type=int,
        default=8,
        help="Triton max_batch_size setting.",
    )
    parser.add_argument(
        "--instance-group-count",
        type=int,
        default=1,
        help="Number of Triton instances to request in config.pbtxt.",
    )
    parser.add_argument(
        "--sample-frame-count",
        type=int,
        default=12,
        help="How many frames to emit in the generated sample infer request.",
    )
    parser.add_argument(
        "--preferred-batch-size",
        action="append",
        type=int,
        dest="preferred_batch_sizes",
        default=[],
        help="Preferred dynamic batching size. Can be passed multiple times.",
    )
    parser.add_argument(
        "--max-queue-delay-microseconds",
        type=int,
        default=1_000,
        help="Dynamic batching max queue delay.",
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
    config = load_project_config(
        config_path=args.config,
        overrides=args.override,
        env_file=args.env_file,
    )
    preferred_batch_sizes = (
        tuple(args.preferred_batch_sizes)
        if args.preferred_batch_sizes
        else TritonDynamicBatchingConfig().preferred_batch_sizes
    )
    built = build_triton_model_repository(
        config=config,
        request=TritonRepositoryRequest(
            output_root=args.output_root,
            model_name=args.model_name,
            backend_mode=args.backend_mode,
            engine_path=args.engine_path,
            version=args.version,
            max_batch_size=args.max_batch_size,
            instance_group_count=args.instance_group_count,
            sample_frame_count=args.sample_frame_count,
            dynamic_batching=TritonDynamicBatchingConfig(
                preferred_batch_sizes=preferred_batch_sizes,
                max_queue_delay_microseconds=args.max_queue_delay_microseconds,
            ),
        ),
    )
    if args.output == "json":
        print(json.dumps(built.to_dict(), indent=2, sort_keys=True))
    else:
        print(
            "\n".join(
                [
                    "Triton repository build: PASS",
                    f"Repository: {built.repository_root}",
                    f"Model: {built.model_name} ({built.platform})",
                    f"config.pbtxt: {built.config_path}",
                    f"Model artifact: {built.model_path}",
                    f"Sample request: {built.smoke_request_path}",
                    "",
                    "Smoke curl:",
                    built.sample_curl_command,
                ]
            )
        )


if __name__ == "__main__":
    main()
