"""Export a CAM++ checkpoint into an encoder-only ONNX model bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.runtime.onnx_export import (
    DEFAULT_CAMPP_ONNX_BUNDLE_ROOT,
    CAMPPONNXExportRequest,
    export_campp_checkpoint_to_onnx,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="Path to the project config that defines export names and frontend metadata.",
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
        "--checkpoint",
        required=True,
        help="Checkpoint file or completed CAM++ run directory to export.",
    )
    parser.add_argument(
        "--output-root",
        default=DEFAULT_CAMPP_ONNX_BUNDLE_ROOT,
        help="Where to write the ONNX model bundle.",
    )
    parser.add_argument(
        "--model-version",
        default=None,
        help="Optional metadata model_version override.",
    )
    parser.add_argument(
        "--embedding-stage",
        choices=("eval", "demo", "train"),
        default="eval",
        help="Chunking stage metadata to record in the export bundle.",
    )
    parser.add_argument(
        "--sample-batch-size",
        type=int,
        default=1,
        help="Batch size for the export-time ONNX Runtime smoke input.",
    )
    parser.add_argument(
        "--sample-frame-count",
        type=int,
        default=200,
        help="Frame count for the export-time ONNX Runtime smoke input.",
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
    exported = export_campp_checkpoint_to_onnx(
        config=config,
        request=CAMPPONNXExportRequest(
            checkpoint_path=args.checkpoint,
            output_root=args.output_root,
            model_version=args.model_version,
            sample_batch_size=args.sample_batch_size,
            sample_frame_count=args.sample_frame_count,
            embedding_stage=args.embedding_stage,
        ),
    )
    if args.output == "json":
        print(json.dumps(exported.to_dict(), indent=2, sort_keys=True))
        return
    print(
        "\n".join(
            [
                "CAM++ ONNX export: PASS",
                f"Checkpoint: {exported.source_checkpoint_path}",
                f"Output root: {exported.output_root}",
                f"Model: {exported.model_path}",
                f"Metadata: {exported.metadata_path}",
                f"Report: {exported.report_markdown_path}",
                "Validation: checker=true, onnxruntime_smoke=true",
                f"Max abs diff: {exported.validation.max_abs_diff:.8f}",
                f"Mean abs diff: {exported.validation.mean_abs_diff:.8f}",
            ]
        )
    )


if __name__ == "__main__":
    main()
