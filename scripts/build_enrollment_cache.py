"""Build the offline enrollment-embedding cache consumed by the infer runtime."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.data import AudioLoadRequest
from kryptonite.features import FbankExtractionRequest, UtteranceChunkingRequest
from kryptonite.serve.enrollment_cache import (
    MODEL_BUNDLE_METADATA_NAME,
    build_enrollment_embedding_cache,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/deployment/infer.toml"),
        help="Path to the serving config that defines manifests and artifact roots.",
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
        "--manifest",
        type=Path,
        default=None,
        help=(
            "Optional override for the source manifest. "
            "Defaults to artifacts/manifests/demo_manifest.jsonl."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional override for the enrollment-cache output directory.",
    )
    parser.add_argument(
        "--model-metadata",
        type=Path,
        default=None,
        help="Optional override for the model bundle metadata.json path.",
    )
    parser.add_argument(
        "--stage",
        default="demo",
        help="Chunking stage to use when extracting offline enrollment embeddings.",
    )
    parser.add_argument(
        "--embedding-mode",
        default="mean_std",
        help="Frame-pooling recipe used to build per-sample embeddings.",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Optional torch device override (for example cpu or cuda).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(
        config_path=args.config,
        overrides=args.override,
        env_file=args.env_file,
    )

    manifest_path = args.manifest or Path(config.paths.manifests_root) / "demo_manifest.jsonl"
    output_dir = args.output_dir or Path(config.deployment.enrollment_cache_root)
    model_metadata = args.model_metadata or (
        Path(config.deployment.model_bundle_root) / MODEL_BUNDLE_METADATA_NAME
    )
    written = build_enrollment_embedding_cache(
        project_root=config.paths.project_root,
        manifest_path=manifest_path,
        output_root=output_dir,
        model_metadata_path=model_metadata,
        audio_request=AudioLoadRequest.from_config(config.normalization, vad=config.vad),
        fbank_request=FbankExtractionRequest.from_config(config.features),
        chunking_request=UtteranceChunkingRequest.from_config(config.chunking),
        stage=args.stage,
        embedding_mode=args.embedding_mode,
        device=args.device or config.runtime.device,
    )
    print(json.dumps(written.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
