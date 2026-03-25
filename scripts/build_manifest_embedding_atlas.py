"""Build a baseline embedding atlas directly from a manifests-backed dataset split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.data import AudioLoadRequest
from kryptonite.eval import (
    ALLOWED_PROJECTION_METHODS,
    SUPPORTED_BASELINE_EMBEDDING_MODES,
    SUPPORTED_METADATA_SOURCE_FORMATS,
    ManifestEmbeddingExportArtifacts,
    build_embedding_atlas,
    export_manifest_fbank_embeddings,
    write_embedding_atlas_report,
)
from kryptonite.features import FbankExtractionRequest, UtteranceChunkingRequest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="Path to the base TOML config.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        required=True,
        help="Path to the source manifest JSONL.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval/embedding-atlas/manifest-run"),
        help=(
            "Directory where exported embeddings, metadata, and atlas artifacts should be written."
        ),
    )
    parser.add_argument(
        "--title",
        default="Manifest Embedding Atlas",
        help="Title rendered into the HTML and Markdown reports.",
    )
    parser.add_argument(
        "--projection",
        choices=ALLOWED_PROJECTION_METHODS,
        default="cosine_pca",
        help="Projection method used to build the 2D atlas.",
    )
    parser.add_argument(
        "--stage",
        choices=("eval", "demo", "train"),
        default="eval",
        help="Chunking stage used to build fixed-size baseline embeddings.",
    )
    parser.add_argument(
        "--embedding-mode",
        choices=sorted(SUPPORTED_BASELINE_EMBEDDING_MODES),
        default="mean_std",
        help="Frame-level pooling recipe used to turn chunk Fbanks into fixed-size embeddings.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Torch device for feature extraction, for example auto, cpu, cuda, or cuda:0.",
    )
    parser.add_argument(
        "--metadata-format",
        choices=sorted(SUPPORTED_METADATA_SOURCE_FORMATS),
        default="parquet",
        help="Metadata sidecar format fed into the atlas after export.",
    )
    parser.add_argument(
        "--max-rows",
        type=int,
        default=None,
        help="Optional hard cap on exported manifest rows.",
    )
    parser.add_argument(
        "--max-per-speaker",
        type=int,
        default=None,
        help="Optional cap on rows exported per speaker_id for a more balanced map.",
    )
    parser.add_argument(
        "--point-id-field",
        default="atlas_point_id",
        help="Metadata field used as the atlas point identifier.",
    )
    parser.add_argument(
        "--label-field",
        default="speaker_id",
        help="Metadata field shown as the primary atlas label.",
    )
    parser.add_argument(
        "--color-by",
        default="speaker_id",
        help="Metadata field used for stable point coloring.",
    )
    parser.add_argument(
        "--search-field",
        action="append",
        default=[],
        help="Metadata field added to the atlas text filter. Can be passed multiple times.",
    )
    parser.add_argument(
        "--audio-path-field",
        default="audio_path",
        help="Metadata field that points to an audio file for preview. Use '' to disable.",
    )
    parser.add_argument(
        "--image-path-field",
        default="",
        help="Metadata field that points to an image file for preview. Optional.",
    )
    parser.add_argument(
        "--neighbors",
        type=int,
        default=5,
        help="Number of cosine nearest neighbors to display per point.",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config override in dotted.key=value form. Can be passed multiple times.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_project_config(config_path=args.config, overrides=args.override)

    export = export_manifest_fbank_embeddings(
        project_root=config.paths.project_root,
        manifest_path=args.manifest,
        output_root=args.output_dir,
        audio_request=AudioLoadRequest.from_config(config.normalization, vad=config.vad),
        fbank_request=FbankExtractionRequest.from_config(config.features),
        chunking_request=UtteranceChunkingRequest.from_config(config.chunking),
        stage=args.stage,
        embedding_mode=args.embedding_mode,
        device=args.device,
        max_rows=args.max_rows,
        max_per_speaker=args.max_per_speaker,
    )
    metadata_path = _select_metadata_path(export=export, metadata_format=args.metadata_format)
    search_fields = _dedupe_fields(
        args.search_field
        or [args.label_field, args.color_by, "capture_condition", "pace", "dataset", "split"]
    )
    atlas = build_embedding_atlas(
        project_root=config.paths.project_root,
        output_root=args.output_dir,
        embeddings_path=export.embeddings_path,
        metadata_path=metadata_path,
        title=args.title,
        projection_method=args.projection,
        point_id_field=args.point_id_field,
        label_field=args.label_field,
        color_by_field=args.color_by,
        search_fields=search_fields,
        audio_path_field=args.audio_path_field or None,
        image_path_field=args.image_path_field or None,
        neighbors=args.neighbors,
        embeddings_key="embeddings",
        ids_key="point_ids",
    )
    written = write_embedding_atlas_report(report=atlas, output_root=args.output_dir)
    print(
        json.dumps(
            {
                "export": export.to_dict(),
                "atlas": {
                    **written.to_dict(),
                    "summary": atlas.summary.to_dict(),
                    "metadata_input_path": str(metadata_path),
                },
            },
            indent=2,
            sort_keys=True,
        )
    )


def _select_metadata_path(
    *,
    export: ManifestEmbeddingExportArtifacts,
    metadata_format: str,
) -> Path:
    if metadata_format == "jsonl":
        return Path(export.metadata_jsonl_path)
    if metadata_format == "csv":
        return Path(export.metadata_csv_path)
    return Path(export.metadata_parquet_path)


def _dedupe_fields(fields: list[str]) -> tuple[str, ...]:
    seen: set[str] = set()
    deduped: list[str] = []
    for field in fields:
        normalized = field.strip()
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        deduped.append(normalized)
    return tuple(deduped)


if __name__ == "__main__":
    main()
