"""Project precomputed embeddings into an interactive atlas report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from kryptonite.config import load_project_config
from kryptonite.eval import (
    ALLOWED_PROJECTION_METHODS,
    build_embedding_atlas,
    write_embedding_atlas_report,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.toml"),
        help="Path to the base TOML config.",
    )
    parser.add_argument(
        "--embeddings",
        type=Path,
        required=True,
        help="Path to a .npy or .npz embeddings matrix.",
    )
    parser.add_argument(
        "--metadata",
        type=Path,
        required=True,
        help="Path to a .jsonl or .csv metadata table aligned with the embeddings.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/eval/embedding-atlas"),
        help="Directory where the atlas HTML, points JSONL, and reports should be written.",
    )
    parser.add_argument(
        "--title",
        default="Speaker Embedding Atlas",
        help="Title rendered into the HTML and Markdown reports.",
    )
    parser.add_argument(
        "--projection",
        choices=ALLOWED_PROJECTION_METHODS,
        default="cosine_pca",
        help="Projection method used to build the 2D atlas.",
    )
    parser.add_argument(
        "--point-id-field",
        default="utterance_id",
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
        "--embeddings-key",
        default="embeddings",
        help="Array key to use when the embeddings source is a .npz file.",
    )
    parser.add_argument(
        "--ids-key",
        default="",
        help="Optional array key with point ids when the embeddings source is a .npz file.",
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
    search_fields = _dedupe_fields(
        args.search_field or [args.label_field, args.color_by, "dataset", "split"]
    )
    report = build_embedding_atlas(
        project_root=config.paths.project_root,
        output_root=args.output_dir,
        embeddings_path=args.embeddings,
        metadata_path=args.metadata,
        title=args.title,
        projection_method=args.projection,
        point_id_field=args.point_id_field,
        label_field=args.label_field,
        color_by_field=args.color_by,
        search_fields=search_fields,
        audio_path_field=args.audio_path_field or None,
        image_path_field=args.image_path_field or None,
        neighbors=args.neighbors,
        embeddings_key=args.embeddings_key,
        ids_key=args.ids_key or None,
    )
    written = write_embedding_atlas_report(report=report, output_root=args.output_dir)
    print(
        json.dumps(
            {
                **written.to_dict(),
                "summary": report.summary.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
    )


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
