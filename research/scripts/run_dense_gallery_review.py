"""Build dense-gallery ablation and hubness review package."""

from __future__ import annotations

import argparse
from pathlib import Path

from kryptonite.eda.dense_gallery import run_dense_gallery_package


def main() -> None:
    args = _parse_args()
    run_dense_gallery_package(
        dataset_root=Path(args.dataset_root),
        onnx_path=Path(args.onnx_path),
        audio_artifact_dir=Path(args.audio_artifact_dir),
        public_artifact_dir=Path(args.public_artifact_dir),
        bucket_file_stats_path=Path(args.bucket_file_stats_path),
        output_dir=Path(args.output_dir),
        zip_path=Path(args.zip_path),
        query_speakers=args.query_speakers,
        distractor_speakers=args.distractor_speakers,
        utts_per_speaker=args.utts_per_speaker,
        seed=args.seed,
        batch_size=args.batch_size,
        query_batch_size=args.query_batch_size,
        synthetic_shift=args.synthetic_shift,
        shift_mode=args.shift_mode,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", default="datasets/Для участников")
    parser.add_argument("--onnx-path", default="datasets/Для участников/baseline.onnx")
    parser.add_argument("--audio-artifact-dir", default="artifacts/eda/participants_audio6")
    parser.add_argument(
        "--public-artifact-dir", default="artifacts/eda/participants_public_baseline"
    )
    parser.add_argument(
        "--bucket-file-stats-path",
        default="artifacts/eda/eda_review_package/05_local_file_stats.parquet",
    )
    parser.add_argument("--output-dir", default="artifacts/eda/dense_gallery_review_package")
    parser.add_argument("--zip-path", default="artifacts/eda/dense_gallery_review_package.zip")
    parser.add_argument("--query-speakers", type=int, default=500)
    parser.add_argument("--distractor-speakers", type=int, default=3000)
    parser.add_argument("--utts-per-speaker", type=int, default=11)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=4096)
    parser.add_argument(
        "--synthetic-shift",
        action="store_true",
        help="Add deterministic public-empirical leading/trailing silence before embeddings.",
    )
    parser.add_argument(
        "--shift-mode",
        choices=["none", "edge_silence", "v2"],
        default="none",
        help="Synthetic shift transform for dense validation.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    main()
