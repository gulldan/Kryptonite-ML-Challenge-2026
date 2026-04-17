"""Run public B2/B4/B7 submissions and public hubness reports."""

from __future__ import annotations

import argparse
from pathlib import Path

from kryptonite.eda.public_ablation import run_public_ablation_package


def main() -> None:
    args = _parse_args()
    run_public_ablation_package(
        manifest_csv=Path(args.manifest_csv),
        template_csv=Path(args.template_csv),
        onnx_path=Path(args.onnx_path),
        file_stats_path=Path(args.file_stats_path),
        output_dir=Path(args.output_dir),
        zip_path=Path(args.zip_path),
        batch_size=args.batch_size,
        search_batch_size=args.search_batch_size,
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest-csv",
        default="artifacts/eda/participants_public_baseline/test_public_manifest.csv",
    )
    parser.add_argument("--template-csv", default="datasets/Для участников/test_public.csv")
    parser.add_argument(
        "--onnx-path", default="artifacts/baseline_fixed_participants/model_embeddings.onnx"
    )
    parser.add_argument(
        "--file-stats-path", default="artifacts/eda/participants_audio6/file_stats.parquet"
    )
    parser.add_argument("--output-dir", default="artifacts/eda/public_ablation_cycle")
    parser.add_argument("--zip-path", default="artifacts/eda/public_ablation_cycle.zip")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--search-batch-size", type=int, default=2048)
    return parser.parse_args()


if __name__ == "__main__":
    main()
