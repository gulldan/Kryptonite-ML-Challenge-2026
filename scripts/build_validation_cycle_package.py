"""Build validation-cycle review package for baseline_fixed ablations."""

from __future__ import annotations

import argparse
from pathlib import Path

from kryptonite.eda.validation_cycle import build_validation_cycle_package


def main() -> None:
    args = _parse_args()
    build_validation_cycle_package(
        dense_dir=Path(args.dense_dir),
        shifted_dir=Path(args.shifted_dir),
        baseline_fixed_dir=Path(args.baseline_fixed_dir),
        audio_artifact_dir=Path(args.audio_artifact_dir),
        output_dir=Path(args.output_dir),
        zip_path=Path(args.zip_path),
        shifted_v2_dir=Path(args.shifted_v2_dir),
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dense-dir", default="artifacts/eda/baseline_fixed_dense_gallery")
    parser.add_argument(
        "--shifted-dir", default="artifacts/eda/baseline_fixed_dense_shifted_gallery"
    )
    parser.add_argument("--baseline-fixed-dir", default="artifacts/baseline_fixed_participants")
    parser.add_argument("--audio-artifact-dir", default="artifacts/eda/participants_audio6")
    parser.add_argument("--output-dir", default="artifacts/eda/validation_cycle_package")
    parser.add_argument("--zip-path", default="artifacts/eda/validation_cycle_package.zip")
    parser.add_argument("--shifted-v2-dir", default="artifacts/eda/baseline_fixed_dense_shifted_v2")
    return parser.parse_args()


if __name__ == "__main__":
    main()
