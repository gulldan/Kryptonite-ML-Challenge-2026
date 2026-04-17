"""Export a CSV-only EDA pack from offline artifacts."""

from __future__ import annotations

import argparse
from pathlib import Path

from kryptonite.eda.csv_exports import export_eda_csv_pack


def main() -> None:
    args = _parse_args()
    written = export_eda_csv_pack(
        artifact_dir=Path(args.artifact_dir),
        output_dir=Path(args.output_dir),
        dataset_root=Path(args.dataset_root),
        baseline_onnx=Path(args.baseline_onnx) if args.baseline_onnx else None,
    )
    print(f"Wrote {len(written)} CSV artifacts to {args.output_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", default="artifacts/eda/participants")
    parser.add_argument("--output-dir", default="artifacts/eda/participants_csv")
    parser.add_argument("--dataset-root", default="datasets/Для участников")
    parser.add_argument("--baseline-onnx", default="datasets/Для участников/baseline.onnx")
    return parser.parse_args()


if __name__ == "__main__":
    main()
