"""Build pseudo-label manifests from public cluster assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import polars as pl


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    clusters = pl.read_csv(args.clusters_csv)
    public_manifest = pl.read_csv(args.public_manifest_csv)
    selected = (
        clusters.join(public_manifest.select(["row_index", "filepath"]), on="row_index")
        .filter(
            (pl.col("cluster_size") >= args.min_cluster_size)
            & (pl.col("cluster_size") <= args.max_cluster_size)
        )
        .sort(["cluster_id", "row_index"])
    )
    pseudo_path = output_dir / f"{args.experiment_id}_pseudo_manifest.jsonl"
    mixed_path = output_dir / f"{args.experiment_id}_mixed_train_manifest.jsonl"
    summary_path = output_dir / f"{args.experiment_id}_summary.json"

    with pseudo_path.open("w", encoding="utf-8") as handle:
        for row in selected.iter_rows(named=True):
            cluster_id = int(row["cluster_id"])
            row_index = int(row["row_index"])
            payload = {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": args.dataset_name,
                "source_dataset": "test_public_pseudo_labels",
                "speaker_id": f"{args.label_prefix}{cluster_id:06d}",
                "utterance_id": f"{args.label_prefix}{cluster_id:06d}:{row_index:06d}",
                "split": "pseudo_train",
                "audio_path": f"{args.public_audio_prefix.rstrip('/')}/{row['filepath']}",
                "channel": "mono",
            }
            handle.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")

    with mixed_path.open("w", encoding="utf-8") as mixed:
        if args.base_train_manifest:
            base_path = Path(args.base_train_manifest)
            for line in base_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    mixed.write(line.rstrip() + "\n")
        mixed.write(pseudo_path.read_text(encoding="utf-8"))

    cluster_count = selected.select("cluster_id").n_unique()
    summary = {
        "experiment_id": args.experiment_id,
        "clusters_csv": args.clusters_csv,
        "public_manifest_csv": args.public_manifest_csv,
        "base_train_manifest": args.base_train_manifest,
        "pseudo_manifest": str(pseudo_path),
        "mixed_train_manifest": str(mixed_path),
        "min_cluster_size": args.min_cluster_size,
        "max_cluster_size": args.max_cluster_size,
        "pseudo_row_count": selected.height,
        "pseudo_cluster_count": cluster_count,
        "mixed_row_count": sum(
            1 for line in mixed_path.read_text(encoding="utf-8").splitlines() if line.strip()
        ),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clusters-csv", required=True)
    parser.add_argument("--public-manifest-csv", required=True)
    parser.add_argument("--base-train-manifest", default="")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--min-cluster-size", type=int, default=8)
    parser.add_argument("--max-cluster-size", type=int, default=80)
    parser.add_argument("--label-prefix", default="pseudo_g6_")
    parser.add_argument("--dataset-name", default="participants_g6_pseudo")
    parser.add_argument("--public-audio-prefix", default="datasets/Для участников")
    return parser.parse_args()


if __name__ == "__main__":
    main()
