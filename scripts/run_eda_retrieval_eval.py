"""Evaluate local retrieval P@K from precomputed embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import polars as pl

from kryptonite.eda import evaluate_retrieval_embeddings


def main() -> None:
    args = _parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    embeddings = np.load(args.embeddings)
    metadata = pl.read_csv(args.metadata_csv)
    required = {"filepath", args.label_column}
    missing = sorted(required.difference(metadata.columns))
    if missing:
        raise ValueError(f"metadata_csv is missing columns: {missing}")
    if args.max_rows is not None:
        embeddings = embeddings[: args.max_rows]
        metadata = metadata.head(args.max_rows)

    result = evaluate_retrieval_embeddings(
        embeddings,
        labels=metadata.get_column(args.label_column).cast(pl.Utf8).to_list(),
        filepaths=metadata.get_column("filepath").cast(pl.Utf8).to_list(),
        top_k=args.top_k,
        normalize=not args.no_normalize,
    )
    result.per_query.write_parquet(output_dir / "embedding_eval.parquet")
    result.worst_queries.write_csv(output_dir / "worst_queries.csv")
    result.confused_speaker_pairs.write_csv(output_dir / "confused_speaker_pairs.csv")
    (output_dir / "retrieval_summary.json").write_text(
        json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result.summary, ensure_ascii=False, indent=2, sort_keys=True))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--embeddings", required=True, help="Path to .npy embedding matrix.")
    parser.add_argument(
        "--metadata-csv",
        default="datasets/Для участников/train.csv",
        help="CSV aligned to embedding rows; must contain filepath and label column.",
    )
    parser.add_argument("--label-column", default="speaker_id")
    parser.add_argument("--output-dir", default="artifacts/eda/participants")
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--max-rows", type=int, default=None, help="Smoke-test row limit.")
    parser.add_argument("--no-normalize", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
