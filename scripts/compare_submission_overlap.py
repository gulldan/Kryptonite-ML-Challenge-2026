"""Compare two retrieval submission CSV files by top-k neighbour overlap."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

from kryptonite.eda.rerank import gini
from kryptonite.eda.submission import validate_submission


def main() -> None:
    args = _parse_args()
    left = _load_neighbours(Path(args.left_csv), k=args.k)
    right = _load_neighbours(Path(args.right_csv), k=args.k)
    if left["paths"] != right["paths"]:
        raise ValueError("Submission filepath order/content differs between left and right CSV.")

    payload = {
        "left_stats": _submission_stats(
            path=Path(args.left_csv),
            neighbours=left["neighbours"],
            row_count=len(left["paths"]),
            k=args.k,
        ),
        "right_stats": _submission_stats(
            path=Path(args.right_csv),
            neighbours=right["neighbours"],
            row_count=len(right["paths"]),
            k=args.k,
        ),
        "comparison": _compare_neighbours(left["neighbours"], right["neighbours"], k=args.k),
    }
    if args.template_csv:
        payload["validation"] = {
            "left": validate_submission(
                template_csv=Path(args.template_csv),
                submission_csv=Path(args.left_csv),
                k=args.k,
            ),
            "right": validate_submission(
                template_csv=Path(args.template_csv),
                submission_csv=Path(args.right_csv),
                k=args.k,
            ),
        }

    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))
    print(f"Wrote {output_json}")


def _load_neighbours(path: Path, *, k: int) -> dict[str, Any]:
    frame = pl.read_csv(path)
    if "filepath" not in frame.columns or "neighbours" not in frame.columns:
        raise ValueError(f"{path} must contain filepath and neighbours columns.")
    paths = frame.get_column("filepath").cast(pl.Utf8).to_list()
    neighbour_cells = frame.get_column("neighbours").cast(pl.Utf8).fill_null("").to_list()
    neighbours = np.empty((len(neighbour_cells), k), dtype=np.int64)
    for row_index, cell in enumerate(neighbour_cells):
        values = [int(part.strip()) for part in cell.split(",") if part.strip()]
        if len(values) < k:
            raise ValueError(f"{path} row {row_index + 1} has {len(values)} neighbours < {k}.")
        neighbours[row_index] = np.asarray(values[:k], dtype=np.int64)
    return {"paths": paths, "neighbours": neighbours}


def _submission_stats(
    *,
    path: Path,
    neighbours: np.ndarray,
    row_count: int,
    k: int,
) -> dict[str, Any]:
    indegree = np.bincount(neighbours[:, :k].ravel(), minlength=row_count)
    return {
        "path": str(path),
        "shape": [int(neighbours.shape[0]), int(min(k, neighbours.shape[1]))],
        "first_row": [int(value) for value in neighbours[0, :k].tolist()],
        "indegree_gini": float(gini(indegree)),
        "indegree_max": int(indegree.max()),
    }


def _compare_neighbours(left: np.ndarray, right: np.ndarray, *, k: int) -> dict[str, Any]:
    if left.shape != right.shape:
        raise ValueError(f"Neighbour matrix shapes differ: {left.shape} != {right.shape}.")
    left_k = left[:, :k]
    right_k = right[:, :k]
    ordered_equal = left_k == right_k
    overlap_counts = np.fromiter(
        (
            len(set(left_row.tolist()) & set(right_row.tolist()))
            for left_row, right_row in zip(left_k, right_k, strict=True)
        ),
        dtype=np.int64,
        count=left_k.shape[0],
    )
    hist = np.bincount(overlap_counts, minlength=k + 1)
    return {
        "rows": int(left_k.shape[0]),
        "mean_overlap_at_k": float(overlap_counts.mean()),
        "median_overlap_at_k": float(np.median(overlap_counts)),
        "ordered_cell_equal_share": float(ordered_equal.mean()),
        "row_exact_same_order_share": float(np.all(ordered_equal, axis=1).mean()),
        "row_same_set_share": float((overlap_counts == k).mean()),
        "top1_equal_share": float((left_k[:, 0] == right_k[:, 0]).mean()),
        "overlap_count_hist": {str(index): int(value) for index, value in enumerate(hist)},
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--left-csv", required=True)
    parser.add_argument("--right-csv", required=True)
    parser.add_argument("--template-csv", default="")
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--k", type=int, default=10)
    return parser.parse_args()


if __name__ == "__main__":
    main()
