import argparse
import json
import os

import numpy as np
import pandas as pd
from src.metrics import precision_at_k_from_indices


def load_submission(indices_csv: str) -> pd.DataFrame:
    df = pd.read_csv(indices_csv)
    missing = {"filepath", "neighbours"}.difference(df.columns)
    if missing:
        raise ValueError(f"submission.csv is missing columns: {sorted(missing)}")
    return df


def load_indices(indices_csv: str, expected_columns: int) -> np.ndarray:
    df = load_submission(indices_csv)
    rows = []
    for row_idx, value in enumerate(df["neighbours"].fillna("").astype(str).tolist()):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) != expected_columns:
            raise ValueError(
                "submission.csv must contain exactly "
                f"{expected_columns} neighbours in each row; "
                f"data row {row_idx + 1} has {len(parts)}."
            )
        if any(part == "" for part in parts):
            raise ValueError(
                f"submission.csv contains empty neighbour ids in data row {row_idx + 1}."
            )
        if any(not part.isdigit() for part in parts):
            raise ValueError(
                f"submission.csv contains non-integer neighbour ids in data row {row_idx + 1}."
            )
        rows.append([int(part) for part in parts])

    return np.asarray(rows, dtype=np.int64)


def validate_indices(indices: np.ndarray) -> np.ndarray:
    """
    Проверяет, что в каждой строке submission.csv:
      - нет отрицательных индексов
      - все индексы уникальны
      - нет self-index
    """
    if indices.ndim != 2:
        raise ValueError("indices must be a 2D array of shape (N, K)")

    n, m = indices.shape

    for i in range(n):
        row = indices[i]
        if np.any(row < 0):
            raise ValueError(f"indices.csv contains negative indices in data row {i + 1}.")
        if np.any(row >= n):
            raise ValueError(f"indices.csv contains out-of-range indices in data row {i + 1}.")
        if np.any(row == i):
            raise ValueError(f"indices.csv contains self-index in data row {i + 1}.")
        if np.unique(row).size != m:
            raise ValueError(f"indices.csv contains duplicated indices in data row {i + 1}.")

    return indices


def validate_filepath_order(indices_csv: str, template_csv: str) -> None:
    submission = load_submission(indices_csv)
    template = pd.read_csv(template_csv)
    if "filepath" not in template.columns:
        raise ValueError("template_csv is missing column: filepath")
    if len(submission) != len(template):
        raise ValueError(
            "submission row count differs from template: "
            f"submission={len(submission)} template={len(template)}"
        )
    submitted_paths = submission["filepath"].astype(str).tolist()
    template_paths = template["filepath"].astype(str).tolist()
    if submitted_paths != template_paths:
        raise ValueError("submission filepath order/content differs from template_csv.")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        type=str,
        required=True,
        help="Path to submission.csv",
    )
    parser.add_argument("--labels", type=str, default="", help="Path to labels.npy")
    parser.add_argument("--k", type=int, default=10, help="K for Precision@K (default: 10)")
    parser.add_argument(
        "--template_csv",
        type=str,
        default="",
        help="Optional CSV whose filepath column must exactly match submission order.",
    )
    args = parser.parse_args()

    labels_path = args.labels
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels.npy not found: {labels_path}")

    labels = np.load(labels_path, allow_pickle=True)
    if args.template_csv:
        validate_filepath_order(args.indices, args.template_csv)
    indices = load_indices(args.indices, expected_columns=args.k)
    indices = validate_indices(indices)
    if indices.shape[0] != labels.shape[0]:
        raise ValueError(f"labels length {labels.shape[0]} != submission rows {indices.shape[0]}")
    metrics = precision_at_k_from_indices(indices, labels, ks=(args.k,))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
