import argparse
import json
import os
import numpy as np
import pandas as pd

from src.metrics import precision_at_k_from_indices


def load_indices(indices_csv: str, expected_columns: int) -> np.ndarray:
    df = pd.read_csv(indices_csv)
    rows = []
    for row_idx, value in enumerate(df["neighbours"].fillna("").astype(str).tolist()):
        parts = [part.strip() for part in value.split(",")]
        if len(parts) < expected_columns:
            raise ValueError(
                "submission.csv must contain at least "
                f"{expected_columns} neighbours in each row "
            )
        if any(part == "" for part in parts[:expected_columns]):
            raise ValueError(
                f"submission.csv contains empty neighbour ids in data row {row_idx + 1}."
            )
        if any(not part.isdigit() for part in parts[:expected_columns]):
            raise ValueError(
                f"submission.csv contains non-integer neighbour ids in data row {row_idx + 1}."
            )
        rows.append([int(part) for part in parts[:expected_columns]])

    return np.asarray(rows, dtype=np.int64)


def validate_indices(I: np.ndarray) -> np.ndarray:
    """
    Проверяет, что в каждой строке submission.csv:
      - нет отрицательных индексов
      - все индексы уникальны
      - нет self-index
    """
    if I.ndim != 2:
        raise ValueError("indices must be a 2D array of shape (N, K)")

    n, m = I.shape

    for i in range(n):
        row = I[i]
        if np.any(row < 0):
            raise ValueError(
                f"indices.csv contains negative indices in data row {i + 1}."
            )
        if np.any(row == i):
            raise ValueError(
                f"indices.csv contains self-index in data row {i + 1}."
            )
        if np.unique(row).size != m:
            raise ValueError(
                f"indices.csv contains duplicated indices in data row {i + 1}."
            )

    return I


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--indices",
        type=str,
        required=True,
        help="Path to submission.csv",
    )
    parser.add_argument("--labels", type=str, default="", help="Path to labels.npy")
    parser.add_argument(
        "--k", type=int, default=10, help="K for Precision@K (default: 10)"
    )
    args = parser.parse_args()

    labels_path = args.labels
    if not os.path.isfile(labels_path):
        raise FileNotFoundError(f"labels.npy not found: {labels_path}")

    labels = np.load(labels_path, allow_pickle=True)
    indices = load_indices(args.indices, expected_columns=args.k)
    indices = validate_indices(indices)
    metrics = precision_at_k_from_indices(indices, labels, ks=(args.k,))
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
