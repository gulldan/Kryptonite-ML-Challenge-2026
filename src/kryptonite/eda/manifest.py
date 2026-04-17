"""Manifest loading helpers for participant challenge data."""

from __future__ import annotations

from pathlib import Path

import polars as pl


def load_train_manifest(dataset_root: Path | str) -> pl.DataFrame:
    """Load the participant train manifest with stable row ids."""

    root = Path(dataset_root)
    manifest_path = root / "train.csv"
    return _load_manifest_csv(
        manifest_path=manifest_path,
        dataset_root=root,
        split="train",
        speaker_col="speaker_id",
    )


def load_test_manifest(dataset_root: Path | str, *, name: str = "test_public") -> pl.DataFrame:
    """Load a participant test manifest with stable row ids."""

    root = Path(dataset_root)
    manifest_path = root / f"{name}.csv"
    return _load_manifest_csv(
        manifest_path=manifest_path,
        dataset_root=root,
        split=name,
        speaker_col=None,
    )


def _load_manifest_csv(
    *,
    manifest_path: Path,
    dataset_root: Path,
    split: str,
    speaker_col: str | None,
) -> pl.DataFrame:
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest does not exist: {manifest_path}")

    frame = pl.read_csv(manifest_path)
    if "filepath" not in frame.columns:
        raise ValueError(f"{manifest_path} must contain a filepath column.")
    if speaker_col is not None and speaker_col not in frame.columns:
        raise ValueError(f"{manifest_path} must contain a {speaker_col!r} column.")

    selected = frame.select(
        pl.col("filepath").cast(pl.Utf8),
        (
            pl.col(speaker_col).cast(pl.Utf8)
            if speaker_col is not None
            else pl.lit(None, dtype=pl.Utf8).alias("speaker_id")
        ),
    )
    return selected.with_row_index("row_index").with_columns(
        pl.lit(split).alias("split"),
        pl.col("filepath")
        .map_elements(lambda value: str((dataset_root / value).resolve()), return_dtype=pl.Utf8)
        .alias("resolved_path"),
    )
