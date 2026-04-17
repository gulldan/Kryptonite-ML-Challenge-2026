"""Build training manifests from organizer participant CSV splits."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from kryptonite.data.manifest_artifacts import write_manifest_inventory, write_tabular_artifact
from kryptonite.data.schema import MANIFEST_RECORD_TYPE, MANIFEST_SCHEMA_VERSION


def build_participant_training_manifests(
    *,
    train_split_csv: Path,
    dev_split_csv: Path,
    output_dir: Path,
    project_root: Path,
    audio_root: Path = Path("datasets/Для участников"),
    dataset_name: str = "participants",
) -> dict[str, str]:
    """Convert speaker-id/filepath CSV splits into schema-compliant JSONL manifests."""

    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = _csv_to_manifest_rows(
        train_split_csv,
        audio_root=audio_root,
        dataset_name=dataset_name,
        split="train",
    )
    dev_rows = _csv_to_manifest_rows(
        dev_split_csv,
        audio_root=audio_root,
        dataset_name=dataset_name,
        split="dev",
    )
    train_artifact = write_tabular_artifact(
        name="participants_train",
        kind="manifest",
        rows=train_rows,
        jsonl_path=output_dir / "train_manifest.jsonl",
        project_root=project_root.as_posix(),
    )
    dev_artifact = write_tabular_artifact(
        name="participants_dev",
        kind="manifest",
        rows=dev_rows,
        jsonl_path=output_dir / "dev_manifest.jsonl",
        project_root=project_root.as_posix(),
    )
    inventory_path = write_manifest_inventory(
        dataset=dataset_name,
        inventory_path=output_dir / "manifest_inventory.json",
        project_root=project_root.as_posix(),
        manifest_tables=[train_artifact, dev_artifact],
    )
    return {
        "train_manifest": train_artifact.jsonl_path,
        "dev_manifest": dev_artifact.jsonl_path,
        "inventory": inventory_path,
    }


def _csv_to_manifest_rows(
    csv_path: Path,
    *,
    audio_root: Path,
    dataset_name: str,
    split: str,
) -> list[dict[str, object]]:
    frame = pl.read_csv(csv_path)
    required = {"speaker_id", "filepath"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{csv_path} is missing required columns: {', '.join(missing)}")
    rows = []
    for index, row in enumerate(frame.iter_rows(named=True)):
        speaker_id = str(row["speaker_id"])
        filepath = str(row["filepath"])
        rows.append(
            {
                "schema_version": MANIFEST_SCHEMA_VERSION,
                "record_type": MANIFEST_RECORD_TYPE,
                "dataset": dataset_name,
                "source_dataset": "kryptonite_participants",
                "speaker_id": speaker_id,
                "utterance_id": f"{speaker_id}:{Path(filepath).stem}:{index}",
                "split": split,
                "audio_path": (audio_root / filepath).as_posix(),
                "channel": "mono",
            }
        )
    if not rows:
        raise ValueError(f"{csv_path} produced an empty manifest")
    return rows
