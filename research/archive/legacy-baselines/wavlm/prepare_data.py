#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from common import (
    ensure_dir,
    load_config,
    load_duration_lookup,
    prepared_root,
    seed_everything,
    speaker_bin,
    stratified_speaker_train_val_test_split,
    write_json,
    write_resolved_config,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare WavLM speaker-retrieval manifests and speaker split."
    )
    parser.add_argument("--config", required=True, help="Path to model YAML config.")
    return parser.parse_args()


def build_manifest_rows(
    frame: pd.DataFrame,
    duration_lookup: dict[str, float],
    data_root: Path,
    write_absolute_paths: bool,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in frame.itertuples(index=False):
        filepath = row.filepath
        duration = float(duration_lookup[filepath])
        audio_path = data_root / filepath
        rows.append(
            {
                "ID": filepath.replace("/", "__"),
                "dur": duration,
                "path": str(audio_path.resolve() if write_absolute_paths else filepath),
                "start": 0.0,
                "stop": duration,
                "spk": row.speaker_id,
                "orig_filepath": filepath,
            }
        )
    return rows


def summarize_manifest(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "rows": int(len(frame)),
        "speakers": int(frame["spk"].nunique()) if not frame.empty else 0,
    }


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    seed_everything(int(config["data_prep"]["seed"]))

    data_root = config["paths"]["data_root"]
    train_csv = config["paths"]["train_csv"]
    out_root = prepared_root(config)
    ensure_dir(out_root)
    write_resolved_config(config, out_root / "config_resolved.yaml")

    train_df = pd.read_csv(train_csv)
    duration_lookup = load_duration_lookup(config)
    bins = list(config["data_prep"]["speaker_count_bins"])
    train_split_df, val_df, test_df, speaker_counts = stratified_speaker_train_val_test_split(
        train_df=train_df,
        validation_fraction=float(config["data_prep"]["validation_speaker_fraction"]),
        test_fraction=float(config["data_prep"]["test_speaker_fraction"]),
        min_eval_utterances=int(config["data_prep"]["min_eval_utterances"]),
        bins=bins,
        seed=int(config["data_prep"]["seed"]),
    )

    write_absolute_paths = bool(config["data_prep"]["write_absolute_paths"])
    train_manifest = pd.DataFrame(
        build_manifest_rows(train_split_df, duration_lookup, data_root, write_absolute_paths)
    )
    val_manifest = pd.DataFrame(
        build_manifest_rows(val_df, duration_lookup, data_root, write_absolute_paths)
    )
    test_manifest = pd.DataFrame(
        build_manifest_rows(test_df, duration_lookup, data_root, write_absolute_paths)
    )

    train_manifest_path = out_root / "train_manifest.csv"
    val_manifest_path = out_root / "val_manifest.csv"
    test_manifest_path = out_root / "test_manifest.csv"
    train_manifest.to_csv(train_manifest_path, index=False)
    val_manifest.to_csv(val_manifest_path, index=False)
    test_manifest.to_csv(test_manifest_path, index=False)

    train_speakers = sorted(train_manifest["spk"].unique().tolist())
    label_mapping = {speaker: idx for idx, speaker in enumerate(train_speakers)}
    label_mapping_path = out_root / "speaker_to_index.json"
    write_json(label_mapping_path, label_mapping)

    speaker_counts["speaker_bin"] = speaker_counts["utterance_count"].map(
        lambda value: speaker_bin(int(value), bins)
    )
    speaker_counts_path = out_root / "speaker_counts.csv"
    speaker_counts.to_csv(speaker_counts_path, index=False)

    train_summary = summarize_manifest(train_manifest)
    val_summary = summarize_manifest(val_manifest)
    test_summary = summarize_manifest(test_manifest)
    summary = {
        "seed": int(config["data_prep"]["seed"]),
        "validation_speaker_fraction": float(config["data_prep"]["validation_speaker_fraction"]),
        "test_speaker_fraction": float(config["data_prep"]["test_speaker_fraction"]),
        "min_eval_utterances": int(config["data_prep"]["min_eval_utterances"]),
        "write_absolute_paths": bool(config["data_prep"]["write_absolute_paths"]),
        "train_rows": train_summary["rows"],
        "validation_rows": val_summary["rows"],
        "test_rows": test_summary["rows"],
        "train_speakers": train_summary["speakers"],
        "validation_speakers": val_summary["speakers"],
        "test_speakers": test_summary["speakers"],
        "eligible_eval_speakers": int(speaker_counts["eligible_for_eval"].sum()),
        "train_only_low_count_speakers": int(
            (speaker_counts["split_role"] == "train_only_lt11").sum()
        ),
        "train_manifest": str(train_manifest_path),
        "validation_manifest": str(val_manifest_path),
        "test_manifest": str(test_manifest_path),
        "speaker_to_index": str(label_mapping_path),
        "speaker_counts": str(speaker_counts_path),
    }
    write_json(out_root / "split_summary.json", summary)

    print("Prepared manifests:")
    print(f"  train rows: {train_summary['rows']} | speakers: {train_summary['speakers']}")
    print(f"  val rows:   {val_summary['rows']} | speakers: {val_summary['speakers']}")
    print(f"  test rows:  {test_summary['rows']} | speakers: {test_summary['speakers']}")
    print(f"  train manifest: {train_manifest_path}")
    print(f"  val manifest:   {val_manifest_path}")
    print(f"  test manifest:  {test_manifest_path}")


if __name__ == "__main__":
    main()
