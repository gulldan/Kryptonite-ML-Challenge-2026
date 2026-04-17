"""Build CN-Celeb manifests for external speaker adaptation experiments."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    args = _parse_args()
    root = Path(args.root)
    if not root.exists():
        raise SystemExit(f"CN-Celeb root does not exist: {root}")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    audio_suffixes = tuple(suffix.lower() for suffix in args.audio_suffix)
    excluded_parts = {part.strip() for part in args.exclude_part if part.strip()}
    rows = _scan_audio_rows(
        root=root,
        dataset_name=args.dataset_name,
        source_dataset=args.source_dataset,
        speaker_prefix=args.speaker_prefix,
        speaker_source=args.speaker_source,
        speaker_level=args.speaker_level,
        filename_speaker_separator=args.filename_speaker_separator,
        audio_suffixes=audio_suffixes,
        excluded_parts=excluded_parts,
        channel=args.channel,
    )
    if not rows:
        raise SystemExit(f"No audio rows found under {root}")

    train_speakers, dev_speakers = _split_speakers(
        rows,
        dev_speaker_fraction=args.dev_speaker_fraction,
        seed=args.seed,
        min_dev_speakers=args.min_dev_speakers,
    )
    train_rows = [
        row | {"split": "external_train"} for row in rows if row["speaker_id"] in train_speakers
    ]
    dev_rows = [
        row | {"split": "external_dev"} for row in rows if row["speaker_id"] in dev_speakers
    ]

    prefix = args.experiment_id
    all_path = output_dir / f"{prefix}_all_manifest.jsonl"
    train_path = output_dir / f"{prefix}_train_manifest.jsonl"
    dev_path = output_dir / f"{prefix}_dev_manifest.jsonl"
    mixed_path = output_dir / f"{prefix}_mixed_train_manifest.jsonl"
    summary_path = output_dir / f"{prefix}_summary.json"

    _write_jsonl(all_path, rows)
    _write_jsonl(train_path, train_rows)
    _write_jsonl(dev_path, dev_rows)
    _write_csv(all_path.with_suffix(".csv"), rows)
    _write_csv(train_path.with_suffix(".csv"), train_rows)
    _write_csv(dev_path.with_suffix(".csv"), dev_rows)

    mixed_row_count = 0
    if args.base_train_manifest:
        mixed_row_count = _write_mixed_manifest(
            mixed_path,
            base_train_manifest=Path(args.base_train_manifest),
            external_train_manifest=train_path,
        )

    summary = {
        "experiment_id": args.experiment_id,
        "root": str(root),
        "dataset_name": args.dataset_name,
        "source_dataset": args.source_dataset,
        "speaker_source": args.speaker_source,
        "speaker_level": args.speaker_level,
        "filename_speaker_separator": args.filename_speaker_separator,
        "speaker_prefix": args.speaker_prefix,
        "audio_suffixes": list(audio_suffixes),
        "excluded_parts": sorted(excluded_parts),
        "row_count": len(rows),
        "speaker_count": len({row["speaker_id"] for row in rows}),
        "train_row_count": len(train_rows),
        "train_speaker_count": len(train_speakers),
        "dev_row_count": len(dev_rows),
        "dev_speaker_count": len(dev_speakers),
        "base_train_manifest": args.base_train_manifest,
        "mixed_train_manifest": str(mixed_path) if args.base_train_manifest else "",
        "mixed_row_count": mixed_row_count,
        "all_manifest": str(all_path),
        "train_manifest": str(train_path),
        "dev_manifest": str(dev_path),
        "top_speakers": _top_speakers(rows, limit=10),
    }
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="datasets/CN-Celeb_flac")
    parser.add_argument("--output-dir", default="artifacts/manifests/cnceleb_v2")
    parser.add_argument("--experiment-id", default="cnceleb_v2")
    parser.add_argument("--dataset-name", default="cnceleb_v2")
    parser.add_argument("--source-dataset", default="cnceleb_v2_openslr82")
    parser.add_argument("--speaker-prefix", default="cnceleb_")
    parser.add_argument(
        "--speaker-source",
        choices=("filename-prefix", "path-part"),
        default="filename-prefix",
        help="CN-Celeb v2 stores speaker ids as filename prefixes such as id00939-...",
    )
    parser.add_argument("--speaker-level", type=int, default=1)
    parser.add_argument("--filename-speaker-separator", default="-")
    parser.add_argument("--audio-suffix", action="append", default=[".flac", ".wav"])
    parser.add_argument("--exclude-part", action="append", default=["lists", ".ipynb_checkpoints"])
    parser.add_argument("--channel", default="mono")
    parser.add_argument("--dev-speaker-fraction", type=float, default=0.02)
    parser.add_argument("--min-dev-speakers", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260414)
    parser.add_argument("--base-train-manifest", default="")
    return parser.parse_args()


def _scan_audio_rows(
    *,
    root: Path,
    dataset_name: str,
    source_dataset: str,
    speaker_prefix: str,
    speaker_source: str,
    speaker_level: int,
    filename_speaker_separator: str,
    audio_suffixes: tuple[str, ...],
    excluded_parts: set[str],
    channel: str,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen_utterances: Counter[str] = Counter()
    for audio_path in sorted(path for path in root.rglob("*") if path.is_file()):
        if audio_path.suffix.lower() not in audio_suffixes:
            continue
        relative = audio_path.relative_to(root)
        if any(part in excluded_parts for part in relative.parts):
            continue
        raw_speaker_id = _speaker_id_from_relative_path(
            relative,
            speaker_source=speaker_source,
            speaker_level=speaker_level,
            filename_speaker_separator=filename_speaker_separator,
        )
        if not raw_speaker_id:
            continue
        speaker_id = f"{speaker_prefix}{raw_speaker_id}"
        utterance_key = f"{speaker_id}:{relative.with_suffix('').as_posix()}"
        duplicate_index = seen_utterances[utterance_key]
        seen_utterances[utterance_key] += 1
        utterance_id = (
            utterance_key if duplicate_index == 0 else f"{utterance_key}:{duplicate_index}"
        )
        rows.append(
            {
                "schema_version": "kryptonite.manifest.v1",
                "record_type": "utterance",
                "dataset": dataset_name,
                "source_dataset": source_dataset,
                "speaker_id": speaker_id,
                "utterance_id": utterance_id,
                "split": "external",
                "audio_path": audio_path.as_posix(),
                "channel": channel,
            }
        )
    rows.sort(key=lambda row: (row["speaker_id"], row["audio_path"]))
    return rows


def _speaker_id_from_relative_path(
    relative: Path,
    *,
    speaker_source: str,
    speaker_level: int,
    filename_speaker_separator: str,
) -> str:
    if speaker_source == "filename-prefix":
        stem = relative.stem
        if filename_speaker_separator and filename_speaker_separator in stem:
            return stem.split(filename_speaker_separator, 1)[0]
        return stem
    if speaker_source == "path-part":
        if speaker_level < 0 or speaker_level >= len(relative.parts) - 1:
            return ""
        return relative.parts[speaker_level]
    raise ValueError(f"unknown speaker_source: {speaker_source}")


def _split_speakers(
    rows: list[dict[str, Any]],
    *,
    dev_speaker_fraction: float,
    seed: int,
    min_dev_speakers: int,
) -> tuple[set[str], set[str]]:
    speakers = sorted({str(row["speaker_id"]) for row in rows})
    if len(speakers) < 2 or dev_speaker_fraction <= 0.0:
        return set(speakers), set()
    target_dev = round(len(speakers) * dev_speaker_fraction)
    target_dev = max(1, min_dev_speakers, target_dev)
    target_dev = min(target_dev, len(speakers) - 1)
    ranked = sorted(
        speakers,
        key=lambda speaker: hashlib.sha256(f"{seed}:{speaker}".encode()).hexdigest(),
    )
    dev_speakers = set(ranked[:target_dev])
    train_speakers = set(speakers) - dev_speakers
    return train_speakers, dev_speakers


def _write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_mixed_manifest(
    path: Path,
    *,
    base_train_manifest: Path,
    external_train_manifest: Path,
) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as output:
        for source in (base_train_manifest, external_train_manifest):
            for line in source.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    output.write(line.rstrip() + "\n")
                    count += 1
    return count


def _top_speakers(rows: list[dict[str, Any]], *, limit: int) -> list[dict[str, Any]]:
    counts = Counter(str(row["speaker_id"]) for row in rows)
    return [
        {"speaker_id": speaker_id, "row_count": count}
        for speaker_id, count in counts.most_common(limit)
    ]


if __name__ == "__main__":
    main()
