"""Reusable helpers for speaker-disjoint train/dev splits."""

from __future__ import annotations

import random
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeakerDisjointManifestSummary:
    train_speaker_count: int
    dev_speaker_count: int
    train_utterance_count: int
    dev_utterance_count: int
    split_row_counts: dict[str, int]
    split_speaker_counts: dict[str, int]
    missing_speaker_count: int
    unexpected_speakers: tuple[str, ...]
    overlapping_speakers: tuple[str, ...]
    split_mismatch_count: int

    @property
    def is_valid(self) -> bool:
        return (
            self.missing_speaker_count == 0
            and not self.unexpected_speakers
            and not self.overlapping_speakers
            and self.split_mismatch_count == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "train_speaker_count": self.train_speaker_count,
            "dev_speaker_count": self.dev_speaker_count,
            "train_utterance_count": self.train_utterance_count,
            "dev_utterance_count": self.dev_utterance_count,
            "split_row_counts": dict(self.split_row_counts),
            "split_speaker_counts": dict(self.split_speaker_counts),
            "missing_speaker_count": self.missing_speaker_count,
            "unexpected_speakers": list(self.unexpected_speakers),
            "overlapping_speakers": list(self.overlapping_speakers),
            "split_mismatch_count": self.split_mismatch_count,
            "is_valid": self.is_valid,
        }


def build_speaker_holdout_split(
    *,
    speaker_ids: Iterable[str],
    dev_speaker_count: int,
    seed: int,
) -> tuple[set[str], set[str]]:
    speakers = sorted({speaker_id for speaker_id in speaker_ids if speaker_id})
    if dev_speaker_count <= 0 or dev_speaker_count >= len(speakers):
        raise ValueError(
            f"dev_speaker_count must be between 1 and {len(speakers) - 1}, got {dev_speaker_count}"
        )

    shuffled = list(speakers)
    random.Random(seed).shuffle(shuffled)
    dev_speakers = set(shuffled[:dev_speaker_count])
    train_speakers = set(shuffled[dev_speaker_count:])
    return train_speakers, dev_speakers


def summarize_speaker_disjoint_entries(
    *,
    rows: Sequence[Mapping[str, object]],
    train_speakers: Iterable[str],
    dev_speakers: Iterable[str],
    speaker_field: str = "speaker_id",
    split_field: str = "split",
) -> SpeakerDisjointManifestSummary:
    train_speaker_set = set(train_speakers)
    dev_speaker_set = set(dev_speakers)
    overlap = train_speaker_set & dev_speaker_set
    if overlap:
        overlap_display = ", ".join(sorted(overlap))
        raise ValueError(
            f"train_speakers and dev_speakers must be disjoint, got overlap: {overlap_display}"
        )

    split_row_counts: Counter[str] = Counter()
    split_speakers: defaultdict[str, set[str]] = defaultdict(set)
    observed_speaker_splits: defaultdict[str, set[str]] = defaultdict(set)
    missing_speaker_count = 0
    unexpected_speakers: set[str] = set()
    split_mismatch_count = 0

    for row in rows:
        split_name = _coerce_string(row.get(split_field)) or "unknown"
        split_row_counts[split_name] += 1

        speaker_id = _coerce_string(row.get(speaker_field))
        if speaker_id is None:
            missing_speaker_count += 1
            continue

        split_speakers[split_name].add(speaker_id)
        expected_split: str | None
        if speaker_id in train_speaker_set:
            expected_split = "train"
        elif speaker_id in dev_speaker_set:
            expected_split = "dev"
        else:
            expected_split = None
            unexpected_speakers.add(speaker_id)

        if expected_split is None:
            continue
        observed_speaker_splits[speaker_id].add(split_name)
        if split_name != expected_split:
            split_mismatch_count += 1

    overlapping_speakers = tuple(
        sorted(
            speaker_id
            for speaker_id, split_names in observed_speaker_splits.items()
            if len(split_names & {"train", "dev"}) > 1
        )
    )

    ordered_split_row_counts = _sorted_counts(split_row_counts)
    ordered_split_speaker_counts = _sorted_counts(
        Counter({split_name: len(speakers) for split_name, speakers in split_speakers.items()})
    )
    return SpeakerDisjointManifestSummary(
        train_speaker_count=len(train_speaker_set),
        dev_speaker_count=len(dev_speaker_set),
        train_utterance_count=ordered_split_row_counts.get("train", 0),
        dev_utterance_count=ordered_split_row_counts.get("dev", 0),
        split_row_counts=ordered_split_row_counts,
        split_speaker_counts=ordered_split_speaker_counts,
        missing_speaker_count=missing_speaker_count,
        unexpected_speakers=tuple(sorted(unexpected_speakers)),
        overlapping_speakers=overlapping_speakers,
        split_mismatch_count=split_mismatch_count,
    )


def _sorted_counts(counter: Counter[str]) -> dict[str, int]:
    return {key: counter[key] for key in sorted(counter)}


def _coerce_string(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)
