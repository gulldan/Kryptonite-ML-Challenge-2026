"""FFSVC-specific duplicate policy and verification-trial support helpers."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from .speaker_splits import SpeakerDisjointManifestSummary
from .verification_trials import VerificationTrialSummary, VerificationUtterance

ManifestEntry = dict[str, object]


@dataclass(frozen=True, slots=True)
class FfsvcDuplicateResolution:
    group_id: str
    canonical_utterance_id: str
    duplicate_utterance_ids: tuple[str, ...]
    reason: str


@dataclass(frozen=True, slots=True)
class FfsvcSplitTrialCoverageSummary:
    official_trial_count: int
    speaker_disjoint_trial_count: int
    positive_trial_count: int
    negative_trial_count: int
    negative_trial_requirement_enabled: bool
    covered_dev_speaker_count: int
    missing_dev_speakers: tuple[str, ...]
    covered_dev_audio_count: int
    uncovered_dev_audio_count: int
    missing_trial_reference_count: int

    @property
    def is_valid(self) -> bool:
        return (
            self.speaker_disjoint_trial_count > 0
            and self.positive_trial_count > 0
            and (not self.negative_trial_requirement_enabled or self.negative_trial_count > 0)
            and not self.missing_dev_speakers
            and self.missing_trial_reference_count == 0
        )

    def to_dict(self) -> dict[str, object]:
        return {
            "official_trial_count": self.official_trial_count,
            "speaker_disjoint_trial_count": self.speaker_disjoint_trial_count,
            "positive_trial_count": self.positive_trial_count,
            "negative_trial_count": self.negative_trial_count,
            "negative_trial_requirement_enabled": self.negative_trial_requirement_enabled,
            "covered_dev_speaker_count": self.covered_dev_speaker_count,
            "missing_dev_speakers": list(self.missing_dev_speakers),
            "covered_dev_audio_count": self.covered_dev_audio_count,
            "uncovered_dev_audio_count": self.uncovered_dev_audio_count,
            "missing_trial_reference_count": self.missing_trial_reference_count,
            "is_valid": self.is_valid,
        }


KNOWN_FFSVC_DUPLICATE_RESOLUTIONS: Final[tuple[FfsvcDuplicateResolution, ...]] = (
    FfsvcDuplicateResolution(
        group_id="s0449-i1m-session-1-0211-0212",
        canonical_utterance_id="ffsvc22_dev_043388",
        duplicate_utterance_ids=("ffsvc22_dev_002177",),
        reason=(
            "Confirmed byte-identical upstream duplicate in the FFSVC 2022 surrogate bundle; "
            "keep the earlier metadata row and quarantine the duplicate training example."
        ),
    ),
    FfsvcDuplicateResolution(
        group_id="s0449-pad5m-session-1-0233-0234",
        canonical_utterance_id="ffsvc22_dev_063782",
        duplicate_utterance_ids=("ffsvc22_dev_063743",),
        reason=(
            "Confirmed byte-identical upstream duplicate in the FFSVC 2022 surrogate bundle; "
            "keep the earlier metadata row and quarantine the duplicate training example."
        ),
    ),
)


def split_quarantined_ffsvc_entries(
    entries: list[ManifestEntry],
) -> tuple[list[ManifestEntry], list[ManifestEntry]]:
    resolution_by_duplicate_id = _build_duplicate_resolution_index(
        KNOWN_FFSVC_DUPLICATE_RESOLUTIONS
    )
    entry_by_utterance_id = {str(entry["utterance_id"]): entry for entry in entries}
    for resolution in KNOWN_FFSVC_DUPLICATE_RESOLUTIONS:
        group_utterance_ids = (
            resolution.canonical_utterance_id,
            *resolution.duplicate_utterance_ids,
        )
        present_utterance_ids = [
            utterance_id
            for utterance_id in group_utterance_ids
            if utterance_id in entry_by_utterance_id
        ]
        if present_utterance_ids and len(present_utterance_ids) != len(group_utterance_ids):
            missing_utterance_ids = sorted(
                utterance_id
                for utterance_id in group_utterance_ids
                if utterance_id not in entry_by_utterance_id
            )
            raise ValueError(
                "Known FFSVC duplicate policy references utterance ids that are partially "
                f"missing from the prepared metadata rows for group {resolution.group_id!r}: "
                f"{missing_utterance_ids}"
            )

    active_entries: list[ManifestEntry] = []
    quarantined_entries: list[ManifestEntry] = []
    for entry in entries:
        utterance_id = str(entry["utterance_id"])
        resolution = resolution_by_duplicate_id.get(utterance_id)
        if resolution is None:
            active_entries.append(entry)
            continue

        canonical_entry = entry_by_utterance_id[resolution.canonical_utterance_id]
        quarantined_entries.append(
            {
                **entry,
                "quality_issue_code": "duplicate_audio_content",
                "duplicate_group_id": resolution.group_id,
                "duplicate_policy": "quarantine",
                "duplicate_reason": resolution.reason,
                "duplicate_canonical_utterance_id": resolution.canonical_utterance_id,
                "duplicate_canonical_original_name": str(canonical_entry["original_name"]),
            }
        )
    return active_entries, quarantined_entries


def build_verification_utterance(entry: ManifestEntry) -> VerificationUtterance:
    audio_path = str(entry["audio_path"])
    capture_condition = str(entry.get("capture_condition") or "")
    return VerificationUtterance(
        audio_basename=Path(audio_path).name,
        speaker_id=str(entry["speaker_id"]),
        duration_seconds=_coerce_float(entry.get("duration_seconds")),
        domain=str(entry.get("source_prefix") or "unknown"),
        channel=_extract_channel_label(capture_condition),
    )


def build_ffsvc_split_trial_coverage_summary(
    *,
    dev_entries: list[ManifestEntry],
    split_trial_entries: list[dict[str, object]],
    official_trial_count: int,
    dev_speakers: set[str],
) -> FfsvcSplitTrialCoverageSummary:
    filename_to_speaker: dict[str, str] = {
        Path(str(entry["audio_path"])).name: str(entry["speaker_id"]) for entry in dev_entries
    }
    covered_audio: set[str] = set()
    covered_speakers: set[str] = set()
    positive_trial_count = 0
    negative_trial_count = 0
    missing_trial_reference_count = 0

    for trial in split_trial_entries:
        label = _coerce_trial_label(trial["label"])
        if label == 1:
            positive_trial_count += 1
        elif label == 0:
            negative_trial_count += 1
        else:
            raise ValueError(f"Unexpected FFSVC trial label: {label}")

        for field_name in ("left_audio", "right_audio"):
            filename = str(trial[field_name])
            speaker_id = filename_to_speaker.get(filename)
            if speaker_id is None:
                missing_trial_reference_count += 1
                continue
            covered_audio.add(filename)
            covered_speakers.add(speaker_id)

    missing_dev_speakers = tuple(sorted(set(dev_speakers) - covered_speakers))
    return FfsvcSplitTrialCoverageSummary(
        official_trial_count=official_trial_count,
        speaker_disjoint_trial_count=len(split_trial_entries),
        positive_trial_count=positive_trial_count,
        negative_trial_count=negative_trial_count,
        negative_trial_requirement_enabled=len(dev_speakers) > 1,
        covered_dev_speaker_count=len(covered_speakers),
        missing_dev_speakers=missing_dev_speakers,
        covered_dev_audio_count=len(covered_audio),
        uncovered_dev_audio_count=len(filename_to_speaker) - len(covered_audio),
        missing_trial_reference_count=missing_trial_reference_count,
    )


def format_manifest_split_summary_error(summary: SpeakerDisjointManifestSummary) -> str:
    issues: list[str] = []
    if summary.missing_speaker_count:
        issues.append(f"missing speaker_id rows={summary.missing_speaker_count}")
    if summary.unexpected_speakers:
        issues.append("unexpected speakers=" + ",".join(summary.unexpected_speakers[:10]))
    if summary.overlapping_speakers:
        issues.append("overlapping speakers=" + ",".join(summary.overlapping_speakers[:10]))
    if summary.split_mismatch_count:
        issues.append(f"split mismatches={summary.split_mismatch_count}")
    joined = "; ".join(issues) if issues else "unknown split validation failure"
    return f"Prepared manifests are not speaker-disjoint: {joined}"


def format_trial_coverage_error(summary: FfsvcSplitTrialCoverageSummary) -> str:
    issues: list[str] = []
    if summary.speaker_disjoint_trial_count == 0:
        issues.append("speaker_disjoint_dev_trials is empty")
    if summary.positive_trial_count == 0:
        issues.append("missing positive dev-only trials")
    if summary.negative_trial_requirement_enabled and summary.negative_trial_count == 0:
        issues.append("missing negative dev-only trials")
    if summary.missing_dev_speakers:
        issues.append(
            "dev speakers without strict-dev trial coverage="
            + ",".join(summary.missing_dev_speakers[:10])
        )
    if summary.missing_trial_reference_count:
        issues.append(
            f"missing dev-manifest trial references={summary.missing_trial_reference_count}"
        )
    joined = "; ".join(issues) if issues else "unknown strict-dev validation failure"
    return f"Prepared strict-dev trials are not threshold-tuning ready: {joined}"


def format_verification_trial_summary_error(summary: VerificationTrialSummary) -> str:
    issues: list[str] = []
    if summary.label_counts.get("positive", 0) == 0:
        issues.append("missing positive verification trials")
    if summary.label_counts.get("negative", 0) == 0:
        issues.append("missing negative verification trials")
    if summary.bucket_shortfalls:
        first_key = next(iter(sorted(summary.bucket_shortfalls)))
        issues.append(f"bucket shortfall {first_key}={summary.bucket_shortfalls[first_key]}")
    if summary.missing_speakers:
        issues.append(
            "dev speakers without generated trial coverage="
            + ",".join(summary.missing_speakers[:10])
        )
    if summary.duplicate_pair_count:
        issues.append(f"duplicate generated pairs={summary.duplicate_pair_count}")
    if summary.self_pair_count:
        issues.append(f"self-pairs={summary.self_pair_count}")
    if summary.label_balance_gap > 1:
        issues.append(f"label balance gap={summary.label_balance_gap}")
    joined = "; ".join(issues) if issues else "unknown verification-trial validation failure"
    return f"Prepared verification trials are not balanced-eval ready: {joined}"


def _build_duplicate_resolution_index(
    resolutions: tuple[FfsvcDuplicateResolution, ...],
) -> dict[str, FfsvcDuplicateResolution]:
    index: dict[str, FfsvcDuplicateResolution] = {}
    for resolution in resolutions:
        for duplicate_utterance_id in resolution.duplicate_utterance_ids:
            if duplicate_utterance_id == resolution.canonical_utterance_id:
                raise ValueError(
                    f"Duplicate resolution group {resolution.group_id!r} reuses the canonical "
                    f"utterance {resolution.canonical_utterance_id!r} as a quarantined duplicate."
                )
            if duplicate_utterance_id in index:
                raise ValueError(
                    f"Duplicate utterance id {duplicate_utterance_id!r} is assigned to multiple "
                    "FFSVC duplicate-resolution groups."
                )
            index[duplicate_utterance_id] = resolution
    return index


def _extract_channel_label(capture_condition: str) -> str:
    match = re.match(
        r"^(?P<channel>[A-Z]+(?:[LR])?)(?P<distance>-?\d+(?:\.\d+)?M)$",
        capture_condition,
    )
    if match is None:
        return capture_condition or "unknown"
    return str(match.group("channel"))


def _coerce_trial_label(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Unexpected trial label type: {type(value)!r}")


def _coerce_float(value: object | None) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unexpected float-like value: {type(value)!r}")
