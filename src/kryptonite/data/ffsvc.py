"""FFSVC 2022 surrogate parsing and manifest preparation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from kryptonite.deployment import resolve_project_path

from .manifest_artifacts import (
    build_file_artifact,
    inspect_wav_audio_file,
    write_manifest_inventory,
    write_tabular_artifact,
)
from .schema import ManifestRow
from .speaker_splits import (
    SpeakerDisjointManifestSummary,
    build_speaker_holdout_split,
    summarize_speaker_disjoint_entries,
)

ORIGINAL_NAME_PATTERN = re.compile(
    r"^(?P<source_prefix>[A-Z])(?P<speaker_id>\d{4})_"
    r"(?P<condition_core>[A-Z0-9.\-]+)_"
    r"(?P<session_index>\d+)_"
    r"(?P<utterance_index>\d+)_"
    r"(?P<pace>[a-z]+)$"
)


@dataclass(frozen=True, slots=True)
class FfsvcUtterance:
    original_name: str
    ffsvc_name: str
    speaker_id: str
    source_prefix: str
    capture_condition: str
    session_index: str
    utterance_index: str
    pace: str

    @property
    def audio_filename(self) -> str:
        return f"{self.ffsvc_name}.wav"


@dataclass(frozen=True, slots=True)
class FfsvcTrial:
    label: int
    left_filename: str
    right_filename: str


@dataclass(frozen=True, slots=True)
class FfsvcDuplicateResolution:
    group_id: str
    canonical_utterance_id: str
    duplicate_utterance_ids: tuple[str, ...]
    reason: str


@dataclass(slots=True)
class PreparedFfsvcArtifacts:
    manifests_root: str
    all_manifest_file: str
    train_manifest_file: str
    dev_manifest_file: str
    quarantine_manifest_file: str
    official_trials_file: str
    split_trials_file: str
    speaker_split_file: str
    speaker_split_summary_file: str
    manifest_inventory_file: str
    utterance_count: int
    source_utterance_count: int
    quarantined_utterance_count: int
    train_speaker_count: int
    dev_speaker_count: int
    official_trial_count: int
    split_trial_count: int

    def to_dict(self) -> dict[str, object]:
        return {
            "manifests_root": self.manifests_root,
            "all_manifest_file": self.all_manifest_file,
            "train_manifest_file": self.train_manifest_file,
            "dev_manifest_file": self.dev_manifest_file,
            "quarantine_manifest_file": self.quarantine_manifest_file,
            "official_trials_file": self.official_trials_file,
            "split_trials_file": self.split_trials_file,
            "speaker_split_file": self.speaker_split_file,
            "speaker_split_summary_file": self.speaker_split_summary_file,
            "manifest_inventory_file": self.manifest_inventory_file,
            "utterance_count": self.utterance_count,
            "source_utterance_count": self.source_utterance_count,
            "quarantined_utterance_count": self.quarantined_utterance_count,
            "train_speaker_count": self.train_speaker_count,
            "dev_speaker_count": self.dev_speaker_count,
            "official_trial_count": self.official_trial_count,
            "split_trial_count": self.split_trial_count,
        }


ManifestEntry = dict[str, object]


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


KNOWN_FFSVC_DUPLICATE_RESOLUTION_BY_DUPLICATE_ID: Final[dict[str, FfsvcDuplicateResolution]] = (
    _build_duplicate_resolution_index(KNOWN_FFSVC_DUPLICATE_RESOLUTIONS)
)


def load_ffsvc_dev_metadata(path: Path) -> list[FfsvcUtterance]:
    rows = path.read_text().splitlines()
    if not rows:
        raise ValueError(f"Metadata file is empty: {path}")
    header = rows[0].split()
    if header != ["Original_Name", "FFSVC2022_Name"]:
        raise ValueError(f"Unexpected metadata header in {path}: {rows[0]!r}")

    utterances: list[FfsvcUtterance] = []
    for line in rows[1:]:
        if not line.strip():
            continue
        original_name, ffsvc_name = line.split()
        utterances.append(parse_ffsvc_utterance(original_name=original_name, ffsvc_name=ffsvc_name))
    return utterances


def load_ffsvc_trials(path: Path) -> list[FfsvcTrial]:
    trials: list[FfsvcTrial] = []
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        raw_label, left_filename, right_filename = line.split()
        trials.append(
            FfsvcTrial(
                label=int(raw_label),
                left_filename=left_filename,
                right_filename=right_filename,
            )
        )
    return trials


def parse_ffsvc_utterance(*, original_name: str, ffsvc_name: str) -> FfsvcUtterance:
    match = ORIGINAL_NAME_PATTERN.match(original_name)
    if match is None:
        raise ValueError(f"Unexpected FFSVC original name format: {original_name!r}")

    speaker_id = str(match.group("speaker_id"))
    condition_core = str(match.group("condition_core"))
    candidate_prefixes: list[str] = list(
        {speaker_id, speaker_id[-3:], speaker_id.lstrip("0") or "0"}
    )
    candidate_prefixes.sort(key=len, reverse=True)
    prefix_length = next(
        (
            len(candidate)
            for candidate in candidate_prefixes
            if condition_core.startswith(candidate)
        ),
        None,
    )
    if prefix_length is None:
        raise ValueError(
            f"Condition token does not repeat speaker id for {original_name!r}: {condition_core!r}"
        )

    return FfsvcUtterance(
        original_name=original_name,
        ffsvc_name=ffsvc_name,
        speaker_id=speaker_id,
        source_prefix=match.group("source_prefix"),
        capture_condition=condition_core[prefix_length:],
        session_index=match.group("session_index"),
        utterance_index=match.group("utterance_index"),
        pace=match.group("pace"),
    )


def prepare_ffsvc2022_surrogate(
    *,
    project_root: str,
    dataset_root: str,
    manifests_root: str,
    metadata_relpath: str = "ffsvc2022-surrogate/metadata/dev_meta_list.txt",
    trials_relpath: str = "ffsvc2022-surrogate/metadata/trials_dev_keys.txt",
    audio_root_relpath: str = "ffsvc2022-surrogate/raw/dev",
    output_relpath: str = "ffsvc2022-surrogate",
    dev_speaker_count: int = 6,
    seed: int = 42,
) -> PreparedFfsvcArtifacts:
    dataset_root_path = resolve_project_path(project_root, dataset_root)
    manifests_root_path = resolve_project_path(project_root, manifests_root)
    metadata_path = dataset_root_path / metadata_relpath
    trials_path = dataset_root_path / trials_relpath
    audio_root_path = dataset_root_path / audio_root_relpath
    output_root = manifests_root_path / output_relpath
    output_root.mkdir(parents=True, exist_ok=True)

    utterances = load_ffsvc_dev_metadata(metadata_path)
    trials = load_ffsvc_trials(trials_path)
    audio_map = build_audio_map(audio_root_path)

    source_entries: list[ManifestEntry] = []
    train_speakers, dev_speakers = build_speaker_splits(
        utterances=utterances,
        dev_speaker_count=dev_speaker_count,
        seed=seed,
    )
    for utterance in utterances:
        audio_filename = utterance.audio_filename
        audio_path = audio_map.get(audio_filename)
        if audio_path is None:
            raise FileNotFoundError(
                f"Audio file {audio_filename!r} is missing under extracted root {audio_root_path}"
            )
        audio_metadata = inspect_wav_audio_file(audio_path)
        split = "dev" if utterance.speaker_id in dev_speakers else "train"
        source_entries.append(
            ManifestRow(
                dataset="ffsvc2022-surrogate",
                source_dataset="ffsvc2022",
                speaker_id=utterance.speaker_id,
                utterance_id=utterance.ffsvc_name,
                session_id=f"{utterance.speaker_id}:{utterance.session_index}",
                split=split,
                audio_path=_relative_to_project(audio_path, project_root),
                duration_seconds=audio_metadata.duration_seconds,
                sample_rate_hz=audio_metadata.sample_rate_hz,
                num_channels=audio_metadata.num_channels,
            ).to_dict(
                extra_fields={
                    "original_name": utterance.original_name,
                    "source_prefix": utterance.source_prefix,
                    "capture_condition": utterance.capture_condition,
                    "session_index": utterance.session_index,
                    "utterance_index": utterance.utterance_index,
                    "pace": utterance.pace,
                }
            )
        )

    entries, quarantined_entries = split_quarantined_ffsvc_entries(source_entries)
    train_entries = [entry for entry in entries if entry["split"] == "train"]
    dev_entries = [entry for entry in entries if entry["split"] == "dev"]
    split_by_filename: dict[str, str] = {
        Path(str(entry["audio_path"])).name: str(entry["split"]) for entry in entries
    }

    all_manifest = output_root / "all_manifest.jsonl"
    train_manifest = output_root / "train_manifest.jsonl"
    dev_manifest = output_root / "dev_manifest.jsonl"
    quarantine_manifest = output_root / "quarantine_manifest.jsonl"
    official_trials = output_root / "official_dev_trials.jsonl"
    split_trials = output_root / "speaker_disjoint_dev_trials.jsonl"
    speaker_split = output_root / "speaker_splits.json"
    speaker_split_summary = output_root / "speaker_split_summary.json"
    manifest_inventory = output_root / "manifest_inventory.json"

    all_manifest_artifact = write_tabular_artifact(
        name="all_manifest",
        kind="data_manifest",
        rows=entries,
        jsonl_path=all_manifest,
        project_root=project_root,
    )
    train_manifest_artifact = write_tabular_artifact(
        name="train_manifest",
        kind="data_manifest",
        rows=train_entries,
        jsonl_path=train_manifest,
        project_root=project_root,
    )
    dev_manifest_artifact = write_tabular_artifact(
        name="dev_manifest",
        kind="data_manifest",
        rows=dev_entries,
        jsonl_path=dev_manifest,
        project_root=project_root,
    )
    quarantine_manifest_artifact = write_tabular_artifact(
        name="quarantine_manifest",
        kind="data_manifest",
        rows=quarantined_entries,
        jsonl_path=quarantine_manifest,
        project_root=project_root,
    )

    official_trial_entries: list[dict[str, object]] = [
        {
            "label": trial.label,
            "left_audio": trial.left_filename,
            "right_audio": trial.right_filename,
        }
        for trial in trials
    ]
    split_trial_entries: list[dict[str, object]] = [
        {
            "label": trial.label,
            "left_audio": trial.left_filename,
            "right_audio": trial.right_filename,
        }
        for trial in trials
        if split_by_filename.get(trial.left_filename) == "dev"
        and split_by_filename.get(trial.right_filename) == "dev"
    ]
    official_trials_artifact = write_tabular_artifact(
        name="official_dev_trials",
        kind="trial_list",
        rows=official_trial_entries,
        jsonl_path=official_trials,
        project_root=project_root,
        field_order=("label", "left_audio", "right_audio"),
    )
    split_trials_artifact = write_tabular_artifact(
        name="speaker_disjoint_dev_trials",
        kind="trial_list",
        rows=split_trial_entries,
        jsonl_path=split_trials,
        project_root=project_root,
        field_order=("label", "left_audio", "right_audio"),
    )
    manifest_summary = summarize_speaker_disjoint_entries(
        rows=entries,
        train_speakers=train_speakers,
        dev_speakers=dev_speakers,
    )
    if not manifest_summary.is_valid:
        raise ValueError(_format_manifest_split_summary_error(manifest_summary))

    trial_coverage = _build_ffsvc_split_trial_coverage_summary(
        dev_entries=dev_entries,
        split_trial_entries=split_trial_entries,
        official_trial_count=len(official_trial_entries),
        dev_speakers=dev_speakers,
    )
    if not trial_coverage.is_valid:
        raise ValueError(_format_trial_coverage_error(trial_coverage))

    speaker_split.write_text(
        json.dumps(
            {
                "seed": seed,
                "train_speakers": sorted(train_speakers),
                "dev_speakers": sorted(dev_speakers),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    speaker_split_summary.write_text(
        json.dumps(
            {
                "seed": seed,
                "selection_strategy": "speaker_shuffle",
                "requested_dev_speaker_count": dev_speaker_count,
                "train_speakers": sorted(train_speakers),
                "dev_speakers": sorted(dev_speakers),
                "manifest_summary": manifest_summary.to_dict(),
                "trial_coverage": trial_coverage.to_dict(),
            },
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    write_manifest_inventory(
        dataset="ffsvc2022-surrogate",
        inventory_path=manifest_inventory,
        project_root=project_root,
        manifest_tables=(
            all_manifest_artifact,
            train_manifest_artifact,
            dev_manifest_artifact,
            quarantine_manifest_artifact,
        ),
        auxiliary_tables=(official_trials_artifact, split_trials_artifact),
        auxiliary_files=(
            build_file_artifact(
                name="speaker_splits",
                kind="metadata",
                path=speaker_split,
                project_root=project_root,
            ),
            build_file_artifact(
                name="speaker_split_summary",
                kind="metadata",
                path=speaker_split_summary,
                project_root=project_root,
            ),
        ),
    )

    return PreparedFfsvcArtifacts(
        manifests_root=str(output_root),
        all_manifest_file=str(all_manifest),
        train_manifest_file=str(train_manifest),
        dev_manifest_file=str(dev_manifest),
        quarantine_manifest_file=str(quarantine_manifest),
        official_trials_file=str(official_trials),
        split_trials_file=str(split_trials),
        speaker_split_file=str(speaker_split),
        speaker_split_summary_file=str(speaker_split_summary),
        manifest_inventory_file=str(manifest_inventory),
        utterance_count=len(entries),
        source_utterance_count=len(source_entries),
        quarantined_utterance_count=len(quarantined_entries),
        train_speaker_count=len(train_speakers),
        dev_speaker_count=len(dev_speakers),
        official_trial_count=len(official_trial_entries),
        split_trial_count=len(split_trial_entries),
    )


def build_speaker_splits(
    *,
    utterances: list[FfsvcUtterance],
    dev_speaker_count: int,
    seed: int,
) -> tuple[set[str], set[str]]:
    return build_speaker_holdout_split(
        speaker_ids=(utterance.speaker_id for utterance in utterances),
        dev_speaker_count=dev_speaker_count,
        seed=seed,
    )


def build_audio_map(audio_root: Path) -> dict[str, Path]:
    if not audio_root.exists():
        raise FileNotFoundError(f"Extracted audio root does not exist: {audio_root}")
    mapping: dict[str, Path] = {}
    for audio_path in audio_root.rglob("*.wav"):
        mapping[audio_path.name] = audio_path
    if not mapping:
        raise FileNotFoundError(f"No WAV files found under extracted audio root: {audio_root}")
    return mapping


def split_quarantined_ffsvc_entries(
    entries: list[ManifestEntry],
) -> tuple[list[ManifestEntry], list[ManifestEntry]]:
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
        resolution = KNOWN_FFSVC_DUPLICATE_RESOLUTION_BY_DUPLICATE_ID.get(utterance_id)
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


def _relative_to_project(path: Path, project_root: str) -> str:
    root = resolve_project_path(project_root, ".")
    return str(path.resolve().relative_to(root))


def _build_ffsvc_split_trial_coverage_summary(
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


def _format_manifest_split_summary_error(summary: SpeakerDisjointManifestSummary) -> str:
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


def _format_trial_coverage_error(summary: FfsvcSplitTrialCoverageSummary) -> str:
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


def _coerce_trial_label(value: object) -> int:
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        return int(value)
    raise TypeError(f"Unexpected trial label type: {type(value)!r}")
