"""FFSVC 2022 surrogate parsing and manifest preparation."""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Final

from kryptonite.deployment import resolve_project_path

from .schema import ManifestRow

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
            "utterance_count": self.utterance_count,
            "source_utterance_count": self.source_utterance_count,
            "quarantined_utterance_count": self.quarantined_utterance_count,
            "train_speaker_count": self.train_speaker_count,
            "dev_speaker_count": self.dev_speaker_count,
            "official_trial_count": self.official_trial_count,
            "split_trial_count": self.split_trial_count,
        }


ManifestEntry = dict[str, object]

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

    _write_jsonl(all_manifest, entries)
    _write_jsonl(train_manifest, [entry for entry in entries if entry["split"] == "train"])
    _write_jsonl(dev_manifest, [entry for entry in entries if entry["split"] == "dev"])
    _write_jsonl(quarantine_manifest, quarantined_entries)

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
    _write_jsonl(official_trials, official_trial_entries)
    _write_jsonl(split_trials, split_trial_entries)
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

    return PreparedFfsvcArtifacts(
        manifests_root=str(output_root),
        all_manifest_file=str(all_manifest),
        train_manifest_file=str(train_manifest),
        dev_manifest_file=str(dev_manifest),
        quarantine_manifest_file=str(quarantine_manifest),
        official_trials_file=str(official_trials),
        split_trials_file=str(split_trials),
        speaker_split_file=str(speaker_split),
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
    speakers = sorted({utterance.speaker_id for utterance in utterances})
    if dev_speaker_count <= 0 or dev_speaker_count >= len(speakers):
        raise ValueError(
            f"dev_speaker_count must be between 1 and {len(speakers) - 1}, got {dev_speaker_count}"
        )
    shuffled = list(speakers)
    random.Random(seed).shuffle(shuffled)
    dev_speakers = set(shuffled[:dev_speaker_count])
    train_speakers = set(shuffled[dev_speaker_count:])
    return train_speakers, dev_speakers


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


def _write_jsonl(path: Path, entries: list[ManifestEntry]) -> None:
    path.write_text("".join(json.dumps(entry, sort_keys=True) + "\n" for entry in entries))


def _relative_to_project(path: Path, project_root: str) -> str:
    root = resolve_project_path(project_root, ".")
    return str(path.resolve().relative_to(root))
