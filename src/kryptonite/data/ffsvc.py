"""FFSVC 2022 surrogate parsing and manifest preparation."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path

from kryptonite.deployment import resolve_project_path

from .ffsvc_support import (
    ManifestEntry,
    build_ffsvc_split_trial_coverage_summary,
    build_verification_utterance,
    format_manifest_split_summary_error,
    format_trial_coverage_error,
    format_verification_trial_summary_error,
    split_quarantined_ffsvc_entries,
)
from .manifest_artifacts import (
    build_file_artifact,
    inspect_wav_audio_file,
    write_manifest_inventory,
    write_tabular_artifact,
)
from .schema import ManifestRow
from .speaker_splits import (
    build_speaker_holdout_split,
    summarize_speaker_disjoint_entries,
)
from .verification_trials import build_balanced_verification_trials

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


@dataclass(slots=True)
class PreparedFfsvcArtifacts:
    manifests_root: str
    all_manifest_file: str
    train_manifest_file: str
    dev_manifest_file: str
    quarantine_manifest_file: str
    official_trials_file: str
    split_trials_file: str
    split_trial_summary_file: str
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
            "split_trial_summary_file": self.split_trial_summary_file,
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
    trial_target_per_bucket: int = 128,
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

    all_manifest = output_root / "all_manifest.jsonl"
    train_manifest = output_root / "train_manifest.jsonl"
    dev_manifest = output_root / "dev_manifest.jsonl"
    quarantine_manifest = output_root / "quarantine_manifest.jsonl"
    official_trials = output_root / "official_dev_trials.jsonl"
    split_trials = output_root / "speaker_disjoint_dev_trials.jsonl"
    split_trial_summary = output_root / "speaker_disjoint_dev_trials_summary.json"
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
    verification_utterances = [
        build_verification_utterance(entry) for entry in dev_entries if entry.get("audio_path")
    ]
    split_trial_entries, verification_trial_summary = build_balanced_verification_trials(
        utterances=verification_utterances,
        seed=seed,
        target_trials_per_bucket=trial_target_per_bucket,
    )
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
        field_order=(
            "label",
            "left_audio",
            "right_audio",
            "duration_bucket",
            "domain_relation",
            "channel_relation",
            "channel_mismatch",
            "left_speaker_id",
            "right_speaker_id",
            "left_domain",
            "right_domain",
            "left_channel",
            "right_channel",
        ),
    )
    manifest_summary = summarize_speaker_disjoint_entries(
        rows=entries,
        train_speakers=train_speakers,
        dev_speakers=dev_speakers,
    )
    if not manifest_summary.is_valid:
        raise ValueError(format_manifest_split_summary_error(manifest_summary))

    trial_coverage = build_ffsvc_split_trial_coverage_summary(
        dev_entries=dev_entries,
        split_trial_entries=split_trial_entries,
        official_trial_count=len(official_trial_entries),
        dev_speakers=dev_speakers,
    )
    if not trial_coverage.is_valid:
        raise ValueError(format_trial_coverage_error(trial_coverage))
    if not verification_trial_summary.is_valid:
        raise ValueError(format_verification_trial_summary_error(verification_trial_summary))

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
    split_trial_summary.write_text(
        json.dumps(verification_trial_summary.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    speaker_split_summary.write_text(
        json.dumps(
            {
                "seed": seed,
                "selection_strategy": "speaker_shuffle",
                "trial_target_per_bucket": trial_target_per_bucket,
                "requested_dev_speaker_count": dev_speaker_count,
                "train_speakers": sorted(train_speakers),
                "dev_speakers": sorted(dev_speakers),
                "manifest_summary": manifest_summary.to_dict(),
                "trial_coverage": trial_coverage.to_dict(),
                "trial_balance": verification_trial_summary.to_dict(),
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
                name="speaker_disjoint_dev_trials_summary",
                kind="metadata",
                path=split_trial_summary,
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
        split_trial_summary_file=str(split_trial_summary),
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


def _relative_to_project(path: Path, project_root: str) -> str:
    root = resolve_project_path(project_root, ".")
    return str(path.resolve().relative_to(root))
