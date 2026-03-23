"""Manifest-driven audio normalization and quarantine workflow."""

from __future__ import annotations

import json
import shutil
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from kryptonite.config import NormalizationConfig
from kryptonite.deployment import resolve_project_path

from .manifest_artifacts import (
    build_file_artifact,
    write_manifest_inventory,
    write_tabular_artifact,
)
from .schema import ManifestRow, normalize_manifest_entry

DATA_MANIFEST_PRIORITY: tuple[str, ...] = (
    "all_manifest.jsonl",
    "train_manifest.jsonl",
    "dev_manifest.jsonl",
)
TRIAL_FIELD_ORDER: tuple[str, ...] = ("label", "left_audio", "right_audio")
QUARANTINE_MANIFEST_NAME = "quarantine_manifest.jsonl"


@dataclass(frozen=True, slots=True)
class AudioNormalizationPolicy:
    target_sample_rate_hz: int
    target_channels: int
    output_format: str
    output_pcm_bits_per_sample: int
    peak_headroom_db: float
    dc_offset_threshold: float
    clipped_sample_threshold: float

    @classmethod
    def from_config(cls, config: NormalizationConfig) -> AudioNormalizationPolicy:
        return cls(
            target_sample_rate_hz=config.target_sample_rate_hz,
            target_channels=config.target_channels,
            output_format=config.output_format,
            output_pcm_bits_per_sample=config.output_pcm_bits_per_sample,
            peak_headroom_db=config.peak_headroom_db,
            dc_offset_threshold=config.dc_offset_threshold,
            clipped_sample_threshold=config.clipped_sample_threshold,
        )

    @property
    def output_suffix(self) -> str:
        return f".{self.output_format.lower()}"

    @property
    def normalization_profile(self) -> str:
        return (
            f"{self.target_sample_rate_hz}hz-"
            f"{self.target_channels}ch-"
            f"pcm{self.output_pcm_bits_per_sample}-"
            f"{self.output_format.lower()}"
        )

    @property
    def target_peak_amplitude(self) -> float:
        return float(10 ** (-self.peak_headroom_db / 20.0))

    def to_dict(self) -> dict[str, object]:
        return {
            "target_sample_rate_hz": self.target_sample_rate_hz,
            "target_channels": self.target_channels,
            "output_format": self.output_format,
            "output_pcm_bits_per_sample": self.output_pcm_bits_per_sample,
            "peak_headroom_db": self.peak_headroom_db,
            "dc_offset_threshold": self.dc_offset_threshold,
            "clipped_sample_threshold": self.clipped_sample_threshold,
            "normalization_profile": self.normalization_profile,
        }


@dataclass(frozen=True, slots=True)
class SourceManifestTable:
    name: str
    path: Path
    rows: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class AuxiliaryTable:
    name: str
    path: Path
    rows: list[dict[str, object]]


@dataclass(frozen=True, slots=True)
class NormalizedAudioRecord:
    source_audio_path: str
    normalized_audio_path: str
    source_sample_rate_hz: int
    source_num_channels: int
    source_duration_seconds: float
    normalized_duration_seconds: float
    source_peak_amplitude: float
    source_dc_offset_ratio: float
    source_clipped_sample_ratio: float
    resampled: bool
    downmixed: bool
    dc_offset_removed: bool
    peak_scaled: bool


@dataclass(frozen=True, slots=True)
class QuarantineDecision:
    issue_code: str
    reason: str
    source_audio_path: str | None


@dataclass(slots=True)
class AudioNormalizationSummary:
    dataset: str
    source_manifests_root: str
    output_root: str
    output_manifests_root: str
    output_audio_root: str
    manifest_inventory_file: str
    report_json_file: str
    report_markdown_file: str
    source_manifest_count: int
    source_row_count: int
    normalized_row_count: int
    normalized_audio_count: int
    generated_quarantine_row_count: int
    carried_quarantine_row_count: int
    auxiliary_table_count: int
    copied_metadata_file_count: int
    resampled_row_count: int
    downmixed_row_count: int
    dc_offset_fixed_row_count: int
    peak_scaled_row_count: int
    source_clipping_row_count: int
    quarantine_issue_counts: dict[str, int]
    policy: AudioNormalizationPolicy

    def to_dict(self) -> dict[str, object]:
        return {
            "dataset": self.dataset,
            "source_manifests_root": self.source_manifests_root,
            "output_root": self.output_root,
            "output_manifests_root": self.output_manifests_root,
            "output_audio_root": self.output_audio_root,
            "manifest_inventory_file": self.manifest_inventory_file,
            "report_json_file": self.report_json_file,
            "report_markdown_file": self.report_markdown_file,
            "source_manifest_count": self.source_manifest_count,
            "source_row_count": self.source_row_count,
            "normalized_row_count": self.normalized_row_count,
            "normalized_audio_count": self.normalized_audio_count,
            "generated_quarantine_row_count": self.generated_quarantine_row_count,
            "carried_quarantine_row_count": self.carried_quarantine_row_count,
            "auxiliary_table_count": self.auxiliary_table_count,
            "copied_metadata_file_count": self.copied_metadata_file_count,
            "resampled_row_count": self.resampled_row_count,
            "downmixed_row_count": self.downmixed_row_count,
            "dc_offset_fixed_row_count": self.dc_offset_fixed_row_count,
            "peak_scaled_row_count": self.peak_scaled_row_count,
            "source_clipping_row_count": self.source_clipping_row_count,
            "quarantine_issue_counts": dict(self.quarantine_issue_counts),
            "policy": self.policy.to_dict(),
        }


def normalize_audio_manifest_bundle(
    *,
    project_root: str,
    dataset_root: str,
    source_manifests_root: str,
    output_root: str,
    policy: AudioNormalizationPolicy,
) -> AudioNormalizationSummary:
    _validate_policy(policy)

    project_root_path = resolve_project_path(project_root, ".")
    dataset_root_path = resolve_project_path(project_root, dataset_root)
    source_root_path = resolve_project_path(project_root, source_manifests_root)
    output_root_path = resolve_project_path(project_root, output_root)
    manifests_output_root = output_root_path / "manifests"
    audio_output_root = output_root_path / "audio"
    reports_output_root = output_root_path / "reports"

    source_tables, carried_quarantine_rows, auxiliary_tables, metadata_files = _load_source_bundle(
        source_root_path
    )
    if not source_tables:
        raise FileNotFoundError(
            f"No active data manifests found under source manifests root {source_root_path}"
        )

    dataset_name = _detect_dataset_name(source_tables)
    normalizer = _ManifestAudioNormalizer(
        project_root=project_root_path,
        dataset_root=dataset_root_path,
        audio_output_root=audio_output_root,
        policy=policy,
    )

    manifest_artifacts = []
    unique_source_row_keys: set[str] = set()
    unique_normalized_row_keys: set[str] = set()
    generated_quarantine_rows_by_key: dict[str, dict[str, object]] = {}

    for table in source_tables:
        normalized_rows: list[dict[str, object]] = []
        for row in table.rows:
            row_key = _row_identity_key(row)
            unique_source_row_keys.add(row_key)
            decision = normalizer.normalize_row(row)
            if isinstance(decision, QuarantineDecision):
                generated_quarantine_rows_by_key.setdefault(
                    row_key,
                    _build_generated_quarantine_row(
                        row,
                        decision=decision,
                    ),
                )
                continue

            unique_normalized_row_keys.add(row_key)
            normalized_rows.append(
                _build_normalized_manifest_row(
                    row,
                    record=decision,
                    policy=policy,
                )
            )

        manifest_artifacts.append(
            write_tabular_artifact(
                name=table.name.removesuffix(".jsonl"),
                kind="data_manifest",
                rows=normalized_rows,
                jsonl_path=manifests_output_root / table.name,
                project_root=str(project_root_path),
            )
        )

    prepared_carried_quarantine_rows = _deduplicate_rows(
        _prepare_carried_quarantine_row(row) for row in carried_quarantine_rows
    )
    generated_quarantine_rows = list(generated_quarantine_rows_by_key.values())
    all_quarantine_rows = [*prepared_carried_quarantine_rows, *generated_quarantine_rows]
    quarantine_artifact = write_tabular_artifact(
        name=QUARANTINE_MANIFEST_NAME.removesuffix(".jsonl"),
        kind="data_manifest",
        rows=all_quarantine_rows,
        jsonl_path=manifests_output_root / QUARANTINE_MANIFEST_NAME,
        project_root=str(project_root_path),
    )
    manifest_artifacts.append(quarantine_artifact)

    auxiliary_artifacts = []
    for table in auxiliary_tables:
        rewritten_rows = _rewrite_trial_rows(
            rows=table.rows,
            basename_mapping=normalizer.output_basename_by_source_basename,
        )
        auxiliary_artifacts.append(
            write_tabular_artifact(
                name=table.name.removesuffix(".jsonl"),
                kind="trial_list",
                rows=rewritten_rows,
                jsonl_path=manifests_output_root / table.name,
                project_root=str(project_root_path),
                field_order=TRIAL_FIELD_ORDER,
            )
        )

    metadata_artifacts = []
    for path in metadata_files:
        copied = manifests_output_root / path.name
        copied.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, copied)
        metadata_artifacts.append(
            build_file_artifact(
                name=path.name,
                kind="metadata",
                path=copied,
                project_root=str(project_root_path),
            )
        )

    reports_output_root.mkdir(parents=True, exist_ok=True)
    report_json_path = reports_output_root / "audio_normalization_report.json"
    report_markdown_path = reports_output_root / "audio_normalization_report.md"

    quarantine_issue_counts = Counter(
        str(row.get("quality_issue_code", "unknown")) for row in generated_quarantine_rows
    )
    report_summary = AudioNormalizationSummary(
        dataset=dataset_name,
        source_manifests_root=_relative_to_project(source_root_path, project_root_path),
        output_root=_relative_to_project(output_root_path, project_root_path),
        output_manifests_root=_relative_to_project(manifests_output_root, project_root_path),
        output_audio_root=_relative_to_project(audio_output_root, project_root_path),
        manifest_inventory_file="",
        report_json_file=_relative_to_project(report_json_path, project_root_path),
        report_markdown_file=_relative_to_project(report_markdown_path, project_root_path),
        source_manifest_count=len(source_tables),
        source_row_count=len(unique_source_row_keys),
        normalized_row_count=len(unique_normalized_row_keys),
        normalized_audio_count=len(normalizer.success_by_source_audio_path),
        generated_quarantine_row_count=len(generated_quarantine_rows),
        carried_quarantine_row_count=len(prepared_carried_quarantine_rows),
        auxiliary_table_count=len(auxiliary_tables),
        copied_metadata_file_count=len(metadata_files),
        resampled_row_count=sum(
            1 for record in normalizer.success_by_source_audio_path.values() if record.resampled
        ),
        downmixed_row_count=sum(
            1 for record in normalizer.success_by_source_audio_path.values() if record.downmixed
        ),
        dc_offset_fixed_row_count=sum(
            1
            for record in normalizer.success_by_source_audio_path.values()
            if record.dc_offset_removed
        ),
        peak_scaled_row_count=sum(
            1 for record in normalizer.success_by_source_audio_path.values() if record.peak_scaled
        ),
        source_clipping_row_count=sum(
            1
            for record in normalizer.success_by_source_audio_path.values()
            if record.source_clipped_sample_ratio > 0.0
        ),
        quarantine_issue_counts=dict(sorted(quarantine_issue_counts.items())),
        policy=policy,
    )

    report_json_path.write_text(
        json.dumps(report_summary.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    report_markdown_path.write_text(_render_summary_markdown(report_summary))
    report_artifacts = [
        build_file_artifact(
            name="audio_normalization_report_json",
            kind="report",
            path=report_json_path,
            project_root=str(project_root_path),
        ),
        build_file_artifact(
            name="audio_normalization_report_markdown",
            kind="report",
            path=report_markdown_path,
            project_root=str(project_root_path),
        ),
    ]

    inventory_path = manifests_output_root / "manifest_inventory.json"
    inventory_file = write_manifest_inventory(
        dataset=dataset_name,
        inventory_path=inventory_path,
        project_root=str(project_root_path),
        manifest_tables=manifest_artifacts,
        auxiliary_tables=auxiliary_artifacts,
        auxiliary_files=(*metadata_artifacts, *report_artifacts),
    )
    report_summary.manifest_inventory_file = inventory_file
    report_json_path.write_text(
        json.dumps(report_summary.to_dict(), indent=2, sort_keys=True) + "\n"
    )
    report_markdown_path.write_text(_render_summary_markdown(report_summary))
    return report_summary


class _ManifestAudioNormalizer:
    def __init__(
        self,
        *,
        project_root: Path,
        dataset_root: Path,
        audio_output_root: Path,
        policy: AudioNormalizationPolicy,
    ) -> None:
        self.project_root = project_root
        self.dataset_root = dataset_root
        self.audio_output_root = audio_output_root
        self.policy = policy
        self.success_by_source_audio_path: dict[str, NormalizedAudioRecord] = {}
        self.failure_by_source_audio_path: dict[str, QuarantineDecision] = {}
        self.output_basename_by_source_basename: dict[str, str] = {}

    def normalize_row(
        self, row: Mapping[str, object]
    ) -> NormalizedAudioRecord | QuarantineDecision:
        source_audio_path = _coerce_str(row.get("audio_path"))
        if source_audio_path is None:
            return QuarantineDecision(
                issue_code="missing_audio_path",
                reason="manifest row does not define a non-empty audio_path",
                source_audio_path=None,
            )

        cached_success = self.success_by_source_audio_path.get(source_audio_path)
        if cached_success is not None:
            return cached_success

        cached_failure = self.failure_by_source_audio_path.get(source_audio_path)
        if cached_failure is not None:
            return cached_failure

        decision = self._normalize_source_audio(source_audio_path)
        if isinstance(decision, QuarantineDecision):
            self.failure_by_source_audio_path[source_audio_path] = decision
        else:
            self.success_by_source_audio_path[source_audio_path] = decision
            self.output_basename_by_source_basename[Path(source_audio_path).name] = Path(
                decision.normalized_audio_path
            ).name
        return decision

    def _normalize_source_audio(
        self,
        source_audio_path: str,
    ) -> NormalizedAudioRecord | QuarantineDecision:
        source_path = resolve_project_path(str(self.project_root), source_audio_path)
        if not source_path.exists():
            return QuarantineDecision(
                issue_code="missing_audio_file",
                reason=f"audio file is missing from disk: {source_audio_path}",
                source_audio_path=source_audio_path,
            )

        try:
            waveform, sample_rate_hz = _read_audio_file(source_path)
        except Exception as exc:
            return QuarantineDecision(
                issue_code="audio_decode_error",
                reason=f"{type(exc).__name__}: {exc}",
                source_audio_path=source_audio_path,
            )

        if waveform.ndim != 2 or int(waveform.shape[-1]) == 0:
            return QuarantineDecision(
                issue_code="empty_audio_signal",
                reason="decoded audio tensor is empty",
                source_audio_path=source_audio_path,
            )
        if not _all_finite(waveform).all():
            return QuarantineDecision(
                issue_code="invalid_audio_signal",
                reason="decoded audio tensor contains non-finite values",
                source_audio_path=source_audio_path,
            )
        if sample_rate_hz <= 0:
            return QuarantineDecision(
                issue_code="invalid_sample_rate",
                reason=f"decoded audio reports a non-positive sample rate: {sample_rate_hz}",
                source_audio_path=source_audio_path,
            )

        source_num_channels = int(waveform.shape[0])
        source_duration_seconds = round(float(waveform.shape[-1]) / float(sample_rate_hz), 6)
        source_peak_amplitude = _peak_amplitude(waveform)
        source_dc_offset_ratio = _channel_dc_offset_ratio(waveform)
        source_clipped_sample_ratio = _clipped_sample_ratio(
            waveform,
            threshold=self.policy.clipped_sample_threshold,
        )

        normalized = waveform
        downmixed = False
        if self.policy.target_channels == 1 and source_num_channels != 1:
            normalized = normalized.mean(axis=0, keepdims=True)
            downmixed = True
        elif source_num_channels != self.policy.target_channels:
            return QuarantineDecision(
                issue_code="unsupported_channel_layout",
                reason=(
                    f"cannot normalize {source_num_channels} channels into "
                    f"{self.policy.target_channels} channels"
                ),
                source_audio_path=source_audio_path,
            )

        resampled = False
        if sample_rate_hz != self.policy.target_sample_rate_hz:
            normalized = _resample_waveform(
                normalized,
                orig_freq=sample_rate_hz,
                new_freq=self.policy.target_sample_rate_hz,
            )
            resampled = True

        dc_offset_removed = False
        normalized_dc_offset_ratio = _channel_dc_offset_ratio(normalized)
        if normalized_dc_offset_ratio >= self.policy.dc_offset_threshold:
            normalized = normalized - normalized.mean(axis=-1, keepdims=True)
            dc_offset_removed = True

        peak_scaled = False
        normalized_peak = _peak_amplitude(normalized)
        if normalized_peak > self.policy.target_peak_amplitude and normalized_peak > 0.0:
            normalized = normalized * (self.policy.target_peak_amplitude / normalized_peak)
            peak_scaled = True

        normalized_audio_path = self._build_normalized_audio_path(source_path)
        try:
            normalized_audio_path.parent.mkdir(parents=True, exist_ok=True)
            _write_audio_file(
                path=normalized_audio_path,
                waveform=normalized,
                sample_rate_hz=self.policy.target_sample_rate_hz,
                output_format=self.policy.output_format,
                pcm_bits_per_sample=self.policy.output_pcm_bits_per_sample,
            )
        except Exception as exc:
            return QuarantineDecision(
                issue_code="audio_write_error",
                reason=f"{type(exc).__name__}: {exc}",
                source_audio_path=source_audio_path,
            )

        return NormalizedAudioRecord(
            source_audio_path=source_audio_path,
            normalized_audio_path=_relative_to_project(normalized_audio_path, self.project_root),
            source_sample_rate_hz=sample_rate_hz,
            source_num_channels=source_num_channels,
            source_duration_seconds=source_duration_seconds,
            normalized_duration_seconds=round(
                float(normalized.shape[-1]) / float(self.policy.target_sample_rate_hz),
                6,
            ),
            source_peak_amplitude=source_peak_amplitude,
            source_dc_offset_ratio=source_dc_offset_ratio,
            source_clipped_sample_ratio=source_clipped_sample_ratio,
            resampled=resampled,
            downmixed=downmixed,
            dc_offset_removed=dc_offset_removed,
            peak_scaled=peak_scaled,
        )

    def _build_normalized_audio_path(self, source_path: Path) -> Path:
        if source_path.is_relative_to(self.dataset_root):
            relative_source = source_path.relative_to(self.dataset_root)
        elif source_path.is_relative_to(self.project_root):
            relative_source = source_path.relative_to(self.project_root)
        else:
            relative_source = Path(source_path.name)
        return (self.audio_output_root / relative_source).with_suffix(self.policy.output_suffix)


def _load_source_bundle(
    source_root: Path,
) -> tuple[list[SourceManifestTable], list[dict[str, object]], list[AuxiliaryTable], list[Path]]:
    if not source_root.exists():
        raise FileNotFoundError(f"Source manifests root does not exist: {source_root}")

    source_tables: list[SourceManifestTable] = []
    carried_quarantine_rows: list[dict[str, object]] = []
    auxiliary_tables: list[AuxiliaryTable] = []
    metadata_files: list[Path] = []

    for path in sorted(source_root.iterdir()):
        if path.suffix == ".jsonl":
            rows = _read_jsonl_rows(path)
            lower_name = path.name.lower()
            if "trial" in lower_name:
                auxiliary_tables.append(AuxiliaryTable(name=path.name, path=path, rows=rows))
            elif "quarantine" in lower_name:
                carried_quarantine_rows.extend(rows)
            else:
                source_tables.append(SourceManifestTable(name=path.name, path=path, rows=rows))
        elif path.is_file() and path.name != "manifest_inventory.json":
            metadata_files.append(path)

    source_tables.sort(key=_source_manifest_sort_key)
    auxiliary_tables.sort(key=lambda table: table.name)
    metadata_files.sort(key=lambda path: path.name)
    return source_tables, carried_quarantine_rows, auxiliary_tables, metadata_files


def _build_normalized_manifest_row(
    row: Mapping[str, object],
    *,
    record: NormalizedAudioRecord,
    policy: AudioNormalizationPolicy,
) -> dict[str, object]:
    normalized_entry = normalize_manifest_entry(row)
    base_row = ManifestRow(
        dataset=str(normalized_entry["dataset"]),
        source_dataset=str(normalized_entry.get("source_dataset") or normalized_entry["dataset"]),
        speaker_id=str(normalized_entry["speaker_id"]),
        audio_path=record.normalized_audio_path,
        utterance_id=_coerce_str(normalized_entry.get("utterance_id")),
        session_id=_coerce_str(normalized_entry.get("session_id")),
        split=_coerce_str(normalized_entry.get("split")),
        role=_coerce_str(normalized_entry.get("role")),
        language=_coerce_str(normalized_entry.get("language")),
        device=_coerce_str(normalized_entry.get("device")),
        channel="mono"
        if policy.target_channels == 1
        else _coerce_str(normalized_entry.get("channel")),
        snr_db=_coerce_float(normalized_entry.get("snr_db")),
        rir_id=_coerce_str(normalized_entry.get("rir_id")),
        duration_seconds=record.normalized_duration_seconds,
        sample_rate_hz=policy.target_sample_rate_hz,
        num_channels=policy.target_channels,
    )
    canonical_fields = set(base_row.to_dict())
    extra_fields = {key: value for key, value in row.items() if key not in canonical_fields}
    extra_fields.update(
        {
            "source_audio_path": record.source_audio_path,
            "source_sample_rate_hz": record.source_sample_rate_hz,
            "source_num_channels": record.source_num_channels,
            "source_peak_amplitude": round(record.source_peak_amplitude, 6),
            "source_dc_offset_ratio": round(record.source_dc_offset_ratio, 6),
            "source_clipped_sample_ratio": round(record.source_clipped_sample_ratio, 6),
            "normalization_profile": policy.normalization_profile,
            "normalization_resampled": record.resampled,
            "normalization_downmixed": record.downmixed,
            "normalization_dc_offset_removed": record.dc_offset_removed,
            "normalization_peak_scaled": record.peak_scaled,
        }
    )
    return base_row.to_dict(extra_fields=extra_fields)


def _build_generated_quarantine_row(
    row: Mapping[str, object],
    *,
    decision: QuarantineDecision,
) -> dict[str, object]:
    payload = dict(row)
    payload["quality_issue_code"] = decision.issue_code
    payload["quarantine_policy"] = "quarantine"
    payload["quarantine_stage"] = "audio_normalization"
    payload["quarantine_reason"] = decision.reason
    if decision.source_audio_path is not None:
        payload.setdefault("source_audio_path", decision.source_audio_path)
    return payload


def _prepare_carried_quarantine_row(row: Mapping[str, object]) -> dict[str, object]:
    payload = dict(row)
    payload.setdefault("quarantine_stage", "source_manifest")
    payload.setdefault("quarantine_policy", "quarantine")
    return payload


def _rewrite_trial_rows(
    *,
    rows: Sequence[Mapping[str, object]],
    basename_mapping: Mapping[str, str],
) -> list[dict[str, object]]:
    rewritten_rows: list[dict[str, object]] = []
    for row in rows:
        payload = dict(row)
        for field_name in ("left_audio", "right_audio"):
            current = _coerce_str(payload.get(field_name))
            if current is None:
                continue
            payload[field_name] = basename_mapping.get(current, current)
        rewritten_rows.append(payload)
    return rewritten_rows


def _render_summary_markdown(summary: AudioNormalizationSummary) -> str:
    issue_lines = [
        f"- `{issue_code}`: {count}"
        for issue_code, count in summary.quarantine_issue_counts.items()
    ] or ["- none"]
    return "\n".join(
        [
            "# Audio Normalization Report",
            "",
            "## Scope",
            "",
            f"- dataset: `{summary.dataset}`",
            f"- source manifests root: `{summary.source_manifests_root}`",
            f"- output root: `{summary.output_root}`",
            "",
            "## Policy",
            "",
            f"- normalization profile: `{summary.policy.normalization_profile}`",
            f"- peak headroom: `{summary.policy.peak_headroom_db:.2f} dB`",
            f"- dc offset threshold: `{summary.policy.dc_offset_threshold:.4f}`",
            f"- clipped sample threshold: `{summary.policy.clipped_sample_threshold:.4f}`",
            "",
            "## Results",
            "",
            f"- source manifests: `{summary.source_manifest_count}`",
            f"- source rows: `{summary.source_row_count}`",
            f"- normalized rows: `{summary.normalized_row_count}`",
            f"- normalized audio files: `{summary.normalized_audio_count}`",
            f"- generated quarantine rows: `{summary.generated_quarantine_row_count}`",
            f"- carried quarantine rows: `{summary.carried_quarantine_row_count}`",
            f"- resampled rows: `{summary.resampled_row_count}`",
            f"- downmixed rows: `{summary.downmixed_row_count}`",
            f"- dc-offset fixes: `{summary.dc_offset_fixed_row_count}`",
            f"- peak-scaled rows: `{summary.peak_scaled_row_count}`",
            f"- source clipping detections: `{summary.source_clipping_row_count}`",
            "",
            "## Quarantine Issues",
            "",
            *issue_lines,
            "",
        ]
    )


def _detect_dataset_name(source_tables: Sequence[SourceManifestTable]) -> str:
    for table in source_tables:
        for row in table.rows:
            dataset_name = _coerce_str(normalize_manifest_entry(row).get("dataset"))
            if dataset_name is not None:
                return dataset_name
    return source_tables[0].path.parent.name


def _source_manifest_sort_key(table: SourceManifestTable) -> tuple[int, str]:
    try:
        return (DATA_MANIFEST_PRIORITY.index(table.name), table.name)
    except ValueError:
        return (len(DATA_MANIFEST_PRIORITY), table.name)


def _read_jsonl_rows(path: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(path.read_text().splitlines(), start=1):
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if not isinstance(payload, dict):
            raise ValueError(f"Expected object JSONL rows in {path}:{line_number}")
        rows.append(payload)
    return rows


def _deduplicate_rows(rows: Iterable[Mapping[str, object]]) -> list[dict[str, object]]:
    deduplicated: dict[str, dict[str, object]] = {}
    for row in rows:
        deduplicated.setdefault(_row_identity_key(row), dict(row))
    return list(deduplicated.values())


def _row_identity_key(row: Mapping[str, object]) -> str:
    normalized = normalize_manifest_entry(row)
    parts = (
        _coerce_str(normalized.get("dataset")) or "unknown-dataset",
        _coerce_str(normalized.get("split")) or "unknown-split",
        _coerce_str(normalized.get("speaker_id")) or "unknown-speaker",
        _coerce_str(normalized.get("utterance_id"))
        or _coerce_str(normalized.get("audio_path"))
        or json.dumps(dict(row), sort_keys=True),
    )
    return "|".join(parts)


def _validate_policy(policy: AudioNormalizationPolicy) -> None:
    output_format = policy.output_format.lower()
    if output_format not in {"wav", "flac"}:
        raise ValueError(f"Unsupported output format: {policy.output_format!r}")
    if policy.target_sample_rate_hz <= 0:
        raise ValueError("target_sample_rate_hz must be positive")
    if policy.target_channels <= 0:
        raise ValueError("target_channels must be positive")
    if not 0.0 < policy.clipped_sample_threshold <= 1.0:
        raise ValueError("clipped_sample_threshold must be within (0, 1]")
    if policy.dc_offset_threshold < 0.0:
        raise ValueError("dc_offset_threshold must be non-negative")
    if policy.peak_headroom_db < 0.0:
        raise ValueError("peak_headroom_db must be non-negative")


def _coerce_str(value: object | None) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _coerce_float(value: object | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        return float(value)
    raise TypeError(f"Unsupported float value type: {type(value)!r}")


def _relative_to_project(path: Path, project_root: Path) -> str:
    return str(path.resolve().relative_to(project_root.resolve()))


def _channel_dc_offset_ratio(waveform: Any) -> float:
    import numpy as np

    channel_means = np.abs(waveform.mean(axis=-1))
    return float(channel_means.max(initial=0.0))


def _clipped_sample_ratio(waveform: Any, *, threshold: float) -> float:
    import numpy as np

    sample_count = int(waveform.size)
    if sample_count == 0:
        return 0.0
    clipped_count = int(np.count_nonzero(np.abs(waveform) >= threshold))
    return clipped_count / sample_count


def _peak_amplitude(waveform: Any) -> float:
    import numpy as np

    if waveform.size == 0:
        return 0.0
    return float(np.abs(waveform).max(initial=0.0))


def _read_audio_file(path: Path) -> tuple[Any, int]:
    import soundfile as sf

    waveform, sample_rate_hz = sf.read(
        str(path),
        always_2d=True,
        dtype="float32",
    )
    return waveform.T, int(sample_rate_hz)


def _resample_waveform(waveform: Any, *, orig_freq: int, new_freq: int) -> Any:
    import numpy as np
    import soxr

    if waveform.ndim == 1:
        return soxr.resample(waveform, orig_freq, new_freq, quality="HQ").astype(
            "float32",
            copy=False,
        )
    channels = [soxr.resample(channel, orig_freq, new_freq, quality="HQ") for channel in waveform]
    return np.stack(channels, axis=0).astype("float32", copy=False)


def _write_audio_file(
    *,
    path: Path,
    waveform: Any,
    sample_rate_hz: int,
    output_format: str,
    pcm_bits_per_sample: int,
) -> None:
    import numpy as np
    import soundfile as sf

    output_format = output_format.lower()
    sf.write(
        str(path),
        np.clip(waveform, -1.0, 1.0).T,
        sample_rate_hz,
        format=output_format.upper(),
        subtype=_pcm_subtype(bits_per_sample=pcm_bits_per_sample),
    )


def _all_finite(waveform: Any) -> Any:
    import numpy as np

    return np.isfinite(waveform)


def _pcm_subtype(*, bits_per_sample: int) -> str:
    return {
        8: "PCM_U8",
        16: "PCM_16",
        24: "PCM_24",
        32: "PCM_32",
    }.get(bits_per_sample) or _raise_unsupported_pcm_bits(bits_per_sample)


def _raise_unsupported_pcm_bits(bits_per_sample: int) -> str:
    raise ValueError(f"Unsupported PCM bits per sample: {bits_per_sample}")
