"""Manifest-driven silence and pause augmentation ablation reports."""

from __future__ import annotations

import json
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from kryptonite.config import NormalizationConfig, SilenceAugmentationConfig, VADConfig
from kryptonite.data import (
    AudioLoadRequest,
    analyze_silence_profile,
    apply_silence_augmentation,
    iter_manifest_audio,
)
from kryptonite.deployment import resolve_project_path


@dataclass(frozen=True, slots=True)
class SilenceAugmentationComparisonRecord:
    manifest_path: str
    line_number: int | None
    audio_path: str
    speaker_id: str | None
    utterance_id: str | None
    input_duration_seconds: float
    output_duration_seconds: float
    input_silence_ratio: float
    output_silence_ratio: float
    input_pause_count: int
    output_pause_count: int
    input_pause_total_seconds: float
    output_pause_total_seconds: float
    leading_padding_seconds: float
    trailing_padding_seconds: float
    inserted_pause_count: int
    inserted_pause_total_seconds: float
    perturbed_pause_count: int
    stretched_pause_count: int
    compressed_pause_count: int
    applied: bool
    skip_reason: str

    @property
    def duration_delta_seconds(self) -> float:
        return round(self.output_duration_seconds - self.input_duration_seconds, 6)

    @property
    def silence_ratio_delta(self) -> float:
        return round(self.output_silence_ratio - self.input_silence_ratio, 6)

    @property
    def pause_count_delta(self) -> int:
        return self.output_pause_count - self.input_pause_count

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "line_number": self.line_number,
            "audio_path": self.audio_path,
            "speaker_id": self.speaker_id,
            "utterance_id": self.utterance_id,
            "input_duration_seconds": self.input_duration_seconds,
            "output_duration_seconds": self.output_duration_seconds,
            "duration_delta_seconds": self.duration_delta_seconds,
            "input_silence_ratio": self.input_silence_ratio,
            "output_silence_ratio": self.output_silence_ratio,
            "silence_ratio_delta": self.silence_ratio_delta,
            "input_pause_count": self.input_pause_count,
            "output_pause_count": self.output_pause_count,
            "pause_count_delta": self.pause_count_delta,
            "input_pause_total_seconds": self.input_pause_total_seconds,
            "output_pause_total_seconds": self.output_pause_total_seconds,
            "leading_padding_seconds": self.leading_padding_seconds,
            "trailing_padding_seconds": self.trailing_padding_seconds,
            "inserted_pause_count": self.inserted_pause_count,
            "inserted_pause_total_seconds": self.inserted_pause_total_seconds,
            "perturbed_pause_count": self.perturbed_pause_count,
            "stretched_pause_count": self.stretched_pause_count,
            "compressed_pause_count": self.compressed_pause_count,
            "applied": self.applied,
            "skip_reason": self.skip_reason,
        }


@dataclass(frozen=True, slots=True)
class SilenceAugmentationSummary:
    row_count: int
    changed_row_count: int
    rows_with_padding: int
    rows_with_inserted_pauses: int
    rows_with_pause_perturbation: int
    mean_input_duration_seconds: float
    mean_output_duration_seconds: float
    mean_duration_delta_seconds: float
    mean_input_silence_ratio: float
    mean_output_silence_ratio: float
    mean_silence_ratio_delta: float
    mean_input_pause_count: float
    mean_output_pause_count: float
    max_silence_ratio_delta: float

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "changed_row_count": self.changed_row_count,
            "rows_with_padding": self.rows_with_padding,
            "rows_with_inserted_pauses": self.rows_with_inserted_pauses,
            "rows_with_pause_perturbation": self.rows_with_pause_perturbation,
            "mean_input_duration_seconds": self.mean_input_duration_seconds,
            "mean_output_duration_seconds": self.mean_output_duration_seconds,
            "mean_duration_delta_seconds": self.mean_duration_delta_seconds,
            "mean_input_silence_ratio": self.mean_input_silence_ratio,
            "mean_output_silence_ratio": self.mean_output_silence_ratio,
            "mean_silence_ratio_delta": self.mean_silence_ratio_delta,
            "mean_input_pause_count": self.mean_input_pause_count,
            "mean_output_pause_count": self.mean_output_pause_count,
            "max_silence_ratio_delta": self.max_silence_ratio_delta,
        }


@dataclass(frozen=True, slots=True)
class SilenceAugmentationReport:
    project_root: str
    manifest_path: str
    seed: int
    limit: int | None
    vad_mode: str
    config: SilenceAugmentationConfig
    summary: SilenceAugmentationSummary
    records: list[SilenceAugmentationComparisonRecord]

    def to_dict(self, *, include_records: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "project_root": self.project_root,
            "manifest_path": self.manifest_path,
            "seed": self.seed,
            "limit": self.limit,
            "vad_mode": self.vad_mode,
            "config": {
                "enabled": self.config.enabled,
                "max_leading_padding_seconds": self.config.max_leading_padding_seconds,
                "max_trailing_padding_seconds": self.config.max_trailing_padding_seconds,
                "max_inserted_pauses": self.config.max_inserted_pauses,
                "min_inserted_pause_seconds": self.config.min_inserted_pause_seconds,
                "max_inserted_pause_seconds": self.config.max_inserted_pause_seconds,
                "pause_ratio_min": self.config.pause_ratio_min,
                "pause_ratio_max": self.config.pause_ratio_max,
                "min_detected_pause_seconds": self.config.min_detected_pause_seconds,
                "max_perturbed_pause_seconds": self.config.max_perturbed_pause_seconds,
                "analysis_frame_ms": self.config.analysis_frame_ms,
                "silence_threshold_dbfs": self.config.silence_threshold_dbfs,
            },
            "summary": self.summary.to_dict(),
        }
        if include_records:
            payload["records"] = [record.to_dict() for record in self.records]
        return payload


@dataclass(frozen=True, slots=True)
class WrittenSilenceAugmentationReport:
    output_root: str
    json_path: str
    markdown_path: str
    rows_path: str

    def to_dict(self) -> dict[str, str]:
        return {
            "output_root": self.output_root,
            "json_path": self.json_path,
            "markdown_path": self.markdown_path,
            "rows_path": self.rows_path,
        }


def build_silence_augmentation_report(
    *,
    project_root: Path | str,
    manifest_path: Path | str,
    normalization: NormalizationConfig,
    vad: VADConfig | None,
    silence_augmentation: SilenceAugmentationConfig,
    seed: int,
    limit: int | None = None,
) -> SilenceAugmentationReport:
    request = AudioLoadRequest.from_config(normalization, vad=vad)
    records: list[SilenceAugmentationComparisonRecord] = []

    for index, loaded in enumerate(
        iter_manifest_audio(
            manifest_path,
            project_root=project_root,
            request=request,
        )
    ):
        if limit is not None and index >= limit:
            break

        input_profile = analyze_silence_profile(
            loaded.audio.waveform,
            sample_rate_hz=loaded.audio.sample_rate_hz,
            config=silence_augmentation,
        )
        augmented_waveform, decision = apply_silence_augmentation(
            loaded.audio.waveform,
            sample_rate_hz=loaded.audio.sample_rate_hz,
            config=silence_augmentation,
            rng=random.Random(seed + index),
        )
        output_profile = analyze_silence_profile(
            augmented_waveform,
            sample_rate_hz=loaded.audio.sample_rate_hz,
            config=silence_augmentation,
        )
        records.append(
            SilenceAugmentationComparisonRecord(
                manifest_path=loaded.manifest_path or str(manifest_path),
                line_number=loaded.line_number,
                audio_path=loaded.row.audio_path,
                speaker_id=loaded.row.speaker_id,
                utterance_id=loaded.row.utterance_id,
                input_duration_seconds=input_profile.duration_seconds,
                output_duration_seconds=output_profile.duration_seconds,
                input_silence_ratio=input_profile.silence_ratio,
                output_silence_ratio=output_profile.silence_ratio,
                input_pause_count=input_profile.interior_pause_count,
                output_pause_count=output_profile.interior_pause_count,
                input_pause_total_seconds=input_profile.interior_pause_total_seconds,
                output_pause_total_seconds=output_profile.interior_pause_total_seconds,
                leading_padding_seconds=decision.leading_padding_seconds,
                trailing_padding_seconds=decision.trailing_padding_seconds,
                inserted_pause_count=decision.inserted_pause_count,
                inserted_pause_total_seconds=decision.inserted_pause_total_seconds,
                perturbed_pause_count=decision.perturbed_pause_count,
                stretched_pause_count=decision.stretched_pause_count,
                compressed_pause_count=decision.compressed_pause_count,
                applied=decision.applied,
                skip_reason=decision.skip_reason,
            )
        )

    return SilenceAugmentationReport(
        project_root=str(resolve_project_path(str(project_root), ".")),
        manifest_path=str(resolve_project_path(str(project_root), str(manifest_path))),
        seed=seed,
        limit=limit,
        vad_mode="none" if vad is None else vad.mode,
        config=silence_augmentation,
        summary=_build_summary(records),
        records=records,
    )


def write_silence_augmentation_report(
    *,
    report: SilenceAugmentationReport,
    output_root: Path | str,
) -> WrittenSilenceAugmentationReport:
    output_root_path = resolve_project_path(report.project_root, str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)

    json_path = output_root_path / "silence_augmentation_report.json"
    markdown_path = output_root_path / "silence_augmentation_report.md"
    rows_path = output_root_path / "silence_augmentation_rows.jsonl"

    json_path.write_text(
        json.dumps(report.to_dict(include_records=True), indent=2, sort_keys=True) + "\n"
    )
    markdown_path.write_text(render_silence_augmentation_markdown(report))
    rows_path.write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in report.records)
    )
    return WrittenSilenceAugmentationReport(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        rows_path=str(rows_path),
    )


def render_silence_augmentation_markdown(report: SilenceAugmentationReport) -> str:
    summary = report.summary
    config = report.config
    lines = [
        "# Silence Augmentation Report",
        "",
        "## Scope",
        "",
        f"- manifest: `{report.manifest_path}`",
        f"- rows analyzed: `{summary.row_count}`",
        f"- seed: `{report.seed}`",
        f"- vad mode: `{report.vad_mode}`",
        f"- enabled: `{config.enabled}`",
        f"- max leading padding: `{config.max_leading_padding_seconds:.2f}s`",
        f"- max trailing padding: `{config.max_trailing_padding_seconds:.2f}s`",
        (
            f"- inserted pauses: `<= {config.max_inserted_pauses}` "
            f"(`{config.min_inserted_pause_seconds:.2f}s` .. "
            f"`{config.max_inserted_pause_seconds:.2f}s`)"
        ),
        f"- pause ratio range: `{config.pause_ratio_min:.2f}` .. `{config.pause_ratio_max:.2f}`",
        f"- detected pause floor: `{config.min_detected_pause_seconds:.2f}s`",
        f"- perturbed pause cap: `{config.max_perturbed_pause_seconds:.2f}s`",
        f"- analysis frame: `{config.analysis_frame_ms:.1f} ms`",
        f"- silence threshold: `{config.silence_threshold_dbfs:.1f} dBFS`",
        "",
        "## Summary",
        "",
        f"- changed rows: `{summary.changed_row_count}`",
        f"- rows with boundary padding: `{summary.rows_with_padding}`",
        f"- rows with inserted pauses: `{summary.rows_with_inserted_pauses}`",
        f"- rows with pause perturbation: `{summary.rows_with_pause_perturbation}`",
        f"- mean input duration: `{summary.mean_input_duration_seconds:.3f}s`",
        f"- mean output duration: `{summary.mean_output_duration_seconds:.3f}s`",
        f"- mean duration delta: `{summary.mean_duration_delta_seconds:.3f}s`",
        f"- mean input silence ratio: `{summary.mean_input_silence_ratio:.3f}`",
        f"- mean output silence ratio: `{summary.mean_output_silence_ratio:.3f}`",
        f"- mean silence ratio delta: `{summary.mean_silence_ratio_delta:.3f}`",
        f"- mean input pause count: `{summary.mean_input_pause_count:.3f}`",
        f"- mean output pause count: `{summary.mean_output_pause_count:.3f}`",
        f"- max silence ratio delta: `{summary.max_silence_ratio_delta:.3f}`",
        "",
    ]

    top_changed = [
        record
        for record in sorted(
            report.records,
            key=lambda item: abs(item.silence_ratio_delta),
            reverse=True,
        )
        if record.applied
    ][:5]
    if top_changed:
        lines.extend(
            [
                "## Largest Silence Deltas",
                "",
            ]
        )
        for record in top_changed:
            lines.extend(
                [
                    f"### `{record.audio_path}`",
                    "",
                    f"- duration delta: `{record.duration_delta_seconds:.3f}s`",
                    (
                        f"- silence ratio: `{record.input_silence_ratio:.3f}` -> "
                        f"`{record.output_silence_ratio:.3f}`"
                    ),
                    f"- pause count: `{record.input_pause_count}` -> `{record.output_pause_count}`",
                    f"- inserted pauses: `{record.inserted_pause_count}`",
                    f"- perturbed pauses: `{record.perturbed_pause_count}`",
                    "",
                ]
            )
    else:
        lines.extend(
            [
                "## Largest Silence Deltas",
                "",
                "- no rows changed under the current configuration",
                "",
            ]
        )
    return "\n".join(lines)


def _build_summary(
    records: list[SilenceAugmentationComparisonRecord],
) -> SilenceAugmentationSummary:
    changed = [record for record in records if record.applied]
    return SilenceAugmentationSummary(
        row_count=len(records),
        changed_row_count=len(changed),
        rows_with_padding=sum(
            1
            for record in records
            if record.leading_padding_seconds > 0.0 or record.trailing_padding_seconds > 0.0
        ),
        rows_with_inserted_pauses=sum(1 for record in records if record.inserted_pause_count > 0),
        rows_with_pause_perturbation=sum(
            1 for record in records if record.perturbed_pause_count > 0
        ),
        mean_input_duration_seconds=_mean(record.input_duration_seconds for record in records),
        mean_output_duration_seconds=_mean(record.output_duration_seconds for record in records),
        mean_duration_delta_seconds=_mean(record.duration_delta_seconds for record in records),
        mean_input_silence_ratio=_mean(record.input_silence_ratio for record in records),
        mean_output_silence_ratio=_mean(record.output_silence_ratio for record in records),
        mean_silence_ratio_delta=_mean(record.silence_ratio_delta for record in records),
        mean_input_pause_count=_mean(record.input_pause_count for record in records),
        mean_output_pause_count=_mean(record.output_pause_count for record in records),
        max_silence_ratio_delta=round(
            max((abs(record.silence_ratio_delta) for record in records), default=0.0),
            6,
        ),
    )


def _mean(values: Iterable[float | int]) -> float:
    items = [float(value) for value in values]
    if not items:
        return 0.0
    return round(sum(items) / len(items), 6)
