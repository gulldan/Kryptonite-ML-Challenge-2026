"""Manifest-driven VAD/trimming comparison reports."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from kryptonite.config import NormalizationConfig
from kryptonite.data import AudioLoadRequest, iter_manifest_audio
from kryptonite.data.vad import SUPPORTED_VAD_MODES, apply_vad_policy
from kryptonite.deployment import resolve_project_path


@dataclass(frozen=True, slots=True)
class VADModeObservation:
    mode: str
    output_duration_seconds: float
    trim_applied: bool
    speech_detected: bool
    trim_reason: str
    leading_trim_seconds: float
    trailing_trim_seconds: float

    @property
    def removed_duration_seconds(self) -> float:
        return round(self.leading_trim_seconds + self.trailing_trim_seconds, 6)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "output_duration_seconds": self.output_duration_seconds,
            "trim_applied": self.trim_applied,
            "speech_detected": self.speech_detected,
            "trim_reason": self.trim_reason,
            "leading_trim_seconds": self.leading_trim_seconds,
            "trailing_trim_seconds": self.trailing_trim_seconds,
            "removed_duration_seconds": self.removed_duration_seconds,
        }


@dataclass(frozen=True, slots=True)
class VADComparisonRecord:
    manifest_path: str
    line_number: int | None
    audio_path: str
    speaker_id: str | None
    utterance_id: str | None
    input_duration_seconds: float
    observations: dict[str, VADModeObservation]

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "line_number": self.line_number,
            "audio_path": self.audio_path,
            "speaker_id": self.speaker_id,
            "utterance_id": self.utterance_id,
            "input_duration_seconds": self.input_duration_seconds,
            "observations": {
                mode: observation.to_dict()
                for mode, observation in sorted(self.observations.items())
            },
        }


@dataclass(frozen=True, slots=True)
class VADModeSummary:
    mode: str
    row_count: int
    trimmed_row_count: int
    rows_without_detected_speech: int
    total_input_duration_seconds: float
    total_output_duration_seconds: float
    mean_leading_trim_seconds: float
    mean_trailing_trim_seconds: float

    @property
    def retained_duration_ratio(self) -> float:
        if self.total_input_duration_seconds <= 0.0:
            return 0.0
        return round(self.total_output_duration_seconds / self.total_input_duration_seconds, 6)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "row_count": self.row_count,
            "trimmed_row_count": self.trimmed_row_count,
            "rows_without_detected_speech": self.rows_without_detected_speech,
            "total_input_duration_seconds": self.total_input_duration_seconds,
            "total_output_duration_seconds": self.total_output_duration_seconds,
            "retained_duration_ratio": self.retained_duration_ratio,
            "mean_leading_trim_seconds": self.mean_leading_trim_seconds,
            "mean_trailing_trim_seconds": self.mean_trailing_trim_seconds,
        }


@dataclass(frozen=True, slots=True)
class VADComparisonExample:
    mode: str
    audio_path: str
    input_duration_seconds: float
    output_duration_seconds: float
    removed_duration_seconds: float
    trim_reason: str

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode,
            "audio_path": self.audio_path,
            "input_duration_seconds": self.input_duration_seconds,
            "output_duration_seconds": self.output_duration_seconds,
            "removed_duration_seconds": self.removed_duration_seconds,
            "trim_reason": self.trim_reason,
        }


@dataclass(frozen=True, slots=True)
class VADComparisonReport:
    project_root: str
    manifest_path: str
    modes: tuple[str, ...]
    limit: int | None
    records: list[VADComparisonRecord]
    summaries: list[VADModeSummary]
    examples_by_mode: dict[str, list[VADComparisonExample]]

    @property
    def row_count(self) -> int:
        return len(self.records)

    def to_dict(self, *, include_records: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "project_root": self.project_root,
            "manifest_path": self.manifest_path,
            "modes": list(self.modes),
            "limit": self.limit,
            "row_count": self.row_count,
            "summaries": [summary.to_dict() for summary in self.summaries],
            "examples_by_mode": {
                mode: [example.to_dict() for example in examples]
                for mode, examples in sorted(self.examples_by_mode.items())
            },
        }
        if include_records:
            payload["records"] = [record.to_dict() for record in self.records]
        return payload


@dataclass(frozen=True, slots=True)
class WrittenVADComparisonReport:
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


def build_vad_trimming_report(
    *,
    project_root: Path | str,
    manifest_path: Path | str,
    normalization: NormalizationConfig,
    modes: tuple[str, ...] = SUPPORTED_VAD_MODES,
    limit: int | None = None,
) -> VADComparisonReport:
    normalized_modes = _normalize_modes(modes)
    request = AudioLoadRequest.from_config(normalization)
    records: list[VADComparisonRecord] = []
    for index, loaded in enumerate(
        iter_manifest_audio(
            manifest_path,
            project_root=project_root,
            request=request,
        )
    ):
        if limit is not None and index >= limit:
            break

        observations: dict[str, VADModeObservation] = {}
        for mode in normalized_modes:
            if mode == "none":
                observations[mode] = VADModeObservation(
                    mode=mode,
                    output_duration_seconds=loaded.audio.duration_seconds,
                    trim_applied=False,
                    speech_detected=True,
                    trim_reason="disabled",
                    leading_trim_seconds=0.0,
                    trailing_trim_seconds=0.0,
                )
                continue

            trimmed_waveform, decision = apply_vad_policy(
                loaded.audio.waveform,
                sample_rate_hz=loaded.audio.sample_rate_hz,
                mode=mode,
            )
            observations[mode] = VADModeObservation(
                mode=mode,
                output_duration_seconds=round(
                    float(trimmed_waveform.shape[-1]) / float(loaded.audio.sample_rate_hz),
                    6,
                ),
                trim_applied=decision.applied,
                speech_detected=decision.speech_detected,
                trim_reason=decision.reason,
                leading_trim_seconds=round(
                    float(decision.leading_trim_frames) / float(loaded.audio.sample_rate_hz),
                    6,
                ),
                trailing_trim_seconds=round(
                    float(decision.trailing_trim_frames) / float(loaded.audio.sample_rate_hz),
                    6,
                ),
            )

        records.append(
            VADComparisonRecord(
                manifest_path=loaded.manifest_path or str(manifest_path),
                line_number=loaded.line_number,
                audio_path=loaded.row.audio_path,
                speaker_id=loaded.row.speaker_id,
                utterance_id=loaded.row.utterance_id,
                input_duration_seconds=loaded.audio.duration_seconds,
                observations=observations,
            )
        )

    summaries = [_build_summary(records, mode) for mode in normalized_modes]
    examples_by_mode = {
        mode: _build_examples(records, mode) for mode in normalized_modes if mode != "none"
    }
    manifest_location = resolve_project_path(str(project_root), str(manifest_path))
    return VADComparisonReport(
        project_root=str(resolve_project_path(str(project_root), ".")),
        manifest_path=str(manifest_location),
        modes=normalized_modes,
        limit=limit,
        records=records,
        summaries=summaries,
        examples_by_mode=examples_by_mode,
    )


def write_vad_trimming_report(
    *,
    report: VADComparisonReport,
    output_root: Path | str,
) -> WrittenVADComparisonReport:
    output_root_path = resolve_project_path(report.project_root, str(output_root))
    output_root_path.mkdir(parents=True, exist_ok=True)

    json_path = output_root_path / "vad_trimming_report.json"
    markdown_path = output_root_path / "vad_trimming_report.md"
    rows_path = output_root_path / "vad_trimming_rows.jsonl"

    json_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n")
    markdown_path.write_text(render_vad_trimming_markdown(report))
    rows_path.write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in report.records)
    )
    return WrittenVADComparisonReport(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        rows_path=str(rows_path),
    )


def render_vad_trimming_markdown(report: VADComparisonReport) -> str:
    lines = [
        "# VAD / Trimming Comparison",
        "",
        "## Scope",
        "",
        f"- manifest: `{report.manifest_path}`",
        f"- rows analyzed: `{report.row_count}`",
        f"- modes: `{', '.join(report.modes)}`",
    ]
    if report.limit is not None:
        lines.append(f"- limit: `{report.limit}`")

    lines.extend(
        [
            "",
            "## Summary",
            "",
            (
                "| Mode | Rows | Trimmed rows | No speech rows | Input sec | Output sec | "
                "Retained ratio | Mean lead trim | Mean tail trim |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for summary in report.summaries:
        lines.append(
            f"| {summary.mode} | {summary.row_count} | {summary.trimmed_row_count} | "
            f"{summary.rows_without_detected_speech} | "
            f"{summary.total_input_duration_seconds:.3f} | "
            f"{summary.total_output_duration_seconds:.3f} | "
            f"{summary.retained_duration_ratio:.3f} | "
            f"{summary.mean_leading_trim_seconds:.3f} | "
            f"{summary.mean_trailing_trim_seconds:.3f} |"
        )

    for mode in report.modes:
        if mode == "none":
            continue
        lines.extend(["", f"## Top `{mode}` trims", ""])
        examples = report.examples_by_mode.get(mode, [])
        if not examples:
            lines.append("No rows changed under this mode.")
            continue
        for example in examples:
            lines.append(
                f"- `{example.audio_path}`: {example.input_duration_seconds:.3f}s -> "
                f"{example.output_duration_seconds:.3f}s "
                f"(removed {example.removed_duration_seconds:.3f}s, "
                f"reason `{example.trim_reason}`)"
            )
    return "\n".join(lines) + "\n"


def _build_summary(records: list[VADComparisonRecord], mode: str) -> VADModeSummary:
    observations = [record.observations[mode] for record in records]
    row_count = len(observations)
    total_input_duration = round(sum(record.input_duration_seconds for record in records), 6)
    total_output_duration = round(sum(obs.output_duration_seconds for obs in observations), 6)
    mean_leading = round(
        sum(obs.leading_trim_seconds for obs in observations) / row_count if row_count else 0.0,
        6,
    )
    mean_trailing = round(
        sum(obs.trailing_trim_seconds for obs in observations) / row_count if row_count else 0.0,
        6,
    )
    return VADModeSummary(
        mode=mode,
        row_count=row_count,
        trimmed_row_count=sum(1 for obs in observations if obs.trim_applied),
        rows_without_detected_speech=sum(1 for obs in observations if not obs.speech_detected),
        total_input_duration_seconds=total_input_duration,
        total_output_duration_seconds=total_output_duration,
        mean_leading_trim_seconds=mean_leading,
        mean_trailing_trim_seconds=mean_trailing,
    )


def _build_examples(records: list[VADComparisonRecord], mode: str) -> list[VADComparisonExample]:
    examples = [
        VADComparisonExample(
            mode=mode,
            audio_path=record.audio_path,
            input_duration_seconds=record.input_duration_seconds,
            output_duration_seconds=record.observations[mode].output_duration_seconds,
            removed_duration_seconds=record.observations[mode].removed_duration_seconds,
            trim_reason=record.observations[mode].trim_reason,
        )
        for record in records
        if record.observations[mode].removed_duration_seconds > 0.0
    ]
    return sorted(
        examples,
        key=lambda example: (-example.removed_duration_seconds, example.audio_path),
    )[:5]


def _normalize_modes(modes: tuple[str, ...]) -> tuple[str, ...]:
    normalized: list[str] = []
    for mode in modes:
        lowered = mode.lower()
        if lowered not in SUPPORTED_VAD_MODES:
            raise ValueError(
                f"Unsupported VAD mode {mode!r}; expected one of {SUPPORTED_VAD_MODES}"
            )
        if lowered not in normalized:
            normalized.append(lowered)
    if not normalized:
        raise ValueError("At least one VAD mode must be requested")
    return tuple(normalized)
