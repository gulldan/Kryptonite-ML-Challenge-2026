"""Parity reporting helpers for offline vs streaming Fbank extraction."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from kryptonite.config import FeaturesConfig, NormalizationConfig, VADConfig
from kryptonite.data import AudioLoadRequest, iter_manifest_audio
from kryptonite.deployment import resolve_project_path

from .fbank import FbankExtractionRequest, FbankExtractor

REPORT_JSON_NAME = "fbank_parity_report.json"
REPORT_MARKDOWN_NAME = "fbank_parity_report.md"
REPORT_ROWS_NAME = "fbank_parity_rows.jsonl"


@dataclass(frozen=True, slots=True)
class FbankParityRecord:
    manifest_path: str
    line_number: int | None
    audio_path: str
    speaker_id: str | None
    utterance_id: str | None
    input_duration_seconds: float
    offline_frame_count: int
    online_frame_count: int
    feature_dim: int
    max_abs_diff: float | None
    mean_abs_diff: float | None
    parity_passed: bool

    def to_dict(self) -> dict[str, object]:
        return {
            "manifest_path": self.manifest_path,
            "line_number": self.line_number,
            "audio_path": self.audio_path,
            "speaker_id": self.speaker_id,
            "utterance_id": self.utterance_id,
            "input_duration_seconds": self.input_duration_seconds,
            "offline_frame_count": self.offline_frame_count,
            "online_frame_count": self.online_frame_count,
            "feature_dim": self.feature_dim,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "parity_passed": self.parity_passed,
        }


@dataclass(frozen=True, slots=True)
class FbankParitySummary:
    row_count: int
    passed_row_count: int
    frame_mismatch_row_count: int
    max_abs_diff: float | None
    mean_abs_diff: float | None
    atol: float

    @property
    def passed(self) -> bool:
        return self.row_count > 0 and self.passed_row_count == self.row_count

    def to_dict(self) -> dict[str, object]:
        return {
            "row_count": self.row_count,
            "passed_row_count": self.passed_row_count,
            "frame_mismatch_row_count": self.frame_mismatch_row_count,
            "max_abs_diff": self.max_abs_diff,
            "mean_abs_diff": self.mean_abs_diff,
            "atol": self.atol,
            "passed": self.passed,
        }


@dataclass(frozen=True, slots=True)
class FbankParityReport:
    project_root: str
    manifest_path: str
    chunk_duration_ms: float
    limit: int | None
    request: FbankExtractionRequest
    records: list[FbankParityRecord]
    summary: FbankParitySummary

    def to_dict(self, *, include_records: bool = False) -> dict[str, object]:
        payload: dict[str, object] = {
            "project_root": self.project_root,
            "manifest_path": self.manifest_path,
            "chunk_duration_ms": self.chunk_duration_ms,
            "limit": self.limit,
            "request": {
                "sample_rate_hz": self.request.sample_rate_hz,
                "num_mel_bins": self.request.num_mel_bins,
                "frame_length_ms": self.request.frame_length_ms,
                "frame_shift_ms": self.request.frame_shift_ms,
                "fft_size": self.request.fft_size,
                "window_type": self.request.window_type,
                "f_min_hz": self.request.f_min_hz,
                "f_max_hz": self.request.f_max_hz,
                "power": self.request.power,
                "log_offset": self.request.log_offset,
                "pad_end": self.request.pad_end,
                "cmvn_mode": self.request.cmvn_mode,
                "cmvn_window_frames": self.request.cmvn_window_frames,
                "output_dtype": self.request.output_dtype,
            },
            "summary": self.summary.to_dict(),
        }
        if include_records:
            payload["records"] = [record.to_dict() for record in self.records]
        return payload


@dataclass(frozen=True, slots=True)
class WrittenFbankParityReport:
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


def build_fbank_parity_report(
    *,
    project_root: Path | str,
    manifest_path: Path | str,
    normalization: NormalizationConfig,
    features: FeaturesConfig,
    vad: VADConfig | None = None,
    chunk_duration_ms: float = 137.0,
    limit: int | None = None,
    atol: float = 1e-5,
) -> FbankParityReport:
    if chunk_duration_ms <= 0.0:
        raise ValueError("chunk_duration_ms must be positive")
    if atol < 0.0:
        raise ValueError("atol must be non-negative")

    feature_request = FbankExtractionRequest.from_config(features)
    extractor = FbankExtractor(request=feature_request)
    audio_request = AudioLoadRequest.from_config(normalization, vad=vad)
    records: list[FbankParityRecord] = []
    for index, loaded in enumerate(
        iter_manifest_audio(
            manifest_path,
            project_root=project_root,
            request=audio_request,
        )
    ):
        if limit is not None and index >= limit:
            break

        offline_features = extractor.extract(
            loaded.audio.waveform,
            sample_rate_hz=loaded.audio.sample_rate_hz,
        )
        online_features = _extract_online_features(
            extractor=extractor,
            waveform=loaded.audio.waveform,
            sample_rate_hz=loaded.audio.sample_rate_hz,
            chunk_duration_ms=chunk_duration_ms,
        )
        parity_passed = offline_features.shape == online_features.shape
        max_abs_diff: float | None = None
        mean_abs_diff: float | None = None
        if parity_passed:
            delta = (
                offline_features.to(dtype=torch.float32) - online_features.to(dtype=torch.float32)
            ).abs()
            max_abs_diff = round(float(delta.max().item()), 8) if delta.numel() > 0 else 0.0
            mean_abs_diff = round(float(delta.mean().item()), 8) if delta.numel() > 0 else 0.0
            parity_passed = max_abs_diff <= atol

        records.append(
            FbankParityRecord(
                manifest_path=loaded.manifest_path or str(manifest_path),
                line_number=loaded.line_number,
                audio_path=loaded.row.audio_path,
                speaker_id=loaded.row.speaker_id,
                utterance_id=loaded.row.utterance_id,
                input_duration_seconds=loaded.audio.duration_seconds,
                offline_frame_count=int(offline_features.shape[0]),
                online_frame_count=int(online_features.shape[0]),
                feature_dim=int(offline_features.shape[-1]) if offline_features.ndim == 2 else 0,
                max_abs_diff=max_abs_diff,
                mean_abs_diff=mean_abs_diff,
                parity_passed=parity_passed,
            )
        )

    frame_mismatch_row_count = sum(
        1 for record in records if record.offline_frame_count != record.online_frame_count
    )
    valid_max_diffs = [record.max_abs_diff for record in records if record.max_abs_diff is not None]
    valid_mean_diffs = [
        record.mean_abs_diff for record in records if record.mean_abs_diff is not None
    ]
    summary = FbankParitySummary(
        row_count=len(records),
        passed_row_count=sum(1 for record in records if record.parity_passed),
        frame_mismatch_row_count=frame_mismatch_row_count,
        max_abs_diff=max(valid_max_diffs, default=None),
        mean_abs_diff=(
            round(sum(valid_mean_diffs) / len(valid_mean_diffs), 8) if valid_mean_diffs else None
        ),
        atol=atol,
    )
    return FbankParityReport(
        project_root=str(resolve_project_path(str(project_root), ".")),
        manifest_path=str(manifest_path),
        chunk_duration_ms=round(chunk_duration_ms, 6),
        limit=limit,
        request=feature_request,
        records=records,
        summary=summary,
    )


def write_fbank_parity_report(
    *,
    report: FbankParityReport,
    output_root: Path | str,
) -> WrittenFbankParityReport:
    output_root_path = resolve_project_path(str(output_root), ".")
    output_root_path.mkdir(parents=True, exist_ok=True)
    json_path = output_root_path / REPORT_JSON_NAME
    markdown_path = output_root_path / REPORT_MARKDOWN_NAME
    rows_path = output_root_path / REPORT_ROWS_NAME

    json_path.write_text(json.dumps(report.to_dict(include_records=True), indent=2, sort_keys=True))
    markdown_path.write_text(render_fbank_parity_markdown(report))
    rows_path.write_text(
        "".join(json.dumps(record.to_dict(), sort_keys=True) + "\n" for record in report.records)
    )

    return WrittenFbankParityReport(
        output_root=str(output_root_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
        rows_path=str(rows_path),
    )


def render_fbank_parity_markdown(report: FbankParityReport) -> str:
    lines = [
        "# Fbank Parity Report",
        "",
        f"- manifest: `{report.manifest_path}`",
        f"- chunk duration ms: `{report.chunk_duration_ms}`",
        f"- rows analyzed: `{report.summary.row_count}`",
        f"- passed rows: `{report.summary.passed_row_count}`",
        f"- frame mismatches: `{report.summary.frame_mismatch_row_count}`",
        f"- max abs diff: `{report.summary.max_abs_diff}`",
        f"- mean abs diff: `{report.summary.mean_abs_diff}`",
        f"- tolerance: `{report.summary.atol}`",
        f"- cmvn mode: `{report.request.cmvn_mode}`",
        f"- output dtype: `{report.request.output_dtype}`",
        "",
        "## Worst Rows",
        "",
        "| audio_path | offline_frames | online_frames | max_abs_diff | parity_passed |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for record in sorted(
        report.records,
        key=lambda current: (current.max_abs_diff is None, -(current.max_abs_diff or 0.0)),
    )[:10]:
        lines.append(
            "| "
            f"`{record.audio_path}` | "
            f"{record.offline_frame_count} | "
            f"{record.online_frame_count} | "
            f"{record.max_abs_diff} | "
            f"{record.parity_passed} |"
        )
    return "\n".join(lines) + "\n"


def _extract_online_features(
    *,
    extractor: FbankExtractor,
    waveform: Any,
    sample_rate_hz: int,
    chunk_duration_ms: float,
) -> torch.Tensor:
    online_extractor = extractor.create_online_extractor()
    waveform_tensor = torch.as_tensor(waveform)
    if waveform_tensor.ndim != 2 or int(waveform_tensor.shape[0]) != 1:
        raise ValueError("Expected waveform shaped as [1, samples] for manifest-backed parity runs")

    chunk_size_samples = max(1, round(sample_rate_hz * chunk_duration_ms / 1000.0))
    feature_chunks: list[torch.Tensor] = []
    total_samples = int(waveform_tensor.shape[-1])
    for start in range(0, total_samples, chunk_size_samples):
        feature_chunks.append(
            online_extractor.push(
                waveform_tensor[:, start : start + chunk_size_samples],
                sample_rate_hz=sample_rate_hz,
            )
        )
    feature_chunks.append(online_extractor.flush())

    non_empty_chunks = [chunk for chunk in feature_chunks if chunk.numel() > 0]
    if not non_empty_chunks:
        return torch.empty(
            (0, extractor.request.num_mel_bins), dtype=extractor.request.torch_output_dtype
        )
    return torch.cat(non_empty_chunks, dim=0)


__all__ = [
    "FbankParityRecord",
    "FbankParityReport",
    "FbankParitySummary",
    "WrittenFbankParityReport",
    "build_fbank_parity_report",
    "render_fbank_parity_markdown",
    "write_fbank_parity_report",
]
