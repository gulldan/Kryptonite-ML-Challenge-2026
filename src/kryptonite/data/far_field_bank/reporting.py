"""Build and render reproducible far-field simulation artifacts."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from kryptonite.data.audio_io import write_audio_file
from kryptonite.deployment import resolve_project_path

from .models import (
    ALLOWED_DISTANCE_FIELDS,
    MANIFEST_JSONL_NAME,
    PROBE_AUDIO_NAME,
    REPORT_JSON_NAME,
    REPORT_MARKDOWN_NAME,
    FarFieldBankEntry,
    FarFieldBankPlan,
    FarFieldBankReport,
    FarFieldBankSummary,
    WrittenFarFieldArtifacts,
)
from .simulation import render_far_field_preset, write_probe_audio


def build_far_field_bank(
    *,
    project_root: Path | str,
    output_root: Path | str,
    plan: FarFieldBankPlan,
    plan_path: Path | str | None = None,
) -> FarFieldBankReport:
    project_root_path = resolve_project_path(str(project_root), ".")
    output_root_path = resolve_project_path(str(project_root_path), str(output_root))
    probe_path = output_root_path / "probe" / PROBE_AUDIO_NAME
    probe_metrics = write_probe_audio(probe_path, settings=plan.probe)
    probe_waveform = _load_waveform(probe_path)

    entries: list[FarFieldBankEntry] = []
    for preset in plan.presets:
        rendered = render_far_field_preset(
            waveform=probe_waveform,
            sample_rate_hz=plan.probe.sample_rate_hz,
            preset=preset,
            render_settings=plan.render,
        )
        preview_path = output_root_path / "previews" / f"{preset.id}.wav"
        kernel_path = output_root_path / "kernels" / f"{preset.id}.wav"
        preview_path.parent.mkdir(parents=True, exist_ok=True)
        kernel_path.parent.mkdir(parents=True, exist_ok=True)

        write_audio_file(
            path=preview_path,
            waveform=rendered.preview_waveform,
            sample_rate_hz=plan.probe.sample_rate_hz,
            output_format="wav",
            pcm_bits_per_sample=16,
        )
        write_audio_file(
            path=kernel_path,
            waveform=rendered.kernel_waveform,
            sample_rate_hz=plan.probe.sample_rate_hz,
            output_format="wav",
            pcm_bits_per_sample=16,
        )

        entries.append(
            FarFieldBankEntry(
                preset_id=preset.id,
                name=preset.name,
                field=preset.field,
                description=preset.description,
                distance_meters=preset.distance_meters,
                off_axis_angle_deg=preset.off_axis_angle_deg,
                attenuation_db=preset.attenuation_db,
                target_drr_db=preset.target_drr_db,
                reverb_rt60_seconds=preset.reverb_rt60_seconds,
                lowpass_hz=preset.lowpass_hz,
                high_shelf_db=preset.high_shelf_db,
                tags=preset.tags,
                sampling_weight=preset.sampling_weight,
                probe_audio_path=_relative_to_project(probe_path, project_root_path),
                kernel_audio_path=_relative_to_project(kernel_path, project_root_path),
                preview_audio_path=_relative_to_project(preview_path, project_root_path),
                kernel_sha256=_sha256_file(kernel_path),
                preview_sha256=_sha256_file(preview_path),
                source_metrics=probe_metrics,
                output_metrics=rendered.output_metrics,
                kernel_metrics=rendered.kernel_metrics,
            )
        )

    summary = FarFieldBankSummary(
        preset_count=len(plan.presets),
        rendered_preview_count=len(entries),
        field_counts=_ordered_counts(entry.field for entry in entries),
    )
    resolved_plan_path = (
        str(resolve_project_path(str(project_root_path), str(plan_path)))
        if plan_path is not None
        else None
    )
    return FarFieldBankReport(
        generated_at=_utc_now(),
        project_root=str(project_root_path),
        output_root=str(output_root_path),
        plan_path=resolved_plan_path,
        notes=plan.notes,
        probe=plan.probe,
        render=plan.render,
        probe_audio_path=_relative_to_project(probe_path, project_root_path),
        probe_metrics=probe_metrics,
        entries=tuple(entries),
        summary=summary,
    )


def render_far_field_bank_markdown(report: FarFieldBankReport) -> str:
    missing_fields = report.summary.missing_field_coverage
    coverage_text = "complete" if not missing_fields else ", ".join(missing_fields)
    lines = [
        "# Far-Field Simulation Report",
        "",
        f"- Generated at: `{report.generated_at}`",
        f"- Project root: `{report.project_root}`",
        f"- Output root: `{report.output_root}`",
        f"- Plan path: `{report.plan_path or '-'}`",
        f"- Probe audio: `{report.probe_audio_path}`",
        "",
    ]
    if report.notes:
        lines.extend(["## Notes", ""])
        lines.extend(f"- {note}" for note in report.notes)
        lines.append("")

    lines.extend(
        [
            "## Overview",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Presets", str(report.summary.preset_count)],
                    ["Rendered previews", str(report.summary.rendered_preview_count)],
                    ["Field coverage", coverage_text],
                    ["Field counts", _format_counts(report.summary.field_counts)],
                ],
            ),
            "",
            "## Probe Metrics",
            "",
            _markdown_table(
                ["Metric", "Value"],
                [
                    ["Sample rate (Hz)", str(report.probe_metrics.sample_rate_hz)],
                    ["Duration (s)", f"{report.probe_metrics.duration_seconds:.3f}"],
                    ["Peak amplitude", f"{report.probe_metrics.peak_amplitude:.4f}"],
                    ["RMS (dBFS)", f"{report.probe_metrics.rms_dbfs:.3f}"],
                    ["Spectral centroid (Hz)", f"{report.probe_metrics.spectral_centroid_hz:.1f}"],
                    [
                        "Spectral rolloff 95% (Hz)",
                        f"{report.probe_metrics.spectral_rolloff_95_hz:.1f}",
                    ],
                ],
            ),
            "",
            "## Preset Coverage",
            "",
            _markdown_table(
                [
                    "Preset",
                    "Field",
                    "Distance (m)",
                    "Off-axis",
                    "Target DRR (dB)",
                    "Actual DRR (dB)",
                    "RT60 (s)",
                    "Low-pass (Hz)",
                ],
                [
                    [
                        entry.name,
                        entry.field,
                        f"{entry.distance_meters:.2f}",
                        f"{entry.off_axis_angle_deg:.1f} deg",
                        f"{entry.target_drr_db:.2f}",
                        f"{entry.kernel_metrics.actual_drr_db:.2f}",
                        f"{entry.reverb_rt60_seconds:.2f}",
                        f"{entry.lowpass_hz:.0f}",
                    ]
                    for entry in report.entries
                ],
            ),
            "",
            "## Control Examples",
            "",
            _markdown_table(
                [
                    "Preset",
                    "Arrival delay",
                    "RMS delta (dB)",
                    "Rolloff delta (Hz)",
                    "Preview",
                    "Kernel",
                ],
                [
                    [
                        entry.name,
                        f"{entry.kernel_metrics.arrival_delay_ms:.2f} ms",
                        f"{entry.rms_delta_db:.2f}",
                        f"{entry.rolloff_delta_hz:.1f}",
                        entry.preview_audio_path,
                        entry.kernel_audio_path,
                    ]
                    for entry in report.entries
                ],
            ),
            "",
        ]
    )
    return "\n".join(lines)


def write_far_field_bank_report(
    report: FarFieldBankReport,
    *,
    output_root: Path | str | None = None,
) -> WrittenFarFieldArtifacts:
    output_root_path = Path(output_root or report.output_root).resolve()
    manifest_dir = output_root_path / "manifests"
    report_dir = output_root_path / "reports"
    manifest_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = manifest_dir / MANIFEST_JSONL_NAME
    json_path = report_dir / REPORT_JSON_NAME
    markdown_path = report_dir / REPORT_MARKDOWN_NAME

    manifest_path.write_text(
        "".join(json.dumps(entry.to_dict(), sort_keys=True) + "\n" for entry in report.entries)
    )
    json_path.write_text(
        json.dumps(
            report.to_dict(include_entries=True),
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    markdown_path.write_text(render_far_field_bank_markdown(report) + "\n")

    probe_path = resolve_project_path(report.project_root, report.probe_audio_path)
    return WrittenFarFieldArtifacts(
        output_root=str(output_root_path),
        probe_path=str(probe_path),
        manifest_path=str(manifest_path),
        json_path=str(json_path),
        markdown_path=str(markdown_path),
    )


def _load_waveform(path: Path) -> np.ndarray:
    from kryptonite.data.audio_io import read_audio_file

    waveform, _ = read_audio_file(path)
    return np.asarray(waveform, dtype=np.float32)


def _ordered_counts(values: Iterable[str]) -> dict[str, int]:
    counts = Counter(values)
    return {field: counts.get(field, 0) for field in ALLOWED_DISTANCE_FIELDS}


def _format_counts(counts: dict[str, int]) -> str:
    return ", ".join(f"{name}={count}" for name, count in counts.items())


def _markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(row) + " |" for row in rows)
    return "\n".join(lines)


def _relative_to_project(path: Path, project_root: Path) -> str:
    return path.resolve().relative_to(project_root.resolve()).as_posix()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(64 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _utc_now() -> str:
    return datetime.now(tz=UTC).isoformat(timespec="seconds")


__all__ = [
    "build_far_field_bank",
    "render_far_field_bank_markdown",
    "write_far_field_bank_report",
]
