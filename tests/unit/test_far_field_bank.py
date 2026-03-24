from __future__ import annotations

import json
from pathlib import Path

from kryptonite.data.far_field_bank import (
    build_far_field_bank,
    load_far_field_bank_plan,
    render_far_field_bank_markdown,
    write_far_field_bank_report,
)


def test_load_far_field_bank_plan_parses_probe_render_and_presets(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)

    plan = load_far_field_bank_plan(plan_path)

    assert plan.notes == ("test note",)
    assert plan.probe.sample_rate_hz == 16_000
    assert plan.render.kernel_duration_seconds == 1.25
    assert plan.presets[2].field == "far"
    assert plan.presets[1].early_reflection_gains_db == (-10.0, -13.5, -17.0)


def test_build_far_field_bank_renders_near_mid_far_controls(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    report = build_far_field_bank(
        project_root=tmp_path,
        output_root="artifacts/corruptions/far-field-bank",
        plan=load_far_field_bank_plan(plan_path),
        plan_path=plan_path,
    )

    entries_by_field = {entry.field: entry for entry in report.entries}

    assert report.summary.preset_count == 3
    assert report.summary.rendered_preview_count == 3
    assert report.summary.field_counts == {"near": 1, "mid": 1, "far": 1}
    assert report.summary.missing_field_coverage == ()
    assert entries_by_field["near"].kernel_metrics.actual_drr_db > (
        entries_by_field["mid"].kernel_metrics.actual_drr_db
    )
    assert entries_by_field["mid"].kernel_metrics.actual_drr_db > (
        entries_by_field["far"].kernel_metrics.actual_drr_db
    )
    assert entries_by_field["near"].output_metrics.spectral_rolloff_95_hz > (
        entries_by_field["far"].output_metrics.spectral_rolloff_95_hz
    )
    assert entries_by_field["near"].kernel_metrics.arrival_delay_ms < (
        entries_by_field["far"].kernel_metrics.arrival_delay_ms
    )
    assert (tmp_path / entries_by_field["far"].preview_audio_path).is_file()
    assert (tmp_path / entries_by_field["far"].kernel_audio_path).is_file()


def test_write_far_field_bank_report_emits_manifest_and_markdown(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    report = build_far_field_bank(
        project_root=tmp_path,
        output_root="artifacts/corruptions/far-field-bank",
        plan=load_far_field_bank_plan(plan_path),
        plan_path=plan_path,
    )

    written = write_far_field_bank_report(report=report)
    manifest_rows = _read_jsonl(Path(written.manifest_path))
    report_payload = json.loads(Path(written.json_path).read_text())
    markdown = render_far_field_bank_markdown(report)

    assert Path(written.probe_path).is_file()
    assert Path(written.manifest_path).is_file()
    assert Path(written.markdown_path).is_file()
    assert len(manifest_rows) == 3
    assert report_payload["summary"]["rendered_preview_count"] == 3
    assert "## Control Examples" in markdown
    assert "Far field hall tail" in markdown


def _write_plan(tmp_path: Path) -> Path:
    plan_path = tmp_path / "far-field-bank.toml"
    plan_path.write_text(
        """
notes = ["test note"]

[probe]
sample_rate_hz = 16000
duration_seconds = 2.0
peak_amplitude = 0.82

[render]
kernel_duration_seconds = 1.25
speed_of_sound_mps = 343.0
output_peak_limit = 0.92
high_shelf_pivot_hz = 1800.0

[[presets]]
id = "near-reference"
name = "Near reference"
field = "near"
description = "Close-mic reference with limited room spill."
distance_meters = 0.7
off_axis_angle_deg = 8.0
attenuation_db = 1.0
target_drr_db = 9.0
reverb_rt60_seconds = 0.28
late_reverb_start_ms = 18.0
lowpass_hz = 7200.0
high_shelf_db = -0.5
base_weight = 1.0
early_reflection_delays_ms = [8.0, 15.0]
early_reflection_gains_db = [-18.0, -20.0]
tags = ["near", "reference"]

[[presets]]
id = "mid-room-coupled"
name = "Mid room coupled"
field = "mid"
description = "Meeting-room style capture with audible off-axis coloration."
distance_meters = 2.2
off_axis_angle_deg = 32.0
attenuation_db = 4.0
target_drr_db = 2.0
reverb_rt60_seconds = 0.48
late_reverb_start_ms = 26.0
lowpass_hz = 5400.0
high_shelf_db = -2.0
base_weight = 1.2
early_reflection_delays_ms = [10.0, 22.0, 36.0]
early_reflection_gains_db = [-10.0, -13.5, -17.0]
tags = ["mid", "meeting-room"]

[[presets]]
id = "far-hall-tail"
name = "Far field hall tail"
field = "far"
description = "Low-direct far-field capture with strong room tail and steeper HF loss."
distance_meters = 4.8
off_axis_angle_deg = 65.0
attenuation_db = 7.0
target_drr_db = -4.0
reverb_rt60_seconds = 0.92
late_reverb_start_ms = 35.0
lowpass_hz = 3400.0
high_shelf_db = -4.5
base_weight = 1.4
early_reflection_delays_ms = [14.0, 28.0, 47.0, 66.0]
early_reflection_gains_db = [-7.0, -8.5, -10.0, -12.0]
tags = ["far", "hall"]
"""
    )
    return plan_path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
