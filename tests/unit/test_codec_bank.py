from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

from kryptonite.data.codec_bank import (
    build_codec_bank,
    load_codec_bank_plan,
    render_codec_bank_markdown,
    write_codec_bank_report,
)


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None and shutil.which("ffprobe") is not None


def test_load_codec_bank_plan_parses_probe_profiles_and_presets(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)

    plan = load_codec_bank_plan(plan_path)

    assert plan.notes == ("test note",)
    assert plan.probe.sample_rate_hz == 16_000
    assert plan.severity_profiles["heavy"].weight_multiplier == 2.0
    assert plan.presets[1].codec_name == "pcm_mulaw"
    assert plan.presets[2].eq_bands[0].gain_db == 6.0


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg and ffprobe are required")
def test_build_codec_bank_renders_previews_and_is_reproducible(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    plan = load_codec_bank_plan(plan_path)

    report_one = build_codec_bank(
        project_root=tmp_path,
        output_root="artifacts/corruptions/codec-bank",
        plan=plan,
        plan_path=plan_path,
    )
    report_two = build_codec_bank(
        project_root=tmp_path,
        output_root="artifacts/corruptions/codec-bank",
        plan=plan,
        plan_path=plan_path,
    )

    telephony_entry = next(
        entry for entry in report_one.entries if entry.preset_id == "telephony-ulaw"
    )
    clipped_entry = next(
        entry for entry in report_one.entries if entry.preset_id == "channel-clipped"
    )

    assert report_one.ffmpeg.ffmpeg_available is True
    assert report_one.summary.preset_count == 3
    assert report_one.summary.rendered_preview_count == 3
    assert report_one.summary.failure_count == 0
    assert report_one.summary.family_counts == {
        "band_limit": 1,
        "telephony": 1,
        "channel": 1,
    }
    assert telephony_entry.output_metrics.spectral_rolloff_95_hz < (
        report_one.probe_metrics.spectral_rolloff_95_hz
    )
    assert clipped_entry.output_metrics.clipped_sample_ratio >= 0.0001
    assert (tmp_path / telephony_entry.preview_audio_path).is_file()
    assert report_one.entries[0].preview_sha256 == report_two.entries[0].preview_sha256


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg and ffprobe are required")
def test_write_codec_bank_report_emits_manifest_failures_and_markdown(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)
    report = build_codec_bank(
        project_root=tmp_path,
        output_root="artifacts/corruptions/codec-bank",
        plan=load_codec_bank_plan(plan_path),
        plan_path=plan_path,
    )

    written = write_codec_bank_report(report=report)
    manifest_rows = _read_jsonl(Path(written.manifest_path))
    failure_rows = _read_jsonl(Path(written.failures_path))
    report_payload = json.loads(Path(written.json_path).read_text())
    markdown = render_codec_bank_markdown(report)

    assert Path(written.probe_path).is_file()
    assert Path(written.manifest_path).is_file()
    assert Path(written.failures_path).is_file()
    assert Path(written.markdown_path).is_file()
    assert len(manifest_rows) == 3
    assert failure_rows == []
    assert report_payload["summary"]["rendered_preview_count"] == 3
    assert "## FFmpeg Environment" in markdown
    assert "Telephony u-law" in markdown


def _write_plan(tmp_path: Path) -> Path:
    plan_path = tmp_path / "codec-bank.toml"
    plan_path.write_text(
        """
notes = ["test note"]

[probe]
sample_rate_hz = 16000
duration_seconds = 1.5
peak_amplitude = 0.82

[severity_profiles.light]
description = "gentle"
weight_multiplier = 1.0

[severity_profiles.medium]
description = "moderate"
weight_multiplier = 1.4

[severity_profiles.heavy]
description = "aggressive"
weight_multiplier = 2.0

[[presets]]
id = "band-limit-light"
name = "Band-limit light"
family = "band_limit"
severity = "light"
description = "Gentle band limit."
highpass_hz = 120.0
lowpass_hz = 7200.0
tags = ["light"]

[[presets]]
id = "telephony-ulaw"
name = "Telephony u-law"
family = "telephony"
severity = "medium"
description = "Narrowband u-law."
highpass_hz = 300.0
lowpass_hz = 3400.0
pre_gain_db = 2.0
codec_name = "pcm_mulaw"
container_extension = "wav"
encode_sample_rate_hz = 8000
tags = ["telephony"]

[[presets]]
id = "channel-clipped"
name = "Channel clipped"
family = "channel"
severity = "heavy"
description = "Clipped device coloration."
highpass_hz = 160.0
lowpass_hz = 5000.0
pre_gain_db = 11.0
soft_clip = true
bitcrusher_bits = 7
bitcrusher_mix = 0.65
tags = ["clipped"]

[[presets.eq_bands]]
frequency_hz = 1300.0
width_hz = 900.0
gain_db = 6.0
"""
    )
    return plan_path


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
