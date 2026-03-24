from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf

from kryptonite.data.normalization import AudioNormalizationPolicy
from kryptonite.data.rir_bank import (
    build_rir_bank,
    load_rir_bank_plan,
    render_rir_bank_markdown,
    write_rir_bank_report,
)


def test_load_rir_bank_plan_parses_sources_and_analysis_ranges(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)

    plan = load_rir_bank_plan(plan_path)

    assert plan.notes == ("test note",)
    assert plan.analysis.preview_bins == 16
    assert plan.analysis.rt60_buckets["long"].minimum == 0.9
    assert plan.sources[1].room_family == "simulated"
    assert plan.sources[1].classification_rules[0].room_size == "small"


def test_build_rir_bank_normalizes_audio_and_derives_room_configs(tmp_path: Path) -> None:
    _write_rir_fixture(
        tmp_path / "datasets" / "rirs_noises" / "simulated_rirs" / "smallroom" / "near.wav",
        sample_rate_hz=8_000,
        channels=2,
        rt60_seconds=0.25,
        drr_db=12.0,
    )
    _write_rir_fixture(
        tmp_path / "datasets" / "rirs_noises" / "simulated_rirs" / "mediumroom" / "mid.wav",
        sample_rate_hz=16_000,
        channels=1,
        rt60_seconds=0.6,
        drr_db=3.0,
    )
    _write_rir_fixture(
        tmp_path
        / "datasets"
        / "rirs_noises"
        / "real_rirs_isotropic_noises"
        / "real_rirs_isotropic_noises"
        / "largeroom"
        / "far.wav",
        sample_rate_hz=22_050,
        channels=2,
        rt60_seconds=1.1,
        drr_db=-10.0,
    )
    plan_path = _write_plan(tmp_path)

    report = build_rir_bank(
        project_root=tmp_path,
        dataset_root="datasets",
        output_root="artifacts/corruptions/rir-bank",
        plan=load_rir_bank_plan(plan_path),
        plan_path=plan_path,
        policy=_policy(),
    )

    entries_by_field = {entry.field: entry for entry in report.entries}

    assert report.summary.entry_count == 3
    assert report.summary.config_count == 3
    assert report.summary.field_counts == {"near": 1, "mid": 1, "far": 1}
    assert report.summary.room_size_counts == {"small": 1, "medium": 1, "large": 1}
    assert report.summary.rt60_counts == {"short": 1, "medium": 1, "long": 1}
    assert report.summary.direct_condition_counts == {"high": 1, "medium": 1, "low": 1}
    assert report.summary.missing_field_coverage == ()
    assert entries_by_field["near"].normalization_resampled is True
    assert entries_by_field["near"].normalization_downmixed is True
    assert entries_by_field["far"].room_family == "real"
    assert entries_by_field["mid"].rt60_bucket == "medium"
    assert all(entry.envelope_preview for entry in report.entries)
    assert len(report.room_configs) == 3
    assert (tmp_path / entries_by_field["near"].normalized_audio_path).is_file()


def test_write_rir_bank_report_emits_manifests_and_markdown(tmp_path: Path) -> None:
    _write_rir_fixture(
        tmp_path / "datasets" / "rirs_noises" / "simulated_rirs" / "smallroom" / "near.wav",
        sample_rate_hz=16_000,
        channels=1,
        rt60_seconds=0.25,
        drr_db=12.0,
    )
    plan_path = _write_plan(tmp_path)

    report = build_rir_bank(
        project_root=tmp_path,
        dataset_root="datasets",
        output_root="artifacts/corruptions/rir-bank",
        plan=load_rir_bank_plan(plan_path),
        plan_path=plan_path,
        policy=_policy(),
    )
    written = write_rir_bank_report(report=report)
    manifest_rows = _read_jsonl(Path(written.manifest_path))
    room_config_rows = _read_jsonl(Path(written.room_config_path))
    report_payload = json.loads(Path(written.json_path).read_text())
    markdown = render_rir_bank_markdown(report)

    assert Path(written.manifest_path).is_file()
    assert Path(written.room_config_path).is_file()
    assert Path(written.markdown_path).is_file()
    assert len(manifest_rows) == 1
    assert len(room_config_rows) == 1
    assert report_payload["summary"]["entry_count"] == 1
    assert "## Room Configs" in markdown
    assert "## Visual Sanity Checks" in markdown


def _write_plan(tmp_path: Path) -> Path:
    plan_path = tmp_path / "rir-bank.toml"
    plan_path.write_text(
        """
notes = ["test note"]

[analysis]
direct_window_ms = 3.0
reverb_start_ms = 10.0
preview_duration_ms = 250.0
preview_bins = 16

[analysis.rt60_buckets.short]
maximum = 0.45

[analysis.rt60_buckets.medium]
minimum = 0.45
maximum = 0.9

[analysis.rt60_buckets.long]
minimum = 0.9

[analysis.field_buckets.near]
minimum = 6.0

[analysis.field_buckets.mid]
minimum = 0.0
maximum = 6.0

[analysis.field_buckets.far]
maximum = 0.0

[analysis.direct_buckets.high]
minimum = 6.0

[analysis.direct_buckets.medium]
minimum = 0.0
maximum = 6.0

[analysis.direct_buckets.low]
maximum = 0.0

[[sources]]
id = "rirs-real"
name = "OpenSLR real room impulse responses"
inventory_source_id = "rirs-noises"
room_family = "real"
root_candidates = ["datasets/rirs_noises/real_rirs_isotropic_noises/real_rirs_isotropic_noises"]
default_room_size = "medium"
tags = ["rirs", "real"]

[[sources.classification_rules]]
match_any = ["largeroom"]
room_size = "large"
tags = ["large-room"]

[[sources]]
id = "rirs-simulated"
name = "OpenSLR simulated room impulse responses"
inventory_source_id = "rirs-noises"
room_family = "simulated"
root_candidates = ["datasets/rirs_noises/simulated_rirs"]
default_room_size = "medium"
tags = ["rirs", "simulated"]

[[sources.classification_rules]]
match_any = ["smallroom"]
room_size = "small"
tags = ["small-room"]
"""
    )
    return plan_path


def _policy() -> AudioNormalizationPolicy:
    return AudioNormalizationPolicy(
        target_sample_rate_hz=16_000,
        target_channels=1,
        output_format="wav",
        output_pcm_bits_per_sample=16,
        peak_headroom_db=1.0,
        dc_offset_threshold=0.01,
        clipped_sample_threshold=0.999,
    )


def _write_rir_fixture(
    path: Path,
    *,
    sample_rate_hz: int,
    channels: int,
    rt60_seconds: float,
    drr_db: float,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    waveform = _synthetic_rir(
        sample_rate_hz=sample_rate_hz,
        channels=channels,
        rt60_seconds=rt60_seconds,
        drr_db=drr_db,
    )
    sf.write(path, waveform, sample_rate_hz, format="WAV")


def _synthetic_rir(
    *,
    sample_rate_hz: int,
    channels: int,
    rt60_seconds: float,
    drr_db: float,
    duration_seconds: float = 1.5,
    peak_time_ms: float = 5.0,
) -> np.ndarray:
    frame_count = int(sample_rate_hz * duration_seconds)
    peak_index = int(round(sample_rate_hz * peak_time_ms / 1000.0))
    waveform = np.zeros((frame_count, channels), dtype=np.float32)
    waveform[peak_index, :] = 0.9

    tail_start = peak_index + int(round(sample_rate_hz * 0.003))
    tail_time = np.arange(frame_count - tail_start, dtype=np.float64) / float(sample_rate_hz)
    tau = rt60_seconds / math.log(1000.0)
    decay_envelope = np.exp(-tail_time / tau)
    rng = np.random.default_rng(123)
    tail = rng.normal(size=(frame_count - tail_start, channels)) * decay_envelope[:, None]

    direct_energy = float(waveform[peak_index, 0] ** 2)
    tail_energy = float(np.square(tail, dtype=np.float64).sum())
    target_tail_energy = direct_energy / (10.0 ** (drr_db / 10.0))
    tail *= math.sqrt(target_tail_energy / tail_energy)
    waveform[tail_start:, :] += tail.astype(np.float32)
    return waveform


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
