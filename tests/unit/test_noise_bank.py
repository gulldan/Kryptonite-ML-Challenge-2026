from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import soundfile as sf

from kryptonite.data.noise_bank import (
    build_noise_bank,
    load_noise_bank_plan,
    render_noise_bank_markdown,
    write_noise_bank_report,
)
from kryptonite.data.normalization import AudioNormalizationPolicy


def test_load_noise_bank_plan_parses_sources_and_severity_profiles(tmp_path: Path) -> None:
    plan_path = _write_plan(tmp_path)

    plan = load_noise_bank_plan(plan_path)

    assert plan.notes == ("test note",)
    assert plan.severity_profiles["heavy"].snr_db_max == 8.0
    assert plan.sources[0].id == "musan-noise"
    assert plan.sources[0].classification_rules[0].category == "impulsive"


def test_build_noise_bank_normalizes_audio_and_tracks_missing_sources(tmp_path: Path) -> None:
    _write_noise_fixture(
        tmp_path / "datasets" / "musan" / "noise" / "alarms" / "alarm.wav", 8_000, 2
    )
    _write_noise_fixture(
        tmp_path / "datasets" / "musan" / "noise" / "traffic" / "street.wav",
        16_000,
        1,
    )
    _write_noise_fixture(
        tmp_path / "datasets" / "musan" / "speech" / "crowd" / "babble.wav", 16_000, 1
    )
    _write_noise_fixture(tmp_path / "datasets" / "musan" / "music" / "loop.wav", 22_050, 2)
    plan_path = _write_plan(tmp_path)

    report = build_noise_bank(
        project_root=tmp_path,
        dataset_root="datasets",
        output_root="artifacts/corruptions/noise-bank",
        plan=load_noise_bank_plan(plan_path),
        plan_path=plan_path,
        policy=_policy(),
    )

    entries_by_category = {entry.category: entry for entry in report.entries}

    assert report.summary.entry_count == 4
    assert report.summary.missing_source_count == 1
    assert report.summary.category_counts == {
        "babble": 1,
        "music": 1,
        "impulsive": 1,
        "low_snr": 1,
    }
    assert report.summary.severity_counts == {"medium": 2, "heavy": 2}
    assert entries_by_category["impulsive"].severity == "heavy"
    assert entries_by_category["low_snr"].mix_mode == "additive"
    assert entries_by_category["music"].normalization_downmixed is True
    assert entries_by_category["impulsive"].normalization_resampled is True
    assert (tmp_path / entries_by_category["babble"].normalized_audio_path).is_file()

    waveform, sample_rate_hz = sf.read(
        str(tmp_path / entries_by_category["impulsive"].normalized_audio_path),
        always_2d=True,
        dtype="float32",
    )
    assert sample_rate_hz == 16_000
    assert waveform.shape[1] == 1


def test_write_noise_bank_report_emits_manifest_and_markdown(tmp_path: Path) -> None:
    _write_noise_fixture(tmp_path / "datasets" / "musan" / "music" / "loop.wav", 16_000, 1)
    plan_path = _write_plan(tmp_path)
    report = build_noise_bank(
        project_root=tmp_path,
        dataset_root="datasets",
        output_root="artifacts/corruptions/noise-bank",
        plan=load_noise_bank_plan(plan_path),
        plan_path=plan_path,
        policy=_policy(),
    )

    written = write_noise_bank_report(report=report)
    manifest_rows = _read_jsonl(Path(written.manifest_path))
    report_payload = json.loads(Path(written.json_path).read_text())
    markdown = render_noise_bank_markdown(report)

    assert Path(written.manifest_path).is_file()
    assert Path(written.markdown_path).is_file()
    assert len(manifest_rows) == 1
    assert report_payload["summary"]["entry_count"] == 1
    assert "## Source Status" in markdown
    assert "MUSAN music" in markdown


def _write_plan(tmp_path: Path) -> Path:
    plan_path = tmp_path / "noise-bank.toml"
    plan_path.write_text(
        """
notes = ["test note"]

[severity_profiles.light]
snr_db_min = 15.0
snr_db_max = 25.0

[severity_profiles.medium]
snr_db_min = 8.0
snr_db_max = 15.0

[severity_profiles.heavy]
snr_db_min = 0.0
snr_db_max = 8.0

[[sources]]
id = "musan-noise"
name = "MUSAN noise"
inventory_source_id = "musan"
root_candidates = ["datasets/musan/noise"]
default_category = "stationary"
default_severity = "medium"
tags = ["musan"]

[[sources.classification_rules]]
match_any = ["alarm"]
category = "impulsive"
severity = "heavy"
tags = ["transient"]

[[sources.classification_rules]]
match_any = ["street"]
category = "low_snr"
severity = "heavy"
tags = ["ambient"]

[[sources]]
id = "musan-speech"
name = "MUSAN speech"
inventory_source_id = "musan"
root_candidates = ["datasets/musan/speech"]
default_category = "babble"
default_severity = "medium"

[[sources]]
id = "musan-music"
name = "MUSAN music"
inventory_source_id = "musan"
root_candidates = ["datasets/musan/music"]
default_category = "music"
default_severity = "medium"

[[sources]]
id = "rirs-pointsource-noises"
name = "OpenSLR point-source noises"
inventory_source_id = "rirs-noises"
root_candidates = ["datasets/rirs_noises/pointsource_noises"]
default_category = "low_snr"
default_severity = "heavy"
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


def _write_noise_fixture(path: Path, sample_rate_hz: int, channels: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sample_count = sample_rate_hz
    base = (
        np.random.default_rng(123).normal(0.0, 0.1, size=(sample_count, channels)).astype("float32")
    )
    sf.write(path, base, sample_rate_hz, format="WAV")


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
