from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import cast

import numpy as np
import pytest

from kryptonite.config import NormalizationConfig, SilenceAugmentationConfig
from kryptonite.data.audio_io import write_audio_file
from kryptonite.data.schema import ManifestRow
from kryptonite.eval import build_corrupted_dev_suites, load_corrupted_dev_suites_plan


def _ffmpeg_available() -> bool:
    return shutil.which("ffmpeg") is not None


def test_load_corrupted_dev_suites_plan_parses_weighted_suite_specs(tmp_path: Path) -> None:
    plan_path = _write_suite_plan(tmp_path)

    plan = load_corrupted_dev_suites_plan(plan_path)

    assert plan.seed == 17
    assert plan.output_root == "artifacts/eval/corrupted-dev-suites"
    assert plan.suites[1].family == "reverb"
    assert plan.suites[2].codec_families == ("band_limit", "telephony")
    assert plan.suites[3].distance_field_weights is not None
    assert plan.suites[4].codec_families == ("channel",)


@pytest.mark.skipif(not _ffmpeg_available(), reason="ffmpeg is required")
def test_build_corrupted_dev_suites_writes_manifests_trials_and_catalog(tmp_path: Path) -> None:
    source_manifest_path, source_trial_paths = _write_source_bundle(tmp_path)
    noise_manifest_path = _write_noise_manifest(tmp_path)
    rir_manifest_path, room_config_manifest_path = _write_rir_bundle(tmp_path)
    codec_plan_path = _write_codec_plan(tmp_path)
    far_field_plan_path = _write_far_field_plan(tmp_path)
    suite_plan_path = _write_suite_plan(
        tmp_path,
        source_manifest_path=source_manifest_path,
        source_trial_paths=source_trial_paths,
    )

    report = build_corrupted_dev_suites(
        project_root=tmp_path,
        plan=load_corrupted_dev_suites_plan(suite_plan_path),
        normalization_config=_normalization_config(),
        silence_config=_silence_config(),
        plan_path=suite_plan_path,
        noise_manifest_path=noise_manifest_path,
        rir_manifest_path=rir_manifest_path,
        room_config_manifest_path=room_config_manifest_path,
        codec_plan_path=codec_plan_path,
        far_field_plan_path=far_field_plan_path,
    )

    assert len(report.suites) == 6
    catalog_payload = json.loads((tmp_path / report.catalog_json_path).read_text())
    assert catalog_payload["seed"] == 17
    assert len(catalog_payload["suites"]) == 6

    suites_by_id = {suite.suite_id: suite for suite in report.suites}
    noise_rows = _read_jsonl(tmp_path / suites_by_id["dev_snr"].manifest_path)
    silence_rows = _read_jsonl(tmp_path / suites_by_id["dev_silence"].manifest_path)
    channel_trials = _read_jsonl(tmp_path / suites_by_id["dev_channel"].trial_manifest_paths[0])

    assert len(noise_rows) == 4
    assert noise_rows[0]["corruption_suite"] == "dev_snr"
    assert noise_rows[0]["corruption_family"] == "noise"
    assert "target_snr_db" in _as_dict(noise_rows[0]["corruption_metadata"])
    assert all(str(row["audio_path"]).endswith(".wav") for row in noise_rows)
    assert any(_as_float(row["duration_seconds"]) > 1.0 for row in silence_rows)
    assert str(channel_trials[0]["left_audio"]).endswith(".wav")
    assert (tmp_path / suites_by_id["dev_reverb"].inventory_path).is_file()
    assert (tmp_path / suites_by_id["dev_distance"].suite_summary_markdown_path).is_file()


def _write_source_bundle(tmp_path: Path) -> tuple[Path, tuple[Path, ...]]:
    audio_root = tmp_path / "datasets" / "synthetic-dev"
    audio_root.mkdir(parents=True)
    manifest_path = tmp_path / "artifacts" / "manifests" / "synthetic-dev" / "dev_manifest.jsonl"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    trial_one = manifest_path.parent / "official_dev_trials.jsonl"
    trial_two = manifest_path.parent / "speaker_disjoint_dev_trials.jsonl"

    rows: list[dict[str, object]] = []
    for speaker_index, speaker_id in enumerate(("spk-a", "spk-b"), start=1):
        for utterance_index in range(2):
            audio_name = f"{speaker_id}-{utterance_index + 1}.wav"
            audio_path = audio_root / audio_name
            waveform = _sine_wave(
                sample_rate_hz=16_000,
                duration_seconds=1.0 + 0.05 * utterance_index,
                frequency_hz=220.0 * speaker_index,
            )
            write_audio_file(
                path=audio_path,
                waveform=waveform,
                sample_rate_hz=16_000,
                output_format="wav",
                pcm_bits_per_sample=16,
            )
            rows.append(
                ManifestRow(
                    dataset="synthetic-dev",
                    source_dataset="synthetic-dev-clean",
                    speaker_id=speaker_id,
                    utterance_id=f"{speaker_id}:{utterance_index + 1}",
                    session_id=f"{speaker_id}:session",
                    split="dev",
                    audio_path=audio_path.relative_to(tmp_path).as_posix(),
                    duration_seconds=round(float(waveform.shape[-1]) / 16_000.0, 6),
                    sample_rate_hz=16_000,
                    num_channels=1,
                ).to_dict()
            )

    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    trial_rows = [
        {"label": 1, "left_audio": "spk-a-1.wav", "right_audio": "spk-a-2.wav"},
        {"label": 0, "left_audio": "spk-a-1.wav", "right_audio": "spk-b-1.wav"},
    ]
    speaker_disjoint_rows = [
        {
            "label": 1,
            "left_audio": "spk-b-1.wav",
            "right_audio": "spk-b-2.wav",
            "duration_bucket": "short",
        }
    ]
    trial_one.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in trial_rows))
    trial_two.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in speaker_disjoint_rows)
    )
    return manifest_path, (trial_one, trial_two)


def _write_noise_manifest(tmp_path: Path) -> Path:
    noise_root = tmp_path / "artifacts" / "corruptions" / "noise-bank" / "audio"
    noise_root.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        tmp_path
        / "artifacts"
        / "corruptions"
        / "noise-bank"
        / "manifests"
        / "noise_bank_manifest.jsonl"
    )
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []
    for severity, scale in (("light", 0.02), ("medium", 0.05), ("heavy", 0.08)):
        audio_path = noise_root / f"noise-{severity}.wav"
        write_audio_file(
            path=audio_path,
            waveform=_noise_wave(duration_seconds=1.5, scale=scale),
            sample_rate_hz=16_000,
            output_format="wav",
            pcm_bits_per_sample=16,
        )
        rows.append(
            {
                "noise_id": f"noise-{severity}",
                "normalized_audio_path": audio_path.relative_to(tmp_path).as_posix(),
                "category": "stationary",
                "severity": severity,
                "mix_mode": "additive",
                "sampling_weight": 1.0,
                "recommended_snr_db_min": 4.0,
                "recommended_snr_db_max": 12.0,
            }
        )
    manifest_path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
    return manifest_path


def _write_rir_bundle(tmp_path: Path) -> tuple[Path, Path]:
    audio_root = tmp_path / "artifacts" / "corruptions" / "rir-bank" / "audio"
    audio_root.mkdir(parents=True, exist_ok=True)
    manifests_root = tmp_path / "artifacts" / "corruptions" / "rir-bank" / "manifests"
    manifests_root.mkdir(parents=True, exist_ok=True)
    rir_manifest_path = manifests_root / "rir_bank_manifest.jsonl"
    room_config_path = manifests_root / "room_simulation_configs.jsonl"

    rir_rows = []
    for index, direct_condition in enumerate(("high", "medium", "low"), start=1):
        audio_path = audio_root / f"rir-{direct_condition}.wav"
        write_audio_file(
            path=audio_path,
            waveform=_rir_wave(decay=0.96 - 0.01 * index),
            sample_rate_hz=16_000,
            output_format="wav",
            pcm_bits_per_sample=16,
        )
        rir_rows.append(
            {
                "rir_id": f"rir-{direct_condition}",
                "normalized_audio_path": audio_path.relative_to(tmp_path).as_posix(),
                "room_size": "medium",
                "field": "mid"
                if direct_condition == "medium"
                else ("near" if direct_condition == "high" else "far"),
                "rt60_bucket": "medium",
                "direct_condition": direct_condition,
            }
        )
    room_rows = [
        {
            "config_id": "room-high",
            "room_size": "medium",
            "field": "near",
            "rt60_bucket": "medium",
            "direct_condition": "high",
            "sample_rir_ids": ["rir-high"],
        },
        {
            "config_id": "room-medium",
            "room_size": "medium",
            "field": "mid",
            "rt60_bucket": "medium",
            "direct_condition": "medium",
            "sample_rir_ids": ["rir-medium"],
        },
        {
            "config_id": "room-low",
            "room_size": "medium",
            "field": "far",
            "rt60_bucket": "medium",
            "direct_condition": "low",
            "sample_rir_ids": ["rir-low"],
        },
    ]
    rir_manifest_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rir_rows)
    )
    room_config_path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in room_rows)
    )
    return rir_manifest_path, room_config_path


def _write_codec_plan(tmp_path: Path) -> Path:
    plan_path = tmp_path / "configs" / "corruption" / "codec-bank.toml"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        """
notes = ["codec test"]

[probe]
sample_rate_hz = 16000
duration_seconds = 1.0
peak_amplitude = 0.8

[severity_profiles.light]
description = "gentle"
weight_multiplier = 1.0

[severity_profiles.medium]
description = "moderate"
weight_multiplier = 1.5

[severity_profiles.heavy]
description = "aggressive"
weight_multiplier = 2.0

[[presets]]
id = "band-limit-light"
name = "Band-limit light"
family = "band_limit"
severity = "light"
description = "Filter-only band limit."
highpass_hz = 120.0
lowpass_hz = 7000.0

[[presets]]
id = "telephony-medium"
name = "Telephony medium"
family = "telephony"
severity = "medium"
description = "Narrowband u-law."
highpass_hz = 300.0
lowpass_hz = 3400.0
codec_name = "pcm_mulaw"
container_extension = "wav"
encode_sample_rate_hz = 8000

[[presets]]
id = "channel-heavy"
name = "Channel heavy"
family = "channel"
severity = "heavy"
description = "Clipped channel coloration."
highpass_hz = 180.0
lowpass_hz = 5000.0
pre_gain_db = 9.0
soft_clip = true
"""
    )
    return plan_path


def _write_far_field_plan(tmp_path: Path) -> Path:
    plan_path = tmp_path / "configs" / "corruption" / "far-field-bank.toml"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    plan_path.write_text(
        """
notes = ["distance test"]

[probe]
sample_rate_hz = 16000
duration_seconds = 1.0
peak_amplitude = 0.8

[render]
kernel_duration_seconds = 0.75

[[presets]]
id = "near-reference"
name = "Near reference"
field = "near"
description = "Near field."
distance_meters = 0.8
off_axis_angle_deg = 8.0
attenuation_db = 1.0
target_drr_db = 8.0
reverb_rt60_seconds = 0.20
late_reverb_start_ms = 16.0
lowpass_hz = 7200.0
high_shelf_db = -0.5
early_reflection_delays_ms = [8.0]
early_reflection_gains_db = [-18.0]

[[presets]]
id = "mid-room"
name = "Mid room"
field = "mid"
description = "Mid field."
distance_meters = 2.0
off_axis_angle_deg = 25.0
attenuation_db = 3.0
target_drr_db = 2.0
reverb_rt60_seconds = 0.40
late_reverb_start_ms = 22.0
lowpass_hz = 5600.0
high_shelf_db = -2.0
early_reflection_delays_ms = [10.0, 20.0]
early_reflection_gains_db = [-12.0, -15.0]

[[presets]]
id = "far-hall"
name = "Far hall"
field = "far"
description = "Far field."
distance_meters = 4.5
off_axis_angle_deg = 60.0
attenuation_db = 6.0
target_drr_db = -3.0
reverb_rt60_seconds = 0.80
late_reverb_start_ms = 30.0
lowpass_hz = 3600.0
high_shelf_db = -4.0
early_reflection_delays_ms = [14.0, 26.0, 40.0]
early_reflection_gains_db = [-8.0, -10.0, -12.0]
"""
    )
    return plan_path


def _write_suite_plan(
    tmp_path: Path,
    *,
    source_manifest_path: Path | None = None,
    source_trial_paths: tuple[Path, ...] | None = None,
) -> Path:
    plan_path = tmp_path / "configs" / "corruption" / "corrupted-dev-suites.toml"
    plan_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_value = (
        "artifacts/manifests/synthetic-dev/dev_manifest.jsonl"
        if source_manifest_path is None
        else source_manifest_path.relative_to(tmp_path).as_posix()
    )
    trial_lines = ""
    if source_trial_paths is not None:
        trial_entries = ",\n  ".join(
            f'"{path.relative_to(tmp_path).as_posix()}"' for path in source_trial_paths
        )
        trial_lines = f"trial_manifest_paths = [\n  {trial_entries},\n]\n"
    plan_path.write_text(
        f"""
output_root = "artifacts/eval/corrupted-dev-suites"
source_manifest_path = "{manifest_value}"
{trial_lines}seed = 17

[[suites]]
suite_id = "dev_snr"
family = "noise"
description = "Noise suite."

[[suites]]
suite_id = "dev_reverb"
family = "reverb"
description = "Reverb suite."

[suites.reverb_direct_weights]
high = 1.0
medium = 1.0
low = 1.0

[[suites]]
suite_id = "dev_codec"
family = "codec"
description = "Codec suite."
codec_families = ["band_limit", "telephony"]

[[suites]]
suite_id = "dev_distance"
family = "distance"
description = "Distance suite."

[suites.distance_field_weights]
near = 1.0
mid = 1.0
far = 1.0

[[suites]]
suite_id = "dev_channel"
family = "codec"
description = "Channel suite."
codec_families = ["channel"]

[suites.severity_weights]
light = 0.0
medium = 0.0
heavy = 1.0

[[suites]]
suite_id = "dev_silence"
family = "silence"
description = "Silence suite."
"""
    )
    return plan_path


def _normalization_config() -> NormalizationConfig:
    return NormalizationConfig(
        target_sample_rate_hz=16_000,
        target_channels=1,
        output_format="wav",
        output_pcm_bits_per_sample=16,
        peak_headroom_db=1.0,
        dc_offset_threshold=0.01,
        clipped_sample_threshold=0.999,
    )


def _silence_config() -> SilenceAugmentationConfig:
    return SilenceAugmentationConfig(
        enabled=False,
        max_leading_padding_seconds=0.10,
        max_trailing_padding_seconds=0.12,
        max_inserted_pauses=2,
        min_inserted_pause_seconds=0.06,
        max_inserted_pause_seconds=0.20,
        pause_ratio_min=0.9,
        pause_ratio_max=1.3,
        min_detected_pause_seconds=0.06,
        max_perturbed_pause_seconds=0.40,
    )


def _sine_wave(*, sample_rate_hz: int, duration_seconds: float, frequency_hz: float) -> np.ndarray:
    frame_count = int(round(sample_rate_hz * duration_seconds))
    timeline = np.arange(frame_count, dtype=np.float64) / float(sample_rate_hz)
    waveform = 0.35 * np.sin(2.0 * np.pi * frequency_hz * timeline)
    return waveform[np.newaxis, :].astype("float32")


def _noise_wave(*, duration_seconds: float, scale: float) -> np.ndarray:
    frame_count = int(round(16_000 * duration_seconds))
    rng = np.random.default_rng(17)
    waveform = scale * rng.normal(size=frame_count)
    return waveform[np.newaxis, :].astype("float32")


def _rir_wave(*, decay: float) -> np.ndarray:
    frame_count = 2_400
    values = np.zeros(frame_count, dtype=np.float32)
    values[0] = 1.0
    for index in range(1, frame_count):
        values[index] = (decay**index) * (0.2 if index % 97 == 0 else 0.02)
    return values[np.newaxis, :]


def _read_jsonl(path: Path) -> list[dict[str, object]]:
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def _as_dict(value: object) -> dict[str, object]:
    assert isinstance(value, dict)
    return cast(dict[str, object], value)


def _as_float(value: object) -> float:
    assert isinstance(value, (int, float, str))
    return float(value)
