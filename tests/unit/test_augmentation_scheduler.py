from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

from kryptonite.config import (
    AugmentationSchedulerConfig,
    SilenceAugmentationConfig,
)
from kryptonite.training.augmentation_scheduler import (
    AugmentationCandidate,
    AugmentationCatalog,
    AugmentationFamily,
    AugmentationScheduler,
    build_augmentation_scheduler_report,
    resolve_bank_manifest_paths,
    write_augmentation_scheduler_report,
)


def test_augmentation_scheduler_plan_reduces_clean_ratio_and_raises_heavy_ratio() -> None:
    scheduler = AugmentationScheduler(
        config=AugmentationSchedulerConfig(enabled=True),
        catalog=_catalog(),
        total_epochs=6,
    )

    first = scheduler.plan_for_epoch(1)
    last = scheduler.plan_for_epoch(6)

    assert first.stage == "warmup"
    assert last.stage == "steady"
    assert first.max_augmentations_per_sample == 1
    assert last.max_augmentations_per_sample == 2
    assert first.intensity_probabilities["clean"] > last.intensity_probabilities["clean"]
    assert first.intensity_probabilities["heavy"] < last.intensity_probabilities["heavy"]
    assert last.family_probabilities["codec"] > 0.0


def test_build_augmentation_scheduler_report_tracks_family_and_severity_coverage(
    tmp_path: Path,
) -> None:
    manifest_paths = _write_scheduler_manifests(tmp_path)
    report = build_augmentation_scheduler_report(
        project_root=tmp_path,
        scheduler_config=AugmentationSchedulerConfig(enabled=True),
        silence_config=_silence_config(),
        total_epochs=6,
        samples_per_epoch=256,
        seed=11,
        noise_manifest_path=manifest_paths["noise"],
        room_config_manifest_path=manifest_paths["room"],
        distance_manifest_path=manifest_paths["distance"],
        codec_manifest_path=manifest_paths["codec"],
    )

    written = write_augmentation_scheduler_report(
        report=report,
        output_root=tmp_path / "artifacts" / "reports" / "augmentation-scheduler",
    )

    assert report.catalog.candidate_counts_by_family["silence"] == 3
    assert report.summary.missing_families == ()
    assert (
        report.epochs[0].empirical_intensity_ratios["clean"]
        > (report.epochs[-1].empirical_intensity_ratios["clean"])
    )
    assert (
        report.epochs[-1].empirical_intensity_ratios["heavy"]
        > (report.epochs[0].empirical_intensity_ratios["heavy"])
    )
    assert set(report.epochs[-1].family_coverage) == {
        "noise",
        "reverb",
        "distance",
        "codec",
        "silence",
    }
    assert report.summary.overall_family_counts["codec"] > 0
    assert report.summary.overall_severity_counts["heavy"] > 0
    assert Path(written.json_path).is_file()
    assert Path(written.epochs_path).is_file()
    assert "Epoch 6" in Path(written.markdown_path).read_text()


def test_resolve_bank_manifest_paths_falls_back_to_rir_glob(tmp_path: Path) -> None:
    room_config_path = (
        tmp_path
        / "artifacts"
        / "corruptions"
        / "rir-bank-smoke"
        / "manifests"
        / "room_simulation_configs.jsonl"
    )
    room_config_path.parent.mkdir(parents=True)
    room_config_path.write_text("")

    resolved = resolve_bank_manifest_paths(project_root=tmp_path)

    assert resolved.room_config_manifest_path == str(room_config_path)


def _catalog() -> AugmentationCatalog:
    families: tuple[AugmentationFamily, ...] = (
        "noise",
        "reverb",
        "distance",
        "codec",
        "silence",
    )
    severities: tuple[Literal["light", "medium", "heavy"], ...] = ("light", "medium", "heavy")
    candidates_by_family: dict[AugmentationFamily, tuple[AugmentationCandidate, ...]] = {}
    for family in families:
        candidates = []
        for severity in severities:
            candidates.append(
                AugmentationCandidate(
                    family=family,
                    candidate_id=f"{family}-{severity}",
                    label=f"{family}/{severity}",
                    severity=severity,
                    weight=1.0,
                )
            )
        candidates_by_family[family] = tuple(candidates)
    return AugmentationCatalog(candidates_by_family=candidates_by_family)


def _silence_config() -> SilenceAugmentationConfig:
    return SilenceAugmentationConfig(
        enabled=False,
        max_leading_padding_seconds=0.15,
        max_trailing_padding_seconds=0.20,
        max_inserted_pauses=2,
        min_inserted_pause_seconds=0.08,
        max_inserted_pause_seconds=0.25,
        pause_ratio_min=0.9,
        pause_ratio_max=1.4,
        min_detected_pause_seconds=0.08,
        max_perturbed_pause_seconds=0.6,
    )


def _write_scheduler_manifests(tmp_path: Path) -> dict[str, Path]:
    manifests_root = tmp_path / "artifacts" / "corruptions"

    noise_path = manifests_root / "noise-bank" / "manifests" / "noise_bank_manifest.jsonl"
    noise_rows: list[dict[str, object]] = [
        {
            "noise_id": "noise-light",
            "category": "stationary",
            "severity": "light",
            "sampling_weight": 1.0,
            "mix_mode": "additive",
            "recommended_snr_db_min": 15.0,
            "recommended_snr_db_max": 24.0,
        },
        {
            "noise_id": "noise-medium",
            "category": "babble",
            "severity": "medium",
            "sampling_weight": 1.1,
            "mix_mode": "babble_overlay",
            "recommended_snr_db_min": 8.0,
            "recommended_snr_db_max": 15.0,
        },
        {
            "noise_id": "noise-heavy",
            "category": "low_snr",
            "severity": "heavy",
            "sampling_weight": 1.2,
            "mix_mode": "additive",
            "recommended_snr_db_min": 0.0,
            "recommended_snr_db_max": 8.0,
        },
    ]
    _write_jsonl(noise_path, noise_rows)

    room_path = manifests_root / "rir-bank" / "manifests" / "room_simulation_configs.jsonl"
    room_rows: list[dict[str, object]] = [
        {
            "config_id": "room-high",
            "room_size": "small",
            "field": "near",
            "rt60_bucket": "short",
            "direct_condition": "high",
            "rir_count": 3,
        },
        {
            "config_id": "room-medium",
            "room_size": "medium",
            "field": "mid",
            "rt60_bucket": "medium",
            "direct_condition": "medium",
            "rir_count": 4,
        },
        {
            "config_id": "room-low",
            "room_size": "large",
            "field": "far",
            "rt60_bucket": "long",
            "direct_condition": "low",
            "rir_count": 5,
        },
    ]
    _write_jsonl(room_path, room_rows)

    distance_path = (
        manifests_root / "far-field-bank" / "manifests" / "far_field_bank_manifest.jsonl"
    )
    distance_rows: list[dict[str, object]] = [
        {
            "preset_id": "distance-near",
            "field": "near",
            "sampling_weight": 1.0,
            "distance_meters": 0.8,
            "target_drr_db": 9.0,
        },
        {
            "preset_id": "distance-mid",
            "field": "mid",
            "sampling_weight": 1.1,
            "distance_meters": 2.0,
            "target_drr_db": 2.0,
        },
        {
            "preset_id": "distance-far",
            "field": "far",
            "sampling_weight": 1.3,
            "distance_meters": 4.5,
            "target_drr_db": -4.0,
        },
    ]
    _write_jsonl(distance_path, distance_rows)

    codec_path = manifests_root / "codec-bank" / "manifests" / "codec_bank_manifest.jsonl"
    codec_rows: list[dict[str, object]] = [
        {
            "preset_id": "codec-light",
            "family": "band_limit",
            "severity": "light",
            "sampling_weight": 1.0,
            "ffmpeg_encode_codec": None,
        },
        {
            "preset_id": "codec-medium",
            "family": "telephony",
            "severity": "medium",
            "sampling_weight": 1.1,
            "ffmpeg_encode_codec": "pcm_mulaw",
        },
        {
            "preset_id": "codec-heavy",
            "family": "compression",
            "severity": "heavy",
            "sampling_weight": 1.2,
            "ffmpeg_encode_codec": "aac",
        },
    ]
    _write_jsonl(codec_path, codec_rows)

    return {
        "noise": noise_path,
        "room": room_path,
        "distance": distance_path,
        "codec": codec_path,
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
