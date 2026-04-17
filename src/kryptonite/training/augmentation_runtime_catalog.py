"""Catalog construction helpers for scheduled training augmentations."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import cast

from kryptonite.config import AugmentationSchedulerConfig, SilenceAugmentationConfig
from kryptonite.deployment import resolve_project_path

from .augmentation_scheduler import (
    AugmentationCandidate,
    AugmentationCatalog,
    AugmentationFamily,
    AugmentationSeverity,
)


def build_augmentation_catalog_from_manifests(
    *,
    project_root: Path | str,
    scheduler_config: AugmentationSchedulerConfig,
    silence_config: SilenceAugmentationConfig,
    noise_manifest_path: Path | str,
    room_config_manifest_path: Path | str,
    rir_manifest_path: Path | str,
    distance_manifest_path: Path | str,
    codec_manifest_path: Path | str,
    require_audio_files: bool = False,
) -> AugmentationCatalog:
    project_root_path = Path(project_root)
    candidates_by_family: dict[AugmentationFamily, tuple[AugmentationCandidate, ...]] = {
        "noise": (
            _load_noise_candidates(
                project_root_path,
                Path(noise_manifest_path),
                require_audio_files=require_audio_files,
            )
            + _load_musan_candidates(
                project_root_path,
                require_audio_files=require_audio_files,
            )
        ),
        "reverb": (
            _load_reverb_candidates(
                project_root_path,
                room_config_manifest=Path(room_config_manifest_path),
                rir_manifest=Path(rir_manifest_path),
                require_audio_files=require_audio_files,
            )
            + _load_direct_rir_candidates(
                project_root_path,
                require_audio_files=require_audio_files,
            )
        ),
        "distance": _load_distance_candidates(project_root_path, Path(distance_manifest_path)),
        "codec": _load_codec_candidates(project_root_path, Path(codec_manifest_path)),
        "silence": _load_silence_candidates(silence_config),
        "speed": (
            _load_speed_candidates()
            if float(getattr(scheduler_config.family_weights, "speed", 0.0)) > 0.0
            else ()
        ),
    }
    return AugmentationCatalog(candidates_by_family=candidates_by_family)


def _candidate_map(
    catalog: AugmentationCatalog,
    family: AugmentationFamily,
) -> dict[str, AugmentationCandidate]:
    return {
        candidate.candidate_id: candidate
        for candidate in catalog.candidates_by_family.get(family, ())
    }


def _load_noise_candidates(
    project_root: Path,
    manifest_path: Path,
    *,
    require_audio_files: bool,
) -> tuple[AugmentationCandidate, ...]:
    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl_records(manifest_path):
        audio_path = _coerce_str(record.get("normalized_audio_path"))
        if not audio_path:
            audio_path = ""
        resolved_audio_path = (
            resolve_project_path(str(project_root), audio_path) if audio_path else None
        )
        if require_audio_files and (
            resolved_audio_path is None or not resolved_audio_path.exists()
        ):
            continue
        severity = _coerce_severity(record.get("severity"))
        category = _coerce_str(record.get("category")) or "noise"
        candidate_id = _coerce_str(record.get("noise_id")) or f"noise-{len(candidates)}"
        candidates.append(
            AugmentationCandidate(
                family="noise",
                candidate_id=candidate_id,
                label=f"noise/{category}/{severity}",
                severity=severity,
                weight=_coerce_float(record.get("sampling_weight"), 1.0),
                metadata={
                    "normalized_audio_path": audio_path,
                    "category": category,
                    "mix_mode": _coerce_str(record.get("mix_mode")) or "additive",
                    "snr_db_min": _coerce_float(record.get("recommended_snr_db_min"), 8.0),
                    "snr_db_max": _coerce_float(record.get("recommended_snr_db_max"), 18.0),
                },
            )
        )
    return tuple(candidates)


def _load_musan_candidates(
    project_root: Path,
    *,
    require_audio_files: bool,
) -> tuple[AugmentationCandidate, ...]:
    musan_root = project_root / "datasets/musan"
    if require_audio_files and not musan_root.exists():
        return ()
    category_specs = (
        ("noise", "musan-noise", "noise", "medium", 1.2, 4.0, 18.0),
        ("music", "musan-music", "music", "medium", 1.0, 8.0, 20.0),
        ("speech", "musan-speech", "babble", "medium", 1.1, 5.0, 15.0),
    )
    candidates: list[AugmentationCandidate] = []
    for directory_name, source_id, category, severity, weight, snr_min, snr_max in category_specs:
        directory = musan_root / directory_name
        if not directory.exists():
            continue
        audio_paths = sorted(
            path
            for pattern in ("*.wav", "*.flac")
            for path in directory.rglob(pattern)
            if path.is_file()
        )
        for index, audio_path in enumerate(audio_paths):
            relative_path = audio_path.relative_to(project_root).as_posix()
            candidates.append(
                AugmentationCandidate(
                    family="noise",
                    candidate_id=f"{source_id}-{index:05d}",
                    label=f"noise/{category}/{severity}",
                    severity=cast(AugmentationSeverity, severity),
                    weight=weight,
                    metadata={
                        "normalized_audio_path": relative_path,
                        "category": category,
                        "mix_mode": "additive",
                        "snr_db_min": snr_min,
                        "snr_db_max": snr_max,
                        "source": "musan",
                    },
                )
            )
    return tuple(candidates)


def _load_reverb_candidates(
    project_root: Path,
    *,
    room_config_manifest: Path,
    rir_manifest: Path,
    require_audio_files: bool,
) -> tuple[AugmentationCandidate, ...]:
    rir_paths: dict[str, str] = {}
    for record in _read_jsonl_records(rir_manifest):
        rir_id = _coerce_str(record.get("rir_id"))
        audio_path = _coerce_str(record.get("normalized_audio_path"))
        if rir_id and audio_path and resolve_project_path(str(project_root), audio_path).exists():
            rir_paths[rir_id] = audio_path

    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl_records(room_config_manifest):
        sample_ids = _coerce_string_list(record.get("sample_rir_ids"))
        audio_paths = tuple(rir_paths[rir_id] for rir_id in sample_ids if rir_id in rir_paths)
        if require_audio_files and not audio_paths:
            continue
        config_id = _coerce_str(record.get("config_id")) or f"reverb-{len(candidates)}"
        direct_condition = _coerce_str(record.get("direct_condition")) or "medium"
        severity = _severity_from_direct_condition(direct_condition)
        candidates.append(
            AugmentationCandidate(
                family="reverb",
                candidate_id=config_id,
                label=(
                    "reverb/"
                    f"{_coerce_str(record.get('room_size')) or 'room'}/"
                    f"{_coerce_str(record.get('rt60_bucket')) or 'rt60'}/"
                    f"{severity}"
                ),
                severity=severity,
                weight=max(1.0, _coerce_float(record.get("rir_count"), 1.0) ** 0.25),
                metadata={
                    "rir_audio_paths": audio_paths,
                    "direct_condition": direct_condition,
                    "field": _coerce_str(record.get("field")) or "",
                    "rt60_bucket": _coerce_str(record.get("rt60_bucket")) or "",
                },
            )
        )
    return tuple(candidates)


def _load_direct_rir_candidates(
    project_root: Path,
    *,
    require_audio_files: bool,
) -> tuple[AugmentationCandidate, ...]:
    candidate_roots = (
        project_root / "datasets/rirs_noises/RIRS_NOISES",
        project_root / "datasets/RIRS_NOISES",
    )
    roots = tuple(root for root in candidate_roots if root.exists())
    if require_audio_files and not roots:
        return ()
    audio_paths = sorted(
        path
        for root in roots
        for directory in (
            root / "simulated_rirs",
            root / "real_rirs_isotropic_noises",
        )
        if directory.exists()
        for path in directory.rglob("*.wav")
        if path.is_file()
    )
    candidates: list[AugmentationCandidate] = []
    for index, audio_path in enumerate(audio_paths):
        relative_path = audio_path.relative_to(project_root).as_posix()
        severity, field, direct_condition = _direct_rir_profile(audio_path)
        candidates.append(
            AugmentationCandidate(
                family="reverb",
                candidate_id=f"direct-rir-{index:06d}",
                label=f"reverb/raw-rirs/{field}/{severity}",
                severity=severity,
                weight=1.0,
                metadata={
                    "rir_audio_paths": (relative_path,),
                    "direct_condition": direct_condition,
                    "field": field,
                    "rt60_bucket": "raw",
                    "source": "rirs_noises_raw",
                },
            )
        )
    return tuple(candidates)


def _direct_rir_profile(path: Path) -> tuple[AugmentationSeverity, str, str]:
    parts = {part.lower() for part in path.parts}
    if "largeroom" in parts:
        return "heavy", "far", "low"
    if "mediumroom" in parts:
        return "medium", "mid", "medium"
    if "smallroom" in parts:
        return "light", "near", "high"
    return "medium", "mid", "medium"


def _load_distance_candidates(
    project_root: Path,
    manifest_path: Path,
) -> tuple[AugmentationCandidate, ...]:
    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl_records(manifest_path):
        kernel_path = _coerce_str(record.get("kernel_audio_path"))
        if kernel_path and not resolve_project_path(str(project_root), kernel_path).exists():
            kernel_path = ""
        preset_id = _coerce_str(record.get("preset_id")) or f"distance-{len(candidates)}"
        field = _coerce_str(record.get("field")) or "mid"
        severity = _severity_from_distance_field(field)
        candidates.append(
            AugmentationCandidate(
                family="distance",
                candidate_id=preset_id,
                label=f"distance/{field}/{severity}",
                severity=severity,
                weight=_coerce_float(record.get("sampling_weight"), 1.0),
                metadata={
                    "kernel_audio_path": kernel_path,
                    "field": field,
                    "lowpass_hz": _coerce_float(record.get("lowpass_hz"), 5600.0),
                    "attenuation_db": _coerce_float(record.get("attenuation_db"), 3.0),
                    "target_drr_db": _coerce_float(record.get("target_drr_db"), 0.0),
                },
            )
        )
    return tuple(candidates)


def _load_codec_candidates(
    project_root: Path,
    manifest_path: Path,
) -> tuple[AugmentationCandidate, ...]:
    del project_root
    candidates: list[AugmentationCandidate] = []
    for record in _read_jsonl_records(manifest_path):
        preset_id = _coerce_str(record.get("preset_id")) or f"codec-{len(candidates)}"
        family = _coerce_str(record.get("family")) or "codec"
        severity = _coerce_severity(record.get("severity"))
        candidates.append(
            AugmentationCandidate(
                family="codec",
                candidate_id=preset_id,
                label=f"codec/{family}/{severity}",
                severity=severity,
                weight=_coerce_float(record.get("sampling_weight"), 1.0),
                metadata={
                    "codec_family": family,
                    "pre_filter_graph": _coerce_str(record.get("ffmpeg_pre_filter_graph")) or "",
                    "post_filter_graph": _coerce_str(record.get("ffmpeg_post_filter_graph")) or "",
                    "codec_name": _coerce_str(record.get("ffmpeg_encode_codec")) or "",
                },
            )
        )
    return tuple(candidates)


def _load_silence_candidates(
    silence_config: SilenceAugmentationConfig,
) -> tuple[AugmentationCandidate, ...]:
    if not _has_effective_silence_config(silence_config):
        return ()
    return (
        AugmentationCandidate(
            family="silence",
            candidate_id="silence-light",
            label="silence/edge-pause/light",
            severity="light",
            weight=1.0,
            metadata={"scale": 0.45},
        ),
        AugmentationCandidate(
            family="silence",
            candidate_id="silence-medium",
            label="silence/edge-pause/medium",
            severity="medium",
            weight=1.0,
            metadata={"scale": 0.70},
        ),
        AugmentationCandidate(
            family="silence",
            candidate_id="silence-heavy",
            label="silence/edge-pause/heavy",
            severity="heavy",
            weight=1.0,
            metadata={"scale": 1.0},
        ),
    )


def _load_speed_candidates() -> tuple[AugmentationCandidate, ...]:
    return (
        AugmentationCandidate(
            family="speed",
            candidate_id="speed-0.95",
            label="speed/0.95/light",
            severity="light",
            weight=1.0,
            metadata={"speed_factor": 0.95},
        ),
        AugmentationCandidate(
            family="speed",
            candidate_id="speed-1.05",
            label="speed/1.05/light",
            severity="light",
            weight=1.0,
            metadata={"speed_factor": 1.05},
        ),
        AugmentationCandidate(
            family="speed",
            candidate_id="speed-0.90",
            label="speed/0.90/medium",
            severity="medium",
            weight=0.8,
            metadata={"speed_factor": 0.90},
        ),
        AugmentationCandidate(
            family="speed",
            candidate_id="speed-1.10",
            label="speed/1.10/medium",
            severity="medium",
            weight=0.8,
            metadata={"speed_factor": 1.10},
        ),
    )


def _read_jsonl_records(path: Path) -> tuple[dict[str, object], ...]:
    if not path.exists():
        return ()
    rows: list[dict[str, object]] = []
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        payload = json.loads(line)
        if isinstance(payload, dict):
            rows.append(payload)
    return tuple(rows)


def _coerce_str(value: object) -> str:
    if isinstance(value, str):
        return value.strip()
    return ""


def _coerce_float(value: object, default: float) -> float:
    if isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except ValueError:
            return default
    return default


def _coerce_string_list(value: object) -> tuple[str, ...]:
    if isinstance(value, list):
        return tuple(item.strip() for item in value if isinstance(item, str) and item.strip())
    return ()


def _coerce_severity(value: object) -> AugmentationSeverity:
    normalized = _coerce_str(value).lower()
    if normalized in {"light", "medium", "heavy"}:
        return cast(AugmentationSeverity, normalized)
    return "medium"


def _severity_from_direct_condition(value: str) -> AugmentationSeverity:
    return cast(
        AugmentationSeverity,
        {"high": "light", "medium": "medium", "low": "heavy"}.get(value.lower(), "medium"),
    )


def _severity_from_distance_field(value: str) -> AugmentationSeverity:
    return cast(
        AugmentationSeverity,
        {"near": "light", "mid": "medium", "far": "heavy"}.get(value.lower(), "medium"),
    )


def _has_effective_silence_config(config: SilenceAugmentationConfig) -> bool:
    return any(
        (
            config.max_leading_padding_seconds > 0.0,
            config.max_trailing_padding_seconds > 0.0,
            config.max_inserted_pauses > 0,
            not math.isclose(config.pause_ratio_min, 1.0),
            not math.isclose(config.pause_ratio_max, 1.0),
        )
    )
