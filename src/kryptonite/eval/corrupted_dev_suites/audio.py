"""Candidate loading and waveform transforms for corrupted dev suites."""

from __future__ import annotations

import hashlib
import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Protocol, cast

import numpy as np

from kryptonite.config import SilenceAugmentationConfig
from kryptonite.data.audio_io import read_audio_file, resample_waveform
from kryptonite.data.codec_bank import load_codec_bank_plan
from kryptonite.data.codec_bank.ffmpeg import apply_codec_preset
from kryptonite.data.far_field_bank import load_far_field_bank_plan
from kryptonite.data.far_field_bank.simulation import render_far_field_preset
from kryptonite.data.silence_augmentation import apply_silence_augmentation
from kryptonite.deployment import resolve_project_path

from .models import CorruptedDevSuiteSpec, SuiteSeverity


@dataclass(frozen=True, slots=True)
class TransformOutcome:
    waveform: np.ndarray
    sample_rate_hz: int
    candidate_id: str
    severity: SuiteSeverity
    metadata: dict[str, object]


@dataclass(frozen=True, slots=True)
class NoiseSuiteCandidate:
    candidate_id: str
    severity: SuiteSeverity
    weight: float
    audio_path: str
    category: str
    mix_mode: str
    snr_db_min: float
    snr_db_max: float
    tags: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ReverbSuiteCandidate:
    candidate_id: str
    severity: SuiteSeverity
    weight: float
    room_size: str
    field: str
    rt60_bucket: str
    direct_condition: str
    rir_ids: tuple[str, ...]
    rir_audio_paths: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class CodecSuiteCandidate:
    candidate_id: str
    severity: SuiteSeverity
    weight: float
    family: str
    preset: Any


@dataclass(frozen=True, slots=True)
class DistanceSuiteCandidate:
    candidate_id: str
    severity: SuiteSeverity
    weight: float
    field: str
    preset: Any
    render_settings: Any


@dataclass(frozen=True, slots=True)
class SilenceSuiteCandidate:
    candidate_id: str
    severity: SuiteSeverity
    weight: float
    config: SilenceAugmentationConfig


class _WeightedCandidate(Protocol):
    weight: float


def load_noise_candidates(
    *,
    project_root: Path,
    manifest_path: Path | str,
    suite_spec: CorruptedDevSuiteSpec,
) -> tuple[NoiseSuiteCandidate, ...]:
    records = _read_jsonl_records(resolve_project_path(str(project_root), str(manifest_path)))
    candidates: list[NoiseSuiteCandidate] = []
    for record in records:
        severity = _coerce_severity(record.get("severity"))
        normalized_audio_path = _coerce_str(record.get("normalized_audio_path"))
        if severity is None or normalized_audio_path is None:
            continue
        severity_weight = getattr(suite_spec.severity_weights, severity)
        if severity_weight <= 0.0:
            continue
        candidates.append(
            NoiseSuiteCandidate(
                candidate_id=_coerce_str(record.get("noise_id")) or normalized_audio_path,
                severity=severity,
                weight=_coerce_float(record.get("sampling_weight"), default=1.0) * severity_weight,
                audio_path=normalized_audio_path,
                category=_coerce_str(record.get("category")) or "unknown",
                mix_mode=_coerce_str(record.get("mix_mode")) or "additive",
                snr_db_min=_coerce_float(record.get("recommended_snr_db_min"), default=0.0),
                snr_db_max=_coerce_float(record.get("recommended_snr_db_max"), default=0.0),
                tags=tuple(_coerce_string_list(record.get("tags"))),
            )
        )
    return tuple(candidates)


def load_reverb_candidates(
    *,
    project_root: Path,
    rir_manifest_path: Path | str,
    room_config_manifest_path: Path | str,
    suite_spec: CorruptedDevSuiteSpec,
) -> tuple[ReverbSuiteCandidate, ...]:
    rir_records = _read_jsonl_records(
        resolve_project_path(str(project_root), str(rir_manifest_path))
    )
    room_config_records = _read_jsonl_records(
        resolve_project_path(str(project_root), str(room_config_manifest_path))
    )
    rir_lookup = {
        rir_id: record
        for record in rir_records
        if (rir_id := _coerce_str(record.get("rir_id"))) is not None
    }
    candidates: list[ReverbSuiteCandidate] = []
    for record in room_config_records:
        direct_condition = _coerce_str(record.get("direct_condition"))
        if direct_condition is None:
            continue
        direct_weight = getattr(suite_spec.reverb_direct_weights, direct_condition)
        severity = _severity_from_direct_condition(direct_condition)
        severity_weight = getattr(suite_spec.severity_weights, severity)
        if direct_weight <= 0.0 or severity_weight <= 0.0:
            continue
        rir_ids = tuple(
            rir_id
            for rir_id in _coerce_string_list(record.get("sample_rir_ids"))
            if rir_id in rir_lookup
        )
        if not rir_ids:
            rir_ids = tuple(
                rir_id
                for rir_id, rir_record in rir_lookup.items()
                if _coerce_str(rir_record.get("room_size")) == _coerce_str(record.get("room_size"))
                and _coerce_str(rir_record.get("field")) == _coerce_str(record.get("field"))
                and _coerce_str(rir_record.get("rt60_bucket"))
                == _coerce_str(record.get("rt60_bucket"))
                and _coerce_str(rir_record.get("direct_condition")) == direct_condition
            )
        rir_audio_paths = tuple(
            _coerce_str(rir_lookup[rir_id].get("normalized_audio_path")) or ""
            for rir_id in rir_ids
            if _coerce_str(rir_lookup[rir_id].get("normalized_audio_path"))
        )
        if not rir_audio_paths:
            continue
        candidates.append(
            ReverbSuiteCandidate(
                candidate_id=_coerce_str(record.get("config_id")) or "-".join(rir_ids),
                severity=severity,
                weight=direct_weight * severity_weight * max(float(len(rir_audio_paths)), 1.0),
                room_size=_coerce_str(record.get("room_size")) or "unknown",
                field=_coerce_str(record.get("field")) or "unknown",
                rt60_bucket=_coerce_str(record.get("rt60_bucket")) or "unknown",
                direct_condition=direct_condition,
                rir_ids=rir_ids,
                rir_audio_paths=rir_audio_paths,
            )
        )
    return tuple(candidates)


def load_codec_candidates(
    *,
    plan_path: Path | str,
    suite_spec: CorruptedDevSuiteSpec,
) -> tuple[CodecSuiteCandidate, ...]:
    plan = load_codec_bank_plan(plan_path)
    allowed_families = set(suite_spec.codec_families)
    candidates: list[CodecSuiteCandidate] = []
    for preset in plan.presets:
        if preset.family not in allowed_families:
            continue
        severity_weight = getattr(suite_spec.severity_weights, preset.severity)
        if severity_weight <= 0.0:
            continue
        candidates.append(
            CodecSuiteCandidate(
                candidate_id=preset.id,
                severity=preset.severity,
                weight=preset.sampling_weight(plan.severity_profiles) * severity_weight,
                family=preset.family,
                preset=preset,
            )
        )
    return tuple(candidates)


def load_distance_candidates(
    *,
    plan_path: Path | str,
    suite_spec: CorruptedDevSuiteSpec,
) -> tuple[DistanceSuiteCandidate, ...]:
    plan = load_far_field_bank_plan(plan_path)
    candidates: list[DistanceSuiteCandidate] = []
    for preset in plan.presets:
        field_weight = getattr(suite_spec.distance_field_weights, preset.field)
        severity = _severity_from_distance_field(preset.field)
        severity_weight = getattr(suite_spec.severity_weights, severity)
        if field_weight <= 0.0 or severity_weight <= 0.0:
            continue
        candidates.append(
            DistanceSuiteCandidate(
                candidate_id=preset.id,
                severity=severity,
                weight=preset.sampling_weight * field_weight * severity_weight,
                field=preset.field,
                preset=preset,
                render_settings=plan.render,
            )
        )
    return tuple(candidates)


def load_silence_candidates(
    *,
    suite_spec: CorruptedDevSuiteSpec,
    silence_config: SilenceAugmentationConfig,
) -> tuple[SilenceSuiteCandidate, ...]:
    if not _has_effective_silence_profile(silence_config):
        return ()
    candidates: list[SilenceSuiteCandidate] = []
    for severity, scale in (("light", 0.5), ("medium", 0.8), ("heavy", 1.0)):
        severity_weight = getattr(suite_spec.severity_weights, severity)
        if severity_weight <= 0.0:
            continue
        candidates.append(
            SilenceSuiteCandidate(
                candidate_id=f"silence-{severity}",
                severity=severity,
                weight=severity_weight,
                config=_scaled_silence_config(silence_config, scale=scale),
            )
        )
    return tuple(candidates)


def apply_noise_transform(
    *,
    waveform: np.ndarray,
    sample_rate_hz: int,
    candidate: NoiseSuiteCandidate,
    rng: random.Random,
    project_root: Path,
) -> TransformOutcome:
    noise_waveform = _load_audio_for_transform(
        path=resolve_project_path(str(project_root), candidate.audio_path),
        sample_rate_hz=sample_rate_hz,
    )
    aligned_noise = _match_waveform_length(
        noise_waveform, target_frames=waveform.shape[-1], rng=rng
    )
    snr_db = round(rng.uniform(candidate.snr_db_min, candidate.snr_db_max), 6)
    mixed = _mix_at_snr_db(signal=waveform, noise=aligned_noise, snr_db=snr_db)
    return TransformOutcome(
        waveform=_limit_peak(mixed),
        sample_rate_hz=sample_rate_hz,
        candidate_id=candidate.candidate_id,
        severity=candidate.severity,
        metadata={
            "corruption_category": candidate.category,
            "corruption_mix_mode": candidate.mix_mode,
            "corruption_tags": list(candidate.tags),
            "target_snr_db": snr_db,
        },
    )


def apply_reverb_transform(
    *,
    waveform: np.ndarray,
    sample_rate_hz: int,
    candidate: ReverbSuiteCandidate,
    rng: random.Random,
    project_root: Path,
) -> TransformOutcome:
    rir_index = rng.randrange(len(candidate.rir_audio_paths))
    rir_path = resolve_project_path(str(project_root), candidate.rir_audio_paths[rir_index])
    rir_waveform = _load_audio_for_transform(path=rir_path, sample_rate_hz=sample_rate_hz)
    rir_mono = rir_waveform.mean(axis=0)
    rir_energy = float(np.sqrt(np.mean(np.square(rir_mono), dtype=np.float64)))
    if rir_energy > 0.0:
        rir_mono = rir_mono / rir_energy
    source_mono = waveform.mean(axis=0)
    reverberated = np.convolve(source_mono, rir_mono, mode="full")
    output = _limit_peak(reverberated[np.newaxis, :].astype("float32"))
    return TransformOutcome(
        waveform=output,
        sample_rate_hz=sample_rate_hz,
        candidate_id=candidate.candidate_id,
        severity=candidate.severity,
        metadata={
            "rir_id": candidate.rir_ids[rir_index],
            "room_size": candidate.room_size,
            "field": candidate.field,
            "rt60_bucket": candidate.rt60_bucket,
            "direct_condition": candidate.direct_condition,
        },
    )


def apply_codec_transform(
    *,
    waveform: np.ndarray,
    sample_rate_hz: int,
    candidate: CodecSuiteCandidate,
    ffmpeg_path: str,
) -> TransformOutcome:
    with TemporaryDirectory() as tmpdir:
        temp_root = Path(tmpdir)
        input_path = temp_root / "source.wav"
        output_path = temp_root / "output.wav"
        _write_temp_waveform(
            path=input_path,
            waveform=waveform,
            sample_rate_hz=sample_rate_hz,
        )
        apply_codec_preset(
            input_path=input_path,
            output_path=output_path,
            preset=candidate.preset,
            final_sample_rate_hz=sample_rate_hz,
            ffmpeg_path=ffmpeg_path,
        )
        rendered = _load_audio_for_transform(path=output_path, sample_rate_hz=sample_rate_hz)
    return TransformOutcome(
        waveform=_limit_peak(rendered),
        sample_rate_hz=sample_rate_hz,
        candidate_id=candidate.candidate_id,
        severity=candidate.severity,
        metadata={
            "codec_family": candidate.family,
            "codec_name": candidate.preset.codec_name,
            "codec_tags": list(candidate.preset.tags),
        },
    )


def apply_distance_transform(
    *,
    waveform: np.ndarray,
    sample_rate_hz: int,
    candidate: DistanceSuiteCandidate,
) -> TransformOutcome:
    rendered = render_far_field_preset(
        waveform=waveform,
        sample_rate_hz=sample_rate_hz,
        preset=candidate.preset,
        render_settings=candidate.render_settings,
    )
    return TransformOutcome(
        waveform=_limit_peak(rendered.preview_waveform),
        sample_rate_hz=sample_rate_hz,
        candidate_id=candidate.candidate_id,
        severity=candidate.severity,
        metadata={
            "distance_field": candidate.field,
            "distance_meters": candidate.preset.distance_meters,
            "target_drr_db": candidate.preset.target_drr_db,
            "off_axis_angle_deg": candidate.preset.off_axis_angle_deg,
        },
    )


def apply_silence_transform(
    *,
    waveform: np.ndarray,
    sample_rate_hz: int,
    candidate: SilenceSuiteCandidate,
    rng: random.Random,
) -> TransformOutcome:
    rendered, decision = apply_silence_augmentation(
        waveform,
        sample_rate_hz=sample_rate_hz,
        config=candidate.config,
        rng=rng,
    )
    return TransformOutcome(
        waveform=_limit_peak(np.asarray(rendered, dtype=np.float32)),
        sample_rate_hz=sample_rate_hz,
        candidate_id=candidate.candidate_id,
        severity=candidate.severity,
        metadata={"silence_decision": decision.to_dict()},
    )


def pick_weighted_candidate[TWeighted: _WeightedCandidate](
    candidates: tuple[TWeighted, ...],
    *,
    rng: random.Random,
) -> TWeighted:
    if not candidates:
        raise ValueError("Expected at least one candidate.")
    total = sum(float(candidate.weight) for candidate in candidates)
    if total <= 0.0:
        raise ValueError("Candidate weights must sum to a positive number.")
    threshold = rng.random() * total
    cumulative = 0.0
    for candidate in candidates:
        cumulative += float(candidate.weight)
        if threshold <= cumulative:
            return candidate
    return candidates[-1]


def stable_rng(*, seed: int, namespace: str, item_id: str) -> random.Random:
    token = hashlib.sha256(f"{seed}:{namespace}:{item_id}".encode()).digest()
    derived_seed = int.from_bytes(token[:8], byteorder="big", signed=False)
    return random.Random(derived_seed)


def _read_jsonl_records(path: Path) -> list[dict[str, object]]:
    return [
        payload
        for line in path.read_text().splitlines()
        if line.strip()
        for payload in [json.loads(line)]
        if isinstance(payload, dict)
    ]


def _load_audio_for_transform(*, path: Path, sample_rate_hz: int) -> np.ndarray:
    waveform, info = read_audio_file(path)
    output = np.asarray(waveform, dtype=np.float32)
    if info.sample_rate_hz != sample_rate_hz:
        output = resample_waveform(output, orig_freq=info.sample_rate_hz, new_freq=sample_rate_hz)
    if output.shape[0] > 1:
        output = output.mean(axis=0, keepdims=True)
    return output.astype("float32", copy=False)


def _write_temp_waveform(*, path: Path, waveform: np.ndarray, sample_rate_hz: int) -> None:
    from kryptonite.data.audio_io import write_audio_file

    write_audio_file(
        path=path,
        waveform=waveform,
        sample_rate_hz=sample_rate_hz,
        output_format="wav",
        pcm_bits_per_sample=16,
    )


def _match_waveform_length(
    waveform: np.ndarray,
    *,
    target_frames: int,
    rng: random.Random,
) -> np.ndarray:
    if waveform.shape[-1] == target_frames:
        return waveform.astype("float32", copy=False)
    if waveform.shape[-1] > target_frames:
        start = rng.randrange(waveform.shape[-1] - target_frames + 1)
        return waveform[:, start : start + target_frames].astype("float32", copy=False)
    repeats = math.ceil(target_frames / waveform.shape[-1])
    tiled = np.tile(waveform, (1, repeats))
    return tiled[:, :target_frames].astype("float32", copy=False)


def _mix_at_snr_db(*, signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    signal_mono = signal.mean(axis=0)
    noise_mono = noise.mean(axis=0)
    signal_rms = _rms(signal_mono)
    noise_rms = _rms(noise_mono)
    if signal_rms <= 0.0 or noise_rms <= 0.0:
        return signal.astype("float32", copy=False)
    target_noise_rms = signal_rms / (10.0 ** (snr_db / 20.0))
    scaled_noise = noise * (target_noise_rms / noise_rms)
    return (signal + scaled_noise).astype("float32", copy=False)


def _rms(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values), dtype=np.float64)))


def _limit_peak(waveform: np.ndarray, *, peak_limit: float = 0.98) -> np.ndarray:
    peak = float(np.max(np.abs(waveform)))
    if peak <= 0.0 or peak <= peak_limit:
        return waveform.astype("float32", copy=False)
    return (waveform * (peak_limit / peak)).astype("float32", copy=False)


def _coerce_str(value: object) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    return str(value)


def _coerce_float(value: object, *, default: float) -> float:
    if value is None:
        return default
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        raise TypeError(f"Expected numeric value, got {type(value)!r}")
    return float(value)


def _coerce_string_list(value: object) -> list[str]:
    if value is None:
        return []
    if not isinstance(value, list):
        return []
    return [str(item) for item in value if str(item).strip()]


def _coerce_severity(value: object) -> SuiteSeverity | None:
    normalized = _coerce_str(value)
    if normalized in {"light", "medium", "heavy"}:
        return cast(SuiteSeverity, normalized)
    return None


def _severity_from_direct_condition(value: str) -> SuiteSeverity:
    return cast(
        SuiteSeverity,
        {
            "high": "light",
            "medium": "medium",
            "low": "heavy",
        }.get(value, "medium"),
    )


def _severity_from_distance_field(value: str) -> SuiteSeverity:
    return cast(
        SuiteSeverity,
        {
            "near": "light",
            "mid": "medium",
            "far": "heavy",
        }.get(value, "medium"),
    )


def _has_effective_silence_profile(config: SilenceAugmentationConfig) -> bool:
    return any(
        (
            config.max_leading_padding_seconds > 0.0,
            config.max_trailing_padding_seconds > 0.0,
            config.max_inserted_pauses > 0 and config.max_inserted_pause_seconds > 0.0,
            config.pause_ratio_min < 1.0,
            config.pause_ratio_max > 1.0,
        )
    )


def _scaled_silence_config(
    config: SilenceAugmentationConfig,
    *,
    scale: float,
) -> SilenceAugmentationConfig:
    max_inserted_pauses = 0
    if config.max_inserted_pauses > 0:
        max_inserted_pauses = max(1, round(config.max_inserted_pauses * scale))
    return SilenceAugmentationConfig(
        enabled=True,
        max_leading_padding_seconds=round(config.max_leading_padding_seconds * scale, 6),
        max_trailing_padding_seconds=round(config.max_trailing_padding_seconds * scale, 6),
        max_inserted_pauses=max_inserted_pauses,
        min_inserted_pause_seconds=round(config.min_inserted_pause_seconds, 6),
        max_inserted_pause_seconds=round(config.max_inserted_pause_seconds * scale, 6),
        pause_ratio_min=round(_scale_pause_ratio(config.pause_ratio_min, scale), 6),
        pause_ratio_max=round(_scale_pause_ratio(config.pause_ratio_max, scale), 6),
        min_detected_pause_seconds=round(config.min_detected_pause_seconds, 6),
        max_perturbed_pause_seconds=round(
            config.min_detected_pause_seconds
            + (config.max_perturbed_pause_seconds - config.min_detected_pause_seconds) * scale,
            6,
        ),
        analysis_frame_ms=round(config.analysis_frame_ms, 6),
        silence_threshold_dbfs=round(config.silence_threshold_dbfs, 6),
    )


def _scale_pause_ratio(value: float, scale: float) -> float:
    if value >= 1.0:
        return 1.0 + ((value - 1.0) * scale)
    return 1.0 - ((1.0 - value) * scale)
